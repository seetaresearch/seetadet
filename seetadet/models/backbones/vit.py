# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""ViT backbone."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from dragon.vm import torch
from dragon.vm.torch import nn

from seetadet.core.config import cfg
from seetadet.models.build import BACKBONES
from seetadet.ops.build import build_norm


def call_inplace(x, method, *args):
    """Call the instance method in-place if required."""
    if x.requires_grad and cfg.TRAIN.COMPRESS_MEMORY:
        return getattr(x, method + '_')(*args)
    return getattr(x, method)(*args)


def space_to_depth(input, block_size):
    """Rearrange blocks of spatial data into depth."""
    if input.dim() == 3:
        hXw, c = input.size()[1:]
        h = w = int(hXw ** 0.5)
    else:
        h, w, c = input.size()[1:]
    h1, w1 = h // block_size, w // block_size
    c1 = (block_size ** 2) * c
    input.reshape_((-1, h1, block_size, w1, block_size, c))
    out = call_inplace(input, 'permute', 0, 1, 3, 2, 4, 5)
    input.reshape_((-1, h, w, c))
    return out.reshape_((-1, h1, w1, c1))


def depth_to_space(input, block_size):
    """Rearrange blocks of depth data into spatial."""
    h1, w1, c1 = input.size()[1:]
    h, w = h1 * block_size, w1 * block_size
    c = c1 // (block_size ** 2)
    input.reshape_((-1, h1, w1, block_size, block_size, c))
    out = call_inplace(input, 'permute', 0, 1, 3, 2, 4, 5)
    input.reshape_((-1, h1, w1, c1))
    return out.reshape_((-1, h, w, c))


class MLP(nn.Module):
    """Two layers MLP."""

    def __init__(self, dim, mlp_ratio=4):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.activation = nn.GELU()

    def forward(self, x):
        if x.requires_grad and cfg.TRAIN.COMPRESS_MEMORY:
            return torch.utils.checkpoint.checkpoint_sequential(
                [self.fc1, self.activation, self.fc2], x)
        return self.fc2(self.activation(self.fc1(x)))


class Attention(nn.Module):
    """Multihead attention."""

    def __init__(self, dim, num_heads, qkv_bias=True):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        qkv_shape = (-1, x.size(1), 3, self.num_heads, self.head_dim)
        qkv = call_inplace(self.qkv(x).reshape_(qkv_shape), 'permute', 2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0, copy=x.device.type == 'mps')
        attn = q @ call_inplace(k, 'transpose', -2, -1).mul_(self.scale)
        attn = nn.functional.softmax(attn, dim=-1, inplace=True)
        return self.proj(call_inplace(attn @ v, 'transpose', 1, 2).flatten_(2))


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=True, drop_path=0):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio)
        self.drop_path = nn.DropPath(p=drop_path, inplace=True)

    def forward(self, x):
        x = self.drop_path(self.attn(self.norm1(x))).add_(x)
        return self.drop_path(self.mlp(self.norm2(x))).add_(x)


class Bottleneck(nn.Module):
    """The bottleneck block."""

    def __init__(self, dim, expansion=2, width=None):
        super(Bottleneck, self).__init__()
        width = width or dim // expansion
        self.conv1 = nn.Conv2d(dim, width, 1, bias=False)
        self.norm1 = build_norm(width, cfg.BACKBONE.NORM)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.norm2 = build_norm(width, cfg.BACKBONE.NORM)
        self.conv3 = nn.Conv2d(width, dim, 1, bias=False)
        self.norm3 = build_norm(dim, cfg.BACKBONE.NORM)
        self.activation = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        return self.norm3(self.conv3(x)).add_(shortcut)


class PatchEmbed(nn.Module):
    """Patch embedding layer."""

    def __init__(self, dim=768, patch_size=16):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d(3, dim, patch_size, patch_size)

    def forward(self, x):
        return self.proj(x)


class PosEmbed(nn.Module):
    """Position embedding layer."""

    def __init__(self, dim, num_patches):
        super(PosEmbed, self).__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.weight = nn.Parameter(torch.zeros(num_patches, dim))
        self.weight.no_weight_decay = True
        nn.init.normal_(self.weight, std=0.02)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        weight = state_dict[prefix + 'weight']
        num_patches, dim = weight.shape
        if num_patches != self.num_patches:
            h = w = int(num_patches ** 0.5)
            new_h = new_w = int(self.num_patches ** 0.5)
            if not isinstance(weight, torch.Tensor):
                weight = torch.from_numpy(weight)
            weight = weight.reshape_(1, h, w, dim).permute(0, 3, 1, 2)
            weight = nn.functional.interpolate(
                weight, size=(new_h, new_w), mode='bilinear')
            weight = weight.flatten_(2).transpose(1, 2).squeeze_(0)
            state_dict[prefix + 'weight'] = weight
        super(PosEmbed, self)._load_from_state_dict(
            state_dict,
            prefix,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        return x.add_(self.weight)


class SimpleFeaturePyramid(nn.Module):
    """Module to create pyramid features."""

    def __init__(self, dim, patch_size):
        super(SimpleFeaturePyramid, self).__init__()
        self.output_conv = nn.ModuleList()
        patch_strides = int(math.log2(patch_size))
        for i in range(4):
            if i + 2 < patch_strides:
                stride, layers = 2 ** (patch_strides - i - 2), []
                while stride > 1:
                    layers += [nn.ConvTranspose2d(dim, dim, 2, 2)]
                    if stride > 2:
                        layers += [build_norm(dim, cfg.BACKBONE.NORM), nn.GELU()]
                    stride /= 2
                self.output_conv.append(nn.Sequential(*layers))
            elif i + 2 == patch_strides:
                self.output_conv += [nn.Identity()]
            elif i + 2 > patch_strides:
                stride = 2 ** (i + 2 - patch_strides)
                self.output_conv += [nn.MaxPool2d(stride, stride)]

    def forward(self, inputs):
        inputs = inputs + [inputs[-1]] * (4 - len(inputs))
        return [conv(x) for conv, x in zip(self.output_conv, inputs)]


class VisionTransformer(nn.Module):
    """Vision Transformer."""

    def __init__(self, depth, dim, num_heads, patch_size, window_size):
        super(VisionTransformer, self).__init__()
        drop_path = cfg.BACKBONE.DROP_PATH_RATE
        drop_path = (torch.linspace(
            0, drop_path, depth, dtype=torch.float32).tolist()
            if drop_path > 0 else [drop_path] * depth)
        image_size = cfg.TRAIN.CROP_SIZE
        self.window_size = window_size or image_size // patch_size
        self.num_windows = (image_size // patch_size // self.window_size) ** 2
        self.patch_embed = PatchEmbed(dim, patch_size)
        self.pos_embed = PosEmbed(dim, (image_size // patch_size) ** 2)
        self.blocks = nn.ModuleList(Block(dim, num_heads, drop_path=p) for p in drop_path)
        self.norm = nn.LayerNorm(dim)
        self.cross_conv = nn.ModuleList(Bottleneck(dim) for _ in range(4))
        self.cross_indices = list(range(depth // 4 - 1, depth, depth // 4))
        self.fpn = SimpleFeaturePyramid(dim, patch_size)
        self.out_indices, self.out_dims = [depth - 1] * 4, (dim,) * 5
        self.reset_parameters()

    def reset_parameters(self):
        gelu_approximate = 'none'
        if torch.backends.mps.is_available():
            gelu_approximate = 'tanh'
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GELU):
                m.approximate = gelu_approximate
        for m in self.cross_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, Bottleneck):
                nn.init.constant_(m.norm3.weight, 0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten_(2).transpose(1, 2)
        x = self.pos_embed(x)
        x = space_to_depth(x, self.window_size)
        wmsa_shape = (-1,) + x.shape[1:]
        msa_shape = (-1, self.window_size ** 2, self.out_dims[0])
        x = x.reshape_(msa_shape)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.cross_indices or i == len(self.blocks) - 1:
                x = self.norm(x) if i == len(self.blocks) - 1 else x
                x = depth_to_space(x.reshape_(wmsa_shape), self.window_size)
                x = call_inplace(x, 'permute', 0, 3, 1, 2)
            if i in self.cross_indices:
                x = self.cross_conv[self.cross_indices.index(i)](x)
            if i in self.cross_indices and i < len(self.blocks) - 1:
                x = call_inplace(x, 'permute', 0, 2, 3, 1)
                x = space_to_depth(x, self.window_size).reshape_(msa_shape)
        return [None] + self.fpn([x])

    def get_lr_scale(self, name, decay):
        values = list(decay ** (len(self.blocks) + 1 - i)
                      for i in range(len(self.blocks) + 2))
        if name.startswith('backbone.pos_embed'):
            return values[0]
        elif name.startswith('backbone.patch_embed'):
            return values[0]
        elif name.startswith('backbone.blocks'):
            return values[int(name.split('.')[2]) + 1]
        return values[-1]


BACKBONES.register('vit_small_patch16_window16', VisionTransformer,
                   depth=12, dim=384, num_heads=6,
                   patch_size=16, window_size=16)

BACKBONES.register('vit_base_patch16_window16', VisionTransformer,
                   depth=12, dim=768, num_heads=12,
                   patch_size=16, window_size=16)

BACKBONES.register('vit_large_patch16_window16', VisionTransformer,
                   depths=24, dim=1024, num_heads=16,
                   patch_size=16, window_size=16)
