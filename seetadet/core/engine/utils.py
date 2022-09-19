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
"""Engine utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import importlib.machinery
import os

import dragon
from dragon.core.framework import backend
from dragon.vm import torch


def count_params(module):
    """Return the number of parameters in MB."""
    return sum([v.size().numel() for v in module.parameters()]) / 1e6


def freeze_module(module):
    """Freeze parameters of given module."""
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def get_device(index):
    """Create the available device object."""
    if torch.cuda.is_available():
        return torch.device('cuda', index)
    try:
        if torch.backends.mps.is_available():
            return torch.device('mps', index)
    except AttributeError:
        pass
    return torch.device('cpu')


def get_param_groups(module, lr_scale_getter=None):
    """Separate parameters into groups."""
    memo, groups = {}, collections.OrderedDict()
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        attrs = collections.OrderedDict()
        if lr_scale_getter:
            attrs['lr_scale'] = lr_scale_getter(name)
        memo[name] = param.shape
        no_weight_decay = not (name.endswith('weight') and param.dim() > 1)
        no_weight_decay = getattr(param, 'no_weight_decay', no_weight_decay)
        if no_weight_decay:
            attrs['weight_decay'] = 0
        group_name = '/'.join(['%s:%s' % (v[0], v[1]) for v in list(attrs.items())])
        if group_name not in groups:
            groups[group_name] = {'params': []}
            groups[group_name].update(attrs)
        groups[group_name]['params'].append(param)
    return list(groups.values())


def load_library(library_prefix):
    """Load a shared library."""
    loader_details = (importlib.machinery.ExtensionFileLoader,
                      importlib.machinery.EXTENSION_SUFFIXES)
    library_prefix = os.path.abspath(library_prefix)
    lib_dir, fullname = os.path.split(library_prefix)
    finder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = finder.find_spec(fullname)
    if ext_specs is None:
        raise ImportError('Could not find the pre-built library '
                          'for <%s>.' % library_prefix)
    backend.load_library(ext_specs.origin)


def synchronize_device(device):
    """Synchronize the computation of device."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        dragon.mps.synchronize(device.index)
