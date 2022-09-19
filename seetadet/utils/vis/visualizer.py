# ------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#     <https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/visualizer.py>
#
# ------------------------------------------------------------
"""Visualizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import matplotlib.backends.backend_agg
import matplotlib.colors
import matplotlib.figure
import matplotlib.patches
import matplotlib.pyplot
import numpy as np

from seetadet.utils.mask import mask_from
from seetadet.utils.mask import mask_to_polygons
from seetadet.utils.mask import paste_masks
from seetadet.utils.vis.colormap import colormap

_SMALL_OBJECT_AREA_THRESH = 1000


class VisImage(object):
    """VisImage."""

    def __init__(self, img, scale=1.0):
        self.img = img
        self.scale = scale
        self.shape = (h, w) = img.shape[:2]
        self.font_size = max(np.sqrt(h * w) // 90, 10 // scale)
        self._setup_figure(img)

    def _setup_figure(self, img):
        fig = matplotlib.figure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        fig.set_size_inches((self.shape[1] * self.scale + 1e-2) / self.dpi,
                            (self.shape[0] * self.scale + 1e-2) / self.dpi)
        self.canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis('off')
        self.fig = fig
        self.ax = ax
        self.ax.imshow(img)

    def save(self, filepath):
        cv2.imwrite(filepath, self.get_image())

    def get_image(self, rgb=False):
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        buffer = np.frombuffer(s, dtype='uint8')
        img_rgba = buffer.reshape(height, width, 4)
        img_rgb, _ = np.split(img_rgba, [3], axis=2)
        img_rgb = img_rgb.astype('uint8', copy=False)
        return img_rgb if rgb else img_rgb[:, :, ::-1]


class Visualizer(object):
    """"Visualizer."""

    def __init__(self, class_names=None, score_thresh=0.7):
        self.class_names = class_names
        self.score_thresh = score_thresh
        self.colormap = colormap(rgb=True) / 255.
        self.output = None

    def _convert_from_dict_format(self, objects):
        boxes, masks, labels = [], [], []
        for obj in objects:
            score = obj.get('score', 1.0)
            name = obj.get('class', 'object')
            if score < self.score_thresh:
                continue
            boxes.append(list(obj['bbox']) + [score])
            labels.append('{} {:.0f}%'.format(name, score * 100))
            if 'segmentation' in obj:
                masks.append(mask_from(obj['segmentation']['counts'].encode(),
                                       obj['segmentation']['size']))
        boxes = np.array(boxes, 'float32') if len(boxes) > 0 else boxes
        masks = np.stack(masks) if len(masks) > 0 else masks
        return boxes, masks, labels

    def _convert_from_cls_format(self, cls_boxes=None, cls_masks=None):
        boxes, masks, labels = [], [], []
        for i, name in enumerate(self.class_names):
            if name == '__background__':
                continue
            if cls_boxes is not None and len(cls_boxes[i]) > 0:
                boxes.append(cls_boxes[i])
                scores = cls_boxes[i][:, -1].tolist()
                labels += ['{} {:.0f}%'.format(name, s * 100) for s in scores]
            if cls_masks is not None and len(cls_masks[i]):
                masks.append(cls_masks[i])
        boxes = np.concatenate(boxes) if len(boxes) > 0 else boxes
        masks = np.concatenate(masks) if len(masks) > 0 else masks
        return boxes, masks, labels

    def overlay_instances(self, boxes, masks, labels):
        """Overlay instances."""
        if boxes is None or len(boxes) == 0:
            return self.output
        # Filter instances.
        keep = np.where(boxes[:, -1] > self.score_thresh)[0]
        if len(keep) == 0:
            return self.output
        boxes, labels = boxes[keep], [labels[i] for i in keep]
        masks = masks[keep] if len(masks) > 0 else []
        # Paste masks.
        if len(masks) > 0 and masks.shape[-2:] != self.output.shape[:2]:
            masks = paste_masks(masks, boxes, self.output.shape[:2],
                                channels_last=False)
        # Display in largest to smallest order to reduce occlusion.
        if boxes.shape[1] == 5:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        elif boxes.shape[1] == 6:
            areas = boxes[:, 2] * boxes[:, 3]
        else:
            raise ValueError('Excepted box4d or box5d.')
        keep = np.argsort(-areas)
        boxes, labels = boxes[keep], [labels[i] for i in keep]
        masks = masks[keep] if len(masks) > 0 else []
        colors = self.colormap[np.arange(len(boxes)) % len(self.colormap)]
        for i, box in enumerate(boxes):
            if boxes.shape[1] == 5:
                self.draw_box(box, edge_color=colors[i])
            self.draw_box_label(box, labels[i])
            if len(masks) > 0:
                polygons = mask_to_polygons(masks[i])
                for p in polygons:
                    self.draw_polygon(p.reshape((-1, 2)), color=colors[i])
        return self.output

    def draw_instances(self, img, boxes, masks):
        """Draw instances."""
        self.output = VisImage(img[:, :, ::-1])
        assert len(boxes) == len(self.class_names)
        boxes, masks, labels = self._convert_from_cls_format(boxes, masks)
        self.overlay_instances(boxes, masks, labels)
        return self.output

    def draw_objects(self, img, objects):
        """Draw objects."""
        self.output = VisImage(img[:, :, ::-1])
        boxes, masks, labels = self._convert_from_dict_format(objects)
        self.overlay_instances(boxes, masks, labels)
        return self.output

    def draw_box(self, box, alpha=0.5, edge_color='g', line_style='-'):
        """Draw box."""
        x0, y0, x1, y1 = box[:4]
        width, height = x1 - x0, y1 - y0
        line_width = max(self.output.font_size / 4, 1)
        self.output.ax.add_patch(
            matplotlib.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=line_width * self.output.scale,
                alpha=alpha,
                linestyle=line_style))
        return self.output

    def draw_box_label(self, box, label):
        """Draw box label."""
        x0, y0, x1, y1 = box[:4]
        text_pos = (x0, y0)
        instance_area = (y1 - y0) * (x1 - x0)
        if (instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                or y1 - y0 < 40 * self.output.scale):
            if y1 >= self.output.shape[0] - 5:
                text_pos = (x1, y0)
            else:
                text_pos = (x0, y1)
        height_ratio = (y1 - y0) / np.sqrt(self.output.shape[0] * self.output.shape[1])
        font_size = (np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                     * 0.5 * self.output.font_size)
        self.draw_text(label, text_pos, font_size=font_size)
        return self.output

    def draw_text(
        self,
        text,
        position,
        font_size=None,
        color='w',
        horizontal_alignment='left',
        rotation=0,
    ):
        """Draw text."""
        if not font_size:
            font_size = self.output.font_size
        color = np.maximum(list(matplotlib.colors.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))
        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family='sans-serif',
            bbox={'facecolor': 'black', 'alpha': 0.8,
                  'pad': 0, 'edgecolor': 'none'},
            verticalalignment='top',
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation)
        return self.output

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        """Draw polygon."""
        edge_color = edge_color or color
        edge_color = matplotlib.colors.to_rgb(edge_color) + (1,)
        polygon = matplotlib.patches.Polygon(
            segment,
            fill=True,
            facecolor=matplotlib.colors.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            linewidth=max(self.output.font_size // 15 * self.output.scale, 1))
        self.output.ax.add_patch(polygon)
        return self.output
