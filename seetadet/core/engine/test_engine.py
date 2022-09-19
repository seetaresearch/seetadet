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
"""Testing engine."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import datetime
import multiprocessing as mp

import codewithgpu
from dragon.vm import torch
import numpy as np

from seetadet.core.config import cfg
from seetadet.data.build import build_evaluator
from seetadet.models.build import build_detector
from seetadet.modules.build import build_inference
from seetadet.utils import logging
from seetadet.utils import profiler
from seetadet.utils import vis


class InferenceCommand(codewithgpu.InferenceCommand):
    """Command to run inference."""

    def __init__(self, input_queue, output_queue, kwargs):
        super(InferenceCommand, self).__init__(input_queue, output_queue)
        self.kwargs = kwargs

    def build_env(self):
        """Build the environment."""
        cfg.merge_from_other_cfg(self.kwargs['cfg'])
        cfg.GPU_ID = self.kwargs['device']
        cfg.freeze()
        logging.set_root(self.kwargs.get('verbose', True))
        self.batch_size = cfg.TEST.IMS_PER_BATCH
        self.batch_timeout = self.kwargs.get('batch_timeout', None)
        if self.kwargs.get('deterministic', False):
            torch.backends.cudnn.deterministic = True

    def build_model(self):
        """Build and return the model."""
        return build_detector(self.kwargs['device'], self.kwargs['weights'])

    def build_module(self, model):
        """Build and return the inference module."""
        return build_inference(model)

    def send_results(self, module, indices, imgs):
        """Send the batch results."""
        results = module.get_results(imgs)
        time_diffs = module.get_time_diffs()
        time_diffs['im_detect'] += time_diffs.pop('im_detect_mask', 0.)
        for i, outputs in enumerate(results):
            outputs['im_shape'] = imgs[i].shape
            self.output_queue.put((indices[i], time_diffs, outputs))


def filter_outputs(outputs, max_dets=100):
    """Limit the max number of detections."""
    if max_dets <= 0:
        return outputs
    boxes = outputs.pop('boxes')
    masks = outputs.pop('masks', None)
    scores, num_classes = [], len(boxes)
    for i in range(num_classes):
        if len(boxes[i]) > 0:
            scores.append(boxes[i][:, -1])
    scores = np.hstack(scores) if len(scores) > 0 else []
    if len(scores) > max_dets:
        thr = np.sort(scores)[-max_dets]
        for i in range(num_classes):
            if len(boxes[i]) < 1:
                continue
            keep = np.where(boxes[i][:, -1] >= thr)[0]
            boxes[i] = boxes[i][keep]
            if masks is not None:
                masks[i] = masks[i][keep]
    outputs['boxes'] = boxes
    outputs['masks'] = masks
    return outputs


def extend_results(index, collection, results):
    """Add image results to the collection."""
    if results is None:
        return
    for _ in range(len(results) - len(collection)):
        collection.append([])
    for i in range(1, len(results)):
        for _ in range(index - len(collection[i]) + 1):
            collection[i].append([])
        collection[i][index] = results[i]


def run_test(
    test_cfg,
    weights,
    output_dir,
    devices,
    deterministic=False,
    read_every=100,
    vis_thresh=0,
    vis_output_dir=None,
):
    """Run a model testing.

    Parameters
    ----------
    test_cfg : CfgNode
        The cfg for testing.
    weights : str
        The path of model weights to load.
    output_dir : str
        The path to save results.
    devices : Sequence[int]
        The index of computing devices.
    deterministic : bool, optional, default=False
        Set cudnn deterministic or not.
    read_every : int, optional, default=100
        Read every N images to distribute to devices.
    vis_thresh : float, optional, default=0
        The score threshold for visualization.
    vis_output_dir : str, optional
        The path to save visualizations.

    """
    cfg.merge_from_other_cfg(test_cfg)
    evaluator = build_evaluator(output_dir)
    devices = devices if devices else [cfg.GPU_ID]
    num_devices = len(devices)
    num_images = evaluator.num_images
    max_dets = cfg.TEST.DETECTIONS_PER_IM
    read_stride = float(num_devices * cfg.TEST.IMS_PER_BATCH)
    read_every = int(np.ceil(read_every / read_stride) * read_stride)
    visualizer = vis.Visualizer(cfg.MODEL.CLASSES, vis_thresh)

    queues = [mp.Queue() for _ in range(num_devices + 1)]
    commands = [InferenceCommand(
        queues[i], queues[-1], kwargs={
            'cfg': test_cfg,
            'weights': weights,
            'device': devices[i],
            'deterministic': deterministic,
            'verbose': i == 0,
        }) for i in range(num_devices)]
    actors = [mp.Process(target=command.run) for command in commands]
    for actor in actors:
        actor.start()

    timers = collections.defaultdict(profiler.Timer)
    all_boxes, all_masks, vis_images = [], [], {}

    for count in range(1, num_images + 1):
        img_id, img = evaluator.get_image()
        queues[count % num_devices].put((count - 1, img))
        if vis_thresh > 0 and vis_output_dir:
            filename = vis_output_dir + '/%s.png' % img_id
            vis_images[count - 1] = (filename, img)
        if count % read_every > 0 and count < num_images:
            continue
        if count == num_images:
            for i in range(num_devices):
                queues[i].put((-1, None))
        for _ in range(((count - 1) % read_every + 1)):
            index, time_diffs, outputs = queues[-1].get()
            outputs = filter_outputs(outputs, max_dets)
            extend_results(index, all_boxes, outputs['boxes'])
            extend_results(index, all_masks, outputs.get('masks', None))
            for name, diff in time_diffs.items():
                timers[name].add_diff(diff)
            if vis_thresh > 0 and vis_output_dir:
                filename, img = vis_images[index]
                visualizer.draw_instances(
                    img=img,
                    boxes=outputs['boxes'],
                    masks=outputs.get('masks', None)).save(filename)
                del vis_images[index]
        avg_time = sum([t.average_time for t in timers.values()])
        eta_seconds = avg_time * (num_images - count)
        print('\rim_detect: {:d}/{:d} [{:.3f}s + {:.3f}s] (eta: {})'
              .format(count, num_images,
                      timers['im_detect'].average_time,
                      timers['misc'].average_time,
                      str(datetime.timedelta(seconds=int(eta_seconds)))),
              end='')

    print('\nEvaluating detections...')
    evaluator.eval_bbox(all_boxes)

    if len(all_masks) > 0:
        print('Evaluating segmentations...')
        evaluator.eval_segm(all_boxes, all_masks)
