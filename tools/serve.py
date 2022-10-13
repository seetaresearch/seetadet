# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Serve a detection network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import os
import multiprocessing as mp
import time

import codewithgpu
import numpy as np

from seetadet.core.config import cfg
from seetadet.core.coordinator import Coordinator
from seetadet.core.engine import test_engine
from seetadet.utils import logging
from seetadet.utils import profiler
from seetadet.utils.mask import encode_masks
from seetadet.utils.mask import paste_masks
from seetadet.utils.vis import Visualizer


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description='Serve a detection network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        default=None,
        help='config file')
    parser.add_argument(
        '--exp_dir',
        default='',
        help='experiment dir')
    parser.add_argument(
        '--iter',
        type=int,
        default=None,
        help='iteration of checkpoint')
    parser.add_argument(
        '--model_dir',
        default='',
        help='model dir')
    parser.add_argument(
        '--score_thresh',
        type=float,
        default=0.7,
        help='score threshold for inference')
    parser.add_argument(
        '--batch_timeout',
        type=float,
        default=1,
        help='timeout to wait for a batch')
    parser.add_argument(
        '--queue_size',
        type=int,
        default=512,
        help='size of the memory queue')
    parser.add_argument(
        '--gpu',
        nargs='+',
        type=int,
        default=None,
        help='index of GPUs to use')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='set cudnn deterministic or not')
    parser.add_argument(
        '--app',
        default='gradio',
        help='application framework')
    parser.add_argument(
        '--processes',
        type=int,
        default=1,
        help='number of flask processes')
    parser.add_argument(
        '--port',
        type=int,
        default=5050,
        help='listening port')
    return parser.parse_args()


class ServingCommand(codewithgpu.ServingCommand):
    """Command to run serving."""

    def __init__(self, output_queue, score_thresh=0.7, perf_every=100):
        super(ServingCommand, self).__init__(app_library='flask')
        self.output_queue = output_queue
        self.output_dict = mp.Manager().dict()
        self.score_thresh = score_thresh
        self.perf_every = perf_every
        self.classes = cfg.MODEL.CLASSES
        self.max_dets = cfg.TEST.DETECTIONS_PER_IM

    def make_objects(self, outputs):
        """Main the detection objects."""
        boxes = outputs.pop('boxes')
        masks = outputs.pop('masks', None)
        objects = []
        for j, name in enumerate(self.classes):
            if name == '__background__':
                continue
            inds = np.where(boxes[j][:, 4] > self.score_thresh)[0]
            if len(inds) == 0:
                continue
            for box in boxes[j][inds]:
                objects.append({'bbox': box[:4].astype(float).tolist(),
                                'score': float(box[4]), 'class': name})
            if masks is not None:
                rles = encode_masks(paste_masks(
                    masks[j][inds], boxes[j][inds], outputs['im_shape'][:2]))
                for i, rle in enumerate(rles[::-1]):
                    objects[-(i + 1)]['segmentation'] = rle
        return objects

    def run(self):
        """Main loop to make the serving outputs."""
        count, timers = 0, collections.defaultdict(profiler.Timer)
        while True:
            count += 1
            img_id, time_diffs, outputs = self.output_queue.get()
            outputs = test_engine.filter_outputs(outputs, self.max_dets)
            for name, diff in time_diffs.items():
                timers[name].add_diff(diff)
            self.output_dict[img_id] = self.make_objects(outputs)
            if count % self.perf_every == 0:
                logging.info('im_detect: {:d} [{:.3f}s + {:.3f}s]'
                             .format(count, timers['im_detect'].average_time,
                                     timers['misc'].average_time))


def find_weights(args, coordinator):
    """Return the weights for serving."""
    weights_list = []
    if args.model_dir:
        for file in os.listdir(args.model_dir):
            if file.endswith('.pkl'):
                weights_list.append(os.path.join(args.model_dir, file))
    else:
        checkpoint, _ = coordinator.get_checkpoint(args.iter)
        weights_list.append(checkpoint)
    return weights_list[0]


def build_flask_app(queues, command):
    """Build the flask application."""
    import flask
    app = flask.Flask('seetadet.serve')
    logging._logging.getLogger('werkzeug').setLevel('ERROR')
    debug_objects = os.environ.get('FLASK_DEBUG', False)

    @app.route("/upload", methods=['POST'])
    def upload():
        img_id, img = command.get_image()
        queues[img_id % len(queues)].put((img_id, img))
        return flask.jsonify({'image_id': img_id})

    @app.route("/get", methods=['POST'])
    def get():
        def try_get(retry_time=0.005):
            try:
                req = flask.request.get_json(force=True)
                img_id = req['image_id']
            except KeyError:
                err_msg, img_id = 'Not found "image_id" in data.', ''
                flask.abort(flask.Response(err_msg))
            while img_id not in command.output_dict:
                time.sleep(retry_time)
            return img_id, command.output_dict.pop(img_id)
        img_id, objects = try_get(retry_time=0.005)
        msg = 'ImageId = %d, #Detects = %d' % (img_id, len(objects))
        if debug_objects:
            msg += (('\n * ' if len(objects) > 0 else '') +
                    ('\n * '.join(str(obj) for obj in objects)))
        logging.info(msg)
        return flask.jsonify({'objects': objects})
    return app


def build_gradio_app(queues, command):
    """Build the gradio application."""
    import cv2
    import gradio
    visualizer = Visualizer(class_names=command.classes, score_thresh=0.0)

    def upload_and_get(img_path):
        with command.example_id.get_lock():
            command.example_id.value += 1
            img_id = command.example_id.value
        img = cv2.imread(img_path)
        queues[img_id % len(queues)].put((img_id, img))
        while img_id not in command.output_dict:
            time.sleep(0.005)
        objects = command.output_dict.pop(img_id)
        logging.info('ImageId = %d, #Detects = %d' % (img_id, len(objects)))
        vis_img = visualizer.draw_objects(img, objects).get_image(rgb=True)
        objects_list = [(i, obj['class'], round(obj['score'], 3),
                         str(np.round(obj['bbox'], 2).tolist()))
                        for i, obj in enumerate(objects)]
        return vis_img, objects_list

    app = gradio.Interface(
        fn=upload_and_get,
        inputs=gradio.Image(type='filepath', label='Image', show_label=False),
        outputs=[gradio.Image(label='Visualization', show_label=False),
                 gradio.Dataframe(headers=['Id', 'Category', 'Score', 'BBox'],
                                  label='Objects')],
        examples=['../data/images/' + x for x in os.listdir('../data/images')],
        allow_flagging='never')
    app.temp_dirs.add('../data/images')
    return app


if __name__ == '__main__':
    logging.set_formatter("%(asctime)s %(levelname)s %(message)s")

    args = parse_args()
    logging.info('Called with args:\n' + str(args))

    coordinator = Coordinator(args.cfg_file, args.exp_dir or args.model_dir)
    logging.info('Using config:\n' + str(cfg))

    # Build actors.
    weights = find_weights(args, coordinator)
    devices = args.gpu if args.gpu else [cfg.GPU_ID]
    num_devices = len(devices)
    queues = [mp.Queue(args.queue_size) for _ in range(num_devices + 1)]
    commands = [test_engine.InferenceCommand(
        queues[i], queues[-1], kwargs={
            'cfg': cfg,
            'device': devices[i],
            'weights': weights,
            'deterministic': args.deterministic,
            'batch_timeout': args.batch_timeout,
            'verbose': i == 0,
        }) for i in range(num_devices)]
    commands += [ServingCommand(queues[-1], score_thresh=args.score_thresh)]
    actors = [mp.Process(target=command.run) for command in commands]
    for actor in actors:
        actor.start()

    # Build app.
    if args.app == 'flask':
        app = build_flask_app(queues[:-1], commands[-1])
        app.run(port=args.port, threaded=args.processes == 1,
                processes=args.processes)
    elif args.app == 'gradio':
        app = build_gradio_app(queues[:-1], commands[-1])
        app.queue(concurrency_count=args.processes)
        app.launch(server_port=args.port)
    else:
        raise ValueError('Unsupported application framework: ' + args.app)
