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
"""Data loader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import multiprocessing as mp
import time
import threading
import queue

import codewithgpu
import dragon

from seetadet.core.config import cfg
from seetadet.data.build import build_dataset
from seetadet.utils import logging
from seetadet.utils.blob import blob_vstack


class BalancedQueues(object):
    """Balanced queues."""

    def __init__(self, base_queue, num=1):
        self.queues = [base_queue]
        self.queues += [mp.Queue(base_queue._maxsize) for _ in range(num - 1)]
        self.index = 0

    def put(self, obj, block=True, timeout=None):
        q = self.queues[self.index]
        q.put(obj, block=block, timeout=timeout)
        self.index = (self.index + 1) % len(self.queues)

    def get(self, block=True, timeout=None):
        q = self.queues[self.index]
        obj = q.get(block=block, timeout=timeout)
        self.index = (self.index + 1) % len(self.queues)
        return obj

    def get_n(self, num=1):
        outputs = []
        while len(outputs) < num:
            obj = self.get()
            if obj is not None:
                outputs.append(obj)
        return outputs


class DataLoaderBase(threading.Thread):
    """Base class of data loader."""

    def __init__(self, worker, **kwargs):
        super(DataLoaderBase, self).__init__(daemon=True)
        self.batch_size = kwargs.get('batch_size', 2)
        self.num_readers = kwargs.get('num_readers', 1)
        self.num_workers = kwargs.get('num_workers', 3)
        self.queue_depth = kwargs.get('queue_depth', 2)

        # Initialize distributed group.
        rank, group_size = 0, 1
        dist_group = dragon.distributed.get_group()
        if dist_group is not None:
            group_size = dist_group.size
            rank = dragon.distributed.get_rank(dist_group)

        # Build queues.
        self.reader_queue = mp.Queue(self.queue_depth * self.batch_size)
        self.worker_queue = mp.Queue(self.queue_depth * self.batch_size)
        self.batch_queue = queue.Queue(self.queue_depth)
        self.reader_queue = BalancedQueues(self.reader_queue, self.num_workers)
        self.worker_queue = BalancedQueues(self.worker_queue, self.num_workers)

        # Build readers.
        self.readers = []
        for i in range(self.num_readers):
            partition_id = i
            num_partitions = self.num_readers
            num_partitions *= group_size
            partition_id += rank * self.num_readers
            self.readers.append(codewithgpu.DatasetReader(
                output_queue=self.reader_queue,
                partition_id=partition_id,
                num_partitions=num_partitions,
                seed=cfg.RNG_SEED + partition_id, **kwargs))
            self.readers[i].start()
            time.sleep(0.1)

        # Build workers.
        self.workers = []
        for i in range(self.num_workers):
            p = worker(**kwargs)
            p.seed += (i + rank * self.num_workers)
            p.reader_queue = self.reader_queue.queues[i]
            p.worker_queue = self.worker_queue.queues[i]
            p.start()
            self.workers.append(p)
            time.sleep(0.1)

        # Register cleanup callbacks.
        def cleanup():
            def terminate(processes):
                for p in processes:
                    p.terminate()
                    p.join()
            terminate(self.workers)
            terminate(self.readers)

        import atexit
        atexit.register(cleanup)

        # Start batch prefetching.
        self.start()

    def next(self):
        """Return the next batch of data."""
        return self.__next__()

    def run(self):
        """Main loop."""

    def __call__(self):
        return self.next()

    def __iter__(self):
        """Return the iterator self."""
        return self

    def __next__(self):
        """Return the next batch of data."""
        return self.batch_queue.get()


class DataLoader(DataLoaderBase):
    """Loader to return the batch of data."""

    def __init__(self, dataset, worker, **kwargs):
        dataset = build_dataset(dataset)
        self.dataset_size = dataset.size
        self.contiguous = kwargs.get('contiguous', True)
        self.prefetch_count = kwargs.get('prefetch_count', 50)
        self.img_mean = cfg.MODEL.PIXEL_MEAN
        self.img_align = (cfg.BACKBONE.COARSEST_STRIDE,) * 2
        args = {'path': dataset.source,
                'dataset_getter': dataset.getter,
                'classes': dataset.classes,
                'shuffle': kwargs.get('shuffle', True),
                'batch_size': kwargs.get('batch_size', 1),
                'num_workers': kwargs.get('num_workers', 1)}
        super(DataLoader, self).__init__(worker, **args)

    def run(self):
        """Main loop."""
        logging.info('Prefetch batches...')
        prev_inputs = self.worker_queue.get_n(
            self.prefetch_count * self.batch_size)
        next_inputs = []

        while True:
            # Use cached buffer for next N inputs.
            if len(next_inputs) == 0:
                next_inputs = prev_inputs
                if 'aspect_ratio' in next_inputs[0]:
                    # Inputs are sorted for aspect ratio grouping.
                    next_inputs.sort(key=lambda d: d['aspect_ratio'][0] > 1)
                prev_inputs = []

            # Collect the next batch.
            outputs = collections.defaultdict(list)
            for _ in range(self.batch_size):
                inputs = next_inputs.pop(0)
                for k, v in inputs.items():
                    outputs[k].extend(v)
                prev_inputs += self.worker_queue.get_n(1)

            # Stack batch data.
            if self.contiguous:
                outputs['img'] = blob_vstack(
                    outputs['img'], fill_value=self.img_mean,
                    align=self.img_align)

            # Send batch data to consumer.
            self.batch_queue.put(outputs)
