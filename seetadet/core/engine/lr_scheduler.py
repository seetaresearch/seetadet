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
"""Learning rate schedulers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math


class ConstantLR(object):
    """Constant LR scheduler."""

    def __init__(self, **kwargs):
        self._lr_max = kwargs.pop('lr_max')
        self._lr_min = kwargs.pop('lr_min', 0)
        self._warmup_steps = kwargs.pop('warmup_steps', 0)
        self._warmup_factor = kwargs.pop('warmup_factor', 0)
        if kwargs:
            raise ValueError('Unexpected arguments: ' + ','.join(v for v in kwargs))
        self._step_count = 0
        self._last_decay = 1.

    def step(self):
        self._step_count += 1

    def get_lr(self):
        if self._step_count < self._warmup_steps:
            alpha = (self._step_count + 1.) / self._warmup_steps
            return self._lr_max * (alpha + (1. - alpha) * self._warmup_factor)
        return self._lr_min + (self._lr_max - self._lr_min) * self.get_decay()

    def get_decay(self):
        return self._last_decay


class CosineLR(ConstantLR):
    """LR scheduler with cosine decay."""

    def __init__(self, lr_max, max_steps, lr_min=0, decay_step=1, **kwargs):
        super(CosineLR, self).__init__(lr_max=lr_max, lr_min=lr_min, **kwargs)
        self._decay_step = decay_step
        self._max_steps = max_steps

    def get_decay(self):
        t = self._step_count - self._warmup_steps
        t_max = self._max_steps - self._warmup_steps
        if t > 0 and t % self._decay_step == 0:
            self._last_decay = .5 * (1. + math.cos(math.pi * t / t_max))
        return self._last_decay


class MultiStepLR(ConstantLR):
    """LR scheduler with multi-steps decay."""

    def __init__(self, lr_max, decay_steps, decay_gamma, **kwargs):
        super(MultiStepLR, self).__init__(lr_max=lr_max, **kwargs)
        self._decay_steps = decay_steps
        self._decay_gamma = decay_gamma
        self._stage_count = 0
        self._num_stages = len(decay_steps)

    def get_decay(self):
        if self._stage_count < self._num_stages:
            k = self._decay_steps[self._stage_count]
            while self._step_count >= k:
                self._stage_count += 1
                if self._stage_count >= self._num_stages:
                    break
                k = self._decay_steps[self._stage_count]
            self._last_decay = self._decay_gamma ** self._stage_count
        return self._last_decay


class LinearLR(ConstantLR):
    """LR scheduler with linear decay."""

    def __init__(self, lr_max, max_steps, lr_min=0, decay_step=1, **kwargs):
        super(LinearLR, self).__init__(lr_max=lr_max, lr_min=lr_min, **kwargs)
        self._decay_step = decay_step
        self._max_steps = max_steps

    def get_decay(self):
        t = self._step_count - self._warmup_steps
        t_max = self._max_steps - self._warmup_steps
        if t > 0 and t % self._decay_step == 0:
            self._last_decay = 1. - float(t) / t_max
        return self._last_decay


if __name__ == '__main__':
    def extract_label(scheduler):
        class_name = scheduler.__class__.__name__
        label = class_name + '('
        if class_name == 'CosineLR':
            label += 'α=' + str(scheduler._decay_step)
        elif class_name == 'LinearCosineLR':
            label += 'α=' + str(scheduler._decay_step)
        elif class_name == 'MultiStepLR':
            label += 'α=' + str(scheduler._decay_steps) + ', '
            label += 'γ=' + str(scheduler._decay_gamma)
        elif class_name == 'StepLR':
            label += 'α=' + str(scheduler._decay_step) + ', '
            label += 'γ=' + str(scheduler._decay_gamma)
        label += ')'
        return label

    vis = True
    max_steps = 120
    shared_args = {
        'lr_max': 0.0004,
        'warmup_steps': 0,
        'warmup_factor': 0.,
    }
    schedulers = [
        # CosineLR(lr_min=0., decay_step=1, max_steps=max_steps, **shared_args),
        CosineLR(lr_min=1e-6, decay_step=1, max_steps=140, **shared_args),
    ]

    for i in range(max_steps):
        info = 'Step = %d\n' % i
        for scheduler in schedulers:
            if i == 0:
                scheduler.lr_seq = []
            info += '  * {}: {}\n'.format(
                extract_label(scheduler),
                scheduler.get_lr())
            scheduler.lr_seq.append(scheduler.get_lr())
            scheduler.step()
        if not vis:
            print(info)

    if vis:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.title('Visualization of different LR Schedulers')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        line = '-'
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        for i, scheduler in enumerate(schedulers):
            plt.plot(
                range(max_steps),
                scheduler.lr_seq,
                colors[i] + line,
                linewidth=1.,
                label=extract_label(scheduler),
            )
        plt.legend()
        plt.grid(linestyle='--')
        plt.show()
        plt.savefig('x.png')
