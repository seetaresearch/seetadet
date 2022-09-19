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
"""Base dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codewithgpu

from seetadet.core.config import cfg
from seetadet.data.build import DATASETS


class Dataset(object):
    """Base dataset class."""

    def __init__(self, source):
        self.source = source
        self.classes = cfg.MODEL.CLASSES
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))

    @property
    def getter(self):
        """Return the dataset getter."""
        return type(self)

    @property
    def size(self):
        """Return the dataset size."""
        return 0


@DATASETS.register('default')
class RecordDataset(Dataset):
    def __init__(self, source):
        super(RecordDataset, self).__init__(source)

    @property
    def getter(self):
        """Return the dataset getter."""
        return codewithgpu.RecordDataset

    @property
    def size(self):
        """Return the dataset size."""
        return self.getter(self.source).size
