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
"""Registry class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools


class Registry(object):
    """Registry class."""

    def __init__(self, name):
        self.name = name
        self.registry = collections.OrderedDict()

    def has(self, key):
        return key in self.registry

    def register(self, name, func=None, **kwargs):
        def decorated(inner_function):
            for key in (name if isinstance(
                    name, (tuple, list)) else [name]):
                self.registry[key] = \
                    functools.partial(inner_function, **kwargs)
            return inner_function
        if func is not None:
            return decorated(func)
        return decorated

    def get(self, name, default=None):
        if name is None:
            return None
        if not self.has(name):
            if default is not None:
                return default
            raise KeyError("`%s` is not registered in <%s>."
                           % (name, self.name))
        return self.registry[name]

    def try_get(self, name):
        if self.has(name):
            return self.get(name)
        return None
