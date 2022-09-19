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
"""Build cpp extensions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob

import dragon
from dragon.utils import cpp_extension
from setuptools import setup

Extension = cpp_extension.CppExtension
if (dragon.cuda.is_available() and
        cpp_extension.CUDA_HOME is not None):
    Extension = cpp_extension.CUDAExtension
elif dragon.mps.is_available():
    Extension = cpp_extension.MPSExtension


def find_sources(*dirs):
    ext_suffixes = ['.cc']
    if Extension is cpp_extension.CUDAExtension:
        ext_suffixes.append('.cu')
    elif Extension is cpp_extension.MPSExtension:
        ext_suffixes.append('.mm')
    sources = []
    for path in dirs:
        for ext_suffix in ext_suffixes:
            sources += glob.glob(path + '/*' + ext_suffix, recursive=True)
    return sources


ext_modules = [
    Extension(
        name='seetadet.ops._C',
        sources=find_sources('**'),
    ),
]

setup(
    name='seetadet',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
