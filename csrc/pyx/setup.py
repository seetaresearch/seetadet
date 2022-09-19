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
"""Build cython extensions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from distutils.core import setup
from distutils.extension import Extension
import os

from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np


def clean_builds():
    """Clean the builds."""
    for file in os.listdir('./'):
        if file.endswith('.c'):
            os.remove(file)


ext_modules = [
    Extension(
        'seetadet.utils.bbox.cython_bbox',
        ['cython_bbox.pyx'],
        extra_compile_args=['-w'],
        include_dirs=[np.get_include()],
    ),
    Extension(
        'seetadet.utils.nms.cython_nms',
        ['cython_nms.pyx'],
        extra_compile_args=['-w'],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name='seetadet',
    ext_modules=cythonize(
        ext_modules, compiler_directives={'language_level': '3'}),
    cmdclass={'build_ext': build_ext},
)
clean_builds()
