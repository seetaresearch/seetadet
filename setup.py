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
"""Python setup script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import subprocess
import sys

import setuptools
import setuptools.command.build_py
import setuptools.command.install


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default=None)
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    args.git_version = None
    args.long_description = ''
    if args.version is None and os.path.exists('version.txt'):
        with open('version.txt', 'r') as f:
            args.version = f.read().strip()
    if os.path.exists('.git'):
        try:
            git_version = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd='./')
            args.git_version = git_version.decode('ascii').strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    if os.path.exists('README.md'):
        with open(os.path.join('README.md'), encoding='utf-8') as f:
            args.long_description = f.read()
    return args


def build_extensions(parallel=4):
    """Prepare the package files."""
    # Compile cxx sources.
    py_exec = sys.executable
    if subprocess.call(
        'cd csrc/cxx && '
        '{} setup.py build_ext -b ../../ -f --no-python-abi-suffix=0 -j {} &&'
        '{} setup.py clean'.format(py_exec, parallel, py_exec), shell=True,
    ) > 0:
        raise RuntimeError('Failed to build the cxx sources.')
    # Compile pyx sources.
    if subprocess.call(
        'cd csrc/pyx && '
        '{} setup.py build_ext -b ../../ -f --cython-c-in-temp -j {} &&'
        '{} setup.py clean'.format(py_exec, parallel, py_exec), shell=True,
    ) > 0:
        raise RuntimeError('Failed to build the pyx sources.')


def clean_builds():
    """Clean the builds."""
    for path in ['build', 'seeta_det.egg-info']:
        if os.path.exists(path):
            shutil.rmtree(path)


def find_packages(top):
    """Return the python sources installed to package."""
    packages = []
    for root, _, _ in os.walk(top):
        if os.path.exists(os.path.join(root, '__init__.py')):
            packages.append(root)
    return packages


def find_package_data(top):
    """Return the external data installed to package."""
    headers, libraries = [], []
    if sys.platform == 'win32':
        dylib_suffix = '.pyd'
    else:
        dylib_suffix = '.so'
    for root, _, files in os.walk(top):
        root = root[len(top + '/'):]
        for file in files:
            if file.endswith(dylib_suffix):
                libraries.append(os.path.join(root, file))
    return headers + libraries


class BuildPyCommand(setuptools.command.build_py.build_py):
    """Enhanced 'build_py' command."""

    def build_packages(self):
        with open('seetadet/version.py', 'w') as f:
            f.write("from __future__ import absolute_import\n"
                    "from __future__ import division\n"
                    "from __future__ import print_function\n\n"
                    "version = '{}'\n"
                    "git_version = '{}'\n".format(args.version, args.git_version))
        super(BuildPyCommand, self).build_packages()

    def build_package_data(self):
        parallel = 4
        for k in ('build', 'install'):
            v = self.get_finalized_command(k).parallel
            parallel = max(parallel, (int(v) if v else v) or 1)
        build_extensions(parallel=parallel)
        self.package_data = {'seetadet': find_package_data('seetadet')}
        super(BuildPyCommand, self).build_package_data()


class InstallCommand(setuptools.command.install.install):
    """Enhanced 'install' command."""

    user_options = setuptools.command.install.install.user_options
    user_options += [('parallel=', 'j', "number of parallel build jobs")]

    def initialize_options(self):
        self.parallel = None
        super(InstallCommand, self).initialize_options()
        self.old_and_unmanageable = True


args = parse_args()
setuptools.setup(
    name='seeta-det',
    version=args.version,
    description='SeetaDet: A platform implementing popular object detection algorithms.',
    long_description=args.long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/seetaresearch/seetadet',
    author='SeetaTech',
    license='BSD 2-Clause',
    packages=find_packages('seetadet'),
    cmdclass={'build_py': BuildPyCommand, 'install': InstallCommand},
    install_requires=['opencv-python',
                      'Pillow>=7.1',
                      'pyyaml',
                      'prettytable',
                      'matplotlib',
                      'codewithgpu',
                      'shapely',
                      'Cython',
                      'pycocotools>=2.0.2'],
    classifiers=['Development Status :: 5 - Production/Stable',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Programming Language :: C++',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3 :: Only',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Mathematics',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)
clean_builds()
