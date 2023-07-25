# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup script to build the TAO Toolkit client package."""

import os
import setuptools
import sys


CLI_SOURCE_PATH = os.getcwd()


def get_version_details():
    """Simple function to get packages for setup.py."""
    # Get current __version__.
    version_locals = {}
    with open('version.py', 'r', encoding="utf-8") as version_file:
        exec(version_file.read(), {}, version_locals)  # pylint: disable=W0122

    return version_locals


def get_requirements(package_root):
    """Simple function to get packages."""
    with open(os.path.join(package_root, "requirements-pip.txt"), 'r', encoding="utf-8") as req_file:
        requirements = [r.replace('\n', '') for r in req_file.readlines()]
    return requirements


def find_packages(package_name):
    """List of packages.

    Args:
        package_name (str): Name of the package.

    Returns:
        packages (list): List of packages.
    """
    packages = setuptools.find_packages(package_name)
    packages = [f"{package_name}.{f}" for f in packages]
    packages.append(package_name)
    return packages


def main(args=sys.argv[1:]):
    """Main wrapper to run setup.py"""
    # Get package related information.
    version_locals = get_version_details()
    install_requirements = get_requirements(CLI_SOURCE_PATH)

    print(f'Building wheel with version number {version_locals["__version__"]}')

    PACKAGE_LIST = [
        "tao_cli"
    ]

    setuptools_packages = []
    for package_name in PACKAGE_LIST:
        setuptools_packages.extend(find_packages(package_name))

    # TODO: Modify script entry points
    setuptools.setup(
        name=version_locals["__package_name__"],
        version=version_locals['__version__'],
        description=version_locals["__description__"],
        author=version_locals["__contact_names__"],
        author_email=version_locals["__contact_emails__"],
        classifiers=[
            # How mature is this project? Common values are
            #  3 - Alpha
            #  4 - Beta
            #  5 - Production/Stable
            #  6 - Mature
            #  7 - Inactive
            'Intended Audience :: Developers',
            # Indicate what your project relates to
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Environment :: Console',
            'License :: Other/Proprietary License',
            f'Programming Language :: Python :: {sys.version_info.major}',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
        keywords=version_locals["__keywords__"],
        license="NVIDIA Proprietary Software",
        packages=setuptools_packages,
        package_data={
            '': ['*.pyc', "*.yaml", "*.so"]
        },
        install_requires=install_requirements,
        zip_safe=False,
        entry_points={
            'console_scripts': [
                'tao-client=tao_cli.tao:cli',
            ]
        }
    )


if __name__ == "__main__":
    main()
