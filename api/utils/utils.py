# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""
Helper functions that can't be present in handlers.utilities
"""

import os
import sys
import shutil
import subprocess
import hashlib
from handlers.stateless_handlers import read_base_experiment_metadata


def run_system_command(command):
    """
    Run a linux command - similar to os.system().
    Waits till process ends.
    """
    result = subprocess.run(['/bin/bash', '-c', command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.stdout:
        print("run_system_command stdout", result.stdout.decode("utf-8"), file=sys.stderr)
    if result.stderr:
        print("run_system_command stderr", result.stderr.decode("utf-8"), file=sys.stderr)
    return 0


def empty_ptm_folder(ptm_root):
    """Remove contents of folder recursively"""
    ptm_metadatas = read_base_experiment_metadata()
    ptm_ids = ptm_metadatas.keys()
    for ptm_dir in os.listdir(ptm_root):
        if ptm_dir not in ptm_ids:
            dir_path = os.path.join(ptm_root, ptm_dir)
            if not os.path.isfile(dir_path):
                shutil.rmtree(dir_path)


def sha256_checksum(file_path):
    """Return sh256 checksum of a file"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def remove_key_by_flattened_string(d, key_string, sep='.'):
    """Removes the flattened key from the dictionary"""
    keys = key_string.split(sep)
    current_dict = d
    for key in keys[:-1]:
        current_dict = current_dict.get(key, {})
    if current_dict:
        current_dict.pop(keys[-1], None)


def create_folder_with_permissions(folder_name):
    """Create folder with write permissions"""
    os.makedirs(folder_name, exist_ok=True)
    os.chmod(folder_name, 0o777)


def is_pvc_space_free(threshold_bytes):
    """Check if pvc has required free space"""
    _, _, free_space = shutil.disk_usage('/')
    return free_space > threshold_bytes, free_space
