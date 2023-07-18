#!/usr/bin/env python3

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

"""Download metadata info for all ptm models supported"""
import os
import csv
import copy
import json
import uuid
import subprocess
import datetime

from handlers.utilities import read_network_config


def __get_existing_models(rootdir):
    existing_models = []
    for subdir in os.listdir(rootdir):
        with open(rootdir + '/' + subdir + '/metadata.json', 'r', encoding='utf-8') as infile:
            existing_models.append(json.load(infile))
    return existing_models


def __model_exists(models, ngc_path):
    return bool(next(filter(lambda x: x.get('ngc_path', None) == ngc_path, models), None))


def __get_pretrained_models_from_ngc():
    ngc_models = []
    cached_commands = {}
    with open('pretrained_models.csv', 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            model = dict(row)
            command = f"ngc registry model list {model['ngc_path']} --format_type json"
            print(command)
            cached_command = cached_commands.get(command)
            if cached_command is None:
                model_details = json.loads(subprocess.run(['/bin/bash', '-c', command], stdout=subprocess.PIPE, check=False).stdout.decode('utf-8'))
                assert model_details
                cached_commands[command] = model_details = model_details[0]
            else:
                model_details = cached_command
            metadata = {}
            metadata['public'] = True
            metadata['read_only'] = True
            metadata['network_arch'] = model['network_arch']
            metadata['dataset_type'] = read_network_config(metadata["network_arch"])["api_params"]["dataset_type"]
            metadata['actions'] = read_network_config(metadata["network_arch"])["api_params"]["actions"]
            metadata['name'] = model['displayName']
            metadata['description'] = model_details.get('description', 'TAO Pretrained Model')
            metadata['logo'] = 'https://www.nvidia.com'
            metadata['ptm'] = []
            metadata['train_datasets'] = []
            metadata['eval_dataset'] = None
            metadata['calibration_dataset'] = None
            metadata['inference_dataset'] = None
            metadata['version'] = model_details.get('versionId', '')
            metadata['created_on'] = metadata['last_modified'] = model_details.get('createdDate', datetime.datetime.now().isoformat())
            metadata['ngc_path'] = model['ngc_path']
            metadata['additional_id_info'] = None
            ngc_models.append(metadata.copy())
    return ngc_models


def __create_model(rootdir, metadata):
    metadata['id'] = str(uuid.uuid4())
    ptm_metadatas = [metadata]
    if metadata["network_arch"] == "lprnet":
        ptm_metadatas = []
        for model_type in ("us", "ch"):
            pc_metadata = copy.deepcopy(metadata)
            pc_metadata['id'] = str(uuid.uuid4())
            pc_metadata['additional_id_info'] = model_type
            ptm_metadatas.append(pc_metadata)

    if metadata["network_arch"] == "action_recognition" and metadata["ngc_path"] == "nvidia/tao/actionrecognitionnet:trainable_v1.0":
        ptm_metadatas = []
        for model_type in ("3d", "2d"):
            ac_metadata = copy.deepcopy(metadata)
            ac_metadata['id'] = str(uuid.uuid4())
            ac_metadata['additional_id_info'] = model_type
            ptm_metadatas.append(ac_metadata)

    if metadata["network_arch"] == "action_recognition" and metadata["ngc_path"] == "nvidia/tao/actionrecognitionnet:trainable_v2.0":
        ptm_metadatas = []
        for platform in ("a100", "xavier"):
            for model_type in ("3d", "2d"):
                ac_metadata = copy.deepcopy(metadata)
                ac_metadata['id'] = str(uuid.uuid4())
                ac_metadata['additional_id_info'] = platform + "," + model_type
                ptm_metadatas.append(ac_metadata)

    for ptm_metadata in ptm_metadatas:
        ptm_id = ptm_metadata['id']
        path = rootdir + '/' + ptm_id
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + '/metadata.json', 'w', encoding='utf-8') as outfile:
            json.dump(ptm_metadata, outfile, indent=2, sort_keys=False)


def sync(path='/shared'):
    """Downloads metadata info for ngc hosted ptm models"""
    admin_uuid = uuid.UUID(int=0)
    rootdir = path + '/users/' + str(admin_uuid) + '/models'
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)
    existing_models = __get_existing_models(rootdir)
    pretrained_models = __get_pretrained_models_from_ngc()
    for ptm in pretrained_models:
        if not __model_exists(existing_models, ptm.get('ngc_path', '')):
            __create_model(rootdir, ptm)


if __name__ == '__main__':
    sync('shared')
