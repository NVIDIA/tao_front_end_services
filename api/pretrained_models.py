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
import argparse
import copy
import json
import uuid
import operator
import re
import subprocess
import datetime
import ast
from packaging import version

from handlers.utilities import read_network_config, get_admin_api_key


# Define your version strings
version_string = "1.2.3"

SUPPORTED_OPERATORS = {
    "<=": operator.le,
    "<": operator.lt,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq
}

model_bounds = ["<=5.2", ">=4.0.0"]  # Default model bounds
parsed_model_versions = set([])


def get_version_details():
    """Return current version."""
    version_locals = {}
    with open('version.py', 'r', encoding="utf-8") as version_file:
        exec(version_file.read(), {}, version_locals)  # pylint: disable=W0122

    return version_locals


def is_model_supported(model_bounds):
    """Check if the model is supported for the current release"""
    is_valid = []
    current_version = version.parse(get_version_details()["__version__"])
    for version_string in model_bounds:
        string = re.split(r'(\d)', version_string)
        op = string[0]
        bound_version = version.parse("".join(string[1:]))
        is_valid.append(SUPPORTED_OPERATORS[op](current_version, bound_version))
    return all(is_valid)


def is_model_supported_in_api(endpoint):
    """Check if the model is supported by API"""
    if f"{endpoint}.config.json" in os.listdir(f'{os.path.dirname(os.path.abspath(__file__))}/handlers/network_configs/'):
        return True
    return False


def update_metadata(endpoint, full_model, model_details, model_version, ngc_models):
    """Create metadata for the model"""
    if not is_model_supported_in_api(endpoint):
        print(f"Skipping {full_model},{endpoint} as {endpoint} is not supported by TAO API")
        return
    print(f"Creating {full_model},{endpoint}")
    metadata = {}
    metadata['public'] = True
    metadata['read_only'] = True
    metadata['ptm'] = []
    metadata['train_datasets'] = []
    metadata['eval_dataset'] = None
    metadata['calibration_dataset'] = None
    metadata['inference_dataset'] = None
    metadata['additional_id_info'] = None
    metadata['checkpoint_choose_method'] = "best_model"
    metadata['checkpoint_epoch_number'] = {"id": 0}
    metadata['logo'] = 'https://www.nvidia.com'
    metadata['network_arch'] = endpoint
    metadata['dataset_type'] = read_network_config(metadata["network_arch"])["api_params"]["dataset_type"]
    metadata['actions'] = read_network_config(metadata["network_arch"])["api_params"]["actions"]
    metadata['name'] = model_details.get('displayName', 'TAO PTM')
    metadata['description'] = model_version.get('description', 'TAO Pretrained Model')
    metadata['version'] = model_version.get('versionId', '')
    metadata['created_on'] = metadata['last_modified'] = model_version.get('createdDate', datetime.datetime.now().isoformat())
    metadata['ngc_path'] = full_model
    ngc_models.append(metadata.copy())


def __get_existing_models(rootdir):
    existing_models = []
    for subdir in os.listdir(rootdir):
        with open(rootdir + '/' + subdir + '/metadata.json', 'r', encoding='utf-8') as infile:
            existing_models.append(json.load(infile))
    return existing_models


def __model_exists(models, ngc_path):
    return bool(next(filter(lambda x: x.get('ngc_path', None) == ngc_path, models), None))


def __construct_model_string_to_endpoint_mapping(orgName, teamName):
    model_string_to_endpoint_mapping = {}
    with open(f'{os.path.dirname(os.path.abspath(__file__))}/pretrained_models.csv', 'r', encoding='utf-8') as csv_file:
        csv_file.readline()
        for line in csv_file:
            line = line.strip()
            if line:
                _, model_string, endpoint = line.split(',')
                if not (orgName in model_string and teamName in model_string):
                    continue
                if model_string in model_string_to_endpoint_mapping.keys():
                    model_string_to_endpoint_mapping[model_string].append(endpoint)
                else:
                    model_string_to_endpoint_mapping[model_string] = [endpoint]
    return model_string_to_endpoint_mapping


def __get_pretrained_models_from_ngc(teamNames_string, ngcApiKey):
    ngc_models = []

    for org_teamName in teamNames_string.split(","):
        print(org_teamName)
        orgName = org_teamName.split("/")[0]
        teamName = ""
        if len(org_teamName.split("/")) == 2:
            teamName = org_teamName.split("/")[1]

        model_string_to_endpoint_mapping = __construct_model_string_to_endpoint_mapping(orgName, teamName)
        model_string_to_endpoint_mapping_keys = model_string_to_endpoint_mapping.keys()

        try:
            if not ngcApiKey:
                ngcApiKey = get_admin_api_key()
            else:
                secret_dict = json.loads(ngcApiKey)
                ngcApiKey = secret_dict["auths"]["nvcr.io"]["password"]
        except Exception:
            pass

        curl_command = f"""curl -s -u "\$oauthtoken":"{ngcApiKey}" -H 'Accept:application/json' 'https://authn.nvidia.com/token?service=ngc&scope=group/ngc:'{orgName}'&scope=group/ngc:'{orgName}'/'{teamName}''"""  # noqa: W605 pylint: disable=W1401

        # Execute the command and capture the output
        loginToken = json.loads(subprocess.getoutput(curl_command))["token"]
        authorizationHeader = ""
        if loginToken:
            authorizationHeader = f'-H "Authorization: Bearer {loginToken}"'

        curl_substring_model_list = f"{orgName}"
        if teamName:
            curl_substring_model_list = f"{orgName}%2F{teamName}"
        model_list = json.loads(subprocess.getoutput(f"curl -s {authorizationHeader} https://api.ngc.nvidia.com/v2/search/resources/MODEL?q=%7B%22fields%22%3A+%5B%22attributes%22%2C+%22createdBy%22%2C+%22dateCreated%22%2C+%22dateModified%22%2C+%22description%22%2C+%22displayName%22%2C+%22guestAccess%22%2C+%22isPublic%22%2C+%22labels%22%2C+%22latestVersionId%22%2C+%22name%22%2C+%22orgName%22%2C+%22sharedWithOrgs%22%2C+%22sharedWithTeams%22%2C+%22teamName%22%5D%2C+%22orderBy%22%3A+%5B%7B%22field%22%3A+%22score%22%2C+%22value%22%3A+%22ASC%22%7D%5D%2C+%22page%22%3A+1%2C+%22pageSize%22%3A+100%2C+%22query%22%3A+%22resourceId%3A{curl_substring_model_list}%2F%2A%22%2C+%22queryFields%22%3A+%5B%22name%22%2C+%22displayName%22%2C+%22description%22%2C+%22all%22%5D%7D"))

        for page_number in range(0, model_list["resultPageTotal"] + 1):
            model_list = json.loads(subprocess.getoutput(f"curl -s {authorizationHeader} https://api.ngc.nvidia.com/v2/search/resources/MODEL?q=%7B%22fields%22%3A+%5B%22attributes%22%2C+%22createdBy%22%2C+%22dateCreated%22%2C+%22dateModified%22%2C+%22description%22%2C+%22displayName%22%2C+%22guestAccess%22%2C+%22isPublic%22%2C+%22labels%22%2C+%22latestVersionId%22%2C+%22name%22%2C+%22orgName%22%2C+%22sharedWithOrgs%22%2C+%22sharedWithTeams%22%2C+%22teamName%22%5D%2C+%22orderBy%22%3A+%5B%7B%22field%22%3A+%22score%22%2C+%22value%22%3A+%22ASC%22%7D%5D%2C+%22page%22%3A+{page_number}%2C+%22pageSize%22%3A+100%2C+%22query%22%3A+%22resourceId%3A{curl_substring_model_list}%2F%2A%22%2C+%22queryFields%22%3A+%5B%22name%22%2C+%22displayName%22%2C+%22description%22%2C+%22all%22%5D%7D"))
            if type(model_list["results"] == list):
                for response_results in model_list["results"]:
                    for model_metadata in response_results["resources"]:
                        model_base = f'{model_metadata["orgName"]}/{model_metadata.get("teamName", "")}/{model_metadata["name"]}'.replace("//", "/")
                        curl_substring_model_info = f'{model_metadata["orgName"]}'
                        if teamName:
                            curl_substring_model_info = f'{model_metadata["orgName"]}/team/{model_metadata["teamName"]}'
                        model_info_endpoint = f'curl -s {authorizationHeader} https://api.ngc.nvidia.com/v2/org/{curl_substring_model_info}/models/{model_metadata["name"]}/versions?page-size=1000'.replace("//", "/")
                        model_response = json.loads(subprocess.getoutput(model_info_endpoint))
                        # print(model_response)
                        if "modelVersions" not in model_response.keys():
                            continue
                        model_versions = model_response["modelVersions"]
                        model_details = model_response["model"]

                        for model_version in model_versions:
                            full_model = f'{model_base}:{model_version["versionId"]}'
                            if model_version.get('status', "") != "UPLOAD_COMPLETE":
                                continue
                            if full_model in parsed_model_versions:
                                continue
                            parsed_model_versions.add(full_model)
                            # Custom metadata info (older models)
                            if (full_model in model_string_to_endpoint_mapping_keys):
                                for endpoint in model_string_to_endpoint_mapping[full_model]:
                                    # construct metadata info
                                    update_metadata(endpoint, full_model, model_details, model_version, ngc_models)

                            # Custom metadata info (newer models)
                            tao_version_check = {}
                            endpoints = {}
                            trainable = {}

                            # Adhoc supported models
                            if model_version.get('customMetrics'):
                                for customMetrics in model_version["customMetrics"]:
                                    if customMetrics.get("attributes", "") and model_version.get('versionId', ''):
                                        for key_value in customMetrics["attributes"]:
                                            if key_value["key"] == "tao_version":
                                                tao_version_check[model_version["versionId"]] = is_model_supported(ast.literal_eval(key_value["value"]))
                                            if key_value["key"] == "endpoints":
                                                endpoints[model_version["versionId"]] = ast.literal_eval(key_value["value"])
                                            if key_value["key"] == "trainable":
                                                trainable[model_version["versionId"]] = key_value["value"]
                                for versionId, endpoint_list in endpoints.items():
                                    for endpoint in endpoint_list:
                                        if trainable[versionId] == "true" and tao_version_check[versionId]:  # If model is trainable
                                            update_metadata(endpoint, full_model, model_details, model_version, ngc_models)
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

    if metadata["network_arch"] == "action_recognition" and "tao/actionrecognitionnet:trainable_v1.0" in metadata["ngc_path"]:
        ptm_metadatas = []
        for model_type in ("3d", "2d"):
            ac_metadata = copy.deepcopy(metadata)
            ac_metadata['id'] = str(uuid.uuid4())
            ac_metadata['additional_id_info'] = model_type
            ptm_metadatas.append(ac_metadata)

    if metadata["network_arch"] == "action_recognition" and "tao/actionrecognitionnet:trainable_v2.0" in metadata["ngc_path"]:
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


def sync(args):
    """Downloads metadata info for ngc hosted ptm models"""
    admin_uuid = uuid.UUID(int=0)
    rootdir = os.path.abspath(os.path.join(args.shared_folder_path, 'users', str(admin_uuid), 'models'))
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)
    existing_models = __get_existing_models(rootdir)
    pretrained_models = __get_pretrained_models_from_ngc(args.teamNames, args.ngcApiKey)
    for ptm in pretrained_models:
        if not __model_exists(existing_models, ptm.get('ngc_path', '')):
            __create_model(rootdir, ptm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate PTM metadata info')
    parser.add_argument('--shared_folder_path', required=False, help='PTM root path', default='shared')
    parser.add_argument('--teamNames', required=True, help='Organization Name and Team Name')
    parser.add_argument('--ngcApiKey', required=False, help='NGC API Key', default=None)
    args = parser.parse_args()
    sync(args)
