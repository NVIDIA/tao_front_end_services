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

"""Utility functions"""
import os
import sys
import json
import yaml
import base64
import orjson
import shutil
import hashlib
import requests
import subprocess
import numpy as np
from enum import StrEnum
from filelock import FileLock
from kubernetes import client, config

NUM_OF_RETRY = 3
base_exp_uuid = "00000000-0000-0000-0000-000000000000"


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


def read_network_config(network):
    """Reads the network handler json config file"""
    # CLONE EXISTS AT pretrained_models.py
    _dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # If dataset_format is user_custom, return empty dict. TODO: separate dataset format vs. network config reading
    if network == "user_custom":
        return {}
    config_json_path = os.path.join(_dir_path, "api", "handlers", "network_configs", f"{network}.config.json")
    if not os.path.exists(config_json_path):
        print(f"Network config doesn't exist at {config_json_path}", file=sys.stderr)
        return {}
    cli_config = {}
    with open(config_json_path, mode='r', encoding='utf-8-sig') as f:
        cli_config = json.load(f)
    return cli_config


def find_closest_number(x, arr):
    """Find the closest number to x in arr"""
    return arr[min(range(len(arr)), key=lambda i: abs(arr[i] - x))]


def merge_nested_dicts(dict1, dict2):
    """Merge two nested dictionaries. Overwrite values of dict1 where keys are the same and add new keys.

    Args:
        dict1 (dict): The first nested dictionary.
        dict2 (dict): The second nested dictionary.

    Returns:
        dict: The merged nested dictionary.

    Example:
        dict1 = {'a':1,'b':2, 'c':{'a':1}}
        dict2 = {'a':3,'b':{'c':1,'d':'2'}, 'c':3}
        dict1 = merge_nested_dicts(dict1, dict2)
        {'a': 3, 'b': {'c': 1, 'd': '2'}, 'c': 3}
    """
    merged_dict = dict1.copy()

    for key, value in dict2.items():
        if key in merged_dict and isinstance(value, dict) and isinstance(merged_dict[key], dict):
            # Recursively merge nested dictionaries
            merged_dict[key] = merge_nested_dicts(merged_dict[key], value)
        else:
            # Overwrite values or add new keys
            merged_dict[key] = value

    return merged_dict


def get_admin_api_key():
    """Get admin api key from k8s secret"""
    try:
        if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
            # DEV_MODE, get api key from env. It's used to avoid creating a secret in local dev env
            # same env variable is also used in runtests.sh and build.sh
            api_key = os.environ.get('NGC_KEY')
            if api_key:
                return api_key
            config.load_kube_config()
        else:
            config.load_incluster_config()
        # TODO: Use a better way to get the secret for various deployments
        try:
            secret = client.CoreV1Api().read_namespaced_secret("adminclustersecret", "default")
        except client.exceptions.ApiException as e:
            if e.status == 404:
                print("Secret 'adminclustersecret' not found in 'default' namespace.", file=sys.stderr)
                if os.getenv("DEPLOYMENT_MODE", "PROD") == "PROD":
                    print("Falling back to bcpclustersecret", file=sys.stderr)
                    secret = get_bcp_api_key()
                    if not secret:
                        return ""
                    return secret
            else:
                print(f"Failed to obtain secret from k8s: {e}", file=sys.stderr)
            return ""

        encoded_key = base64.b64decode(next(iter(secret.data.values())))
        api_key = json.loads(encoded_key)["auths"]["nvcr.io"]["password"]

        return api_key
    except Exception as e:
        print(f"Failed to obtain api key from k8s: {e}", file=sys.stderr)
        return ""


def get_bcp_api_key():
    """Get bcp api key from k8s secret"""
    try:
        config.load_incluster_config()
        # TODO: Use a better way to get the secret for various deployments
        try:
            secret = client.CoreV1Api().read_namespaced_secret("bcpclustersecret", "default")
        except client.exceptions.ApiException as e:
            if e.status == 404:
                print("Secret 'bcpclustersecret' not found in 'default' namespace. Falling back to imagepullsecret", file=sys.stderr)
                secret = client.CoreV1Api().read_namespaced_secret(os.getenv('IMAGEPULLSECRET', default='imagepullsecret'), "default")
            else:
                print(f"Failed to obtain secret from k8s: {e}", file=sys.stderr)
                return ""

        encoded_key = base64.b64decode(next(iter(secret.data.values())))
        api_key = json.loads(encoded_key)["auths"]["nvcr.io"]["password"]

        return api_key
    except Exception as e:
        print(f"Failed to obtain api key from k8s: {e}", file=sys.stderr)
        return ""


def check_and_convert(user_spec, schema_spec):
    """
    Check if nested keys in user_spec are present in schema_spec. If present, ensure that the type and value
    of each key in user_spec matches the corresponding key in schema_spec. If the type mismatch is found,
    attempt to convert the value to the correct type based on the schema_spec.

    Args:
        user_spec (dict): The user-specified dictionary to be validated and converted.
        schema_spec (dict): The schema specification dictionary against which user_spec will be validated.
    """
    for key, value in schema_spec.items():
        if key in user_spec:
            if isinstance(value, dict):
                if not isinstance(user_spec[key], dict):
                    # Convert to dictionary if necessary
                    try:
                        user_spec[key] = dict(value)
                    except ValueError:
                        pass  # Unable to convert, leave unchanged
                else:
                    # Recursively check nested dictionaries
                    check_and_convert(user_spec[key], value)
            elif isinstance(value, list):
                if not isinstance(user_spec[key], list):
                    # Convert to list if necessary
                    try:
                        user_spec[key] = list(value)
                    except ValueError:
                        pass  # Unable to convert, leave unchanged
                else:
                    # Check each element of the list
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            # Recursively check nested dictionaries
                            if i < len(user_spec[key]):
                                check_and_convert(user_spec[key][i], item)
                        elif not isinstance(user_spec[key][i], type(item)):
                            # Convert type if necessary
                            try:
                                user_spec[key][i] = type(item)(user_spec[key][i])
                            except ValueError:
                                pass  # Unable to convert, leave unchanged
            else:
                # Convert type if necessary
                if not isinstance(user_spec[key], type(value)):
                    try:
                        user_spec[key] = type(value)(user_spec[key])
                    except ValueError:
                        pass  # Unable to convert, leave unchanged


def get_ngc_artifact_base_url(ngc_path):
    """Construct NGC artifact base url from ngc_path provided"""
    ngc_configs = ngc_path.split('/')
    org = ngc_configs[0]
    model, version = ngc_configs[-1].split(':')
    team = ""
    if len(ngc_configs) == 3:
        team = ngc_configs[1]
    base_url = "https://api.ngc.nvidia.com"
    if os.getenv("DEPLOYMENT_MODE", "PROD") == 'STAGING':
        base_url = "https://api.stg.ngc.nvidia.com"
    url_substring = ""
    if team and team != "no-team":
        url_substring = f"team/{team}"
    endpoint = base_url + f"/v2/org/{org}/{url_substring}/models/{model}/versions/{version}".replace("//", "/")
    return endpoint, model, version


def send_get_request_with_retry(endpoint, headers, retry=0):
    """Send admin GET request with retries"""
    r = requests.get(endpoint, headers=headers)
    if not r.ok:
        if retry < NUM_OF_RETRY:
            print(f"Retrying {retry} time(s) to GET {endpoint}.", file=sys.stderr)
            return send_get_request_with_retry(endpoint, headers, retry + 1)
        print(f"Request to GET {endpoint} failed after {retry} retries.", file=sys.stderr)
    return r


def send_delete_request_with_retry(endpoint, headers, retry=0):
    """Send DELETE request with retries"""
    r = requests.delete(endpoint, headers=headers)
    if not r.ok:
        if retry < NUM_OF_RETRY:
            print(f"Retrying {retry} time(s) to DELETE {endpoint}.", file=sys.stderr)
            return send_delete_request_with_retry(endpoint, headers, retry + 1)
        print(f"Request to DELETE {endpoint} failed after {retry} retries.", file=sys.stderr)
    return r


def get_default_lock_file_path(filepath):
    """Returns the default lock file path"""
    return os.path.splitext(filepath)[0] + "_lock.lock"


def create_lock(filepath, existing_lock=None):
    """Creates a lock file"""
    if existing_lock:
        return existing_lock
    lock_file = get_default_lock_file_path(filepath)
    if not os.path.exists(lock_file):
        with open(lock_file, "w", encoding="utf-8") as _:
            pass
    return FileLock(lock_file, mode=0o666)


def safe_get_file_modified_time(filepath):
    """Returns the modified time of the file"""
    lock_file = get_default_lock_file_path(filepath)

    if not os.path.exists(lock_file):
        with open(lock_file, "w", encoding="utf-8") as _:
            pass

    with FileLock(lock_file, mode=0o666):
        return os.path.getmtime(filepath)


def __convert_keys_to_str(data):
    if isinstance(data, dict):
        return {
            str(key) if isinstance(key, int) else key: __convert_keys_to_str(value)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [__convert_keys_to_str(item) for item in data]
    if isinstance(data, np.float64):
        return float(data)
    return data


def load_file(filepath, attempts=3, file_type="json"):
    """Unsynchronized file load"""
    assert file_type in ("json", "yaml")
    if attempts == 0:
        return {}

    if not os.path.exists(filepath):
        print("File trying to read doesn't exists", filepath, file=sys.stderr)
        return {}

    try:
        if file_type == "json":
            with open(filepath, "r", encoding="utf-8") as f:
                json_data = f.read()
                data = orjson.loads(json_data)
        elif file_type == "yaml":
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        return data
    except:
        data = {}
        print(f"Data not in {file_type} loadable format", filepath, file=sys.stderr)
        with open(filepath, "r", encoding='utf-8') as f:
            file_lines = f.readlines()
            print("Data: \n", file_lines, file=sys.stderr)
        if filepath.endswith("ptm_metadatas.json"):
            ptm_source_path = f"/opt/api/shared/orgs/{base_exp_uuid}/experiments/{base_exp_uuid}"
            if os.path.exists(f"{ptm_source_path}/ptm_metadatas.json"):
                print("Copying corrupt PTM meta file", file=sys.stderr)
                shutil.copy(f"{ptm_source_path}/ptm_metadatas.json", filepath)
        return load_file(filepath, attempts - 1, file_type=file_type)


def safe_load_file(filepath, existing_lock=None, attempts=3, file_type="json"):
    """Loads the json file with synchronization"""
    assert file_type in ("json", "yaml")
    if attempts == 0:
        return {}

    if not os.path.exists(filepath):
        print("File trying to read doesn't exists", filepath, file=sys.stderr)
        return {}

    lock = create_lock(filepath, existing_lock=existing_lock)
    try:
        with lock:
            if file_type == "json":
                with open(filepath, "r", encoding="utf-8") as f:
                    json_data = f.read()
                    data = orjson.loads(json_data)
            elif file_type == "yaml":
                with open(filepath, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            return data
    except:
        data = {}
        print(f"Data not in {file_type} loadable format", filepath, file=sys.stderr)
        with open(filepath, "r", encoding='utf-8') as f:
            file_lines = f.readlines()
            print("Data: \n", file_lines, file=sys.stderr)
        if filepath.endswith("ptm_metadatas.json"):
            ptm_source_path = f"/opt/api/shared/orgs/{base_exp_uuid}/experiments/{base_exp_uuid}"
            if os.path.exists(f"{ptm_source_path}/ptm_metadatas.json"):
                print("Copying corrupt PTM meta file", file=sys.stderr)
                shutil.copy(f"{ptm_source_path}/ptm_metadatas.json", filepath)
        return safe_load_file(filepath, lock, attempts - 1, file_type=file_type)


def safe_dump_file(filepath, data, existing_lock=None, file_type="json"):
    """Dumps the json file"""
    assert file_type in ("json", "yaml", "protobuf")
    parent_folder = os.path.dirname(filepath)
    if not os.path.exists(parent_folder):
        print(f"Parent folder {parent_folder} doesn't exists yet", file=sys.stderr)
        return

    lock = create_lock(filepath, existing_lock=existing_lock)

    with lock:
        tmp_file_path = filepath.replace(f".{file_type}", f"_tmp.{file_type}")
        if file_type == "json":
            json_data = orjson.dumps(__convert_keys_to_str(data))
            with open(tmp_file_path, "w", encoding="utf-8") as f:
                f.write(json_data.decode('utf-8'))
        elif file_type == "yaml":
            with open(tmp_file_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, sort_keys=False)
        elif file_type == "protobuf":
            with open(tmp_file_path, "w", encoding='utf-8') as f:
                f.write(data)
        if os.path.exists(tmp_file_path):
            os.rename(tmp_file_path, filepath)


class DataMonitorLogTypeEnum(StrEnum):
    """Class defining data monitor log type."""

    api = "API"
    tao_job = "TAO_JOB"
    tao_experiment = "TAO_EXPERIMENT"
    tao_dataset = "TAO_DATASET"
    medical_job = "MEDICAL_JOB"
    medical_experiment = "MEDICAL_EXPERIMENT"
    medical_dataset = "MEDICAL_DATASET"


def log_monitor(log_type, log_content):
    """
    Log format information for data monitor servers like Kibana.

    Print the log in a fixed format so that it would be easier for the log monitor
    to analyse or visualize the log like how many times the specific user calls
    the specific API.
    """
    monitor_type = os.getenv("SERVER_MONITOR_TYPE", "DATA_COLLECTION")
    print_string = f"[{monitor_type}][{log_type}] {log_content}"
    print(print_string, file=sys.stderr)


def log_api_error(user_id, org_name, from_ui, schema_dict, log_type, action):
    """Log the api call error."""
    error_desc = schema_dict.get("error_desc", None)
    error_code = schema_dict.get("error_code", None)
    log_content = f"user_id:{user_id}, org_name:{org_name}, from_ui:{from_ui}, action:{action}, error_code:{error_code}, error_desc:{error_desc}"
    log_monitor(log_type=log_type, log_content=log_content)


def is_cookie_request(request):
    """Whether a request contains cookie."""
    try:
        sid_cookie = request.cookies.get('SID')
        ssid_cookie = request.cookies.get('SSID')
        return not (sid_cookie is None and ssid_cookie is None)
    except Exception:
        return False
