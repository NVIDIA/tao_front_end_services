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

"""API Stateless handlers modules"""
import json
import os
import subprocess
import requests
import sys
import io
import zipfile

from handlers.encrypt import NVVaultEncryption
from handlers.stateless_handlers import get_handler_metadata, get_jobs_root, get_workspace_string_identifier, safe_load_file, safe_dump_file, BACKEND
from handlers.cloud_storage import create_cs_instance
from utils import send_delete_request_with_retry, sha256_checksum, read_network_config, get_bcp_api_key, get_ngc_artifact_base_url, send_get_request_with_retry

DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "PROD")
NUM_OF_RETRY = 3
bcp_org_name = "ea-tlt"
if DEPLOYMENT_MODE == "STAGING":
    bcp_org_name = "ygcrk6indslt"
bcp_team_name = "tao_ea"
bcp_ace = "tao-iad2-ace"


def ngc_login(ngc_api_key="", org_name="", team_name=""):
    """Login to NGC and return the token"""
    if not ngc_api_key:
        ngc_api_key = get_bcp_api_key()
        org_name = bcp_org_name
        team_name = bcp_team_name
        curl_command = f"""curl -s -u "\$oauthtoken":"{ngc_api_key}" -H 'Accept:application/json' 'https://authn.nvidia.com/token?service=ngc&scope=group/ngc:'{org_name}'&scope=group/ngc:'{org_name}'/'{team_name}''"""  # noqa: W605 pylint: disable=W1401
    else:
        url_prefix = "stg."
        if DEPLOYMENT_MODE == "PROD":
            url_prefix = ""
        curl_command = f"""curl -s -u "\$oauthtoken":"{ngc_api_key}" -H 'Accept:application/json' 'https://{url_prefix}authn.nvidia.com/token?service=ngc&scope=group/ngc:'{org_name}'&scope=group/ngc:'{org_name}'/'{team_name}''"""  # noqa: W605 pylint: disable=W1401
    token = json.loads(subprocess.getoutput(curl_command))["token"]
    return token


def get_token(user_id, refresh=False, ngc_api_key="", org_name="", team_name=""):
    """Reads the latest token if exists else creates one"""
    ngc_session_cache_file = "/shared/ngc_session_cache.json"
    ngc_session_cache = safe_load_file(ngc_session_cache_file)
    if refresh or not os.path.exists(ngc_session_cache_file):
        key = ngc_session_cache.get(user_id, {}).get("key", "")
        sid_cookie = ngc_session_cache.get(user_id, {}).get("sid_cookie", "")
        ssid_cookie = ngc_session_cache.get(user_id, {}).get("ssid_cookie", "")
        token = ngc_login(ngc_api_key, org_name, team_name)
        ngc_session_cache[user_id] = {"key": key, "ngc_login_token": token, "sid_cookie": sid_cookie, "ssid_cookie": ssid_cookie}
        safe_dump_file(ngc_session_cache_file, ngc_session_cache)
    else:
        token = ngc_session_cache.get(user_id, {}).get("ngc_login_token", "")
        if not token:
            token = get_token(user_id, True, ngc_api_key, org_name, team_name)
    return token


def get_ngc_headers(user_id, refresh=False, ngc_api_key="", org_name="", team_name=""):
    """Return header dictionary required for NGC API calls"""
    token = get_token(user_id, refresh, ngc_api_key, org_name, team_name)
    headers = {"Authorization": f"Bearer {token}"}
    return headers


def get_token_from_cookie(cookie):
    url = 'https://stg.authn.nvidia.com/token?service=ngc'
    if DEPLOYMENT_MODE == "PROD":
        url = 'https://authn.nvidia.com/token?service=ngc'

    headers = {'Accept': 'application/json', 'Cookie': cookie}
    response = send_get_request_with_retry(url, headers=headers)
    return response


class ErrorResponse:
    """Custom error response object"""
    def __init__(self, status_code):
        self.status_code = status_code


def send_ngc_api_request(user_id, endpoint, requests_method, request_body, refresh=False, json=False, retry=0, ngc_api_key="", org_name="", team_name=""):
    """Send NGC API requests with token refresh and retries"""
    try:
        headers = get_ngc_headers(user_id, refresh, ngc_api_key=ngc_api_key, org_name=org_name, team_name=team_name)
        if requests_method == "POST":
            if json:
                headers['accept'] = 'application/json'
                headers['Content-Type'] = 'application/json'
            response = requests.post(url=endpoint, data=request_body, headers=headers)
        if requests_method == "GET":
            response = requests.get(url=endpoint, headers=headers)
        if requests_method == "DELETE":
            response = requests.delete(url=endpoint, headers=headers)
        if response.status_code == 401:  # Token expired
            response = send_ngc_api_request(user_id, endpoint, requests_method, request_body, refresh=True, ngc_api_key=ngc_api_key, org_name=org_name, team_name=team_name)
        if not response.ok:
            # Retry for GET and DELETE requests only. TODO: implement idempotent feature for POST/UPDATE
            if requests_method in ("GET", "DELETE"):
                if retry < NUM_OF_RETRY:
                    print(f"Retrying {retry} time(s) to {requests_method} {endpoint}.", file=sys.stderr)
                    response = send_ngc_api_request(user_id, endpoint, requests_method, request_body, retry=retry + 1, ngc_api_key=ngc_api_key, org_name=org_name, team_name=team_name)
                print(f"Request to {requests_method} {endpoint} failed after {retry} retries.", file=sys.stderr)
        return response
    except Exception as e:
        print(f"Args passed: {user_id}, {endpoint}, {requests_method}, {refresh}, {json}, {retry}, {org_name}, {team_name}", file=sys.stderr)
        print("Exception caught during sending NGC API request", e, file=sys.stderr)
        return ErrorResponse(status_code=404)


def get_user_api_key(user_id):
    """Return user API key"""
    ngc_user_details = safe_load_file("/shared/ngc_session_cache.json")
    encrypted_ngc_api_key = ngc_user_details.get(user_id, {}).get("key")
    encrypted_sid_cookie = ngc_user_details.get(user_id, {}).get("sid_cookie")
    encrypted_ssid_cookie = ngc_user_details.get(user_id, {}).get("ssid_cookie")

    ngc_cookie = False
    # Decrypt the ngc key
    if encrypted_ngc_api_key:
        decrypted_key = encrypted_ngc_api_key
    else:
        decrypted_key = encrypted_ssid_cookie
        if encrypted_sid_cookie:
            decrypted_key = encrypted_sid_cookie
        ngc_cookie = True

    if BACKEND in ("BCP", "NVCF"):
        config_path = os.getenv("VAULT_SECRET_PATH", None)
        encryption = NVVaultEncryption(config_path)
        if decrypted_key and encryption.check_config()[0]:
            decrypted_key = encryption.decrypt(decrypted_key)

    if ngc_cookie:
        cookie = decrypted_key
        decrypted_key = f"SSID={cookie}"
        if encrypted_sid_cookie:
            decrypted_key = f"SID={cookie}"

    return decrypted_key, ngc_cookie


def create_model(org_name, team_name, handler_metadata, source_file, ngc_api_key, ngc_cookie, display_name, description):
    """Create model in ngc private registry"""
    endpoint = "https://api.stg.ngc.nvidia.com/v2"
    if DEPLOYMENT_MODE == "PROD":
        endpoint = "https://api.ngc.nvidia.com/v2"

    endpoint += f"/org/{org_name}"
    if team_name:
        endpoint += f"/team/{team_name}"
    endpoint += "/models"

    user_id = handler_metadata.get("user_id")
    network = handler_metadata.get("network_arch")
    network_config = read_network_config(network)
    framework = network_config.get("api_params", {}).get("image", "tao-pytorch")

    model_format = os.path.splitext(source_file)[1]

    data = {"application": f"TAO {network}",
            "framework": framework,
            "modelFormat": model_format,
            "name": network,
            "precision": "FP32",
            "shortDescription": description,
            "displayName": display_name,
            }

    if ngc_cookie:
        response = get_token_from_cookie(ngc_api_key, ngc_cookie)
        if not response.ok:
            print("API response is not ok", file=sys.stderr)
            return response
        token = response.json()["token"]
        headers = {"Authorization": f"Bearer {token} "}
        response = requests.post(url=endpoint, data=data, headers=headers)
    else:
        response = send_ngc_api_request(user_id, endpoint=endpoint, requests_method="POST", request_body=json.dumps(data), refresh=True, json=True, ngc_api_key=ngc_api_key, org_name=org_name, team_name=team_name)

    status_code = response.status_code
    message = ""
    if status_code != 200:
        message = response.json().get("requestStatus").get("statusDescription")
        if "Model already exists" in message:
            status_code = 201
            message = "Model already exists"
    return status_code, message


def upload_model(org_name, team_name, handler_metadata, source_file, ngc_api_key, job_id, job_action):
    """Upload model to ngc private registry"""
    print("Publishing ", source_file, file=sys.stderr)
    network = handler_metadata.get("network_arch")

    checkpoint_choose_method = handler_metadata.get("checkpoint_choose_method", "best_model")
    epoch_number_dictionary = handler_metadata.get("checkpoint_epoch_number", {})
    epoch_number = epoch_number_dictionary.get(f"{checkpoint_choose_method}_{job_id}", 0)
    num_epochs_trained = handler_metadata["checkpoint_epoch_number"].get(f"latest_model_{job_id}", 0)
    workspace_id = handler_metadata.get("workspace")

    workspace_identifier = get_workspace_string_identifier(org_name, workspace_id, workspace_cache={})
    cloud_path = source_file[len(workspace_identifier):]
    jobs_root = get_jobs_root(handler_metadata.get("user_id"), org_name=org_name)
    local_path = os.path.join(jobs_root, cloud_path[cloud_path.find(job_id):])

    workspace_metadata = get_handler_metadata(org_name, workspace_id, "workspaces")
    cs_instance, _ = create_cs_instance(workspace_metadata)
    cs_instance.download_file(cloud_path, local_path)

    target_version = f"{org_name}/{team_name}/{network}:{job_action}_{job_id}_{epoch_number}"

    publish_model_command = f"NGC_CLI_API_KEY={ngc_api_key} ngc registry model upload-version {target_version} --source {local_path} --num-epochs {num_epochs_trained} --org {org_name}"
    if team_name:
        publish_model_command += f" --team {team_name}"

    if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
        cmd = ['/bin/bash', '-c', publish_model_command]
    else:
        cmd = ['/bin/bash', '-c', 'HOME=/var/www && ' + publish_model_command]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    os.remove(local_path)

    if result.stdout:
        print(result.stdout.decode("utf-8"), file=sys.stderr)
    if result.stderr:
        error_message = result.stderr.decode("utf-8")
        print("Error while uploading", error_message, cloud_path, org_name, team_name, target_version, file=sys.stderr)
        return 404, error_message
    return 200, "Published model into requested org"


def delete_model(org_name, team_name, handler_metadata, ngc_api_key, ngc_cookie, job_id, job_action):
    """Delete model from ngc registry"""
    network = handler_metadata.get("network_arch")

    checkpoint_choose_method = handler_metadata.get("checkpoint_choose_method", "best_model")
    epoch_number_dictionary = handler_metadata.get("checkpoint_epoch_number", {})
    epoch_number = epoch_number_dictionary.get(f"{checkpoint_choose_method}_{job_id}", 0)

    endpoint = "https://api.stg.ngc.nvidia.com/v2"
    if DEPLOYMENT_MODE == "PROD":
        endpoint = "https://api.ngc.nvidia.com/v2"
    endpoint += f"/org/{org_name}"
    if team_name:
        endpoint += f"/team/{team_name}"
    endpoint += f"/models/{network}/versions/{job_action}_{job_id}_{epoch_number}"
    print(f"Deleting: {org_name}/{team_name}/{network}:{job_action}_{job_id}_{epoch_number}", file=sys.stderr)

    if ngc_cookie:
        response = get_token_from_cookie(ngc_api_key, ngc_cookie)
        if not response.ok:
            print("API response is not ok", file=sys.stderr)
            return response
        token = response.json()["token"]
        headers = {"Authorization": f"Bearer {token} "}
        response = send_delete_request_with_retry(endpoint, headers)
    else:
        response = send_ngc_api_request(user_id="", endpoint=endpoint, requests_method="DELETE", request_body={}, refresh=True, ngc_api_key=ngc_api_key, org_name=org_name, team_name=team_name)

    print("Delete model response", response, file=sys.stderr)
    print("Delete model response.text", response.text, file=sys.stderr)
    return response


def validate_ptm_download(base_experiment_folder, sha256_digest):
    """Validate if downloaded files are not corrupt"""
    if sha256_digest:
        for dirpath, _, filenames in os.walk(base_experiment_folder):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.isfile(file_path):
                    if sha256_digest.get(filename):
                        downloaded_file_checksum = sha256_checksum(file_path)
                        if sha256_digest[filename] != downloaded_file_checksum:
                            print(f"{filename} sha256 checksum not matched. Expected checksum is {sha256_digest.get(filename)} wheras downloaded file checksum is {downloaded_file_checksum}", file=sys.stderr)
                        return sha256_digest[filename] == downloaded_file_checksum
    return True


def download_ngc_model(user_id, is_tao_network, base_experiment_id, ngc_path, ptm_root, sha256_digest, retry=0):
    """Download all base experiments that admin can access"""
    if ngc_path == "":
        print("Invalid ngc path.", file=sys.stderr)
        return False
    if retry >= NUM_OF_RETRY:
        print("Retries exceeded", file=sys.stderr)
        return False

    # Get access token using k8s admin secret
    api_key, ngc_cookie = get_user_api_key(user_id)
    if not api_key:
        print("API key/Cookie is None", file=sys.stderr)
        return False

    url = 'https://stg.authn.nvidia.com/token?service=ngc'
    if DEPLOYMENT_MODE == "PROD":
        url = 'https://authn.nvidia.com/token?service=ngc'

    if ngc_cookie:
        headers = {'Accept': 'application/json', 'Cookie': api_key}
    else:
        headers = {'Accept': 'application/json', 'Authorization': 'ApiKey ' + api_key}
        response = send_get_request_with_retry(url, headers=headers)
        if not response.ok:
            print("API response is not ok", file=sys.stderr)
            return False
        token = response.json()["token"]
        headers = {"Authorization": f"Bearer {token} "}

    base_url, model, version = get_ngc_artifact_base_url(ngc_path)
    if is_tao_network:
        download_endpoint = f"{base_url}/files/experiment.yaml/zip/download"
    else:
        download_endpoint = f"{base_url}/zip"
    print("Calling NGC API to download base_experiment", url, file=sys.stderr)
    response = send_get_request_with_retry(download_endpoint, headers=headers)
    if not response.ok:
        print("Download API response is not ok", file=sys.stderr)
        return False

    dest_path = f"{ptm_root}/{model}_v{version}"
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(dest_path)

    sha256_digest_matched = validate_ptm_download(dest_path, sha256_digest)
    if not sha256_digest_matched:
        return download_ngc_model(user_id, is_tao_network, base_experiment_id, ngc_path, ptm_root, sha256_digest, retry + 1)

    return sha256_digest_matched
