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
import shutil
import subprocess
import requests
import sys
import configparser
import threading
import io
import base64
import zipfile
import tempfile
from kubernetes import client, config
from handlers.stateless_handlers import get_root, safe_load_file, safe_dump_file, get_handler_id, get_handler_metadata, ngc_runner, admin_uuid
from utils.utils import run_system_command, sha256_checksum, create_folder_with_permissions
NUM_OF_RETRY = 3


def get_ngc_config():
    """Read NGC config and return the API key"""
    if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
        ngc_config_file = os.path.expanduser("~/.ngc/config")
    else:
        ngc_config_file = "/var/www/.ngc/config"
    # Create a ConfigParser object
    config = configparser.ConfigParser()
    # Read the configuration file
    config.read(ngc_config_file)
    # Check if the 'CURRENT' section exists
    return config


def get_ngc_admin_info():
    """Return the metadata info for the user/admin from the config file"""
    ngc_config = get_ngc_config()
    ngcApiKey = ""
    orgName = "nvidiand"
    teamName = "taos"
    ace = "nv-us-west-2"
    if 'CURRENT' in ngc_config:
        if 'apikey' in ngc_config['CURRENT']:
            ngcApiKey = ngc_config['CURRENT']['apikey']
        if 'org' in ngc_config['CURRENT']:
            orgName = ngc_config['CURRENT']['org']
        if 'team' in ngc_config['CURRENT']:
            teamName = ngc_config['CURRENT']['team']
        if 'ace' in ngc_config['CURRENT']:
            ace = ngc_config['CURRENT']['ace']
    return orgName, teamName, ngcApiKey, ace


def ngc_login():
    """Login to NGC and return the token"""
    orgName, teamName, ngcApiKey, _ = get_ngc_admin_info()
    curl_command = f"""curl -s -u "\$oauthtoken":"{ngcApiKey}" -H 'Accept:application/json' 'https://authn.nvidia.com/token?service=ngc&scope=group/ngc:'{orgName}'&scope=group/ngc:'{orgName}'/'{teamName}''"""  # noqa: W605 pylint: disable=W1401
    token = json.loads(subprocess.getoutput(curl_command))["token"]
    return token


def get_token(user_id, refresh=False):
    """Reads the latest token if exists else creates one"""
    ngc_session_cache_file = "/shared/ngc_session_cache.json"
    if refresh or not os.path.exists(ngc_session_cache_file):
        token = ngc_login()
        cache_data = {user_id: {"ngc_login_token": token}}
        safe_dump_file(ngc_session_cache_file, cache_data)
    else:
        ngc_session_cache = safe_load_file(ngc_session_cache_file)
        token = ngc_session_cache.get(user_id, {}).get("ngc_login_token", "")
        if not token:
            token = get_token(user_id, True)
    return token


def get_ngc_headers(user_id, refresh=False):
    """Return header dictionary required for NGC API calls"""
    token = get_token(user_id, refresh)
    headers = {"Authorization": f"Bearer {token}"}
    return headers


def send_ngc_api_request(user_id, endpoint, requests_method, request_body, refresh=False, json=False, retry=0):
    """Send NGG API requests with token refresh and retries"""
    headers = get_ngc_headers(user_id, refresh)
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
        response = send_ngc_api_request(user_id, endpoint, requests_method, request_body, refresh=True)
    if not response.ok:
        # Retry for GET and DELETE requests only. TODO: implement idempotent feature for POST/UPDATE
        if requests_method in ("GET", "DELETE"):
            if retry < NUM_OF_RETRY:
                print(f"Retrying {retry} time(s) to {requests_method} {endpoint}.", file=sys.stderr)
                response = send_ngc_api_request(user_id, endpoint, requests_method, request_body, retry=retry + 1)
            print(f"Request to {requests_method} {endpoint} failed after {retry} retries.", file=sys.stderr)
    return response


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


def send_admin_get_request(endpoint, headers, retry=0):
    """Send admin GET request with retries"""
    r = requests.get(endpoint, headers=headers)
    if not r.ok:
        if retry < NUM_OF_RETRY:
            print(f"Retrying {retry} time(s) to GET {endpoint}.", file=sys.stderr)
            return send_admin_get_request(endpoint, headers, retry + 1)
        print(f"Request to GET {endpoint} failed after {retry} retries.", file=sys.stderr)
    return r


def is_ngc_workspace_free(user_id, threshold_bytes):
    """Check if ngc workspace has required free space"""
    ngc_workspace_free_space = threshold_bytes + 1
    if os.getenv("NGC_RUNNER", "") == "True":
        ngc_workspace_free_space = 0
        orgName, _, _, aceName = get_ngc_admin_info()

        user_info_endpoint = "https://api.ngc.nvidia.com/v2/users/me"
        user_info_response = send_ngc_api_request(user_id, endpoint=user_info_endpoint, requests_method="GET", request_body={}, refresh=False)
        ngc_user_id = user_info_response.json().get("user", {}).get("id", {})

        storage_quota_endpoint = f"https://api.ngc.nvidia.com/v2/org/{orgName}/users/{ngc_user_id}/quota?ace-name={aceName}"
        storage_quota_response = send_ngc_api_request(user_id, endpoint=storage_quota_endpoint, requests_method="GET", request_body={}, refresh=False)
        storagequotas = storage_quota_response.json().get('userStorageQuotas', [])
        print("storagequotas", storagequotas, file=sys.stderr)
        if not storagequotas:  # In case the NGC API failed
            ngc_workspace_free_space = threshold_bytes + 1
        for storage_quota in storagequotas:
            ngc_workspace_free_space += storage_quota.get('available', threshold_bytes + 1)
    return ngc_workspace_free_space > threshold_bytes, ngc_workspace_free_space


def upload_to_ngc_workspace(workspace_id, source_path, dest_path):
    """Upload data to NGC workspace"""
    orgName, teamName, _, ace = get_ngc_admin_info()
    ptm_folder_upload_command = f"ngc bc workspace upload {workspace_id} --source {source_path} --destination {dest_path} --ace {ace} --org {orgName} --team {teamName}"
    if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
        cmd = ['/bin/bash', '-c', ptm_folder_upload_command]
    else:
        cmd = ['/bin/bash', '-c', 'HOME=/var/www && ' + ptm_folder_upload_command]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return result


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


def download_ngc_model(base_experiment_id, ngc_path, ptm_root, sha256_digest, retry=0):
    """Download all base experiments that admin can access"""
    if ngc_path == "":
        print("Invalid ngc path.", file=sys.stderr)
        return False
    if retry >= NUM_OF_RETRY:
        print("Retries exceeded", file=sys.stderr)
        return False
    ngc_configs = ngc_path.split('/')
    org = ngc_configs[0]
    model, version = ngc_configs[-1].split(':')
    team = ""
    if len(ngc_configs) == 3:
        team = ngc_configs[1]

    # Get access token using k8s admin secret
    api_key = get_admin_api_key()
    if not api_key:
        print("API key is None", file=sys.stderr)
        return False

    url = 'https://authn.nvidia.com/token?service=ngc'
    headers = {'Accept': 'application/json', 'Authorization': 'ApiKey ' + api_key}
    response = send_admin_get_request(url, headers=headers)
    if not response.ok:
        print("API response is not ok", file=sys.stderr)
        return False

    token = response.json()["token"]
    headers = {"Authorization": f"Bearer {token} "}
    url_substring = ""
    if team and team != "no-team":
        url_substring = f"team/{team}"
    base_url = "https://api.ngc.nvidia.com"
    endpoint = f"v2/org/{org}/{url_substring}/models/{model}/versions/{version}/zip".replace("//", "/")
    url = f"{base_url}/{endpoint}"
    print("Calling NGC API to download base_experiment", url, file=sys.stderr)
    response = send_admin_get_request(url, headers=headers)
    if not response.ok:
        print("Download API response is not ok", file=sys.stderr)
        return False

    with tempfile.TemporaryDirectory() as temp_dir:
        print("Saving base_experiment to temp directory", file=sys.stderr)
        tmp_ptm_root = f"{temp_dir}/{ptm_root}/"
        tmp_dest_path = f"{tmp_ptm_root}{model}_v{version}"
        create_folder_with_permissions(tmp_dest_path)

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(tmp_dest_path)

        sha256_digest_matched = validate_ptm_download(tmp_dest_path, sha256_digest)
        if not sha256_digest_matched:
            return download_ngc_model(ngc_path, ptm_root, sha256_digest, retry + 1)

        if sha256_digest_matched:
            if ngc_runner == "True":
                result = upload_to_ngc_workspace(admin_uuid, tmp_ptm_root, base_experiment_id)
                shutil.rmtree(temp_dir)
                if result.stderr:
                    error_message = result.stderr.decode("utf-8")
                    print("Workspace upload error", error_message, file=sys.stderr)
                    sha256_digest_matched = False
                    raise ValueError(f"Workspace upload error: {error_message}")
                if result.stdout:
                    print(result.stdout.decode("utf-8"), file=sys.stderr)
            else:
                dest_path = f"{ptm_root}/{model}_v{version}"
                create_folder_with_permissions(dest_path)
                subprocess.getoutput(f"mv {tmp_dest_path}/* {dest_path}/")
                shutil.rmtree(temp_dir)

        return sha256_digest_matched


def load_user_workspace_metadata(user_id):
    """Load and return workspace belonging to given user_id"""
    workspace_metadata_file = f"/shared/users/{user_id}/workspace_metadata.json"
    workspaces = safe_load_file(workspace_metadata_file)
    return workspace_metadata_file, workspaces


def get_workspaces_for_job(user_id, handler_id, parent_job_id, handler_metadata):
    """Return workspaces belonging to a job"""
    _, workspaces = load_user_workspace_metadata(user_id)
    user_workspaces = workspaces.keys()

    job_workspaces = []
    job_workspace_ids = set([])

    for train_ds in handler_metadata.get("train_datasets", []):
        if train_ds in user_workspaces and train_ds not in job_workspace_ids:
            job_workspace_ids.add(train_ds)
            job_workspaces.append(workspaces[train_ds])

    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds in user_workspaces and eval_ds not in job_workspace_ids:
        job_workspace_ids.add(eval_ds)
        job_workspaces.append(workspaces[eval_ds])

    infer_ds = handler_metadata.get("inference_dataset", None)
    if infer_ds in user_workspaces and infer_ds not in job_workspace_ids:
        job_workspace_ids.add(infer_ds)
        job_workspaces.append(workspaces[infer_ds])

    calibration_ds = handler_metadata.get("calibration_dataset", None)
    if calibration_ds in user_workspaces and calibration_ds not in job_workspace_ids:
        job_workspace_ids.add(calibration_ds)
        job_workspaces.append(workspaces[calibration_ds])

    if handler_id in user_workspaces and handler_id not in job_workspace_ids:
        job_workspace_ids.add(handler_id)
        job_workspaces.append(workspaces[handler_id])

    if parent_job_id:
        parent_handler_id = get_handler_id(user_id, parent_job_id)
        if parent_handler_id and parent_handler_id in workspaces.keys():
            if parent_handler_id not in job_workspace_ids:
                job_workspace_ids.add(parent_handler_id)
                job_workspaces.append(workspaces[parent_handler_id])
        if parent_handler_id not in workspaces.keys():
            print("workspaces for user", workspaces, handler_id, parent_handler_id, file=sys.stderr)

    base_experiment_ids = handler_metadata.get("base_experiment", [])
    for base_experiment_id in base_experiment_ids:
        # For cases when the base experiment is a user experiment/model
        if base_experiment_id in user_workspaces and base_experiment_id not in job_workspace_ids:
            job_workspace_ids.add(base_experiment_id)
            job_workspaces.append(workspaces[base_experiment_id])

    _, ptm_workspace = load_user_workspace_metadata(admin_uuid)
    job_workspaces.append(ptm_workspace[admin_uuid])

    return job_workspaces


def mount_ngc_workspace(user_id, handler_id):
    """Mount NGC workspaces this user owns
    user_id:
    kind:
    handler_id:
    dependent_handler_ids: used for mounting required datasets for this model
    """
    workspace_metadata_file, workspaces = load_user_workspace_metadata(user_id)
    if not workspaces:
        return
    workspace = workspaces.get(handler_id, {})
    if workspace.get("status") != "DELETED" and (workspace.get("user_id") == user_id or workspace.get("user_id") == admin_uuid):
        mount_path = workspace["mount_path"]
        os.makedirs(mount_path, exist_ok=True)
        if not os.path.ismount(mount_path):  # Mount only if already not mounted
            aceName = workspace.get("aceName")
            workspace_mount_command = f"ngc bc workspace mount {workspace['id']} {mount_path} --mode RW --ace {aceName}"
            if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                cmd = ['/bin/bash', '-c', workspace_mount_command]
            else:
                cmd = ['/bin/bash', '-c', 'HOME=/var/www && ' + workspace_mount_command]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if result.stdout:
                print(result.stdout.decode("utf-8"), file=sys.stderr)  # Decode and print the standard output
            if result.stderr:
                error_message = result.stderr.decode("utf-8")
                print("Workspace mount error", error_message, file=sys.stderr)
                if "Workspace was not found" in error_message:
                    print("Workspace was not found", workspace, file=sys.stderr)
                    workspaces[handler_id]["status"] = "DELETED"
                    safe_dump_file(workspace_metadata_file, workspaces)
                raise ValueError(f"Workspace mount error: {error_message}")
            print(f"Mounted workspace {workspace['name']} at {mount_path}", file=sys.stderr)

            if workspace["user_id"] == admin_uuid:
                ptm_source_path = f"/opt/api/shared/users/{admin_uuid}/experiments/{admin_uuid}"
                # The following is comment out for NGC workflow to work according to @rarunachalam and @vennw.
                # if not os.path.exists(f"{mount_path}/ptm_metadatas.json"):
                if os.path.exists(ptm_source_path):
                    shutil.copy(f"{ptm_source_path}/ptm_metadatas.json", f"{mount_path}/ptm_metadatas.json")
                else:
                    print(f"Source path {ptm_source_path}: No such file or directory", file=sys.stderr)


def mount_workspaces_required_for_job(user_id, handler_id, parent_job_id):
    """Mount workspaces required for job"""
    handler_metadata = get_handler_metadata(user_id, handler_id)
    workspaces_for_job = get_workspaces_for_job(user_id, handler_id, parent_job_id, handler_metadata)
    for workspace in workspaces_for_job:
        mount_ngc_workspace(workspace['user_id'], workspace['name'])


def write_workspace_metadata(user_id, handler_id, workspace_metadata):
    """Append the workspace metadata info"""
    # Load existing workspaces info
    workspace_metadata_file, existing_workspaces_metadata = load_user_workspace_metadata(user_id)

    # Check if incoming workspace is already part of the exisiting workspaces
    if existing_workspaces_metadata.get(handler_id):
        # Workspace info already present
        return

    # Append incoming workspace to exisiting workspaces
    existing_workspaces_metadata[handler_id] = workspace_metadata

    # Write the workspaces info onto json file
    os.makedirs(os.path.dirname(workspace_metadata_file), exist_ok=True)
    safe_dump_file(workspace_metadata_file, existing_workspaces_metadata)
    print("Workspace info written to metadata file", file=sys.stderr)


def workspace_info(user_id, handler_id):
    """Returns response of the Workspace info if exists"""
    orgName, _, _, _ = get_ngc_admin_info()
    info_endpoint = f"https://api.ngc.nvidia.com/v2/org/{orgName}/workspaces/{handler_id}"
    info_response = send_ngc_api_request(user_id, endpoint=info_endpoint, requests_method="GET", request_body={}, refresh=False)
    return info_response


def create_workspace(user_id, kind, handler_id):
    """Create workspace followed by logs, jobs_metadata and specs folder"""
    orgName, _, _, aceName = get_ngc_admin_info()
    if handler_id != admin_uuid:
        create_workspace(admin_uuid, "experiments", admin_uuid)
    response = workspace_info(user_id, handler_id)
    response_json = response.json()
    # Create a workspace if it doesn't exist
    if "workspace" not in response_json.keys() or ("workspace" in response_json.keys() and "name" not in response_json["workspace"].keys()):  # Workspace doesn't exist
        endpoint = f"https://api.ngc.nvidia.com/v2/org/{orgName}/workspaces/"
        data = {"aceName": aceName, "name": handler_id}
        response = send_ngc_api_request(user_id, endpoint=endpoint, requests_method="POST", request_body=data, refresh=False)
        assert response.status_code == 200
        response_json = response.json()
        print("Workspace creation successful", file=sys.stderr)
    else:
        aceName = response_json["workspace"]["aceName"]  # Override ace name for exisiting workspaces as ngc mount requires exact ace where the workspace exists

    workspace_metadata = response_json.get("workspace", "")
    assert workspace_metadata
    workspace_metadata["user_id"] = user_id
    workspace_metadata["mount_path"] = f"/users/{user_id}/{kind}/{handler_id}"

    # Write workspace info to file which can be consumed across pods
    write_workspace_metadata(user_id, handler_id, workspace_metadata=workspace_metadata)
    # Mount the workspace in the current pod
    mount_ngc_workspace(user_id, handler_id)

    if user_id != admin_uuid:
        root = get_root(ngc_runner_fetch=True) + f"{user_id}/{kind}/{handler_id}/"
        log_root = root + "logs/"
        jobs_meta_root = root + "jobs_metadata/"
        spec_root = root + "specs/"
        for directory in [root, log_root, jobs_meta_root, spec_root]:
            if not os.path.exists(directory):
                os.makedirs(directory)


def clean_deleted_workspaces():
    """Clean up deleted workspaces in workflow. Unmount and delete local path and remove metadata."""
    if not os.path.exists("/shared/users"):
        print("/shared/users doesn't exist yet", file=sys.stderr)
        return
    for user_id in os.listdir("/shared/users/"):
        if os.path.isfile(f"/shared/users/{user_id}"):
            continue
        workspace_metadata_file, workspaces = load_user_workspace_metadata(user_id)
        if workspaces:
            workspace_metadata_updated = False
            for workspace_name in list(workspaces.keys()):
                workspace = workspaces[workspace_name]
                if workspace["status"] == "DELETED":
                    local_path = workspace["mount_path"]
                    # Unmount and delete local path
                    run_system_command(f"umount {local_path}")
                    if os.path.exists(local_path):
                        deletion_command = f"rm -rf {local_path}"
                        delete_thread = threading.Thread(target=run_system_command, args=(deletion_command,))
                        delete_thread.start()
                        print(f"Unmounted and deleted path from local: {local_path}.", file=sys.stderr)
                    # Remove metadata
                    workspaces.pop(workspace_name)
                    workspace_metadata_updated = True

            if workspace_metadata_updated:
                safe_dump_file(workspace_metadata_file, workspaces)
                print("Cleaned up deleted workspace metadata.", file=sys.stderr)


def unmount_ngc_workspace(unmount_path, user_id=None, kind="models", handler_id=None):
    """Unmount local workpace path.
    unmount_path: Local path of the workspace
    """
    if user_id and handler_id:
        unmount_path = f"/users/{user_id}/{kind}/{handler_id}"
        if not os.path.exists(unmount_path):
            unmount_path = f"/users/{user_id}/datasets/{handler_id}"

    if not (os.path.exists(unmount_path) and os.path.ismount(unmount_path)):
        return

    workspace_unmount_command = f"ngc bc workspace unmount {unmount_path}"
    if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
        cmd = ['/bin/bash', '-c', workspace_unmount_command]
    else:
        cmd = ['/bin/bash', '-c', 'HOME=/var/www/ && ' + workspace_unmount_command]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.stdout:
        print(result.stdout.decode("utf-8"), file=sys.stderr)  # Decode and print the standard output
    if result.stderr:
        error_message = result.stderr.decode("utf-8")
        print("Workspace unmount error", error_message, file=sys.stderr)
        raise ValueError(f"Workspace unmount error: {error_message}")

    print(f"Unmounted {unmount_path}.", file=sys.stderr)


def delete_workspace(user_id, handler_id):
    """Delete and unmount workspace followed by logs, jobs_metadata and specs folder and unmount local path."""
    orgName, _, _, _ = get_ngc_admin_info()
    workspace_metadata_file, workspaces = load_user_workspace_metadata(user_id)
    if not workspaces:
        print("No workspace found to unmount.", file=sys.stderr)
        return

    # Get workspace id
    workspace_id = ""
    workspace_mount_path = ""
    workspace = workspaces.get(handler_id, {})
    if workspace:
        # Update status to DELETED
        workspace["status"] = "DELETED"
        workspace_id = workspace["id"]
        workspace_mount_path = workspace["mount_path"]
        safe_dump_file(workspace_metadata_file, workspaces)
    else:
        print("No workspace found to delete.", file=sys.stderr)
        return

    unmount_ngc_workspace(workspace_mount_path)

    endpoint = f"https://api.ngc.nvidia.com/v2/org/{orgName}/workspaces/{workspace_id}"
    response = send_ngc_api_request(user_id, endpoint=endpoint, requests_method="DELETE", request_body={}, refresh=False)
    assert response.status_code == 200
    print(f"Workspace {handler_id} deleted successfully.", file=sys.stderr)


def get_workspace_id(user_id, handler_id):
    """Get workspace id for given handler_id"""
    _, workspaces = load_user_workspace_metadata(user_id)
    if not workspaces:
        return None
    workspace = workspaces.get(handler_id, {})
    if workspace:
        return workspace["id"]
    return None
