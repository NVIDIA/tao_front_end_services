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

"""TAO-Client base actions module"""
import json
import requests
import os
import time

from configparser import ConfigParser

timeout = 3600 * 24


class Actions:
    """Base class which defines API functions for general actions"""

    def __init__(self):
        """Initialize the actions base class"""
        config = ConfigParser()
        config_file_path = os.path.join(os.path.expanduser('~'), '.tao', 'config')
        config.read(config_file_path)
        default_user = os.getenv('USER', 'nobody')
        default_token = os.getenv('TOKEN', 'invalid')
        default_base_url = os.getenv('BASE_URL', 'https://sqa-tao.metropolis.nvidia.com:32443/api/v1')
        self.user = config.get('main', 'USER', fallback=default_user)
        self.token = config.get('main', 'TOKEN', fallback=default_token)
        self.base_url = config.get('main', 'BASE_URL', fallback=default_base_url) + f"/user/{self.user}"
        self.headers = {"Authorization": f"Bearer {self.token}"}

    # Dataset specific actions
    def dataset_create(self, dataset_type, dataset_format, dataset_pull=None):
        """Create a dataset and return the id"""
        request_dict = {"type": dataset_type, "format": dataset_format}
        if dataset_pull:
            request_dict["pull"] = str(dataset_pull)
        data = json.dumps(request_dict)
        endpoint = self.base_url + "/dataset"
        response = requests.post(endpoint, data=data, headers=self.headers, timeout=timeout)
        assert response.status_code in (200, 201)
        assert "id" in response.json().keys()
        id = response.json()["id"]
        return id

    def dataset_upload(self, dataset_id, dataset_path):
        """Upload a dataset and return the response message"""
        with open(dataset_path, "rb") as data_file:
            files = [("file", data_file)]
            endpoint = f"{self.base_url}/dataset/{dataset_id}/upload"
            response = requests.post(endpoint, files=files, headers=self.headers, timeout=timeout)
            assert response.status_code in (200, 201)
            assert "message" in response.json().keys() and response.json()["message"] == "Data successfully uploaded"
            return response.json()["message"]

    def dataset_delete(self, dataset_id):
        """Delete a dataset"""
        endpoint = f"{self.base_url}/dataset/{dataset_id}"
        response = requests.delete(endpoint, headers=self.headers, timeout=timeout)
        assert response.status_code in (200, 201)

    # Model specific actions
    def model_create(self, network_arch, encryption_key):
        """Create a model and return the id"""
        data = json.dumps({"network_arch": network_arch, "encryption_key": encryption_key})
        endpoint = self.base_url + "/model"
        response = requests.post(endpoint, data=data, headers=self.headers, timeout=timeout)
        assert response.status_code in (200, 201)
        assert "id" in response.json().keys()
        id = response.json()["id"]
        return id

    def model_delete(self, model_id):
        """Delete a model"""
        endpoint = f"{self.base_url}/model/{model_id}"
        response = requests.delete(endpoint, headers=self.headers, timeout=timeout)
        assert response.status_code in (200, 201)

    # Common actions
    def list_artifacts(self, artifact_type):
        """List the available datasets/models"""
        endpoint = self.base_url + f"/{artifact_type}"
        response = requests.get(endpoint, headers=self.headers, timeout=timeout)
        assert response.status_code in (200, 201)
        return response.json()

    def get_artifact_metadata(self, id, artifact_type):
        """Get metadata of model/dataset"""
        endpoint = f"{self.base_url}/{artifact_type}/{id}"
        response = requests.get(endpoint, headers=self.headers, timeout=timeout)
        assert response.status_code in (200, 201)
        return response.json()

    def patch_artifact_metadata(self, id, artifact_type, update_info):
        """Update metadata of a model/dataset"""
        endpoint = f"{self.base_url}/{artifact_type}/{id}"
        response = requests.patch(endpoint, data=update_info, headers=self.headers, timeout=timeout)
        assert response.status_code in (200, 201)
        return response.json()

    def get_action_spec(self, id, action, artifact_type):
        """Return spec dictionary for the action passed"""
        endpoint = self.base_url + f"/{artifact_type}/{id}/specs/{action}/schema"
        response = requests.get(endpoint, headers=self.headers, timeout=timeout)
        assert response.status_code in (200, 201)
        data = response.json()["default"]
        return data

    def post_action_spec(self, id, action, data, artifact_type):
        """Upload the spec dictionary for the action passed"""
        endpoint = self.base_url + f"/{artifact_type}/{id}/specs/{action}"
        response = requests.post(endpoint, data=data, headers=self.headers, timeout=timeout)
        return response.json()

    def get_automl_defaults(self, id, action):
        """Return automl parameters enabled for a network"""
        endpoint = self.base_url + f"/model/{id}/specs/{action}/schema"
        response = requests.get(endpoint, headers=self.headers, timeout=timeout)
        data = response.json()["automl_default_parameters"]
        return data

    def run_action(self, id, parent_job, action, artifact_type, parent_job_type=None, parent_id=None):
        """Submit post request for an action"""
        request_dict = {"job": parent_job, "actions": action}
        if parent_job_type:
            request_dict["parent_job_type"] = parent_job_type
        if parent_id:
            request_dict["parent_id"] = parent_id
        data = json.dumps(request_dict)
        endpoint = self.base_url + f"/{artifact_type}/{id}/job"
        response = requests.post(endpoint, data=data, headers=self.headers, timeout=timeout)
        job_id = response.json()[0]
        return job_id

    def get_action_status(self, id, job, artifact_type):
        """Get status for an action"""
        endpoint = self.base_url + f"/{artifact_type}/{id}/job/{job}"
        response = requests.get(endpoint, headers=self.headers, timeout=timeout)
        return response.json()

    def job_cancel(self, id, job, artifact_type):
        """Cancel a running job"""
        endpoint = self.base_url + f"/{artifact_type}/{id}/job/{job}/cancel"
        response = requests.post(endpoint, headers=self.headers, timeout=timeout)
        assert response.status_code in (200, 201)

    def job_resume(self, id, job, artifact_type):
        """Resume a paused job"""
        endpoint = self.base_url + f"/{artifact_type}/{id}/job/{job}/resume"
        response = requests.post(endpoint, headers=self.headers, timeout=timeout)
        assert response.status_code in (200, 201)

    def list_files_of_job(self, id, job, job_type, retrieve_logs, retrieve_specs):
        endpoint = f'{self.base_url}/{job_type}/{id}/job/{job}/list_files'
        params = {"retrieve_logs": retrieve_logs, "retrieve_specs": retrieve_specs}
        response = requests.get(endpoint, headers=self.headers, params=params, timeout=timeout)
        assert response.status_code in (200, 201)
        return response.json()

    def job_download_selective_files(self, id, job, job_type, workdir, file_lists=[], best_model=False, latest_model=False, tar_files=True):
        """Download a job with the files passed"""
        endpoint = f'{self.base_url}/{job_type}/{id}/job/{job}/download_selective_files'
        params = {"file_lists": file_lists, "best_model": best_model, "latest_model": latest_model, "tar_files": tar_files}

        # Save
        temptar = f'{workdir}/{job}.tar.gz'
        if not tar_files and len(file_lists) == 1:
            temptar = f'{workdir}/{job}_{file_lists[0]}'
        os.makedirs(os.path.dirname(temptar), exist_ok=True)
        with requests.get(endpoint, headers=self.headers, params=params, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(temptar, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return temptar

    def entire_job_download(self, id, job, job_type, workdir):
        """Download a job"""
        endpoint = f'{self.base_url}/{job_type}/{id}/job/{job}/download'

        # Save
        temptar = f'{workdir}/{job}.tar.gz'
        os.makedirs(os.path.dirname(temptar), exist_ok=True)

        # Perform a HEAD request to get the file size without downloading the content
        response = requests.head(endpoint, headers=self.headers, timeout=timeout)

        # Check if the request was successful and the 'Content-Length' header is present
        if response.status_code == 200 and 'Content-Length' in response.headers:
            expected_file_size = int(response.headers['Content-Length'])
        else:
            expected_file_size = None  # Set to None if the size couldn't be determined

        while True:
            # Check if the file already exists
            headers_download_job = dict(self.headers)
            if os.path.exists(temptar):
                # Get the current file size
                file_size = os.path.getsize(temptar)

                # If the file size matches the expected size, break out of the loop
                if file_size >= expected_file_size:
                    break

                # Set the headers to resume the download from where it left off
                headers_download_job['Range'] = f'bytes={file_size}-'
            # Open the file for writing in binary mode
            with open(temptar, 'ab') as f:
                try:
                    response = requests.get(endpoint, headers=headers_download_job, stream=True, timeout=timeout)

                    # Check if the request was successful
                    if response.status_code in [200, 206]:
                        # Iterate over the content in chunks
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:
                                # Write the chunk to the file
                                f.write(chunk)

                                # Flush and sync the file to disk
                                f.flush()
                                os.fsync(f.fileno())
                except requests.exceptions.RequestException:
                    # Connection interrupted during download, resuming download from breaking point
                    time.sleep(5)  # Sleep for a while before retrying the request
                    continue  # Continue the loop to retry the request

        return temptar

    def get_log_file(self, id, job, job_type, workdir):
        """Return logs of a running job"""
        while True:
            time.sleep(10)
            files = self.list_files_of_job(id, job, job_type, retrieve_logs=True, retrieve_specs=False)
            if files and f"logs/{job}.txt" in files:
                break
        log_file = self.job_download_selective_files(id, job, job_type, workdir, file_lists=[f"logs/{job}.txt"], best_model=False, latest_model=False, tar_files=False)
        return log_file
