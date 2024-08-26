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

"""NVCF handlers modules"""
import requests
import sys

NUM_OF_RETRY = 3


def admin_login(client_id, client_secret):
    """Generate NVCF JWT token."""
    url = "https://tbyyhdy8-opimayg5nq78mx1wblbi8enaifkmlqrm8m.ssa.nvidia.com/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "scope": (
            "register_function update_function delete_function list_functions "
            "deploy_function invoke_function queue_details authorize_clients"
        ),
    }

    session = requests.Session()
    session.auth = (client_id, client_secret)
    response = session.post(url=url, headers=headers, data=data)

    if response.status_code != 200:
        return None

    token = response.json()["access_token"]
    return token


def user_login(client_id, client_secret):
    """Generate NVCF JWT token."""
    url = "https://tbyyhdy8-opimayg5nq78mx1wblbi8enaifkmlqrm8m.ssa.nvidia.com/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "scope": (
            "list_functions invoke_function"
        ),
    }

    session = requests.Session()
    session.auth = (client_id, client_secret)
    response = session.post(url=url, headers=headers, data=data)

    if response.status_code != 200:
        return None

    token = response.json()["access_token"]
    return token


def invoke_function(deployment_string, network, action, microservice_action="", cloud_metadata={}, specs={}, ngc_api_key="", job_id="", tao_api_admin_key="", tao_api_base_url="", tao_api_status_callback_url="", tao_api_ui_cookie="", use_ngc_staging="", automl_experiment_number=""):
    """Invoke a NVCF function"""
    token = admin_login()

    if not tao_api_base_url:
        tao_api_base_url = "https://nvidia.com"
    if not tao_api_status_callback_url:
        tao_api_status_callback_url = "https://nvidia.com"

    if action == "retrain":
        action = "train"

    request_metadata = {"api_endpoint": microservice_action,
                        "neural_network_name": network,
                        "action_name": action,
                        "ngc_api_key": ngc_api_key,
                        "storage": cloud_metadata,
                        "specs": specs,
                        "job_id": job_id,
                        "tao_api_admin_key": tao_api_admin_key,
                        "tao_api_base_url": tao_api_base_url,
                        "tao_api_status_callback_url": tao_api_status_callback_url,
                        "tao_api_ui_cookie": tao_api_ui_cookie,
                        "use_ngc_staging": use_ngc_staging,
                        "automl_experiment_number": automl_experiment_number,
                        "hosted_service_interaction": "True"
                        }

    data = {"requestBody": request_metadata}
    function_id, version_id = deployment_string.split(":")

    url = f"https://api.nvcf.nvidia.com/v2/nvcf/exec/functions/{function_id}/versions/{version_id}"
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {token}",
    }

    response = requests.post(url, headers=headers, json=data)

    if not response.ok:
        print("Invocation failed.", file=sys.stderr)
        print("Response status code:", response.status_code, file=sys.stderr)
        print("Response content:", response.text, file=sys.stderr)
    return response


def get_status_of_invoked_function(request_id):
    """Fetch status of invoked function"""
    token = admin_login()
    url = f"https://api.nvcf.nvidia.com/v2/nvcf/exec/status/{request_id}"
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {token}",
    }

    response = requests.get(url, headers=headers)

    if not response.ok:
        print("Request failed.")
        print("Response status code:", response.status_code)
        print("Response content:", response.text)
    return response


def get_function(function_id, version_id):
    """Get function metadata"""
    token = admin_login()
    url = f'https://api.nvcf.nvidia.com/v2/nvcf/functions/{function_id}/versions/{version_id}'
    headers = {
        'Authorization': f'Bearer {token}',
        'accept': 'application/json',
    }

    response = requests.get(url, headers=headers)

    if not response.status_code == 200:
        print(f"Get function request failed with status code: {response.status_code}")
        print(response.text)
    return response


def create_function(container):
    token = admin_login()
    url = 'https://api.nvcf.nvidia.com/v2/nvcf/functions'
    headers = {
        'Authorization': f'Bearer {token}',
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    data = {
        "name": "TAO-API-function",
        "inferenceUrl": "/api/v1/nvcf",
        "containerImage": container,
        "apiBodyFormat": "CUSTOM",
        "containerArgs": "flask run --host 0.0.0.0 --port 8000",
        "healthUri": "/api/v1/health/readiness",
    }

    response = requests.post(url, headers=headers, json=data)

    if not response.ok:
        print("NVCF function create failed", file=sys.stderr)
        print(f"Function create request failed with status code: {response.status_code}", file=sys.stderr)
        print("Response content", response.text, file=sys.stderr)
    return response


def deploy_function(function_details):
    token = admin_login()
    function_id = function_details["function"]["id"]
    version_id = function_details["function"]["versionId"]
    url = f'https://api.nvcf.nvidia.com/v2/nvcf/deployments/functions/{function_id}/versions/{version_id}'

    payload = {
        "deploymentSpecifications": [
            {
                "gpu": "L40S",
                "backend": "GFN",
                "maxInstances": 1,
                "minInstances": 1,
                "instanceType": "gl40s_1x2.br25_4xlarge"
            }
        ]
    }

    headers = {
        'Authorization': f'Bearer {token}',
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, json=payload)

    if not response.ok:
        print("NVCF deploy function failed", file=sys.stderr)
        print("Response status code:", response.status_code, file=sys.stderr)
        print("Response content:", response.text, file=sys.stderr)
    return response


def delete_function_version(function_id, version_id):
    token = admin_login()
    headers = {
        'Authorization': f'Bearer {token}',
        'accept': 'application/json',
    }

    url = f"https://api.nvcf.nvidia.com/v2/nvcf/functions/{function_id}/versions/{version_id}"
    delete_response = requests.delete(url, headers=headers)
    if delete_response.ok:
        print("Deleted function")
    else:
        print(f"Function delete request failed with status code: {delete_response.status_code}", file=sys.stderr)
        print("Response content", delete_response.text, file=sys.stderr)
    return delete_response
