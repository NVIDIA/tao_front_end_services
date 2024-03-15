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

"""Authentication utils credential modules"""
import os
import requests
import uuid


def get_from_ngc(key):
    """Get signing key from token"""
    ngc_api_base_url = os.getenv('NGC_API_BASE_URL', default='https://api.ngc.nvidia.com/v2')
    err = None
    creds = None
    try:
        # Get token
        r = requests.get('https://authn.nvidia.com/token?service=ngc', headers={'Accept': 'application/json', 'Authorization': 'ApiKey ' + key})
        if r.status_code != 200:
            err = 'Credentials error: Invalid NGC_API_KEY'
            return creds, err
        token = r.json().get('token')
        if not token:
            err = 'Credentials error: no token for NGC_API_KEY'
            return creds, err
        # Get user
        r = requests.get(f'{ngc_api_base_url}/users/me', headers={'Accept': 'application/json', 'Authorization': 'Bearer ' + token})
        if r.status_code != 200:
            err = 'Credentials error: ' + r.json().get("requestStatus", {}).get("statusDescription", "Unknown NGC user")
            return creds, err
        ngc_user_id = r.json().get('user', {}).get('id')
        if not ngc_user_id:
            err = 'Credentials error: Unknown NGC user ID'
            return creds, err
        user_id = str(uuid.uuid5(uuid.UUID(int=0), str(ngc_user_id)))
        creds = {'user_id': user_id, 'token': token}
    except Exception as e:
        err = 'Credentials error: ' + str(e)
    return creds, err
