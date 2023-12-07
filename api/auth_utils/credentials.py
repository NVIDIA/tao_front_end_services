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

"""Authentication utils credential modules"""
import jwt
import sys
import uuid
import requests

from auth_utils import __ngc_jwks_client, sessions


def get_from_ngc(key):
    """Get signing key from token"""
    err = None
    creds = None
    try:
        r = requests.get('https://authn.nvidia.com/token?service=ngc', headers={'Accept': 'application/json', 'Authorization': 'ApiKey ' + key})
        if r.status_code != 200:
            err = 'Credentials error: Invalid NGC_API_KEY: ' + key
            return creds, err
        token = r.json().get('token')
        user = None
        payload = {}
        signing_key = __ngc_jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            audience="ngc",
            algorithms=["RS256"]
        )
        user = uuid.uuid5(uuid.UUID(int=0), payload.get('sub'))
        creds = {'user_id': str(user), 'token': token}
    except Exception as e:
        err = 'Credentials error: ' + str(e)
    if not err:
        print('Adding trusted user: ' + str(creds.get('user_id')), file=sys.stderr)
        sessions.set(creds)
    return creds, err
