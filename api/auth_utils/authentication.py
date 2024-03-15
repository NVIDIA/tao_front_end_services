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

"""Authentication utils validation modules"""
import os
import requests
import sys
import uuid

from auth_utils import session

#
# class AuthenticationError(Exception):
#     """Authentication Error"""
#
#     pass
#


def validate(token):
    """Validate Authentication"""
    ngc_api_base_url = os.getenv('NGC_API_BASE_URL', default='https://api.ngc.nvidia.com/v2')
    err = None
    user_id = None
    if not token:
        err = 'Authentication error: mission token'
        return user_id, err
    # Retrieve unexpired session, if any
    user = session.get(token)
    if user:
        user_id = user.get('id')
    if user_id:
        print('Found session for user: ' + str(user_id), file=sys.stderr)
        return str(user_id), err
    # Fall back on NGC to validate
    headers = {'Accept': 'application/json'}
    if token:
        if token.startswith("SID=") or token.startswith("SSID="):
            headers['Cookie'] = token
        else:
            headers['Authorization'] = 'Bearer ' + token
    r = requests.get(f'{ngc_api_base_url}/users/me', headers=headers)
    if r.status_code != 200:
        err = 'Authentication error: ' + r.json().get("requestStatus", {}).get("statusDescription", "Unknown NGC user")
        return user_id, err
    ngc_user_id = r.json().get('user', {}).get('id')
    if not ngc_user_id:
        err = 'Authentication error: Unknown NGC user ID'
        return user_id, err
    user_id = str(uuid.uuid5(uuid.UUID(int=0), str(ngc_user_id)))
    print('New session for user: ' + str(user_id), file=sys.stderr)
    # Create a new or update an expired session
    member_of = []
    roles = r.json().get('user', {}).get('roles', [])
    for role in roles:
        org = role.get('org', {}).get('name', '')
        team = role.get('team', {}).get('name', '')
        member_of.append(f"{org}/{team}")
    extra_user_metadata = {'member_of': member_of}
    session.set(user_id, token, extra_user_metadata)
    return user_id, err


#
# def _validate_ngc(token):
#     """Validate Authentication via ngc"""
#     err = None
#     user = None
#     payload = {}
#     try:
#         signing_key = __ngc_jwks_client.get_signing_key_from_jwt(token)
#         payload = jwt.decode(
#             token,
#             signing_key.key,
#             audience="ngc",
#             algorithms=["RS256"]
#         )
#         user = uuid.uuid5(uuid.UUID(int=0), payload.get('sub'))
#     except Exception as e:
#         err = AuthenticationError(e)
#     return user, err
#
#
# def _validate_starfleet(token):
#     """Validate Authentication via starfleet"""
#     client_id = os.getenv('AUTH_CLIENT_ID', default='bnSePYullXlG-504nOZeNAXemGF6DhoCdYR8ysm088w')
#     payload = {}
#     err = None
#     try:
#         signing_key = __starfleet_jwks_client.get_signing_key_from_jwt(token)
#         payload = jwt.decode(
#             token,
#             signing_key.key,
#             algorithms=["ES256"],
#             audience=client_id
#         )
#     except Exception as e:
#         err = AuthenticationError(e)
#     user = payload.get('external_id')
#     if not err and not user:
#         err = AuthenticationError('Unknown user')
#     return user, err
#
