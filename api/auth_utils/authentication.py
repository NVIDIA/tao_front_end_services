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

DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "PROD")

#
# class AuthenticationError(Exception):
#     """Authentication Error"""
#
#     pass
#


def _remove_prefix(text, prefix):
    """Removes prefix from given text and returns it"""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def get_org_name(url):
    """Extract org name from URL"""
    tmp = _remove_prefix(url, 'http://')
    tmp = _remove_prefix(tmp, 'https://')
    tmp = _remove_prefix(tmp, tmp.split('/')[0])
    tmp = tmp.split('?')[0]
    parts = tmp.split('/')
    # check for user ID match in URL path, with or without domain name in path
    org_name = None
    if (len(parts) >= 5 and parts[3] == 'orgs'):
        org_name = parts[4]
    elif (len(parts) >= 6 and parts[4] == 'orgs'):
        org_name = parts[5]
    return org_name


def validate(url, token):
    """Validate Authentication"""
    ngc_api_base_url = 'https://api.stg.ngc.nvidia.com/v2'
    if DEPLOYMENT_MODE == "PROD":
        ngc_api_base_url = 'https://api.ngc.nvidia.com/v2'

    err = None
    user_id = None
    org_name = None
    if not token:
        err = 'Authentication error: mission token'
        return user_id, org_name, err
    org_name = get_org_name(url)
    if not org_name:
        err = 'Authentication error: mission org_name in url'
        return user_id, org_name, err
    # Retrieve unexpired session, if any
    user = session.get(token, org_name)
    if user:
        user_id = user.get('id')
    if user_id and org_name:
        print(f'Found session for user: {str(user_id)} in org {org_name}',  file=sys.stderr)
        return str(user_id), org_name, err
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
        return user_id, org_name, err
    ngc_user_id = r.json().get('user', {}).get('id')
    if not ngc_user_id:
        err = 'Authentication error: Unknown NGC user ID'
        return user_id, org_name, err
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
    session.set(user_id, org_name, token, extra_user_metadata)
    return user_id, org_name, err


def get_user_id(authorization, cookies, org_name):
    """Checks authorization header and cookies and returns user_id associated with token"""
    # Get user ID
    authorization_parts = authorization.split()
    token = None
    if len(authorization_parts) == 2 and authorization_parts[0].lower() == 'bearer':
        token = authorization_parts[1]
    if not token:
        sid_cookie = cookies.get('SID')
        if sid_cookie:
            token = 'SID=' + sid_cookie
    if not token:
        ssid_cookie = cookies.get('SSID')
        if ssid_cookie:
            token = 'SSID=' + ssid_cookie

    user = session.get(token, org_name)
    user_id = None
    if user:
        user_id = user.get('id')

    return user_id

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
