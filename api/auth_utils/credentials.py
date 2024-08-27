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
import sys
import traceback

from handlers.encrypt import NVVaultEncryption
from handlers.mongo_handler import MongoHandler

BACKEND = os.getenv("BACKEND", "local-k8s")
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "PROD")


def get_from_ngc(key):
    """Get signing key from token"""
    stg_prefix = "stg."
    if DEPLOYMENT_MODE == "PROD":
        stg_prefix = ""

    ngc_api_base_url = f'https://api.{stg_prefix}ngc.nvidia.com/v2'
    err = None
    creds = None
    try:
        # Get token
        r = requests.get(f'https://{stg_prefix}authn.nvidia.com/token?service=ngc', headers={'Accept': 'application/json', 'Authorization': 'ApiKey ' + key})
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
        mongo = MongoHandler("tao", "users")
        encrypted_key = key
        if BACKEND in ("BCP", "NVCF"):
            config_path = os.getenv("VAULT_SECRET_PATH", None)
            encryption = NVVaultEncryption(config_path)
            if encryption.check_config()[0]:
                encrypted_key = encryption.encrypt(key)
            elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                err = "Vault service does not work, can't store API key"
                return creds, err

        user_query = {'id': user_id}
        user = mongo.find_one(user_query)
        if 'key' not in user or encrypted_key != user['key']:
            mongo.upsert(user_query, {'id': user_id, 'key': encrypted_key})

    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        err = 'Credentials error: ' + str(e)
    return creds, err


def save_cookie(user_id, sid_cookie, ssid_cookie):
    """Save the cookie info to User cache"""
    encrypted_sid_cookie = sid_cookie
    encrypted_ssid_cookie = ssid_cookie
    if BACKEND in ("BCP", "NVCF"):
        config_path = os.getenv("VAULT_SECRET_PATH", None)
        encryption = NVVaultEncryption(config_path)
        if encryption.check_config()[0]:
            if sid_cookie:
                encrypted_sid_cookie = encryption.encrypt(sid_cookie)
            if ssid_cookie:
                encrypted_ssid_cookie = encryption.encrypt(ssid_cookie)

    mongo = MongoHandler("tao", "users")
    user_query = {'id': user_id}
    user = mongo.find_one(user_query)
    if sid_cookie:
        if 'sid_cookie' not in user or encrypted_sid_cookie != user['sid_cookie']:
            mongo.upsert(user_query, {'id': user_id, 'sid_cookie': encrypted_sid_cookie})
    if ssid_cookie:
        if 'ssid_cookie' not in user or encrypted_ssid_cookie != user['ssid_cookie']:
            mongo.upsert(user_query, {'id': user_id, 'ssid_cookie': encrypted_ssid_cookie})
