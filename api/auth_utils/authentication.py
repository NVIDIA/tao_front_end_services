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

"""Authentication utils validation modules"""
import jwt
import uuid
import sys
import os

from auth_utils import __ngc_jwks_client, __starfleet_jwks_client, sessions


def validate(token):
    """Validate Authentication"""
    err = None
    user = sessions.get(token)
    if user is not None:
        print('Found trusted user: ' + str(user), file=sys.stderr)
    else:
        user, err = _validate_ngc(token)
        if not err:
            print('Adding trusted user: ' + str(user), file=sys.stderr)
            sessions.set_session({'user_id': str(user), 'token': token})
        else:
            user, err = _validate_starfleet(token)
            if not err:
                print('Adding trusted user: ' + str(user), file=sys.stderr)
                sessions.set_session({'user_id': str(user), 'token': token})
    return user, err


def _validate_ngc(token):
    """Validate Authentication via ngc"""
    err = None
    user = None
    payload = {}
    try:
        signing_key = __ngc_jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            audience="ngc",
            algorithms=["RS256"]
        )
        user = uuid.uuid5(uuid.UUID(int=0), payload.get('sub'))
    except Exception as e:
        err = 'Token error: ' + str(e)
    return user, err


def _validate_starfleet(token):
    """Validate Authentication via starfleet"""
    client_id = os.getenv('AUTH_CLIENT_ID', default='bnSePYullXlG-504nOZeNAXemGF6DhoCdYR8ysm088w')
    payload = {}
    err = None
    try:
        signing_key = __starfleet_jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256"],
            audience=client_id
        )
    except Exception as e:
        err = 'Token error: ' + str(e)
    user = payload.get('external_id')
    if not err and not user:
        err = 'Token error: unknown user'
    return user, err
