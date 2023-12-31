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

# http://<server>:<port>/<namespace>/api/v1/user/<user_id>/model?<params>
# ['', '<namespace', 'api', 'v1', 'user', '<user_id>', 'model']

"""Authentication utils access control modeules"""
import os.path


class AccessControlError(Exception):
    """Access Control Error"""

    pass


def _remove_prefix(text, prefix):
    """Removes prefix from given text and returns it"""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def validate(url, user_id):
    """Validate user_id matches user in URL path and whitelist"""
    user_id = str(user_id)
    err = AccessControlError("Invalid URI path for user " + user_id)
    tmp = _remove_prefix(url, 'http://')
    tmp = _remove_prefix(tmp, 'https://')
    tmp = _remove_prefix(tmp, tmp.split('/')[0])
    tmp = tmp.split('?')[0]
    parts = tmp.split('/')
    # check for user ID match in URL path, with or without domain name in path
    if (len(parts) >= 5 and parts[3] == 'user' and parts[4] == user_id):
        err = None
    elif (len(parts) >= 6 and parts[4] == 'user' and parts[5] == user_id):
        err = None
    if err is None:
        # validate user_id in whitelist
        # if no whitelist, assume everybody is allowed
        filename = '/shared/whitelist.txt'
        if os.path.isfile(filename):
            err = AccessControlError("No access granted for user " + user_id)
            with open(filename, 'r', encoding='utf-8') as f:
                allowed_users = [line.rstrip('\n') for line in f]
                if user_id in allowed_users:
                    err = None
    return err
