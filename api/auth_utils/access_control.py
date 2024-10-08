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

# http://<server>:<port>/<namespace>/api/v1/orgs/<org_name>/experiments?<params>
# ['', '<namespace', 'api', 'v1', 'orgs', '<org_name>', 'experiments']

"""Authentication utils access control modeules"""
import os
from handlers.stateless_handlers import safe_load_file


class AccessControlError(Exception):
    """Access Control Error"""

    pass


def validate(user_id, org_name, url):
    """Validate org_name requested is accessible to the provided user"""
    user_id = str(user_id)
    if url.endswith(":status_update") or url.endswith(":log_update"):
        if user_id != os.environ.get("CALLBACK_UUID"):
            err = AccessControlError("This endpoint is not accessible " + user_id)
            return err
    err = None
    if not org_name:
        err = AccessControlError("Invalid Org requested")
    if err is None:
        # Check org/team membership temporarily for EA programs
        filepath = os.path.join(os.path.sep, 'shared', 'orgs', org_name, 'users', str(user_id), "metadata_token.json")
        user_metadata = safe_load_file(filepath)
        member_of = user_metadata.get('member_of', [])
        if f"{org_name}/" not in member_of:
            err = AccessControlError("No access granted for user in org " + org_name)
    return err
