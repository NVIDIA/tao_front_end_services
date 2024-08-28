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

"""Authentication utils session modules"""
import os
import glob
import functools
import threading
from datetime import datetime

from utils import load_file, safe_dump_file

__SESSION_EXPIRY_HOURS__ = 24


def synchronized(wrapped):
    """Decorator function for thread synchronization"""
    lock = threading.Lock()

    @functools.wraps(wrapped)
    def _wrap(*args, **kwargs):
        with lock:
            return wrapped(*args, **kwargs)
    return _wrap


@synchronized
def set(user_id, org_name, token, extra_user_metadata):
    """Save session in file"""
    session = {'id': user_id}
    if extra_user_metadata:
        session.update(extra_user_metadata)
    user_folder = os.path.join(os.path.sep, 'shared', 'orgs', org_name, 'users', user_id)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    filename = os.path.join(user_folder, "metadata_token.json")
    if os.path.exists(filename):
        tmp_session = load_file(filename)
        if tmp_session:
            tmp_session.update(session)
            session = tmp_session

    token_present = False
    for idx, token_info in enumerate(session.get("token_info", [])):
        if token_info.get("token") == token:
            token_present = True
            session["token_info"][idx]["last_modified"] = datetime.now().isoformat()
            session["token_info"][idx]["timestamp"] = datetime.now().isoformat()
    if not token_present:
        if not session.get("token_info"):
            session["token_info"] = []
        session["token_info"].append({"token": token,
                                      "created_on": datetime.now().isoformat(),
                                      "last_modified": datetime.now().isoformat(),
                                      "timestamp": datetime.timestamp(datetime.now())})
    session['org_name'] = org_name
    session['last_modified'] = datetime.now().isoformat()
    session['timestamp'] = datetime.timestamp(datetime.now())
    safe_dump_file(filename, session)
    return session


@synchronized
def get(token, org_name):
    """Load session from file"""
    session = {}
    users_root = os.path.join(os.path.sep, 'shared', 'orgs', org_name)
    # Only metadata_token.json under users/<user_id>/, instead of sub-directories
    user_roots = glob.glob(os.path.join(users_root, "users", "**"), recursive=False)
    for user_root in user_roots:
        try:
            filepath = os.path.join(user_root, "metadata_token.json")
            if os.path.exists(filepath):
                tmp_session = load_file(filepath)
                for token_info in tmp_session.get('token_info', []):
                    if token_info.get('token', 'invalid') == token:
                        dt_session = datetime.fromtimestamp(tmp_session.get('timestamp', 0))
                        dt_delta = datetime.now() - dt_session
                        if dt_delta.seconds / 3600 < __SESSION_EXPIRY_HOURS__:
                            session = tmp_session
                            break
        except:
            pass
    return session
