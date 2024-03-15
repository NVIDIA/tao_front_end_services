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
import json
import glob
import functools
import threading
from datetime import datetime

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
def set(user_id, token, extra_user_metadata):
    """Save session in file"""
    session = {'id': user_id}
    if extra_user_metadata:
        session.update(extra_user_metadata)
    filename = os.path.join(os.path.sep, 'shared', 'users', user_id, "metadata.json")
    if os.path.exists(filename):
        with open(filename, "r", encoding='utf-8') as infile:
            tmp_session = json.load(infile)
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
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w+", encoding='utf-8') as outfile:
        session['last_modified'] = datetime.now().isoformat()
        session['timestamp'] = datetime.timestamp(datetime.now())
        json.dump(session, outfile, indent=4)
    return session


@synchronized
def get(token):
    """Load session from file"""
    session = {}
    users_root = os.path.join(os.path.sep, 'shared', 'users')
    user_roots = glob.glob(users_root + os.path.sep + "**" + os.path.sep)
    for user_root in user_roots:
        try:
            filepath = os.path.join(user_root, "metadata.json")
            with open(filepath, "r", encoding='utf-8') as infile:
                tmp_session = json.load(infile)
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
