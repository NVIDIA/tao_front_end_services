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

"""Authentication utils session modules"""
import os
import yaml
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
def _load():
    """Load sessions from file"""
    sessions = []
    dt_now = datetime.now()
    filename = os.path.join('/', 'shared', 'users', "sessions.yaml")
    try:
        with open(filename, "r", encoding='utf-8') as infile:
            all_sessions = yaml.safe_load(infile)
            if type(all_sessions) is list:
                for session in all_sessions:
                    dt_session = datetime.fromtimestamp(session.get('timestamp', 0))
                    dt_delta = dt_now - dt_session
                    if dt_delta.seconds / 3600 < __SESSION_EXPIRY_HOURS__:
                        sessions.append(session)
    except:
        pass
    return sessions


@synchronized
def _save(sessions):
    """Save sessions in file"""
    filename = os.path.join('/', 'shared', 'users', "sessions.yaml")
    with open(filename, "w", encoding='utf-8') as outfile:
        yaml.dump(sessions, outfile, sort_keys=False)


def set(session):
    """Add session"""
    sessions = _load()
    session = session.copy()  # avoid adding timestamp key to credentials passed in as a session object
    session['timestamp'] = datetime.timestamp(datetime.now())
    sessions.append(session)
    _save(sessions)


def get(token):
    """Get user from token"""
    user = None
    sessions = _load()
    for session in reversed(sessions):
        if str(session.get('token', 'invalid')) == str(token):
            user = session.get('user_id', None)
            break
    return user
