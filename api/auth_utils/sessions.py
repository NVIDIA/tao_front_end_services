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


def synchronized(wrapped):
    """Decorator function for thread synchronization"""
    lock = threading.Lock()

    @functools.wraps(wrapped)
    def _wrap(*args, **kwargs):
        with lock:
            return wrapped(*args, **kwargs)
    return _wrap


@synchronized
def set_session(creds):
    """Append/Pop sessions"""
    sessions = []
    session = creds
    filename = os.path.join('/', 'shared', 'users', "sessions.yaml")
    try:
        with open(filename, "r", encoding='utf-8') as infile:
            sessions = yaml.safe_load(infile)
            if type(sessions) != list:    # corrupted file?
                sessions = []
            if len(session) > 1000:    # keep a max of 1000 active sessions
                sessions.pop(0)    # remove oldest known session
    except:
        pass
    sessions.append(session)
    with open(filename, "w", encoding='utf-8') as outfile:
        yaml.dump(sessions, outfile, sort_keys=False)


@synchronized
def get(token):
    """Read session from sessions.yaml file"""
    sessions = []
    user = None
    filename = os.path.join('/', 'shared', 'users', "sessions.yaml")
    try:
        with open(filename, "r", encoding='utf-8') as infile:
            sessions = yaml.safe_load(infile)
            if type(sessions) != list:    # corrupted file?
                sessions = []
    except:
        pass
    for session in reversed(sessions):
        if str(session.get('token')) == str(token):
            user = session.get('user_id')
            break
    return user
