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

"""Json to yaml file conversion"""
import yaml
import json


def yml(data):
    """Writes the dictionary data into yaml file"""
    if type(data) is dict:
        data.pop("version", None)
    return yaml.safe_dump(data)


def convert(path):
    """Reads from json and dumps into yaml file"""
    data = '{}'
    with open(path, mode='r', encoding='utf-8-sig') as f:
        data = json.load(f)
    return yml(data)
