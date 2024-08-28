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

"""Login modules"""
import requests
import click
import json
import os

from configparser import ConfigParser


@click.command()
@click.option('--ngc-api-key', prompt='ngc_api_key', help='Your NGC API KEY.', required=True)
@click.option('--ngc-org-name', prompt='ngc_org_name', help='Your NGC ORG.', required=True)
def login(ngc_api_key, ngc_org_name):
    """User login method"""
    config = ConfigParser()
    config_file_path = os.path.join(os.path.expanduser('~'), '.tao', 'config')
    config.read(config_file_path)
    default_base_url = os.getenv('BASE_URL', 'https://sqa-tao.metropolis.nvidia.com:32443/api/v1')
    base_url = config.get('main', 'BASE_URL', fallback=default_base_url)
    endpoint = base_url + "/login"
    data = json.dumps({'ngc_api_key': ngc_api_key})
    response = requests.post(endpoint, data=data, timeout=600)
    assert response.status_code in (200, 201)
    assert response.json()
    creds = response.json()
    token = creds.get('token', 'invalid')
    if 'main' not in config.sections():
        config.add_section('main')
    config.set('main', 'BASE_URL', base_url)
    config.set('main', 'ORG', ngc_org_name)
    config.set('main', 'TOKEN', token)
    os.makedirs(os.path.dirname(config_file_path), mode=0o700, exist_ok=True)
    with open(config_file_path, 'w', encoding='utf-8') as f:
        config.write(f)
    click.echo(json.dumps(creds))
