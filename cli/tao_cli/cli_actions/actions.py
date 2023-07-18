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

"""TAO-Client base actions module"""
import json
import requests
import os


class Actions:
    """Base class which defines API functions for general actions"""

    def __init__(self):
        """Initialize the actions base class"""
        self.user = os.getenv('USER', 'nobody')
        self.base_url = os.getenv('BASE_URL', 'http://localhost/api/v1/') + f"/user/{self.user}"
        self.token = os.getenv('TOKEN', 'invalid')
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.sub_action = self.__class__.__name__.lower()

    def get_action_spec(self, id, action):
        """Return spec dictionary for the action passed"""
        endpoint = self.base_url + f"/{self.sub_action}/{id}/specs/{action}/schema"
        response = requests.get(endpoint, headers=self.headers)
        data = response.json()["default"]
        return data

    def get_automl_defaults(self, id, action):
        """Return automl parameters enabled for a network"""
        endpoint = self.base_url + f"/{self.sub_action}/{id}/specs/{action}/schema"
        response = requests.get(endpoint, headers=self.headers)
        data = response.json()["automl_default_parameters"]
        return data

    def run_action(self, id, job, action):
        """Submit post request for an action"""
        data = json.dumps({"job": job, "actions": action})
        endpoint = self.base_url + f"/{self.sub_action}/{id}/job"
        response = requests.post(endpoint, data=data, headers=self.headers)
        job_id = response.json()[0]
        return job_id

    def model_job_cancel(self, id, job):
        """Pause a running job"""
        endpoint = self.base_url + f"/{self.sub_action}/{id}/job/{job}/cancel"
        requests.post(endpoint, headers=self.headers)

    def model_job_resume(self, id, job):
        """Resume a paused job"""
        endpoint = self.base_url + f"/{self.sub_action}/{id}/job/{job}/resume"
        requests.post(endpoint, headers=self.headers)
