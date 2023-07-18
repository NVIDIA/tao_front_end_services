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

"""TAO-Client dataset module"""
import json
import requests

from tao_cli.cli_actions.actions import Actions


class Dataset(Actions):
    """Class which defines API functions for dataset specific actions"""

    # def __init__(self):
    #     """Intialize Dataset class"""
    #     super().__init__()

    def dataset_create(self, dataset_type, dataset_format):
        """Create a dataset and return the id"""
        data = json.dumps({"type": dataset_type, "format": dataset_format})
        endpoint = self.base_url + "/dataset"
        response = requests.post(endpoint, data=data, headers=self.headers)
        id = response.json()["id"]
        return id
