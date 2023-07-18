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

"""Defines a dictionary mapping docker image to an internal tag"""
import os

DOCKER_IMAGE_MAPPER = {
    "tlt-tf1": os.getenv('IMAGE_TF1', default='nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf1.15.5'),
    "tlt-pytorch": os.getenv('IMAGE_PYT', default='nvcr.io/nvidia/tao/tao-toolkit:5.0.0-pyt'),
    "tlt-tf2": os.getenv('IMAGE_TF2', default='nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf2.11.0'),
    "tao-deploy": os.getenv('IMAGE_TAO_DEPLOY', default='nvcr.io/nvidia/tao/tao-toolkit:5.0.0-deploy'),
    "": os.getenv('IMAGE_DEFAULT', default='nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf1.15.5'),  # Default
    "api": os.getenv('IMAGE_API', default='nvcr.io/nvidia/tao/tao-toolkit:5.0.0-api'),
    "tao-ds": os.getenv('IMAGE_DATA_SERVICES', default='nvcr.io/nvidia/tao/tao-toolkit:5.0.0-data-services')
}
