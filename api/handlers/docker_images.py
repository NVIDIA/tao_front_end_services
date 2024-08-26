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

"""Defines a dictionary mapping docker image to an internal tag"""
import os

DOCKER_IMAGE_MAPPER = {
    "PYTORCH": os.getenv('IMAGE_TAO_PYTORCH', default='nvcr.io/nvidia/tao/tao-toolkit:5.5.0-pyt'),
    "TAO_TF2": os.getenv('IMAGE_TAO_TF2', default='nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf2.11.0'),
    "tao-deploy": os.getenv('IMAGE_TAO_DEPLOY', default='nvcr.io/nvidia/tao/tao-toolkit:5.5.0-deploy'),
    "monai": os.getenv('IMAGE_MONAI', default='nvcr.io/iasixjqzw1hj/monai-service:script-1.1.0.dev.d5d27af'),
    "monai-tis": os.getenv('IMAGE_MONAI_TIS', default='nvcr.io/iasixjqzw1hj/monai-service:infer-1.0.3'),
    "": os.getenv('IMAGE_DEFAULT', default='nvcr.io/nvidia/tao/tao-toolkit:5.5.0-pyt'),  # Default
    "api": os.getenv('IMAGE_API', default='nvcr.io/nvidia/tao/tao-toolkit:5.5.0-api'),
    "TAO_DS": os.getenv('IMAGE_TAO_DS', default='nvcr.io/nvidia/tao/tao-toolkit:5.5.0-data-services'),
    "tensorboard": os.getenv('IMAGE_TF2', default='nvcr.io/nvidia/tensorflow:24.07-tf2-py3')
}


DOCKER_IMAGE_VERSION = {  # (Release tao version for DNN framework, Overriden version for this model)
    "action_recognition": ("5.5.0", "5.5.0"),
    "analytics": ("5.5.0", "5.5.0"),
    "auto_label": ("5.5.0", "5.5.0"),
    "annotations": ("5.5.0", "5.5.0"),
    "augmentation": ("5.5.0", "5.5.0"),
    "centerpose": ("5.5.0", "5.5.0"),
    "classification_tf2": ("5.0.0", "5.0.0"),
    "classification_pyt": ("5.5.0", "5.5.0"),
    "deformable_detr": ("5.5.0", "5.5.0"),
    "detectnet_v2": ("5.0.0", "5.0.0"),
    "dino": ("5.5.0", "5.5.0"),
    "efficientdet_tf2": ("5.0.0", "5.0.0"),
    "mal": ("5.5.0", "5.5.0"),
    "ml_recog": ("5.5.0", "5.5.0"),
    "ocrnet": ("5.5.0", "5.5.0"),
    "ocdnet": ("5.5.0", "5.5.0"),
    "optical_inspection": ("5.5.0", "5.5.0"),
    "pointpillars": ("5.5.0", "5.5.0"),
    "pose_classification": ("5.5.0", "5.5.0"),
    "re_identification": ("5.5.0", "5.5.0"),
    "segformer": ("5.5.0", "5.5.0"),
    "unet": ("5.0.0", "5.0.0"),
    "visual_changenet": ("5.5.0", "5.5.0"),
}
