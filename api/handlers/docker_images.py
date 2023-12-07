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
    "tlt-pytorch": os.getenv('IMAGE_PYT', default='nvcr.io/nvidia/tao/tao-toolkit:5.2.0-pyt2.1.0'),
    "tlt-pytorch-114": os.getenv('IMAGE_PYT_114', default='nvcr.io/nvidia/tao/tao-toolkit:5.2.0-pyt1.14.0'),
    "tlt-tf2": os.getenv('IMAGE_TF2', default='nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf2.11.0'),
    "tao-deploy": os.getenv('IMAGE_TAO_DEPLOY', default='nvcr.io/nvidia/tao/tao-toolkit:5.2.0-deploy'),
    "": os.getenv('IMAGE_DEFAULT', default='nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf1.15.5'),  # Default
    "api": os.getenv('IMAGE_API', default='nvcr.io/nvidia/tao/tao-toolkit:5.2.0-api'),
    "tao-ds": os.getenv('IMAGE_DATA_SERVICES', default='nvcr.io/nvidia/tao/tao-toolkit:5.2.0-data-services')
}


DOCKER_IMAGE_VERSION = {  # (Release tao version for DNN framework, Overriden version for this model)
    "action_recognition": ("5.2.0", "5.2.0"),
    "analytics": ("5.2.0", "5.2.0"),
    "auto_label": ("5.2.0", "5.2.0"),
    "annotations": ("5.2.0", "5.2.0"),
    "augmentation": ("5.2.0", "5.2.0"),
    "bpnet": ("5.0.0", "5.0.0"),
    "centerpose": ("5.2.0", "5.2.0"),
    "classification_tf1": ("5.0.0", "5.0.0"),
    "classification_tf2": ("5.0.0", "5.0.0"),
    "classification_pyt": ("5.2.0", "5.2.0"),
    "deformable_detr": ("5.2.0", "5.2.0"),
    "detectnet_v2": ("5.0.0", "5.0.0"),
    "dino": ("5.2.0", "5.2.0"),
    "dssd": ("5.0.0", "5.0.0"),
    "efficientdet_tf1": ("5.0.0", "5.0.0"),
    "efficientdet_tf2": ("5.0.0", "5.0.0"),
    "faster_rcnn": ("5.0.0", "5.0.0"),
    "fpenet": ("5.0.0", "5.0.0"),
    "lprnet": ("5.0.0", "5.0.0"),
    "mal": ("5.2.0", "5.2.0"),
    "mask_rcnn": ("5.0.0", "5.0.0"),
    "ml_recog": ("5.2.0", "5.2.0"),
    "multitask_classification": ("5.0.0", "5.0.0"),
    "ocrnet": ("5.2.0", "5.2.0"),
    "ocdnet": ("5.2.0", "5.2.0"),
    "optical_inspection": ("5.2.0", "5.2.0"),
    "pointpillars": ("5.2.0", "5.2.0"),
    "pose_classification": ("5.2.0", "5.2.0"),
    "retinanet": ("5.0.0", "5.0.0"),
    "re_identification": ("5.2.0", "5.2.0"),
    "segformer": ("5.2.0", "5.2.0"),
    "ssd": ("5.0.0", "5.0.0"),
    "unet": ("5.0.0", "5.0.0"),
    "visual_changenet": ("5.2.0", "5.2.0"),
    "yolo_v3": ("5.0.0", "5.0.0"),
    "yolo_v4": ("5.0.0", "5.0.0"),
    "yolo_v4_tiny": ("5.0.0", "5.0.0"),
}
