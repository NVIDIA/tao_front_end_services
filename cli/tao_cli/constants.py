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

"""Defining enums for dataset and model formats and types"""
import enum


class dataset_format(str, enum.Enum):
    """Class defining dataset formats in enum"""

    default = "default"
    kitti = "kitti"
    lprnet = "lprnet"
    coco = "coco"
    raw = "raw"
    coco_raw = "coco_raw"
    custom = "custom"
    unet = "unet"
    classification_pyt = "classification_pyt"


class dataset_type(str, enum.Enum):
    """Class defining dataset types in enum"""

    semantic_segmentation = "semantic_segmentation"
    image_classification = "image_classification"
    object_detection = "object_detection"
    character_recognition = "character_recognition"
    instance_segmentation = "instance_segmentation"
    bpnet = "bpnet"
    fpenet = "fpenet"
    action_recognition = "action_recognition"
    ml_recog = "ml_recog"
    ocdnet = "ocdnet"
    ocrnet = "ocrnet"
    optical_inspection = "optical_inspection"
    pointpillars = "pointpillars"
    pose_classification = "pose_classification"
    re_identification = "re_identification"


class network_type(str, enum.Enum):
    """Class defining network types in enum"""

    # TF1 CV networks
    detectnet_v2 = "detectnet_v2"
    dssd = "dssd"
    efficientdet_tf1 = "efficientdet_tf1"
    lprnet = "lprnet"
    unet = "unet"
    multitask_classification = "multitask_classification"
    classification_tf1 = "classification_tf1"
    mask_rcnn = "mask_rcnn"
    ssd = "ssd"
    retinanet = "retinanet"
    faster_rcnn = "faster_rcnn"
    yolo_v3 = "yolo_v3"
    yolo_v4 = "yolo_v4"
    yolo_v4_tiny = "yolo_v4_tiny"
    # TF1 DRIVEIX networks
    bpnet = "bpnet"
    fpenet = "fpenet"
    # TF2 CV networks
    classification_tf2 = "classification_tf2"
    efficientdet_tf2 = "efficientdet_tf2"
    # PYTORCH CV networks
    action_recognition = "action_recognition"
    classification_pyt = "classification_pyt"
    deformable_detr = "deformable_detr"
    dino = "dino"
    mal = "mal"
    ml_recog = "ml_recog"
    ocdnet = "ocdnet"
    ocrnet = "ocrnet"
    optical_inspection = "optical_inspection"
    pointpillars = "pointpillars"
    pose_classification = "pose_classification"
    segformer = "segformer"
    re_identification = "re_identification"
    # PYTORCH TTS
    spectro_gen = "spectro_gen"
    vocoder = "vocoder"
    # Data Services
    annotations = "annotations"
    analytics = "analytics"
    auto_label = "auto_label"
    augmentation = "augmentation"
