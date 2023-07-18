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

"""Add click network groups to the cli command"""
import click

from tao_cli.login import login

# TF1 CV networks
from tao_cli.networks.detectnet_v2 import detectnet_v2
from tao_cli.networks.dssd import dssd
from tao_cli.networks.efficientdet_tf1 import efficientdet_tf1
from tao_cli.networks.lprnet import lprnet
from tao_cli.networks.unet import unet
from tao_cli.networks.multi_task_classification import multitask_classification
from tao_cli.networks.multi_class_classification_tf1 import classification_tf1
from tao_cli.networks.mask_rcnn import mask_rcnn
from tao_cli.networks.ssd import ssd
from tao_cli.networks.retinanet import retinanet
from tao_cli.networks.faster_rcnn import faster_rcnn
from tao_cli.networks.yolo_v3 import yolo_v3
from tao_cli.networks.yolo_v4 import yolo_v4
from tao_cli.networks.yolo_v4_tiny import yolo_v4_tiny
# TF1 DRIVEIX networks
from tao_cli.networks.bpnet import bpnet
from tao_cli.networks.fpenet import fpenet
# TF2 CV networks
from tao_cli.networks.multi_class_classification_tf2 import classification_tf2
from tao_cli.networks.efficientdet_tf2 import efficientdet_tf2
# PYTORCH CV networks
from tao_cli.networks.action_recognition import action_recognition
from tao_cli.networks.multi_class_classification_pyt import classification_pyt
from tao_cli.networks.mal import mal
from tao_cli.networks.ml_recog import ml_recog
from tao_cli.networks.ocdnet import ocdnet
from tao_cli.networks.ocrnet import ocrnet
from tao_cli.networks.optical_inspection import optical_inspection
from tao_cli.networks.pointpillars import pointpillars
from tao_cli.networks.pose_classification import pose_classification
from tao_cli.networks.re_identification import re_identification
from tao_cli.networks.segformer import segformer
from tao_cli.networks.deformable_detr import deformable_detr
from tao_cli.networks.dino import dino
# Data Services
from tao_cli.networks.annotations import annotations
from tao_cli.networks.analytics import analytics
from tao_cli.networks.auto_label import auto_label
from tao_cli.networks.augmentation import augmentation


@click.group()
@click.version_option(package_name='nvidia-tao-client')
@click.pass_context
def cli(ctx):
    """Create base nvidia-tao-client group"""
    pass


cli.add_command(login)
# TF1 CV networks
cli.add_command(detectnet_v2)
cli.add_command(dssd)
cli.add_command(efficientdet_tf2)
cli.add_command(lprnet)
cli.add_command(unet)
cli.add_command(multitask_classification)
cli.add_command(classification_tf1)
cli.add_command(mask_rcnn)
cli.add_command(ssd)
cli.add_command(retinanet)
cli.add_command(faster_rcnn)
cli.add_command(yolo_v3)
cli.add_command(yolo_v4)
cli.add_command(yolo_v4_tiny)
# TF1 DRIVEIX networks
cli.add_command(bpnet)
cli.add_command(fpenet)
# TF2 CV networks
cli.add_command(classification_tf2)
cli.add_command(efficientdet_tf1)
# PYTORCH CV networks
cli.add_command(action_recognition)
cli.add_command(classification_pyt)
cli.add_command(mal)
cli.add_command(ml_recog)
cli.add_command(ocdnet)
cli.add_command(ocrnet)
cli.add_command(optical_inspection)
cli.add_command(pointpillars)
cli.add_command(pose_classification)
cli.add_command(re_identification)
cli.add_command(segformer)
cli.add_command(deformable_detr)
cli.add_command(dino)
# Data Services
cli.add_command(annotations)
cli.add_command(analytics)
cli.add_command(auto_label)
cli.add_command(augmentation)

if __name__ == '__main__':
    cli()
