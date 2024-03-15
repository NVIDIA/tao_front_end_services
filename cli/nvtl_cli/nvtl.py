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

"""Add click network groups to the cli command"""
import sys
import click

from nvtl_cli.login import login

from nvtl_cli.cli_actions.network_click_wrapper import create_click_group


@click.group()
@click.version_option(package_name='nvidia-transfer-learning-client')
@click.pass_context
def cli(ctx):
    """Create base nvtl click group"""
    if len(sys.argv) > 0:
        if "tao-client" in sys.argv[0]:
            print("tao-client entrypoint will be deprecated in the future, use nvtl entrypoint going forward", file=sys.stderr)
        if len(sys.argv) > 1:
            if "tao-client" in sys.argv[1]:
                print("tao-client entrypoint will be deprecated in the future, use nvtl entrypoint going forward", file=sys.stderr)
    pass


cli.add_command(login)

# TF1 CV networks
for tf1_network_name in set(["detectnet_v2", "dssd", "efficientdet_tf1", "lprnet", "unet", "multitask_classification", "classification_tf1", "mask_rcnn", "ssd", "retinanet", "faster_rcnn", "yolo_v3", "yolo_v4", "yolo_v4_tiny"]):
    click_group = create_click_group(tf1_network_name, f'Create {tf1_network_name} click group')
    cli.add_command(click_group)

# TF1 DRIVEIX networks
for tf1_driveix_network_name in set(["bpnet", "fpenet"]):
    click_group = create_click_group(tf1_driveix_network_name, f'Create {tf1_driveix_network_name} click group')
    cli.add_command(click_group)

# TF2 CV networks
for tf2_network_name in set(["efficientdet_tf2", "classification_tf2"]):
    click_group = create_click_group(tf2_network_name, f'Create {tf2_network_name} click group')
    cli.add_command(click_group)

# PYTORCH CV networks
for pyt_network_name in set(["action_recognition", "classification_pyt", "mal", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pointpillars", "pose_classification", "re_identification", "segformer", "deformable_detr", "dino", "visual_changenet", "centerpose"]):
    click_group = create_click_group(pyt_network_name, f'Create {pyt_network_name} click group')
    cli.add_command(click_group)

# Data Services
for ds_network_name in set(["annotations", "analytics", "auto_label", "augmentation"]):
    click_group = create_click_group(ds_network_name, f'Create {ds_network_name} click group')
    cli.add_command(click_group)

if __name__ == '__main__':
    cli()
