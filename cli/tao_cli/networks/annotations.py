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

#
# Data Services - Data Format Conversion
#
import click
import json

from tao_cli.cli_actions.dataset import Dataset
from tao_cli.cli_actions.model import Model

from tao_cli.constants import dataset_format, dataset_type, network_type


dataset_obj = Dataset()
model_obj = Model()


@click.group()
def annotations():
    pass


@annotations.command()
@click.option('--dataset_type', prompt='dataset_type', type=click.Choice(dataset_type), help='The dataset type.', required=True)
@click.option('--dataset_format', prompt='dataset_format', type=click.Choice(dataset_format), help='The dataset format.', required=True)
def dataset_create(dataset_type, dataset_format):
    """Create a dataset and return the id"""
    id = dataset_obj.dataset_create(dataset_type, dataset_format)
    click.echo(f"{id}")


@annotations.command()
@click.option('--network_arch', prompt='network_arch', type=click.Choice(network_type), help='Network architecture.', required=True)
def model_create(network_arch):
    """Create a model and return the id"""
    id = model_obj.model_create(network_arch, "")
    click.echo(f"{id}")


@annotations.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
def model_action_defaults(id):
    """Return default action spec"""
    data = model_obj.get_action_spec(id, "convert")
    click.echo(json.dumps(data, indent=2))


@annotations.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
def execute_action(id):
    """Run action"""
    job_id = model_obj.run_action(id, None, ["convert"])
    click.echo(f"{job_id}")
