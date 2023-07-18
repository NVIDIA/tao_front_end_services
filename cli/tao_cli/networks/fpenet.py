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

"""FPENET tao-client modules"""
import click
import json

from tao_cli.cli_actions.dataset import Dataset
from tao_cli.cli_actions.model import Model

from tao_cli.constants import dataset_format, dataset_type, network_type

dataset_obj = Dataset()
model_obj = Model()


@click.group()
def fpenet():
    """Create FPENET model click group"""
    pass


@fpenet.command()
@click.option('--dataset_type', prompt='dataset_type', type=click.Choice(dataset_type), help='The dataset type.', required=True)
@click.option('--dataset_format', prompt='dataset_format', type=click.Choice(dataset_format), help='The dataset format.', required=True)
def dataset_create(dataset_type, dataset_format):
    """Create a dataset and return the id"""
    id = dataset_obj.dataset_create(dataset_type, dataset_format)
    click.echo(f"{id}")


@fpenet.command()
@click.option('--id', prompt='id', help='The dataset ID.', required=True)
@click.option('--action', prompt='action', help='The dataset convert action.', required=True)
def dataset_convert_defaults(id, action):
    """Return default dataset convert spec"""
    data = dataset_obj.get_action_spec(id, action)
    click.echo(json.dumps(data, indent=2))


@fpenet.command()
@click.option('--id', prompt='id', help='The dataset ID.', required=True)
@click.option('--action', prompt='action', help='The dataset convert action.', required=True)
def dataset_convert(id, action):
    """Run dataset_convert action"""
    job_id = dataset_obj.run_action(id=id, job=None, action=[action])
    click.echo(f"{job_id}")


@fpenet.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
@click.option('--job', prompt='job', help='The job ID.', required=True)
def dataset_job_cancel(id, job):
    """Pause/Cancel a running dataset job"""
    job = dataset_obj.dataset_job_cancel(id, job)
    click.echo(f"{job}")


@fpenet.command()
@click.option('--network_arch', prompt='network_arch', type=click.Choice(network_type), help='Network architecture.', required=True)
@click.option('--encryption_key', prompt='encryption_key', help='Encryption_key.', required=True)
def model_create(network_arch, encryption_key):
    """Create a model and return the id"""
    id = model_obj.model_create(network_arch, encryption_key)
    click.echo(f"{id}")


@fpenet.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
def model_train_defaults(id):
    """Return default train spec"""
    data = model_obj.get_action_spec(id, "train")
    click.echo(json.dumps(data, indent=2))


@fpenet.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
def model_automl_defaults(id):
    """Return default automl parameters"""
    data = model_obj.get_automl_defaults(id, "train")
    click.echo(json.dumps(data, indent=2))


@fpenet.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
@click.option('--job', help='The dataset convert job ID.', required=False, default=None)
def model_train(id, job):
    """Run train action"""
    job_id = model_obj.run_action(id, job, ["train"])
    click.echo(f"{job_id}")


@fpenet.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
def model_evaluate_defaults(id):
    """Return default evaluate spec"""
    data = model_obj.get_action_spec(id, "evaluate")
    click.echo(json.dumps(data, indent=2))


@fpenet.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
@click.option('--job', help='The train, prune or retrain job ID.', required=False, default=None)
def model_evaluate(id, job):
    """Run evaluate action"""
    job_id = model_obj.run_action(id, job, ["evaluate"])
    click.echo(f"{job_id}")


@fpenet.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
def model_export_defaults(id):
    """Return default export spec"""
    data = model_obj.get_action_spec(id, "export")
    click.echo(json.dumps(data, indent=2))


@fpenet.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
@click.option('--job', help='The train, prune or retrain job ID.', required=False, default=None)
def model_export(id, job):
    """Run export action"""
    job_id = model_obj.run_action(id, job, ["export"])
    click.echo(f"{job_id}")


@fpenet.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
def model_inference_defaults(id):
    """Return default inference spec"""
    data = model_obj.get_action_spec(id, "inference")
    click.echo(json.dumps(data, indent=2))


@fpenet.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
@click.option('--job', help='The train, prune, retrain, export or convert job ID.', required=False, default=None)
def model_inference(id, job):
    """Run inference action"""
    job_id = model_obj.run_action(id, job, ["inference"])
    click.echo(f"{job_id}")


@fpenet.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
@click.option('--job', prompt='job', help='The job ID.', required=True)
def model_job_cancel(id, job):
    """Pause a running job"""
    model_obj.model_job_cancel(id, job)
    click.echo(f"{job}")


@fpenet.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
@click.option('--job', prompt='job', help='The job ID.', required=True)
def model_job_resume(id, job):
    """Resume a paused job"""
    model_obj.model_job_resume(id, job)
    click.echo(f"{job}")
