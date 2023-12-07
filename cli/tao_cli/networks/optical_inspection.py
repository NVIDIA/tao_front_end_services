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

"""Optical Inspection tao-client modules"""
import click
import json
import ast

from tao_cli.cli_actions.actions import Actions

from tao_cli.constants import dataset_format, dataset_type, network_type

click_obj = Actions()


@click.group()
def optical_inspection():
    """Create Optical Inspection model click group"""
    pass


@optical_inspection.command()
@click.option('--id', prompt='id', help='Dataset or Model ID', required=True)
@click.option('--job', prompt='job', help='The job ID.', required=True)
@click.option('--job_type', prompt='job', help='model/dataset', required=True)
@click.option('--workdir', prompt='workdir', help='Local path to download the file', required=True)
def get_log_file(id, job, job_type, workdir):
    """Dowload and return the log file path of a job"""
    log_file_path = click_obj.get_log_file(id, job, job_type, workdir)
    click.echo(f"{log_file_path}")


@optical_inspection.command()
@click.option('--id', prompt='id', help='Dataset or Model ID', required=True)
@click.option('--job', prompt='job', help='The job ID.', required=True)
@click.option('--job_type', prompt='job', help='model/dataset', required=True)
@click.option('--retrieve_logs', prompt='retrieve_logs', help='To list log files of the all jobs of the models', required=True, type=bool)
@click.option('--retrieve_specs', prompt='retrieve_specs', help='To list spec files of the all jobs of the models', required=True, type=bool)
def list_job_files(id, job, job_type, retrieve_logs, retrieve_specs):
    """List the files, specs and logs of a job"""
    file_list = click_obj.list_files_of_job(id, job, job_type, bool(retrieve_logs), bool(retrieve_specs))
    click.echo(f"{json.dumps(file_list, indent=2)}")


@optical_inspection.command()
@click.option('--id', prompt='id', help='Dataset or Model ID', required=True)
@click.option('--job', prompt='job', help='The job ID.', required=True)
@click.option('--job_type', prompt='job', help='model/dataset', required=True)
@click.option('--workdir', prompt='workdir', help='Local path to download the files onto', required=True)
@click.option('--file_lists', prompt='file_lists', help='List of files to be downloaded from the list_job_files output', required=True)
@click.option('--best_model', prompt='best_model', help='To add best model in terms of accuracy to the download list', required=True, type=bool)
@click.option('--latest_model', prompt='latest_model', help='To add latest model in terms of accuracy to the download list', required=True, type=bool)
@click.option('--tar_files', prompt='tar_files', help='If the downloaded file should be tar file or not - no need for tars for single file download', required=True, type=bool)
def download_selective_files(id, job, job_type, workdir, file_lists, best_model, latest_model, tar_files):
    """Download job files based on the arguments passed"""
    file_lists = ast.literal_eval(file_lists)
    download_path = click_obj.job_download_selective_files(id, job, job_type, workdir, file_lists, best_model, latest_model, tar_files)
    click.echo(download_path)


@optical_inspection.command()
@click.option('--id', prompt='id', help='Dataset or Model ID', required=True)
@click.option('--job', prompt='job', help='The job ID.', required=True)
@click.option('--job_type', prompt='job', help='model/dataset', required=True)
@click.option('--workdir', prompt='workdir', help='Local path to download the files onto', required=True)
def download_entire_job(id, job, job_type, workdir):
    """Download all files w.r.t to the job"""
    download_path = click_obj.entire_job_download(id, job, job_type, workdir)
    click.echo(download_path)


@optical_inspection.command()
@click.option('--job_type', prompt='job_type', help='model/dataset', required=True)
def list_artifacts(job_type):
    """Return the list of mentioned artifact type"""
    artifacts = click_obj.list_artifacts(job_type)
    click.echo(f"{artifacts}")


@optical_inspection.command()
@click.option('--id', prompt='id', help='Dataset or Model ID', required=True)
@click.option('--job_type', prompt='job_type', help='model/dataset', required=True)
def get_metadata(id, job_type):
    """Get the metadata of the mentioned artifact"""
    metadata = click_obj.get_artifact_metadata(id, job_type)
    click.echo(json.dumps(metadata, indent=2))


@optical_inspection.command()
@click.option('--id', prompt='id', help='Dataset or Model ID', required=True)
@click.option('--job_type', prompt='job_type', help='model/dataset', required=True)
@click.option('--update_info', prompt='update_info', help='Information to be patched', required=True, type=str)
def patch_artifact_metadata(id, job_type, update_info):
    """Patch the metadata of the mentioned artifact"""
    updated_artifact = click_obj.patch_artifact_metadata(id, job_type, update_info)
    click.echo(json.dumps(updated_artifact, indent=2))


@optical_inspection.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
@click.option('--action', prompt='action', help='Model Action.', required=True)
@click.option('--job_type', prompt='job_type', help='model/dataset', required=True)
def get_spec(id, action, job_type):
    """Return default action spec"""
    data = click_obj.get_action_spec(id, action, job_type)
    click.echo(json.dumps(data, indent=2))


@optical_inspection.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
@click.option('--action', prompt='action', help='Model Action.', required=True)
@click.option('--specs', prompt='specs', help='Specs for the Model Action.', required=True)
@click.option('--job_type', prompt='job_type', help='model/dataset', required=True)
def post_spec(id, action, specs, job_type):
    """Post the action spec"""
    data = click_obj.post_action_spec(id, action, specs, job_type)
    click.echo(json.dumps(data, indent=2))


@optical_inspection.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
def model_automl_defaults(id):
    """Return default automl parameters"""
    data = click_obj.get_automl_defaults(id, "train")
    click.echo(json.dumps(data, indent=2))


@optical_inspection.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
@click.option('--action', prompt='action', help='Model Action.', required=True)
@click.option('--job_type', prompt='job_type', help='model/dataset', required=True)
@click.option('--job', help='Parent job.', required=False, default=None)
@click.option('--parent_job_type', help='model/dataset', required=False, default=None)
@click.option('--parent_id', help='Model/Dataset ID', required=False, default=None)
def run_action(id, action, job_type, job, parent_job_type, parent_id):
    """Run action"""
    job_id = click_obj.run_action(id, job, [action], job_type, parent_job_type, parent_id)
    click.echo(f"{job_id}")


@optical_inspection.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
@click.option('--job', help='Job ID.', required=False, default=None)
@click.option('--job_type', prompt='job_type', help='model/dataset', required=True)
def get_action_status(id, job, job_type):
    """Get action status"""
    data = click_obj.get_action_status(id, job, job_type)
    click.echo(json.dumps(data, indent=2))


@optical_inspection.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
@click.option('--job', prompt='job', help='The job ID.', required=True)
@click.option('--job_type', prompt='job_type', help='model/dataset', required=True)
def job_cancel(id, job, job_type):
    """Pause a running job"""
    click_obj.model_job_cancel(id, job, job_type)
    click.echo(f"{job}")


@optical_inspection.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
@click.option('--job', prompt='job', help='The job ID.', required=True)
@click.option('--job_type', prompt='job_type', help='model/dataset', required=True)
def job_resume(id, job, job_type):
    """Resume a paused job"""
    click_obj.model_job_resume(id, job, job_type)
    click.echo(f"{job}")


@optical_inspection.command()
@click.option('--id', prompt='id', help='Dataset ID', required=True)
@click.option('--path', prompt='path', help='Dataset path', required=True)
def dataset_upload(id, path):
    """Upload the file path passed as dataset"""
    response_message = click_obj.dataset_upload(id, path)
    click.echo(f"{response_message}")


@optical_inspection.command()
@click.option('--dataset_type', prompt='dataset_type', type=click.Choice(dataset_type), help='The dataset type.', required=True)
@click.option('--dataset_format', prompt='dataset_format', type=click.Choice(dataset_format), help='The dataset format.', required=True)
@click.option('--dataset_pull', help='The dataset pull URL.', required=False, default=None)
def dataset_create(dataset_type, dataset_format, dataset_pull):
    """Create a dataset and return the id"""
    id = click_obj.dataset_create(dataset_type, dataset_format, dataset_pull)
    click.echo(f"{id}")


@optical_inspection.command()
@click.option('--network_arch', prompt='network_arch', type=click.Choice(network_type), help='Network architecture.', required=True)
@click.option('--encryption_key', prompt='encryption_key', help='Encryption_key.', required=True)
def model_create(network_arch, encryption_key):
    """Create a model and return the id"""
    id = click_obj.model_create(network_arch, encryption_key)
    click.echo(f"{id}")


@optical_inspection.command()
@click.option('--id', prompt='id', help='Dataset ID', required=True)
def dataset_delete(id):
    """Delete a dataset"""
    id = click_obj.dataset_delete(id)
    click.echo(f"{id}")


@optical_inspection.command()
@click.option('--id', prompt='id', help='The model ID.', required=True)
def model_delete(id):
    """Delete a model"""
    id = click_obj.model_delete(id)
    click.echo(f"{id}")
