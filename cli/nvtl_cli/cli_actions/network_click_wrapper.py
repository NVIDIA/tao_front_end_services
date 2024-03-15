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

"""TAO-Client wrapper to add networks to the click CLI support"""
import click
import json
import ast

from nvtl_cli.cli_actions.actions import Actions
from nvtl_cli.constants import dataset_format, dataset_type, network_type

click_obj = Actions()


def create_click_group(group_name, help_text):
    """Wrapper class to create click groups for DNN networks"""
    @click.group(name=group_name, help=help_text)
    def wrapper():
        f"""Create {group_name} click group"""
        pass

    @wrapper.command()
    @click.option('--id', prompt='id', help='Dataset or Experiment ID', required=True)
    @click.option('--job', prompt='job', help='The job ID.', required=True)
    @click.option('--job_type', prompt='job', help='experiment/dataset', required=True)
    @click.option('--workdir', prompt='workdir', help='Local path to download the file', required=True)
    def get_log_file(id, job, job_type, workdir):
        """Dowload and return the log file path of a job"""
        log_file_path = click_obj.get_log_file(id, job, job_type, workdir)
        click.echo(f"{log_file_path}")

    @wrapper.command()
    @click.option('--id', prompt='id', help='Dataset or Experiment ID', required=True)
    @click.option('--job', prompt='job', help='The job ID.', required=True)
    @click.option('--job_type', prompt='job', help='experiment/dataset', required=True)
    @click.option('--retrieve_logs', prompt='retrieve_logs', help='To list log files of the jobs', required=True, type=bool)
    @click.option('--retrieve_specs', prompt='retrieve_specs', help='To list spec files of the jobs', required=True, type=bool)
    def list_job_files(id, job, job_type, retrieve_logs, retrieve_specs):
        """List the files, specs and logs of a job"""
        file_list = click_obj.list_files_of_job(id, job, job_type, bool(retrieve_logs), bool(retrieve_specs))
        click.echo(f"{json.dumps(file_list, indent=2)}")

    @wrapper.command()
    @click.option('--id', prompt='id', help='Dataset or Experiment ID', required=True)
    @click.option('--job', prompt='job', help='The job ID.', required=True)
    @click.option('--job_type', prompt='job', help='experiment/dataset', required=True)
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

    @wrapper.command()
    @click.option('--id', prompt='id', help='Dataset or Experiment ID', required=True)
    @click.option('--job', prompt='job', help='The job ID.', required=True)
    @click.option('--job_type', prompt='job', help='experiment/dataset', required=True)
    @click.option('--workdir', prompt='workdir', help='Local path to download the files onto', required=True)
    def download_entire_job(id, job, job_type, workdir):
        """Download all files w.r.t to the job"""
        download_path = click_obj.entire_job_download(id, job, job_type, workdir)
        click.echo(download_path)

    @wrapper.command()
    @click.option('--filter_params', help='filter_params')
    def list_datasets(filter_params):
        """Return the list of mentioned artifact type"""
        artifacts = click_obj.list_artifacts("dataset", filter_params)
        click.echo(f"{artifacts}")

    @wrapper.command()
    @click.option('--filter_params', help='filter_params')
    def list_experiments(filter_params):
        """Return the list of mentioned artifact type"""
        artifacts = click_obj.list_artifacts("experiment", filter_params)
        click.echo(f"{artifacts}")

    @wrapper.command()
    @click.option('--id', prompt='id', help='Dataset or Experiment ID', required=True)
    @click.option('--job_type', prompt='job_type', help='experiment/dataset', required=True)
    def get_metadata(id, job_type):
        """Get the metadata of the mentioned artifact"""
        metadata = click_obj.get_artifact_metadata(id, job_type)
        click.echo(json.dumps(metadata, indent=2))

    @wrapper.command()
    @click.option('--id', prompt='id', help='Dataset or Experiment ID', required=True)
    @click.option('--job_type', prompt='job_type', help='experiment/dataset', required=True)
    @click.option('--update_info', prompt='update_info', help='Information to be patched', required=True, type=str)
    def patch_artifact_metadata(id, job_type, update_info):
        """Patch the metadata of the mentioned artifact"""
        updated_artifact = click_obj.patch_artifact_metadata(id, job_type, update_info)
        click.echo(json.dumps(updated_artifact, indent=2))

    @wrapper.command()
    @click.option('--id', prompt='id', help='Experiment ID.', required=True)
    @click.option('--action', prompt='action', help='Experiment Action.', required=True)
    @click.option('--job_type', prompt='job_type', help='experiment/dataset', required=True)
    def get_spec(id, action, job_type):
        """Return default spec"""
        data = click_obj.get_action_spec(id, action, job_type)
        click.echo(json.dumps(data, indent=2))

    @wrapper.command()
    @click.option('--id', prompt='id', help='Experiment ID.', required=True)
    def model_automl_defaults(id):
        """Return default automl parameters"""
        data = click_obj.get_automl_defaults(id, "train")
        click.echo(json.dumps(data, indent=2))

    @wrapper.command()
    @click.option('--id', prompt='id', help='Experiment ID.', required=True)
    @click.option('--action', prompt='action', help='Experiment Action.', required=True)
    @click.option('--specs', prompt='specs', help='specs', required=True)
    @click.option('--parent_job_id', help='Parent job.', required=False, default=None)
    def dataset_run_action(id, action, specs, parent_job_id):
        """Run action"""
        job_id = click_obj.run_action(id, parent_job_id, action, "dataset", specs)
        click.echo(f"{job_id}")

    @wrapper.command()
    @click.option('--id', prompt='id', help='Experiment ID.', required=True)
    @click.option('--action', prompt='action', help='Experiment Action.', required=True)
    @click.option('--specs', prompt='specs', help='specs', required=True)
    @click.option('--parent_job_id', help='Parent job.', required=False, default=None)
    def experiment_run_action(id, action, specs, parent_job_id):
        """Run action"""
        job_id = click_obj.run_action(id, parent_job_id, action, "experiment", specs)
        click.echo(f"{job_id}")

    @wrapper.command()
    @click.option('--id', prompt='id', help='Experiment ID.', required=True)
    @click.option('--job', help='Job ID.', required=False, default=None)
    @click.option('--job_type', prompt='job_type', help='experiment/dataset', required=True)
    def get_action_status(id, job, job_type):
        """Get action status"""
        data = click_obj.get_action_status(id, job, job_type)
        click.echo(json.dumps(data, indent=2))

    @wrapper.command()
    @click.option('--id', prompt='id', help='Experiment ID.', required=True)
    @click.option('--job', prompt='job', help='The job ID.', required=True)
    @click.option('--job_type', prompt='job_type', help='experiment/dataset', required=True)
    def job_cancel(id, job, job_type):
        """Pause a running job"""
        click_obj.job_cancel(id, job, job_type)
        click.echo(f"{job}")

    @wrapper.command()
    @click.option('--id', prompt='id', help='Experiment ID.', required=True)
    @click.option('--job', prompt='job', help='The job id.', required=True)
    @click.option('--parent_job_id', help='Parent job.', required=False, default=None)
    @click.option('--specs', prompt='specs', help='specs', required=True)
    def job_resume(id, job, parent_job_id, specs):
        """Resume a paused job"""
        click_obj.job_resume(id, job, parent_job_id, specs)
        click.echo(f"{job}")

    @wrapper.command()
    @click.option('--id', prompt='id', help='Dataset ID', required=True)
    @click.option('--path', prompt='path', help='Dataset path', required=True)
    def dataset_upload(id, path):
        """Upload the file path passed as dataset"""
        response_message = click_obj.dataset_upload(id, path)
        click.echo(f"{response_message}")

    @wrapper.command()
    @click.option('--dataset_type', prompt='dataset_type', type=click.Choice(dataset_type), help='The dataset type.', required=True)
    @click.option('--dataset_format', prompt='dataset_format', type=click.Choice(dataset_format), help='The dataset format.', required=True)
    @click.option('--dataset_pull', help='The dataset pull URL.', required=False, default=None)
    def dataset_create(dataset_type, dataset_format, dataset_pull):
        """Create a dataset and return the id"""
        id = click_obj.dataset_create(dataset_type, dataset_format, dataset_pull)
        click.echo(f"{id}")

    @wrapper.command()
    @click.option('--network_arch', prompt='network_arch', type=click.Choice(network_type), help='Network architecture.', required=True)
    @click.option('--encryption_key', prompt='encryption_key', help='Encryption_key.', required=True)
    def experiment_create(network_arch, encryption_key):
        """Create an experiment and return the id"""
        id = click_obj.experiment_create(network_arch, encryption_key)
        click.echo(f"{id}")

    @wrapper.command()
    @click.option('--id', prompt='id', help='Dataset ID', required=True)
    def dataset_delete(id):
        """Delete a dataset"""
        id = click_obj.artifact_delete("dataset", id)
        click.echo(f"{id}")

    @wrapper.command()
    @click.option('--id', prompt='id', help='Experiment ID.', required=True)
    def experiment_delete(id):
        """Delete an experiment"""
        id = click_obj.artifact_delete("experiment", id)
        click.echo(f"{id}")

    wrapper.add_command(get_log_file)
    wrapper.add_command(list_job_files)
    wrapper.add_command(download_selective_files)
    wrapper.add_command(download_entire_job)
    wrapper.add_command(list_datasets)
    wrapper.add_command(list_experiments)
    wrapper.add_command(get_metadata)
    wrapper.add_command(patch_artifact_metadata)
    wrapper.add_command(get_spec)
    wrapper.add_command(model_automl_defaults)
    wrapper.add_command(dataset_run_action)
    wrapper.add_command(experiment_run_action)
    wrapper.add_command(get_action_status)
    wrapper.add_command(job_cancel)
    wrapper.add_command(job_resume)
    wrapper.add_command(dataset_upload)
    wrapper.add_command(dataset_create)
    wrapper.add_command(experiment_create)
    wrapper.add_command(dataset_delete)
    wrapper.add_command(experiment_delete)
    return wrapper
