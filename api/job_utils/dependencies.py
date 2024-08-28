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

"""
Dependency check modules

1. Dataset - train (tfrecords, labels, images), evaluate, inference, calibration datasets depending on task
2. Model - base_experiment, resume, .tlt from parent, .engine from parent, class map for some tasks, cal cache from parent for convert
3. Platflorm - GPU
4. Specs validation - Use Steve's code hardening
5. Parent job done? - Poll status from metadata

"""
import os
import sys

from constants import NO_PTM_MODELS, TAO_NETWORKS
from handlers.utilities import load_json_spec, search_for_base_experiment, validate_and_update_base_experiment_metadata, get_num_gpus_from_spec
from handlers.stateless_handlers import get_root, get_handler_root, get_jobs_root, get_handler_spec_root, update_job_status, get_handler_job_metadata, get_handler_metadata, get_handler_id, get_base_experiment_metadata, base_exp_uuid
from job_utils import executor
from utils import safe_load_file


def dependency_check_parent(job_context, dependency):
    """Check if parent job is valid and in Done status"""
    parent_job_id = job_context.parent_id
    org_name = job_context.org_name
    # If no parent job, this is always True
    if parent_job_id is None:
        return True, ""
    parent_handler_id = get_handler_id(org_name, parent_job_id)
    # Monai jobs can run without parent_status.
    if parent_handler_id is None and "medical" in job_context.network:
        return True, ""
    parent_job_metadata = get_handler_job_metadata(org_name, parent_handler_id, parent_job_id)
    parent_action = parent_job_metadata.get("action", "")
    if parent_action == "annotation":
        return True, ""
    parent_status = parent_job_metadata.get("status", "")
    parent_root = os.path.join(get_jobs_root(job_context.user_id, org_name), parent_job_id)
    # Parent Job must be done or canceled
    # Parent job output folder must exist
    failure_message = ""

    # Set current job to Error if parent err's out
    if parent_status == "Error":
        handler_kind = "experiments"
        if job_context.handler_id in os.listdir(os.path.join(get_root(), org_name, "datasets")):
            handler_kind = "datasets"
        update_job_status(org_name, job_context.handler_id, job_context.id, parent_status, kind=handler_kind)
        return False, f"Parent job {parent_job_id} errored out"

    if parent_status not in ("Done", "Canceled"):
        failure_message += f"Parent job {parent_job_id}'s status is not Done/Canceled"
    if not os.path.isdir(parent_root):
        failure_message += f" Parent job {parent_job_id}'s folder {parent_root} doesn't exist"
    return bool(parent_status in ("Done", "Canceled") and os.path.isdir(parent_root)), failure_message


def dependency_check_specs(job_context, dependency):
    """Check if valid spec exists for the requested action"""
    # If specs is not None, it means specs is already loaded
    if job_context.specs is not None:
        return True, ""

    network = job_context.network
    action = job_context.action
    handler_id = job_context.handler_id

    handler_spec_root = get_handler_spec_root(job_context.user_id, job_context.org_name, handler_id)
    spec_json_path = os.path.join(handler_spec_root, f"{job_context.id}-{job_context.action}-spec.json")
    load_json_spec(spec_json_path)

    failure_message = ""
    if not os.path.exists(spec_json_path):
        failure_message = f"Specs for network {network} and action {action} can't be found"
    return bool(os.path.exists(spec_json_path)), failure_message


def dependency_check_dataset(job_context, dependency):
    """Returns always true for dataset dependency check"""
    handler_id = job_context.handler_id
    org_name = job_context.org_name

    handler_metadata = get_handler_metadata(org_name, handler_id)
    valid_datset_structure = True
    train_datasets = handler_metadata.get("train_datasets", None)
    eval_dataset = handler_metadata.get("eval_dataset", None)
    inference_dataset = handler_metadata.get("inference_dataset", None)

    def append_dataset_id_to_message(dataset_metadata, valid_datset_structure):
        if not valid_datset_structure:
            return f'{dataset_metadata.get("id")}, '
        return ''

    invalid_datasets = ""
    if handler_metadata.get("type", "vision").lower() == "medical":
        # bypass the checks as the datasets are not downloaded for medical jobs at the time of job creation.
        valid_datset_structure = True

    if not handler_metadata.get("network_arch", ""):  # For dataset convert jobs, we have dataset info directly in metadata
        valid_datset_structure = handler_metadata.get("status") == "pull_complete"
        invalid_datasets += append_dataset_id_to_message(handler_metadata, valid_datset_structure)
    elif train_datasets:
        for train_ds in train_datasets:
            dataset_metadata = get_handler_metadata(org_name, train_ds)
            valid_datset_structure = dataset_metadata.get("status") == "pull_complete"
            invalid_datasets += append_dataset_id_to_message(dataset_metadata, valid_datset_structure)
    if eval_dataset:
        dataset_metadata = get_handler_metadata(org_name, eval_dataset)
        valid_datset_structure = dataset_metadata.get("status") == "pull_complete"
        invalid_datasets += append_dataset_id_to_message(dataset_metadata, valid_datset_structure)
    if inference_dataset:
        dataset_metadata = get_handler_metadata(org_name, inference_dataset)
        valid_datset_structure = dataset_metadata.get("status") == "pull_complete"
        invalid_datasets += append_dataset_id_to_message(dataset_metadata, valid_datset_structure)

    failure_message = ""
    if not valid_datset_structure:
        failure_message = f"Dataset(s) {invalid_datasets} still uploading, or uploaded data doesn't match the directory structure defined for this network"
    return valid_datset_structure, failure_message


def dependency_check_model(job_context, dependency):
    """Checks if valid base_experiment model exists"""
    network = job_context.network
    handler_id = job_context.handler_id

    handler_metadata = get_handler_metadata(job_context.org_name, handler_id)
    # If it is a dataset, no model dependency
    if "train_datasets" not in handler_metadata.keys():
        return True, ""
    if network in NO_PTM_MODELS:
        return True, ""
    base_experiment_ids = handler_metadata.get("base_experiment", None)
    for base_experiment_id in base_experiment_ids:
        if not base_experiment_id:
            return False, "Base Experiment ID is None"
        base_experiment_metadata = get_base_experiment_metadata(base_experiment_id)
        if not base_experiment_metadata:
            # Search in the base_exp_uuid fails, search in the org_name
            base_experiment_metadata = get_handler_metadata(job_context.org_name, base_experiment_id)

        if network in TAO_NETWORKS and not base_experiment_metadata.get("base_experiment_metadata", {}).get("spec_file_present"):  # For TAO models, if the B.E doesn't have a spec file attached in NGC, then we don't need to check anything
            return True, ""

        base_experiment_root = get_handler_root(base_exp_uuid, "experiments", base_exp_uuid, base_experiment_id)
        if not base_experiment_root:
            # Search in the base_exp_uuid fails, search in the org_name
            base_experiment_root = get_handler_root(org_name=job_context.org_name, kind="experiments", handler_id=base_experiment_id)

        if not base_experiment_root:
            return False, f"Base experiment ID {base_experiment_id} is not found"

        spec = False
        if network in TAO_NETWORKS:
            spec = True
        base_experiment_file = search_for_base_experiment(base_experiment_root, network=network, spec=spec)
        if not base_experiment_file:
            return False, f"Base Experiment file for ID {base_experiment_id} is not found"

        if base_experiment_metadata.get("base_experiment_pull_complete") != "pull_complete":
            validate_and_update_base_experiment_metadata(base_experiment_file, base_experiment_id, base_experiment_metadata)
            print("base_experiment_metadata", base_experiment_metadata, file=sys.stderr)
            return False, f"Base Experiment file for ID {base_experiment_id} is being downloaded or downloaded file is corrupt"
    return True, ""


def dependency_check_gpu(job_context, dependency):
    """Check if GPU dependency is met"""
    # If BACKEND is either BCP or NVCF, then we don't need to check for GPU availability if it's not a local job
    local_job = (job_context.specs and "cluster" in job_context.specs and job_context.specs["cluster"] == "local")
    if os.getenv("BACKEND") in ("BCP", "NVCF") and not local_job:
        return True, ""
    num_gpu = get_num_gpus_from_spec(job_context.specs, job_context.action, default=dependency.num)
    gpu_available = executor.dependency_check(num_gpu=num_gpu, accelerator=dependency.name)
    message = ""
    if not gpu_available:
        message = "GPU's needed to run this job is not available yet, please wait for other jobs to complete"
    return gpu_available, message


def dependency_check_default(job_context, dependency):
    """Returns a default value of False when dependency type is not present in dependency_type_map"""
    return False, "Requested dependency not found"


def dependency_check_automl(job_context, dependency):
    """Makes sure the controller.json has the rec_number requested at the time of creation"""
    rec_number = int(dependency.name)
    root = get_jobs_root(job_context.user_id, job_context.org_name)
    # Check if recommendation number is there and can be loaded
    file_path = root + f"/{job_context.id}/controller.json"
    if not os.path.exists(file_path):
        return False, f"Automl controller json for job id {job_context.id} not found yet"
    recs_dict = safe_load_file(file_path)
    try:
        recs_dict[rec_number]
        return True, ""
    except:
        return False, f"Recommendation number {rec_number} requested is not available yet"


dependency_type_map = {
    'parent': dependency_check_parent,
    'specs': dependency_check_specs,
    'dataset': dependency_check_dataset,
    'model': dependency_check_model,
    'gpu':  dependency_check_gpu,
    "automl": dependency_check_automl,
}
