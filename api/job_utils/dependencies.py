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
from handlers.ds_upload import DS_UPLOAD_TO_FUNCTIONS
from handlers.utilities import get_handler_root, load_json_spec, search_for_base_experiment, validate_and_update_base_experiment_metadata, NO_PTM_MODELS
from handlers.stateless_handlers import get_handler_spec_root, get_handler_job_metadata, get_handler_metadata, safe_load_file, get_handler_id, get_base_experiment_metadata, admin_uuid
from job_utils import executor


def dependency_check_parent(job_context, dependency):
    """Check if parent job is valid and in Done status"""
    parent_job_id = job_context.parent_id
    user_id = job_context.user_id
    # If no parent job, this is always True
    if parent_job_id is None:
        return True, ""
    parent_handler_id = get_handler_id(user_id, parent_job_id, ngc_runner_fetch=True)
    # Medical jobs can run without parent_status.
    if parent_handler_id is None and "medical" in job_context.network:
        return True, ""
    parent_action = get_handler_job_metadata(user_id, parent_handler_id, parent_job_id).get("action", "")
    if parent_action == "annotation":
        return True, ""
    parent_status = get_handler_job_metadata(user_id, parent_handler_id, parent_job_id).get("status", "Error")
    parent_root = os.path.join(get_handler_root(user_id, None, parent_handler_id, None, ngc_runner_fetch=True), parent_job_id)
    # Parent Job must be done
    # Parent job output folder must exist
    failure_message = ""
    if not parent_status == "Done":
        failure_message += f"Parent job {parent_job_id}'s status is not Done"
    if not os.path.isdir(parent_root):
        failure_message += f" Parent job {parent_job_id}'s folder {parent_root} doesn't exist"
    return bool(parent_status == "Done" and os.path.isdir(parent_root)), failure_message


def dependency_check_specs(job_context, dependency):
    """Check if valid spec exists for the requested action"""
    # If specs is not None, it means specs is already loaded
    if job_context.specs is not None:
        return True, ""

    network = job_context.network
    action = job_context.action
    handler_id = job_context.handler_id

    handler_spec_root = get_handler_spec_root(job_context.user_id, handler_id)
    spec_json_path = os.path.join(handler_spec_root, f"{job_context.id}-{job_context.action}-spec.json")
    load_json_spec(spec_json_path)

    failure_message = ""
    if not os.path.exists(spec_json_path):
        failure_message = f"Specs for network {network} and action {action} can't be found"
    return bool(os.path.exists(spec_json_path)), failure_message


def dependency_check_dataset(job_context, dependency):
    """Returns always true for dataset dependency check"""
    network = job_context.network
    handler_id = job_context.handler_id
    user_id = job_context.user_id

    handler_metadata = get_handler_metadata(user_id, handler_id)
    valid_datset_structure = True
    ds_upload_keys = DS_UPLOAD_TO_FUNCTIONS.keys()
    if handler_metadata.get("type", "vision").lower() == "medical":
        # bypass the checks as the datasets are not downloaded for medical jobs at the time of job creation.
        valid_datset_structure = True
    elif handler_metadata.get("train_datasets", []):
        train_ds_list = handler_metadata.get("train_datasets", [])
        for train_ds in train_ds_list:
            dataset_metadata = get_handler_metadata(user_id, train_ds)
            if dataset_metadata.get("type", None) in ds_upload_keys:
                valid_datset_structure = DS_UPLOAD_TO_FUNCTIONS[dataset_metadata.get("type")](user_id, dataset_metadata) and dataset_metadata.get("status") == "present"
        if network in ("detectnet_v2", "faster_rcnn", "yolo_v3", "yolo_v4", "yolo_v4_tiny", "ssd", "dssd", "retinanet"):
            eval_ds = handler_metadata.get("eval_dataset", None)
            dataset_metadata = get_handler_metadata(user_id, eval_ds)
            if dataset_metadata.get("type", None) in ds_upload_keys:
                valid_datset_structure = DS_UPLOAD_TO_FUNCTIONS[dataset_metadata.get("type")](user_id, dataset_metadata) and dataset_metadata.get("status") == "present"
        if network in ["detectnet_v2"]:
            infer_ds = handler_metadata.get("inference_dataset", None)
            dataset_metadata = get_handler_metadata(user_id, infer_ds)
            if dataset_metadata.get("type", None) in ds_upload_keys:
                valid_datset_structure = DS_UPLOAD_TO_FUNCTIONS[dataset_metadata.get("type")](user_id, dataset_metadata) and dataset_metadata.get("status") == "present"
    elif handler_metadata.get("type", None) is None:
        eval_valid_datset_structure = True
        test_valid_datset_structure = True
        eval_ds = handler_metadata.get("eval_dataset", None)
        if eval_ds:
            dataset_metadata = get_handler_metadata(user_id, eval_ds)
            if dataset_metadata.get("type", None) in ds_upload_keys:
                eval_valid_datset_structure = DS_UPLOAD_TO_FUNCTIONS[dataset_metadata.get("type")](user_id, dataset_metadata) and dataset_metadata.get("status") == "present"
        infer_ds = handler_metadata.get("inference_dataset", None)
        if infer_ds:
            dataset_metadata = get_handler_metadata(user_id, infer_ds)
            if dataset_metadata.get("type", None) in ds_upload_keys:
                test_valid_datset_structure = DS_UPLOAD_TO_FUNCTIONS[dataset_metadata.get("type")](user_id, dataset_metadata) and dataset_metadata.get("status") == "present"
        valid_datset_structure = eval_valid_datset_structure and test_valid_datset_structure
    else:
        if handler_metadata.get("type", None) in ds_upload_keys:
            valid_datset_structure = DS_UPLOAD_TO_FUNCTIONS[handler_metadata.get("type")](user_id, handler_metadata) and handler_metadata.get("status") == "present"

    failure_message = ""
    if not valid_datset_structure:
        failure_message = "Dataset still uploading, or uploaded data doesn't match the directory structure defined for this network"
    return valid_datset_structure, failure_message


def dependency_check_model(job_context, dependency):
    """Checks if valid base_experiment model exists"""
    network = job_context.network
    handler_id = job_context.handler_id

    handler_metadata = get_handler_metadata(job_context.user_id, handler_id)
    # If it is a dataset, no model dependency
    if "train_datasets" not in handler_metadata.keys():
        return True, ""
    if network in NO_PTM_MODELS:
        return True, ""
    base_experiment_ids = handler_metadata.get("base_experiment", None)
    for base_experiment_id in base_experiment_ids:
        if not base_experiment_id:
            return False, "Base Experiment ID is None"
        base_experiment_root = get_handler_root(admin_uuid, "experiments", admin_uuid, base_experiment_id, ngc_runner_fetch=True)
        if not base_experiment_root:
            # Search in the admin_uuid fails, search in the user_id
            base_experiment_root = get_handler_root(user_id=job_context.user_id, kind="experiments", handler_id=base_experiment_id, ngc_runner_fetch=True)
        if not base_experiment_root:
            return False, f"Base experiment ID {base_experiment_id} is not found"
        base_experiment_file = search_for_base_experiment(base_experiment_root, network=network)
        if not search_for_base_experiment(base_experiment_root, network=network):
            return False, f"Base Experiment file for ID {base_experiment_id} is not found"
        base_experiment_metadata = get_base_experiment_metadata(base_experiment_id)
        if not base_experiment_metadata:
            # Search in the admin_uuid fails, search in the user_id
            base_experiment_metadata = get_handler_metadata(job_context.user_id, base_experiment_id)
        if base_experiment_metadata.get("base_experiment_pull_complete") != "present":
            validate_and_update_base_experiment_metadata(base_experiment_file, base_experiment_id, base_experiment_metadata)
            print("base_experiment_metadata", base_experiment_metadata, file=sys.stderr)
            return False, f"Base Experiment file for ID {base_experiment_id} is being downloaded or downloaded file is corrupt"
    return True, ""


def dependency_check_gpu(job_context, dependency):
    """Check if GPU dependency is met"""
    num_gpu = -1
    if isinstance(job_context.specs, dict):
        spec = job_context.specs
        num_gpu = spec.get("num_gpu", -1) if spec else -1
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
    root = get_handler_root(job_context.user_id, "experiments", job_context.handler_id, None)
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
