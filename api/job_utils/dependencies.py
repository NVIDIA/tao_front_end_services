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

"""
Dependency check modules

1. Dataset - train (tfrecords, labels, images), evaluate, inference, calibration datasets depending on task
2. Model - ptm, resume, .tlt from parent, .engine from parent, class map for some tasks, cal cache from parent for convert
3. Platflorm - GPU
4. Specs validation - Use Steve's code hardening
5. Parent job done? - Poll status from metadata

"""
import os
import json
from handlers.ds_upload import DS_UPLOAD_TO_FUNCTIONS
from handlers.utilities import get_handler_root, load_json_spec, search_for_ptm, NO_PTM_MODELS
from handlers.stateless_handlers import get_handler_spec_root, get_handler_job_metadata, get_handler_metadata
from job_utils import executor


def dependency_check_parent(job_context, dependency):
    """Check if parent job is valid and in Done status"""
    parent_job_id = job_context.parent_id
    # If no parent job, this is always True
    if parent_job_id is None:
        return True, ""
    handler_id = job_context.handler_id
    parent_status = get_handler_job_metadata(handler_id, parent_job_id).get("status", "Error")
    parent_root = os.path.join(get_handler_root(handler_id), parent_job_id)
    # Parent Job must be done
    # Parent job output folder must exist
    failure_message = ""
    if not parent_status == "Done":
        failure_message += f"Parent job {parent_job_id}'s status is not Done"
    if not os.path.isdir(parent_root):
        failure_message += f" Parent job {parent_job_id}'s folder doesn't exist"
    return bool(parent_status == "Done" and os.path.isdir(parent_root)), failure_message


def dependency_check_specs(job_context, dependency):
    """Check if valid spec exists for the requested action"""
    network = job_context.network
    action = job_context.action
    handler_id = job_context.handler_id

    DIR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    CSV_PATH = os.path.join(DIR_PATH, "specs_utils", "specs", network, f"{network} - {action}.csv")
    if not os.path.exists(CSV_PATH):
        metadata = get_handler_metadata(handler_id)
        # Try secondary format for CSV_PATH => "<network> - <action>__<dataset-format>.csv"
        fmt = metadata.get("format", "_")
        CSV_PATH = os.path.join(DIR_PATH, "specs_utils", "specs", network, f"{network} - {action}__{fmt}.csv")

    handler_spec_root = get_handler_spec_root(handler_id)
    spec_json_path = os.path.join(handler_spec_root, action + ".json")
    load_json_spec(spec_json_path)

    failure_message = ""
    if not os.path.exists(spec_json_path):
        failure_message = f"Specs for network {network} and action {action} can't be found"
    return bool(os.path.exists(spec_json_path)), failure_message


def dependency_check_dataset(job_context, dependency):
    """Returns always true for dataset dependency check"""
    network = job_context.network
    handler_id = job_context.handler_id
    handler_metadata = get_handler_metadata(handler_id)
    valid_datset_structure = True
    ds_upload_keys = DS_UPLOAD_TO_FUNCTIONS.keys()
    if handler_metadata.get("train_datasets", []):
        train_ds_list = handler_metadata.get("train_datasets", [])
        for train_ds in train_ds_list:
            dataset_metadata = get_handler_metadata(train_ds)
            if dataset_metadata.get("type", None) in ds_upload_keys:
                valid_datset_structure = DS_UPLOAD_TO_FUNCTIONS[dataset_metadata.get("type")](dataset_metadata)
        if network in ("detectnet_v2", "faster_rcnn", "yolo_v3", "yolo_v4", "yolo_v4_tiny", "ssd", "dssd", "retinanet"):
            eval_ds = handler_metadata.get("eval_dataset", None)
            dataset_metadata = get_handler_metadata(eval_ds)
            if dataset_metadata.get("type", None) in ds_upload_keys:
                valid_datset_structure = DS_UPLOAD_TO_FUNCTIONS[dataset_metadata.get("type")](dataset_metadata)
        if network in ("detectnet_v2"):
            infer_ds = handler_metadata.get("inference_dataset", None)
            dataset_metadata = get_handler_metadata(infer_ds)
            if dataset_metadata.get("type", None) in ds_upload_keys:
                valid_datset_structure = DS_UPLOAD_TO_FUNCTIONS[dataset_metadata.get("type")](dataset_metadata)
    elif handler_metadata.get("type", None) is None:
        eval_valid_datset_structure = True
        test_valid_datset_structure = True
        eval_ds = handler_metadata.get("eval_dataset", None)
        if eval_ds:
            dataset_metadata = get_handler_metadata(eval_ds)
            if dataset_metadata.get("type", None) in ds_upload_keys:
                eval_valid_datset_structure = DS_UPLOAD_TO_FUNCTIONS[dataset_metadata.get("type")](dataset_metadata)
        infer_ds = handler_metadata.get("inference_dataset", None)
        if infer_ds:
            dataset_metadata = get_handler_metadata(infer_ds)
            if dataset_metadata.get("type", None) in ds_upload_keys:
                test_valid_datset_structure = DS_UPLOAD_TO_FUNCTIONS[dataset_metadata.get("type")](dataset_metadata)
        valid_datset_structure = eval_valid_datset_structure and test_valid_datset_structure
    else:
        if handler_metadata.get("type", None) in ds_upload_keys:
            valid_datset_structure = DS_UPLOAD_TO_FUNCTIONS[handler_metadata.get("type")](handler_metadata)

    failure_message = ""
    if not valid_datset_structure:
        failure_message = "Dataset still uploading, or uploaded data doesn't match the directory structure defined for this network"
    return valid_datset_structure, failure_message


def dependency_check_model(job_context, dependency):
    """Checks if valid ptm model exists"""
    network = job_context.network
    handler_id = job_context.handler_id

    handler_metadata = get_handler_metadata(handler_id)
    # If it is a dataset, no model dependency
    if "train_datasets" not in handler_metadata.keys():
        return True, ""
    if network in NO_PTM_MODELS:
        return True, ""
    ptm_ids = handler_metadata.get("ptm", None)
    for ptm_id in ptm_ids:
        if not ptm_id:
            return False, "PTM_ID is None"
        if not search_for_ptm(get_handler_root(ptm_id), network=network):
            return False, f"PTM file path for PTM ID {ptm_id} not found yet"

    return True, ""


def dependency_check_automl_specs(job_context, dependency):
    """Checks if train.json is present for automl job"""
    spec_json_path = os.path.join(get_handler_spec_root(job_context.handler_id), "train.json")
    failure_message = ""
    if not os.path.exists(spec_json_path):
        failure_message = f"AutoML specs path for handler id {job_context.handler_id} not found"
    return bool(os.path.exists(spec_json_path)), failure_message


def dependency_check_gpu(job_context, dependency):
    """Check if GPU dependency is met"""
    gpu_available = executor.dependency_check(accelerator=dependency.name)
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
    root = get_handler_root(job_context.handler_id)
    # Check if recommendation number is there and can be loaded
    file_path = root + f"/{job_context.id}/controller.json"
    if not os.path.exists(file_path):
        return False, f"Automl controller json for job id {job_context.id} not found yet"
    with open(file_path, 'r', encoding='utf-8') as f:
        recs_dict = json.loads(f.read())
    try:
        recs_dict[rec_number]
        return True, ""
    except:
        return False, f"Recommendation number {rec_number} requested is not available yet"


def dependency_check_automl_ptm(job_context, dependency):
    """Checks if valid ptm model exists for automl job"""
    network = job_context.network
    ptm_id = dependency.name
    if ptm_id:
        ptm_exists = bool(search_for_ptm(get_handler_root(ptm_id), network=network))
        failure_message = ""
        if not ptm_exists:
            failure_message = f"AutoML PTM file path for ptm_id {ptm_id} doesn't exists yet"
        return ptm_exists, failure_message
    return True


dependency_type_map = {
    'parent': dependency_check_parent,
    'specs': dependency_check_specs,
    'dataset': dependency_check_dataset,
    'model': dependency_check_model,
    'gpu':  dependency_check_gpu,
    "automl": dependency_check_automl,
    "automl_ptm": dependency_check_automl_ptm,
    "automl_specs": dependency_check_automl_specs
}
