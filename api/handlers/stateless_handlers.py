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

"""API Stateless handlers modules"""
import datetime
import glob
import json
import os
import functools


def get_root():
    """Return root path"""
    return os.environ.get("TAO_ROOT", "/shared/users/")


# Sub for handler.root with handler_root(handler_id)
@functools.lru_cache(maxsize=256)
def get_handler_root(handler_id):
    """Return handler root path"""
    pattern = get_root() + "**/**/**"
    elements = glob.glob(pattern)
    for ele in elements:
        if os.path.basename(ele.rstrip("///")) == handler_id:
            return ele
    return ""


# Sub for handler.spec_root with handler_root(handler_id)
def get_handler_spec_root(handler_id):
    """Return path of specs folder under handler_root"""
    return os.path.join(get_handler_root(handler_id), "specs")


def get_handler_log_root(handler_id):
    """Return path of logs folder under handler_root"""
    return os.path.join(get_handler_root(handler_id), "logs")


def get_handler_job_metadata(handler_id, job_id):
    """Return metadata info present in job_id.json inside jobs_metadata folder"""
    # Only metadata of a particular job
    handler_root = get_handler_root(handler_id)
    job_metadata_file = handler_root + f"/jobs_metadata/{job_id}.json"
    if not os.path.exists(job_metadata_file):
        return {}
    with open(job_metadata_file, "r", encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata


def get_job_files(handler_id, job_id, retrieve_logs=False, retrieve_specs=False):
    """Return metadata info present in job_id.json inside jobs_metadata folder"""
    # Only metadata of a particular job
    logs_folder = ""
    specs_folder = ""
    if retrieve_logs:
        logs_folder = get_handler_log_root(handler_id)
    if retrieve_specs:
        specs_folder = get_handler_spec_root(handler_id)

    handler_root = get_handler_root(handler_id)
    job_folder = os.path.join(handler_root, job_id)
    if not os.path.exists(job_folder) and (retrieve_logs and not os.path.exists(logs_folder)) and (retrieve_specs and not os.path.exists(specs_folder)):
        return []
    files = glob.glob(f"{job_folder}/**", recursive=True)
    if logs_folder and os.path.exists(logs_folder):
        files += glob.glob(f"{logs_folder}/**", recursive=True)
    if specs_folder and os.path.exists(specs_folder):
        files += glob.glob(f"{specs_folder}/**", recursive=True)
    files = [os.path.relpath(file, handler_root) for file in files if not file.endswith('/')]
    return files


def get_toolkit_status(handler_id, job_id):
    """Returns the status of the job reported from the frameworks container"""
    metadata_info = get_handler_job_metadata(handler_id, job_id)
    toolkit_status = ""
    result_dict = metadata_info.get("result", "")
    if result_dict:
        toolkit_detailed_status = result_dict.get("detailed_status", "")
        if toolkit_detailed_status:
            toolkit_status = toolkit_detailed_status.get("status", "")
    return toolkit_status


def json_serializable(response):
    """Check if response is json serializable"""
    try:
        json.dumps(response.json())
        return True
    except:
        return False


# Sub for handler.metadata_file with handler_root(handler_id)
def get_handler_metadata_file(handler_id):
    """Return path of metadata.json under handler_root"""
    return get_handler_root(handler_id) + "/metadata.json"


def get_handler_jobs_metadata_root(handler_id):
    """Return path of job_metadata folder folder under handler_root"""
    return get_handler_root(handler_id) + "/jobs_metadata/"


def load_json_data(json_file):
    """Read data from json file"""
    metadata = {}
    if os.path.exists(json_file):
        with open(json_file, "r", encoding='utf-8') as f:
            metadata = json.load(f)

    return metadata


def get_handler_metadata(handler_id):
    """Return metadata info present in metadata.json inside handler_root"""
    metadata_file = get_handler_metadata_file(handler_id)
    metadata = load_json_data(metadata_file)
    return metadata


def write_handler_metadata(handler_id, metadata):
    """Return metadata info present in metadata.json inside handler_root"""
    metadata_file = get_handler_metadata_file(handler_id)
    with open(metadata_file, "w+", encoding='utf-8') as f:
        f.write(json.dumps(metadata, indent=4))
    return metadata


def get_handler_metadata_with_jobs(handler_id):
    """Return a list of job_metadata info of multiple jobs"""
    metadata = get_handler_metadata(handler_id)
    metadata["jobs"] = []
    job_metadatas_root = get_handler_jobs_metadata_root(handler_id)
    for json_file in glob.glob(job_metadatas_root + "*.json"):
        metadata["jobs"].append(load_json_data(json_file))
    return metadata


def write_job_metadata(handler_id, job_id, metadata):
    """Write job metadata info present in jobs_metadata folder"""
    handler_root = get_handler_root(handler_id)
    job_metadata_file = handler_root + f"/jobs_metadata/{job_id}_tmp.json"
    with open(job_metadata_file, "w+", encoding='utf-8') as f:
        f.write(json.dumps(metadata, indent=4))
    job_metadata_file_orig = handler_root + f"/jobs_metadata/{job_id}.json"
    os.rename(job_metadata_file, job_metadata_file_orig)


def update_job_status(handler_id, job_id, status):
    """Update the job status in jobs_metadata/job_id.json"""
    metadata = get_handler_job_metadata(handler_id, job_id)
    if status != metadata.get("status", ""):
        metadata["last_modified"] = datetime.datetime.now().isoformat()
    metadata["status"] = status

    write_job_metadata(handler_id, job_id, metadata)


def update_job_results(handler_id, job_id, result):
    """Update the job results in jobs_metadata/job_id.json"""
    metadata = get_handler_job_metadata(handler_id, job_id)
    if result != metadata.get("result", {}):
        metadata["last_modified"] = datetime.datetime.now().isoformat()
    metadata["result"] = result

    write_job_metadata(handler_id, job_id, metadata)


def get_handler_user(handler_id):
    """Return the user id for the handler id provided"""
    # Get the handler_root in all the paths
    hander_root = get_handler_root(handler_id)
    # Remove ant final backslashes in the path and take 3rd element from last
    return hander_root.rstrip("///").split("/")[-3]


def get_handler_type(handler_id):
    """Return the handler type"""
    handler_metadata = get_handler_metadata(handler_id)
    network = handler_metadata.get("network_arch", None)
    if not network:
        network = handler_metadata.get("type", None)
    return network


def make_root_dirs(user_id, kind, handler_id):
    """Create root dir followed by logs, jobs_metadata and specs folder"""
    root = get_root() + f"{user_id}/{kind}/{handler_id}/"
    log_root = root + "logs/"
    jobs_meta_root = root + "jobs_metadata/"
    spec_root = root + "specs/"
    for directory in [root, log_root, jobs_meta_root, spec_root]:
        if not os.path.exists(directory):
            os.makedirs(directory)


def check_existence(handler_id, kind):
    """Check if metadata.json exists"""
    if kind not in ["dataset", "model"]:
        return False
    existence = bool(glob.glob(get_root() + f"**/{kind}s/{handler_id}/metadata.json"))
    return existence


def check_read_access(user_id, handler_id):
    """Check if the user has read access to this particular handler"""
    handler_user = get_handler_user(handler_id)
    under_user = handler_user == user_id

    handler_metadata = get_handler_metadata(handler_id)
    public = handler_metadata.get("public", False)  # Default is False

    if under_user:
        return True
    if public:
        return True
    return False


def check_write_access(user_id, handler_id):
    """Check if the user has write access to this particular handler"""
    handler_user = get_handler_user(handler_id)
    under_user = handler_user == user_id

    handler_metadata = get_handler_metadata(handler_id)
    public = handler_metadata.get("public", False)  # Default is False
    read_only = handler_metadata.get("read_only", False)  # Default is False
    if under_user:  # If under user, you can always write - no point in making it un-writable by owner. Read-only is for non-owners
        return True
    if public:
        if read_only:
            return False
        return True
    return False


def get_public_models():
    """Get public models"""
    # Make sure to check if it exists
    public_models = []
    all_models_metadata = get_root() + "**/models/**/metadata.json"
    for metadata_file in glob.glob(all_models_metadata):
        metadata = load_json_data(metadata_file)
        public = metadata.get("public", False)
        if public:
            public_models.append(metadata.get("id"))
    return list(set(public_models))


def get_public_datasets():
    """Get public datasets"""
    public_datasets = []
    return list(set(public_datasets))


def add_public_model(model_id):
    """Add public model"""
    # if model_id in get_public_models():
    #     return
    return


def add_public_dataset(dataset_id):
    """Add public dataset"""
    # if dataset_id in get_public_datasets():
    #     return
    return


def remove_public_model(model_id):
    """Remove public model"""
    # if model_id not in get_public_models():
    #     return
    return


def remove_public_dataset(dataset_id):
    """Remove public dataset"""
    # if dataset_id not in get_public_datasets():
    #     return
    return


def check_dataset_type_match(user_id, model_meta, dataset_id, no_raw=None):
    """Checks if the dataset created for the model is valid dataset_type"""
    # If dataset id is None, then return True
    # Else, if all things match, return True
    # True means replace, False means skip and return a 400 Code
    if dataset_id is None:
        return True
    if not check_existence(dataset_id, "dataset"):
        return False
    if not check_read_access(user_id, dataset_id):
        return False

    dataset_meta = get_handler_metadata(dataset_id)

    model_dataset_type = model_meta.get("dataset_type")
    dataset_type = dataset_meta.get("type")
    dataset_format = dataset_meta.get("format")
    if model_dataset_type != dataset_type:
        return False

    if no_raw:
        if dataset_format in ("raw", "coco_raw"):
            return False

    return True


def check_model_type_match(user_id, model_meta, ptm_ids):
    """Checks if the model created and ptm requested belong to the same network"""
    if ptm_ids is None:
        return True
    for ptm_id in ptm_ids:
        if not check_existence(ptm_id, "model"):
            return False

        if not check_read_access(user_id, ptm_id):
            return False

        ptm_meta = get_handler_metadata(ptm_id)

        model_arch = model_meta.get("network_arch")
        ptm_arch = ptm_meta.get("network_arch")
        if model_arch != ptm_arch:
            return False

    return True


def check_checkpoint_choose_match(technique):
    """Checks if technique chosen for checkpoint retrieve is a valid option"""
    if technique not in ("best_model", "latest_model", "from_epoch_number"):
        return False
    return True


def check_checkpoint_epoch_number_match(epoch_number_dictionary):
    """Checks if the epoch number requested to retrieve checkpoint is a valid number"""
    try:
        for key in epoch_number_dictionary.keys():
            _ = int(epoch_number_dictionary[key])
    except:
        return False
    return True


def model_update_handler_attributes(user_id, model_meta, key, value):
    """Checks if the artifact provided is of the correct type"""
    # Returns value or False
    if key in ["train_datasets"]:
        if type(value) is not list:
            value = [value]
        for dataset_id in value:
            if not check_dataset_type_match(user_id, model_meta, dataset_id, no_raw=True):
                return False
    elif key in ["eval_dataset"]:
        if not check_dataset_type_match(user_id, model_meta, value, no_raw=True):
            return False
    elif key in ["calibration_dataset", "inference_dataset"]:
        if not check_dataset_type_match(user_id, model_meta, value):
            return False
    elif key in ["ptm"]:
        if not check_model_type_match(user_id, model_meta, value):
            return False
    elif key in ["checkpoint_choose_method"]:
        if not check_checkpoint_choose_match(value):
            return False
    elif key in ["checkpoint_epoch_number"]:
        if not check_checkpoint_epoch_number_match(value):
            return False
    else:
        return False
    return value
