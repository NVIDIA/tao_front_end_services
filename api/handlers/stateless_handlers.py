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

"""API Stateless handlers modules"""
import datetime
import glob
import shutil
import numpy as np
import os
import re
import sys
import subprocess
from filelock import FileLock
from pathlib import Path
import orjson
import yaml
import uuid
import traceback

ngc_runner = os.getenv("NGC_RUNNER", "")
tao_root = os.environ.get("TAO_ROOT", "/shared/users/")
admin_uuid = os.environ.get("ADMIN_UUID", "00000000-0000-0000-0000-000000000001")


def get_root(ngc_runner_fetch=False):
    """Return root path"""
    if ngc_runner_fetch and ngc_runner == "True":
        return "/users/"
    return tao_root


def __pathlib_glob(rootdir, handler_id, job_id):
    try:
        for entry in rootdir.glob('**/' + f"{handler_id}/{job_id}"):
            if entry.is_dir():
                return str(entry.resolve())
        return ""
    except:
        print("Issue during finding handler_root", traceback.format_exc(), file=sys.stderr)
        return ""


def get_handler_root(user_id=None, kind=None, handler_id=None, job_id=None, ngc_runner_fetch=False):
    """Return handler root path"""
    handler_id = handler_id if handler_id else ""
    job_id = job_id if job_id else ""
    rootdir = Path(get_root(ngc_runner_fetch))
    if not rootdir.exists():
        return ""
    if user_id:
        rootdir = rootdir / user_id
    else:
        return __pathlib_glob(rootdir, handler_id, job_id)
    if kind:
        rootdir = rootdir / f"{kind}/{handler_id}"
        if job_id:
            rootdir = rootdir / f"{job_id}"
        rootdir = str(rootdir.resolve())
        if os.path.exists(rootdir):
            return rootdir
    else:
        return __pathlib_glob(rootdir, handler_id, job_id)
    return ""


def get_base_experiment_path(handler_id, create_if_not_exist=True):
    """Return base_experiment root path"""
    base_experiment_path = f"{tao_root}/{admin_uuid}/experiments/{admin_uuid}/{handler_id}"
    if ngc_runner == "True":
        base_experiment_path = f"/users/{admin_uuid}/experiments/{admin_uuid}/{handler_id}"
    if not os.path.exists(base_experiment_path) and create_if_not_exist:
        os.makedirs(base_experiment_path, exist_ok=True)
        subprocess.getoutput(f"chmod -R 777 {base_experiment_path}")
    return base_experiment_path


def get_base_experiments_metadata_path():
    """Return base_experiment root path"""
    base_experiments_metadata_path = f"{tao_root}/{admin_uuid}/experiments/{admin_uuid}/ptm_metadatas.json"
    if ngc_runner == "True":
        base_experiments_metadata_path = f"/users/{admin_uuid}/experiments/{admin_uuid}/ptm_metadatas.json"
    return base_experiments_metadata_path


# Sub for handler.spec_root with handler_root(handler_id)
def get_handler_spec_root(user_id, handler_id):
    """Return path of specs folder under handler_root"""
    return os.path.join(get_handler_root(user_id, None, handler_id, None, ngc_runner_fetch=True), "specs")


def get_handler_log_root(user_id, handler_id, ngc_runner_fetch=True):
    """Return path of logs folder under handler_root"""
    return os.path.join(get_handler_root(user_id, None, handler_id, None, ngc_runner_fetch=ngc_runner_fetch), "logs")


def get_handler_job_metadata(user_id, handler_id, job_id):
    """Return metadata info present in job_id.json inside jobs_metadata folder"""
    # Only metadata of a particular job
    metadata = {}
    if job_id:
        handler_root = get_handler_root(user_id, None, handler_id, None)
        job_metadata_file = handler_root + f"/jobs_metadata/{job_id}.json"
        metadata = safe_load_file(job_metadata_file)
    return metadata


def get_job_files(user_id, handler_id, job_id, retrieve_logs=False, retrieve_specs=False, ngc_runner_fetch=False):
    """Return metadata info present in job_id.json inside jobs_metadata folder"""
    # Only metadata of a particular job
    logs_folder = ""
    specs_folder = ""
    if retrieve_logs:
        logs_folder = get_handler_log_root(user_id, handler_id)
    if retrieve_specs:
        specs_folder = get_handler_spec_root(user_id, handler_id)

    handler_root = get_handler_root(user_id, None, handler_id, None, ngc_runner_fetch=ngc_runner_fetch)
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


def get_toolkit_status(user_id, handler_id, job_id):
    """Returns the status of the job reported from the frameworks container"""
    metadata_info = get_handler_job_metadata(user_id, handler_id, job_id)
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
        orjson.dumps(response.json())
        return True
    except:
        return False


# Sub for handler.metadata_file with handler_root(handler_id)
def get_handler_metadata_file(user_id, handler_id, kind=None):
    """Return path of metadata.json under handler_root"""
    return get_handler_root(user_id, kind, handler_id, None) + "/metadata.json"


def get_handler_jobs_metadata_root(user_id, handler_id):
    """Return path of job_metadata folder folder under handler_root"""
    return get_handler_root(user_id, None, handler_id, None) + "/jobs_metadata/"


def read_base_experiment_metadata():
    """Read PTM metadata file and return the entire metadata info"""
    base_experiment_path = get_base_experiments_metadata_path()
    base_experiment_metadatas = safe_load_file(base_experiment_path)
    if not base_experiment_metadatas and os.path.exists(base_experiment_path):  # In-case the file was corrupted
        ptm_source_path = f"/opt/api/shared/users/{admin_uuid}/experiments/{admin_uuid}"
        if os.path.exists(ptm_source_path):
            shutil.copy(f"{ptm_source_path}/ptm_metadatas.json", base_experiment_path)
        else:
            print(f"Source path {ptm_source_path}: No such file or directory", file=sys.stderr)

    return base_experiment_metadatas


def get_base_experiment_metadata(base_experiment_id):
    """Read PTM metadata file and return the metadata info of a particular base_experiment"""
    base_experiment_metadatas = read_base_experiment_metadata()
    return base_experiment_metadatas.get(base_experiment_id, {})


def update_base_experiment_metadata(base_experiment_id, base_experiment_metadata_update):
    """Read PTM metadata file and update the metadata info of a particular base_experiment"""
    base_experiment_metadatas = read_base_experiment_metadata()
    if base_experiment_metadatas.get(base_experiment_id):
        base_experiment_metadatas[base_experiment_id] = base_experiment_metadata_update
    base_experiment_path = get_base_experiments_metadata_path()
    safe_dump_file(base_experiment_path, base_experiment_metadatas)


def get_handler_metadata(user_id, handler_id, kind=None):
    """Return metadata info present in metadata.json inside handler_root"""
    metadata_file = get_handler_metadata_file(user_id, handler_id, kind)
    metadata = safe_load_file(metadata_file)
    return metadata


def write_handler_metadata(user_id, handler_id, metadata):
    """Return metadata info present in metadata.json inside handler_root"""
    metadata_file = get_handler_metadata_file(user_id, handler_id)
    safe_dump_file(metadata_file, metadata)
    return metadata


def get_handler_metadata_with_jobs(user_id, handler_id):
    """Return a list of job_metadata info of multiple jobs"""
    metadata = get_handler_metadata(user_id, handler_id)
    metadata["jobs"] = []
    job_metadatas_root = get_handler_jobs_metadata_root(user_id, handler_id)
    for json_file in glob.glob(job_metadatas_root + "*.json"):
        metadata["jobs"].append(safe_load_file(json_file))
    return metadata


def write_job_metadata(user_id, handler_id, job_id, metadata):
    """Write job metadata info present in jobs_metadata folder"""
    handler_root = get_handler_root(user_id, None, handler_id, None)
    job_metadata_file = handler_root + f"/jobs_metadata/{job_id}.json"
    safe_dump_file(job_metadata_file, metadata, lock_file=None)


def update_job_status(user_id, handler_id, job_id, status):
    """Update the job status in jobs_metadata/job_id.json"""
    metadata = get_handler_job_metadata(user_id, handler_id, job_id)
    current_status = metadata.get("status", "")
    if current_status != "Canceled":
        if status != current_status:
            metadata["last_modified"] = datetime.datetime.now().isoformat()
        metadata["status"] = status

    write_job_metadata(user_id, handler_id, job_id, metadata)


def update_job_metadata(user_id, handler_id, job_id, metadata_key="result", data=""):
    """Update the job status in jobs_metadata/job_id.json"""
    metadata = get_handler_job_metadata(user_id, handler_id, job_id)
    if data != metadata.get(metadata_key, {}):
        metadata["last_modified"] = datetime.datetime.now().isoformat()
    metadata[metadata_key] = data

    write_job_metadata(user_id, handler_id, job_id, metadata)


def update_job_tar_stats(user_id, handler_id, job_id, tar_stats):
    """Update the job results in jobs_metadata/job_id.json"""
    metadata = get_handler_job_metadata(user_id, handler_id, job_id)
    if tar_stats != metadata.get("job_tar_stats", {}):
        metadata["last_modified"] = datetime.datetime.now().isoformat()
    metadata["job_tar_stats"] = tar_stats

    write_job_metadata(user_id, handler_id, job_id, metadata)


def infer_action_from_job(user_id, handler_id, job_id):
    """Takes handler, job_id (UUID / str) and returns action corresponding to that jobID"""
    job_id = str(job_id)
    action = ""
    all_jobs = get_handler_metadata_with_jobs(user_id, handler_id)["jobs"]
    for job in all_jobs:
        if job["id"] == job_id:
            action = job["action"]
            break
    return action


def get_handler_id(user_id, job_id, ngc_runner_fetch=False):
    """Return handler_id of the provided job"""
    job_folder_path = get_handler_root(user_id, None, None, job_id, ngc_runner_fetch=ngc_runner_fetch)
    if job_folder_path:
        return os.path.basename(os.path.abspath(os.path.join(job_folder_path, os.pardir)))
    return None


def get_handler_user(handler_id):
    """Return the user id for the handler id provided"""
    # Get the handler_root in all the paths
    hander_root = get_handler_root(None, None, handler_id, None)
    # Define a regular expression pattern to match UUIDs
    uuid_pattern = r'([a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12})'
    # Search for the UUID pattern in the input string
    match = re.search(uuid_pattern, hander_root)
    # Check if a match is found
    if match:
        return match.group(1)  # Return the matched UUID
    return None  # Return None if no UUID is found


def get_handler_kind(handler_metadata):
    """Return the handler type for the handler id provided"""
    # Get the handler_root in all the paths
    if "network_arch" in handler_metadata.keys():
        return "experiments"
    return "datasets"


def get_handler_type(user_id, handler_id):
    """Return the handler type"""
    handler_metadata = get_handler_metadata(user_id, handler_id)
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


def check_existence(user_id, handler_id, kind):
    """
    Check if dataset or experiment exists
    """
    if kind not in ["dataset", "experiment"]:
        return False

    # first check in the base experiments
    if kind == "experiment":
        model_metadata = get_base_experiment_metadata(handler_id)
        if model_metadata:
            return True
    # get handlers root
    handler_root = get_handler_root(user_id, kind + "s", handler_id, None)
    if handler_root:
        # check if the metadata file exists
        if os.path.exists(os.path.join(handler_root, "metadata.json")):
            return True
        print(f"Directory {handler_root} exists but metadata file not found for {kind} {handler_id}!", file=sys.stderr)
    return False


def check_read_access(user_id, handler_id, base_experiment=False):
    """Check if the user has read access to this particular handler"""
    handler_user = get_handler_user(handler_id)
    under_user = handler_user == user_id

    if base_experiment:
        handler_metadata = get_base_experiment_metadata(handler_id)
    else:
        handler_metadata = get_handler_metadata(user_id, handler_id)
    public = handler_metadata.get("public", False)  # Default is False

    if under_user:
        return True
    if public:
        return True
    return False


def check_write_access(user_id, handler_id, base_experiment=False):
    """Check if the user has write access to this particular handler"""
    handler_user = get_handler_user(handler_id)
    under_user = handler_user == user_id

    if base_experiment:
        handler_metadata = get_base_experiment_metadata(handler_id)
    else:
        handler_metadata = get_handler_metadata(user_id, handler_id)
    public = handler_metadata.get("public", False)  # Default is False
    read_only = handler_metadata.get("read_only", False)  # Default is False
    if under_user:  # If under user, you can always write - no point in making it un-writable by owner. Read-only is for non-owners
        return True
    if public:
        if read_only:
            return False
        return True
    return False


def get_public_experiments():
    """Get public experiments"""
    # Make sure to check if it exists
    public_experiments_metadata = []
    base_experiments_metadata_path = get_base_experiments_metadata_path()
    base_experiment_metadatas = safe_load_file(base_experiments_metadata_path)
    base_experiment_ids = []
    for base_experiment_id, base_experiment_metadata in base_experiment_metadatas.items():
        public = base_experiment_metadata.get("public", False)
        if public and base_experiment_id not in base_experiment_ids:
            base_experiment_ids.append(base_experiment_id)
            public_experiments_metadata.append(base_experiment_metadata)
    return list(public_experiments_metadata)


def get_public_datasets():
    """Get public datasets"""
    public_datasets = []
    return list(set(public_datasets))


def add_public_experiment(experiment_id):
    """Add public experiment"""
    # if experiment_id in get_public_experiments():
    #     return
    return


def add_public_dataset(dataset_id):
    """Add public dataset"""
    # if dataset_id in get_public_datasets():
    #     return
    return


def remove_public_experiment(experiment_id):
    """Remove public experiment"""
    # if experiment_id not in get_public_experiments():
    #     return
    return


def remove_public_dataset(dataset_id):
    """Remove public dataset"""
    # if dataset_id not in get_public_datasets():
    #     return
    return


def check_dataset_type_match(user_id, experiment_meta, dataset_id, no_raw=None):
    """Checks if the dataset created for the experiment is valid dataset_type"""
    # If dataset id is None, then return True
    # Else, if all things match, return True
    # True means replace, False means skip and return a 400 Code
    if dataset_id is None:
        return True
    if not check_existence(user_id, dataset_id, "dataset"):
        return False
    if not check_read_access(user_id, dataset_id):
        return False

    dataset_meta = get_handler_metadata(user_id, dataset_id, "datasets")

    experiment_dataset_type = experiment_meta.get("dataset_type")
    dataset_type = dataset_meta.get("type")
    dataset_format = dataset_meta.get("format")
    if experiment_dataset_type not in (dataset_type, "user_custom"):
        return False

    if no_raw:
        if dataset_format in ("raw", "coco_raw"):
            return False

    return True


def check_experiment_type_match(user_id, experiment_meta, base_experiment_ids):
    """Checks if the experiment created and base_experiment requested belong to the same network"""
    if base_experiment_ids is None:
        return True
    for base_experiment_id in base_experiment_ids:
        if not check_read_access(user_id, base_experiment_id, True):
            return False

        base_experiment_meta = get_base_experiment_metadata(base_experiment_id)
        if not base_experiment_meta:
            # Search in the admin_uuid fails, search in the user_id
            base_experiment_meta = get_handler_metadata(user_id, base_experiment_id)

        experiment_arch = experiment_meta.get("network_arch")
        base_experiment_arch = base_experiment_meta.get("network_arch")
        if experiment_arch != base_experiment_arch:
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


def check_base_experiment_support_realtime_infer(user_id, experiment_meta, realtime_infer):
    """Check if the PTM suppport realtime infer"""
    if realtime_infer is None or realtime_infer is False:  # no need to check
        return True

    base_experiment_ids = experiment_meta.get("base_experiment")
    if len(base_experiment_ids) != 1:
        return False
    base_experiment_id = base_experiment_ids[0]

    if not check_existence(user_id, base_experiment_id, "experiment"):
        return False

    if not check_read_access(user_id, base_experiment_id, True):
        return False

    base_experiment_meta = get_base_experiment_metadata(base_experiment_id)

    if not base_experiment_meta.get("realtime_infer_support", False):
        return False

    return True


def experiment_update_handler_attributes(user_id, experiment_meta, key, value):
    """Checks if the artifact provided is of the correct type"""
    # Returns value or False
    if key in ["train_datasets"]:
        if type(value) is not list:
            value = [value]
        for dataset_id in value:
            if not check_dataset_type_match(user_id, experiment_meta, dataset_id, no_raw=True):
                return False
    elif key in ["eval_dataset"]:
        if not check_dataset_type_match(user_id, experiment_meta, value, no_raw=True):
            return False
    elif key in ["calibration_dataset", "inference_dataset"]:
        if not check_dataset_type_match(user_id, experiment_meta, value):
            return False
    elif key in ["base_experiment"]:
        if not check_experiment_type_match(user_id, experiment_meta, value):
            return False
    elif key in ["checkpoint_choose_method"]:
        if not check_checkpoint_choose_match(value):
            return False
    elif key in ["checkpoint_epoch_number"]:
        if not check_checkpoint_epoch_number_match(value):
            return False
    elif key in ["realtime_infer"]:
        if not check_base_experiment_support_realtime_infer(user_id, experiment_meta, value):
            return False
    else:
        return False
    return True


def list_all_job_metadata(user_id, handler_id):
    """Return a list of job_metadata info of multiple jobs"""
    job_metadatas = []
    job_metadatas_root = get_handler_jobs_metadata_root(user_id, handler_id)
    for json_file in glob.glob(job_metadatas_root + "*.json"):
        # Skip files with 'tmp' in the filename
        if "tmp" in json_file:
            continue
        # Check if the file exists before loading it
        if not os.path.exists(json_file):
            continue
        job_metadata = safe_load_file(json_file)
        if job_metadata.get("id", None) is None or job_metadata.get("status", None) is None:
            continue
        job_metadatas.append(job_metadata)
    return job_metadatas


def resolve_existence(user_id, kind, handler_id):
    """Check if the handler exists"""
    if kind not in ["dataset", "experiment"]:
        return False
    metadata_path = os.path.join(get_root(), user_id, kind + "s", handler_id, "metadata.json")
    return os.path.exists(metadata_path)


def resolve_root(user_id, kind, handler_id):
    """Resolve the root path of the handler"""
    return os.path.join(get_root(), user_id, kind + "s", handler_id)


def resolve_metadata(user_id, kind, handler_id):
    """Resolve the metadata of the handler"""
    metadata_path = os.path.join(resolve_root(user_id, kind, handler_id), "metadata.json")
    metadata = safe_load_file(metadata_path)
    return metadata


def get_latest_ver_folder(tis_model_path):
    """Returns the latest version folder in the model directory"""
    try:
        entries = os.listdir(tis_model_path)

        # Find the maximum numeric value among the folder names
        numeric_folders = [int(folder) for folder in entries if folder.isnumeric()]
        latest_ver = max(numeric_folders)

    except ValueError:
        # If there are no numeric folders, return 0
        return 0

    return latest_ver


def get_default_lock_file_path(filepath):
    """Returns the default lock file path"""
    return os.path.splitext(filepath)[0] + "_lock.lock"


def safe_get_file_modified_time(filepath, lock_file=None):
    """Returns the modified time of the file"""
    if lock_file is None:
        lock_file = get_default_lock_file_path(filepath)

    if not os.path.exists(lock_file):
        with open(lock_file, "w", encoding="utf-8") as _:
            pass

    with FileLock(lock_file, mode=0o666):
        return os.path.getmtime(filepath)


def __convert_keys_to_str(data):
    if isinstance(data, dict):
        return {
            str(key) if isinstance(key, int) else key: __convert_keys_to_str(value)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [__convert_keys_to_str(item) for item in data]
    if isinstance(data, np.float64):
        return float(data)
    return data


def safe_load_file(filepath, lock_file=None, attempts=3, file_type="json"):
    """Loads the json file"""
    assert file_type in ("json", "yaml")
    if attempts == 0:
        return {}

    if not os.path.exists(filepath):
        print("File trying to read doesn't exists", filepath, file=sys.stderr)
        return {}

    if lock_file is None:
        lock_file = get_default_lock_file_path(filepath)

    if not os.path.exists(lock_file):
        with open(lock_file, "w", encoding="utf-8") as _:
            pass
    try:
        with FileLock(lock_file, mode=0o666):
            if file_type == "json":
                with open(filepath, "r", encoding="utf-8") as f:
                    json_data = f.read()
                    data = orjson.loads(json_data)
            elif file_type == "yaml":
                with open(filepath, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            return data
    except:
        data = {}
        print(f"Data not in {file_type} loadable format", filepath, file=sys.stderr)
        with open(filepath, "r", encoding='utf-8') as f:
            file_lines = f.readlines()
            print("Data: \n", file_lines, file=sys.stderr)
        if filepath.endswith("ptm_metadatas.json"):
            ptm_source_path = f"/opt/api/shared/users/{admin_uuid}/experiments/{admin_uuid}"
            if os.path.exists(f"{ptm_source_path}/ptm_metadatas.json"):
                print("Copying corrupt PTM meta file", file=sys.stderr)
                shutil.copy(f"{ptm_source_path}/ptm_metadatas.json", filepath)
        return safe_load_file(filepath, lock_file, attempts - 1, file_type=file_type)


def safe_dump_file(filepath, data, lock_file=None, file_type="json"):
    """Dumps the json file"""
    assert file_type in ("json", "yaml", "protobuf")
    parent_folder = os.path.dirname(filepath)
    if not os.path.exists(parent_folder):
        print(f"Parent folder {parent_folder} doesn't exists yet", file=sys.stderr)
        return
    if lock_file is None:
        lock_file = get_default_lock_file_path(filepath)

    if not os.path.exists(lock_file):
        with open(lock_file, "w", encoding="utf-8") as _:
            pass

    with FileLock(lock_file, mode=0o666):
        tmp_file_path = filepath.replace(f".{file_type}", f"_tmp.{file_type}")
        if file_type == "json":
            json_data = orjson.dumps(__convert_keys_to_str(data))
            with open(tmp_file_path, "w", encoding="utf-8") as f:
                f.write(json_data.decode('utf-8'))
        elif file_type == "yaml":
            with open(tmp_file_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, sort_keys=False)
        elif file_type == "protobuf":
            with open(tmp_file_path, "w", encoding='utf-8') as f:
                f.write(data)
        if os.path.exists(tmp_file_path):
            os.rename(tmp_file_path, filepath)


def safe_update_json(filepath, update_fn, lock_file=None, **kwargs):
    """Read, update and save the json file."""
    if lock_file is None:
        lock_file = get_default_lock_file_path(filepath)

    if not os.path.exists(lock_file):
        with open(lock_file, "w", encoding="utf-8") as _:
            pass

    with FileLock(lock_file, mode=0o666):
        with open(filepath, "r", encoding="utf-8") as f:
            json_data = f.read()
            data = orjson.loads(json_data)

        data = update_fn(data, **kwargs)
        if data is not None:
            json_data = orjson.dumps(data)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_data.decode('utf-8'))


def is_valid_uuid4(uuid_string):
    """Check if the string is a valid UUID4"""
    try:
        val = uuid.UUID(uuid_string, version=4)
    except ValueError:
        return False

    # Return True if the UUID is of version 4
    return val.hex == uuid_string.replace('-', '') and val.version == 4


def printc(*args, **kwargs):
    """Print the contexts (uuid/handler_id/job_id) with the message"""
    context = kwargs.pop("context", {})
    if not isinstance(context, dict):
        print(*args, **kwargs)
        return
    keys = kwargs.pop("keys", ["user_id", "handler_id", "id"])
    keys = keys if isinstance(keys, list) else [keys]
    context_str = ""
    for key, value in context.items():
        if key in keys:
            context_str += f"[{key}:{value}]"
    print(context_str, *args, **kwargs)
