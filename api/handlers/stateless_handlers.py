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
import os
import sys
import subprocess
from pathlib import Path
import orjson
import uuid
import traceback

from handlers.encrypt import NVVaultEncryption
from utils import safe_load_file, safe_dump_file

BACKEND = os.getenv("BACKEND", "local-k8s")
tao_root = os.environ.get("TAO_ROOT", "/shared/orgs/")
base_exp_uuid = "00000000-0000-0000-0000-000000000000"


def get_root():
    """Return root path"""
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


def get_handler_root(org_name=None, kind=None, handler_id=None, job_id=None):
    """Return handler root path"""
    handler_id = handler_id if handler_id else ""
    job_id = job_id if job_id else ""
    rootdir = Path(get_root())
    if not rootdir.exists():
        return ""
    if org_name:
        rootdir = rootdir / org_name
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


def get_jobs_root(user_id=None, org_name=None):
    """Return handler root path"""
    return os.path.join(get_root(), org_name, "users", user_id, "jobs")


def get_base_experiment_path(handler_id, create_if_not_exist=True):
    """Return base_experiment root path"""
    base_experiment_path = f"{tao_root}/{base_exp_uuid}/experiments/{base_exp_uuid}/{handler_id}"
    if not os.path.exists(base_experiment_path) and create_if_not_exist:
        os.makedirs(base_experiment_path, exist_ok=True)
        subprocess.getoutput(f"chmod -R 777 {base_experiment_path}")
    return base_experiment_path


def get_base_experiments_metadata_path():
    """Return base_experiment root path"""
    base_experiments_metadata_path = f"{tao_root}/{base_exp_uuid}/experiments/{base_exp_uuid}/ptm_metadatas.json"
    return base_experiments_metadata_path


# Sub for handler.spec_root with handler_root(handler_id)
def get_handler_spec_root(user_id, org_name, handler_id):
    """Return path of specs folder under handler_root"""
    return os.path.join(get_root(), org_name, "users", user_id, "specs")


def get_handler_log_root(user_id, org_name, handler_id):
    """Return path of logs folder under handler_root"""
    return os.path.join(get_root(), org_name, "users", user_id, "logs")


def get_handler_job_metadata(org_name, handler_id, job_id, kind=""):
    """Return metadata info present in job_id.json inside jobs_metadata folder"""
    # Only metadata of a particular job
    metadata = {}
    if job_id:
        handler_root = get_handler_root(org_name, kind, handler_id, None)
        job_metadata_file = handler_root + f"/jobs_metadata/{job_id}.json"
        if os.path.exists(job_metadata_file):
            metadata = safe_load_file(job_metadata_file)
    return metadata


def get_job_files(user_id, org_name, handler_id, job_id, retrieve_logs=False, retrieve_specs=False):
    """Return metadata info present in job_id.json inside jobs_metadata folder"""
    # Only metadata of a particular job
    logs_folder = ""
    specs_folder = ""
    if retrieve_logs:
        logs_folder = get_handler_log_root(user_id, org_name, handler_id)
    if retrieve_specs:
        specs_folder = get_handler_spec_root(user_id, org_name, handler_id)

    job_root = get_jobs_root(user_id, org_name)
    job_folder = os.path.join(job_root, job_id)
    if not os.path.exists(job_folder) and (retrieve_logs and not os.path.exists(logs_folder)) and (retrieve_specs and not os.path.exists(specs_folder)):
        return []
    files = glob.glob(f"{job_folder}/**", recursive=True)

    # Get log and specs file for that job
    log_file = os.path.join(logs_folder, f"{job_id}.txt")
    if logs_folder and os.path.exists(log_file):
        files += [log_file]
    if specs_folder and os.path.exists(specs_folder):
        files += glob.glob(f"{specs_folder}/*{job_id}.*", recursive=True)  # Ignore lock files and -{action}-spec.json

    files = [os.path.relpath(file, job_root) for file in files if not file.endswith('/')]
    return files


def get_toolkit_status(org_name, handler_id, job_id, kind=""):
    """Returns the status of the job reported from the frameworks container"""
    metadata_info = get_handler_job_metadata(org_name, handler_id, job_id, kind)
    toolkit_status = ""
    result_dict = metadata_info.get("result", {})
    if result_dict:
        toolkit_detailed_status = result_dict.get("detailed_status", {})
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
def get_handler_metadata_file(org_name, handler_id, kind=None):
    """Return path of metadata.json under handler_root"""
    return get_handler_root(org_name, kind, handler_id, None) + "/metadata.json"


def get_handler_jobs_metadata_root(org_name, handler_id, kind=None):
    """Return path of job_metadata folder folder under handler_root"""
    return get_handler_root(org_name, kind, handler_id, None) + "/jobs_metadata/"


def read_base_experiment_metadata():
    """Read PTM metadata file and return the entire metadata info"""
    base_experiment_path = get_base_experiments_metadata_path()
    base_experiment_metadatas = safe_load_file(base_experiment_path)
    if not base_experiment_metadatas and os.path.exists(base_experiment_path):  # In-case the file was corrupted
        ptm_source_path = f"/opt/api/shared/orgs/{base_exp_uuid}/experiments/{base_exp_uuid}"
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


def get_handler_metadata(org_name, handler_id, kind=None):
    """Return metadata info present in metadata.json inside handler_root"""
    metadata_file = get_handler_metadata_file(org_name, handler_id, kind)
    metadata = safe_load_file(metadata_file)
    return metadata


def write_handler_metadata(org_name, handler_id, metadata, kind=""):
    """Return metadata info present in metadata.json inside handler_root"""
    metadata_file = get_handler_metadata_file(org_name, handler_id, kind)
    safe_dump_file(metadata_file, metadata)
    return metadata


def get_handler_metadata_with_jobs(org_name, handler_id, kind=""):
    """Return a list of job_metadata info of multiple jobs"""
    metadata = get_handler_metadata(org_name, handler_id, kind=kind)
    metadata["jobs"] = []
    job_metadatas_root = get_handler_jobs_metadata_root(org_name, handler_id)
    for json_file in glob.glob(job_metadatas_root + "*.json"):
        if os.path.exists(json_file):
            metadata["jobs"].append(safe_load_file(json_file))
    return metadata


def write_job_metadata(org_name, handler_id, job_id, metadata, kind=""):
    """Write job metadata info present in jobs_metadata folder"""
    handler_root = get_handler_root(org_name, kind, handler_id, None)
    job_metadata_file = handler_root + f"/jobs_metadata/{job_id}.json"
    safe_dump_file(job_metadata_file, metadata)


def get_job_id_of_action(org_name, handler_id, kind, action):
    handler_job_id = None
    root = get_handler_root(org_name, kind, handler_id, None)
    handler_jobs_meta_folder = os.path.join(root, "jobs_metadata")
    if os.path.exists(handler_jobs_meta_folder):
        for job_file in os.listdir(handler_jobs_meta_folder):
            if job_file.endswith(".json"):
                job_id = os.path.splitext(os.path.basename(job_file))[0]
                job_metadata = get_handler_job_metadata(org_name, handler_id, job_id, kind=kind)
                if job_metadata.get("action") == action and job_metadata.get("status") == "Done":
                    handler_job_id = job_id
                    break
    if not handler_job_id:
        raise ValueError(f"No job found or no job with status Done found for action:{action}, handler:{handler_id}, kind:{kind}", file=sys.stderr)
    return handler_job_id


def update_handler_with_jobs_info(jobs_metadata, org_name, handler_id, job_id, kind):
    # Update jobs info in handler metadata
    handler_metadata = get_handler_metadata(org_name, handler_id, kind)
    if handler_metadata:
        if "jobs" not in handler_metadata:
            handler_metadata["jobs"] = {}
        if job_id not in handler_metadata["jobs"]:
            handler_metadata["jobs"][job_id] = {}
        if "result" not in handler_metadata["jobs"][job_id]:
            handler_metadata["jobs"][job_id]["result"] = {}
        handler_metadata["jobs"][job_id]["name"] = jobs_metadata.get("name")
        handler_metadata["jobs"][job_id]["status"] = jobs_metadata.get("status")
        handler_metadata["jobs"][job_id]["action"] = jobs_metadata.get("action")
        handler_metadata["jobs"][job_id]["result"]["eta"] = jobs_metadata.get("result", {}).get("eta")
        handler_metadata["jobs"][job_id]["result"]["epoch"] = jobs_metadata.get("result", {}).get("epoch")
        handler_metadata["jobs"][job_id]["result"]["max_epoch"] = jobs_metadata.get("result", {}).get("max_epoch")
        write_handler_metadata(org_name, handler_id, handler_metadata, kind)


def update_job_status(org_name, handler_id, job_id, status, kind=""):
    """Update the job status in jobs_metadata/job_id.json"""
    metadata = get_handler_job_metadata(org_name, handler_id, job_id, kind)
    current_status = metadata.get("status", "")
    if current_status not in ("Canceled", "Canceling", "Pausing", "Paused") or (current_status == "Canceling" and status == "Canceled") or (current_status == "Pausing" and status == "Paused"):
        if status != current_status:
            metadata["last_modified"] = datetime.datetime.now().isoformat()
        metadata["status"] = status
        if kind:
            update_handler_with_jobs_info(metadata, org_name, handler_id, job_id, kind)
        write_job_metadata(org_name, handler_id, job_id, metadata, kind)


def update_job_metadata(org_name, handler_id, job_id, metadata_key="result", data="", kind=""):
    """Update the job status in jobs_metadata/job_id.json"""
    metadata = get_handler_job_metadata(org_name, handler_id, job_id, kind)
    if data != metadata.get(metadata_key, {}):
        metadata["last_modified"] = datetime.datetime.now().isoformat()
    metadata[metadata_key] = data
    if metadata_key == "result" and kind:
        update_handler_with_jobs_info(metadata, org_name, handler_id, job_id, kind)
    write_job_metadata(org_name, handler_id, job_id, metadata, kind)


def infer_action_from_job(org_name, handler_id, job_id):
    """Takes handler, job_id (UUID / str) and returns action corresponding to that jobID"""
    job_id = str(job_id)
    action = ""
    all_jobs = get_handler_metadata_with_jobs(org_name, handler_id)["jobs"]
    for job in all_jobs:
        if job["id"] == job_id:
            action = job["action"]
            break
    return action


def get_handler_id(org_name, parent_job_id):
    """Return handler_id of the provided job"""
    orgs_root = os.path.join(get_root(), org_name)
    for kind in ("experiments", "datasets", "workspaces"):
        handler_folder = os.path.join(orgs_root, kind)
        if not os.path.exists(handler_folder):
            continue
        for handler_id in os.listdir(handler_folder):
            handler_jobs_meta_folder = os.path.join(os.path.join(orgs_root, kind, handler_id, "jobs_metadata"))
            if not os.path.exists(handler_jobs_meta_folder):
                continue
            for job_file in os.listdir(handler_jobs_meta_folder):
                if f"{parent_job_id}.json" == job_file:
                    return handler_id
    return None


def get_handler_org(org_name, handler_id, kind):
    """Return the user id for the handler id provided"""
    # Get the handler_root in all the paths
    handler_root = get_handler_root(org_name, kind, handler_id, None)
    parts = handler_root.split("/")
    org_name = None
    if (len(parts) >= 4 and parts[2] == 'orgs'):
        org_name = parts[3]
    elif (len(parts) >= 3 and parts[1] == 'orgs'):
        org_name = parts[2]
    return org_name


def get_handler_kind(handler_metadata):
    """Return the handler type for the handler id provided"""
    # Get the handler_root in all the paths
    if "network_arch" in handler_metadata.keys():
        return "experiments"
    if "cloud_type" in handler_metadata.keys():
        return "workspaces"
    return "datasets"


def get_handler_type(handler_metadata):
    """Return the handler type"""
    network = handler_metadata.get("network_arch", None)
    if not network:
        network = handler_metadata.get("type", None)
    return network


def make_root_dirs(user_id, org_name, kind, handler_id):
    """Create root dir followed by logs, jobs_metadata and specs folder"""
    log_root = os.path.join(get_root(), org_name, "users", user_id, "logs/")
    spec_root = os.path.join(get_root(), org_name, "users", user_id, "specs/")
    jobs_meta_root = os.path.join(get_root(), org_name, kind, handler_id, "jobs_metadata/")
    for directory in [log_root, spec_root, jobs_meta_root]:
        if not os.path.exists(directory):
            os.makedirs(directory)


def check_existence(org_name, handler_id, kind):
    """
    Check if dataset or experiment exists
    """
    if kind not in ["dataset", "experiment", "workspace"]:
        return False

    # first check in the base experiments
    if kind == "experiment":
        model_metadata = get_base_experiment_metadata(handler_id)
        if model_metadata:
            return True
    # get handlers root
    handler_root = get_handler_root(org_name, kind + "s", handler_id, None)
    if handler_root:
        # check if the metadata file exists
        if os.path.exists(os.path.join(handler_root, "metadata.json")):
            return True
        print(f"Directory {handler_root} exists but metadata file not found for {kind} {handler_id}!", file=sys.stderr)
    return False


def check_read_access(org_name, handler_id, base_experiment=False, kind=""):
    """Check if the user has read access to this particular handler"""
    handler_org = get_handler_org(org_name, handler_id, kind)
    under_user = handler_org is not None

    if base_experiment:
        handler_metadata = get_base_experiment_metadata(handler_id)
    else:
        handler_metadata = get_handler_metadata(org_name, handler_id, kind)
    public = handler_metadata.get("public", False)  # Default is False

    if under_user:
        return True
    if public:
        return True
    return False


def check_write_access(org_name, handler_id, base_experiment=False, kind=""):
    """Check if the user has write access to this particular handler"""
    handler_org = get_handler_org(org_name, handler_id, kind)
    under_user = handler_org is not None
    if base_experiment:
        handler_metadata = get_base_experiment_metadata(handler_id)
    else:
        handler_metadata = get_handler_metadata(org_name, handler_id, kind)
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


def decrypt_handler_metadata(workspace_metadata):
    """Decrypt NvVault encrypted values"""
    if BACKEND in ("BCP", "NVCF"):
        cloud_specific_details = workspace_metadata.get("cloud_specific_details")
        if cloud_specific_details:
            config_path = os.getenv("VAULT_SECRET_PATH", None)
            encryption = NVVaultEncryption(config_path)
            for key, value in cloud_specific_details.items():
                if encryption.check_config()[0]:
                    workspace_metadata["cloud_specific_details"][key] = encryption.decrypt(value)
                else:
                    print("deencryption not possible", file=sys.stderr)


def get_workspace_string_identifier(org_name, workspace_id, workspace_cache):
    """For the given workspace ID, constuct a unique string which can identify this workspace"""
    if workspace_id in workspace_cache:
        workspace_metadata = workspace_cache[workspace_id]
    else:
        workspace_metadata = get_handler_metadata(org_name, workspace_id, kind="workspaces")
        decrypt_handler_metadata(workspace_metadata)
        workspace_cache[workspace_id] = workspace_metadata
    workspace_identifier = ""
    if workspace_metadata:
        workspace_identifier = f"{workspace_metadata.get('cloud_type')}://{workspace_metadata.get('cloud_specific_details', {}).get('cloud_bucket_name')}/"
    return workspace_identifier


def check_dataset_type_match(org_name, experiment_meta, dataset_id, no_raw=None):
    """Checks if the dataset created for the experiment is valid dataset_type"""
    # If dataset id is None, then return True
    # Else, if all things match, return True
    # True means replace, False means skip and return a 400 Code
    if dataset_id is None:
        return True
    if not check_existence(org_name, dataset_id, "dataset"):
        return False
    if not check_read_access(org_name, dataset_id, kind="datasets"):
        return False

    dataset_meta = get_handler_metadata(org_name, dataset_id, "datasets")

    experiment_dataset_type = experiment_meta.get("dataset_type")
    network_arch = experiment_meta.get("network_arch")

    if network_arch == "image" and experiment_dataset_type == "not_restricted":  # Allow Image action from dataservices to run on any model's dataset
        return True

    dataset_type = dataset_meta.get("type")
    dataset_format = dataset_meta.get("format")
    if experiment_dataset_type not in (dataset_type, "user_custom"):
        return False

    if no_raw:
        if dataset_format in ("raw", "coco_raw"):
            return False

    return True


def check_experiment_type_match(org_name, experiment_meta, base_experiment_ids):
    """Checks if the experiment created and base_experiment requested belong to the same network"""
    if base_experiment_ids is None:
        return True
    for base_experiment_id in base_experiment_ids:
        if not check_read_access(org_name, base_experiment_id, True, kind="experiments"):
            return False

        base_experiment_meta = get_base_experiment_metadata(base_experiment_id)
        if not base_experiment_meta:
            # Search in the base_exp_uuid fails, search in the org_name
            base_experiment_meta = get_handler_metadata(org_name, base_experiment_id)

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


def check_base_experiment_support_realtime_infer(org_name, experiment_meta, realtime_infer):
    """Check if the PTM suppport realtime infer"""
    if realtime_infer is None or realtime_infer is False:  # no need to check
        return True

    base_experiment_ids = experiment_meta.get("base_experiment")
    if len(base_experiment_ids) != 1:
        return False
    base_experiment_id = base_experiment_ids[0]

    if not check_existence(org_name, base_experiment_id, "experiment"):
        return False

    if not check_read_access(org_name, base_experiment_id, True, kind="experiments"):
        return False

    base_experiment_meta = get_base_experiment_metadata(base_experiment_id)
    if not base_experiment_meta:
        # Search in the base_exp_uuid fails, search in the user_id
        base_experiment_meta = get_handler_metadata(org_name, base_experiment_id)
    if not base_experiment_meta.get("realtime_infer_support", False):
        return False

    return True


def experiment_update_handler_attributes(org_name, experiment_meta, key, value):
    """Checks if the artifact provided is of the correct type"""
    # Returns value or False
    if key in ["train_datasets"]:
        if type(value) is not list:
            value = [value]
        for dataset_id in value:
            if not check_dataset_type_match(org_name, experiment_meta, dataset_id, no_raw=True):
                return False
    elif key in ["eval_dataset"]:
        if not check_dataset_type_match(org_name, experiment_meta, value, no_raw=True):
            return False
    elif key in ["calibration_dataset", "inference_dataset"]:
        if not check_dataset_type_match(org_name, experiment_meta, value):
            return False
    elif key in ["base_experiment"]:
        if not check_experiment_type_match(org_name, experiment_meta, value):
            return False
    elif key in ["checkpoint_choose_method"]:
        if not check_checkpoint_choose_match(value):
            return False
    elif key in ["checkpoint_epoch_number"]:
        if not check_checkpoint_epoch_number_match(value):
            return False
    elif key in ["realtime_infer"]:
        if not check_base_experiment_support_realtime_infer(org_name, experiment_meta, value):
            return False
    else:
        return False
    return True


def list_all_job_metadata(org_name, handler_id):
    """Return a list of job_metadata info of multiple jobs"""
    job_metadatas = []
    job_metadatas_root = get_handler_jobs_metadata_root(org_name, handler_id)
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


def resolve_existence(org_name, kind, handler_id):
    """Check if the handler exists"""
    if kind not in ["dataset", "experiment", "workspace"]:
        return False
    metadata_path = os.path.join(get_root(), org_name, kind + "s", handler_id, "metadata.json")
    return os.path.exists(metadata_path)


def resolve_root(org_name, kind, handler_id):
    """Resolve the root path of the handler"""
    return os.path.join(get_root(), org_name, kind + "s", handler_id)


def resolve_metadata(org_name, kind, handler_id):
    """Resolve the metadata of the handler"""
    metadata_path = os.path.join(resolve_root(org_name, kind, handler_id), "metadata.json")
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


def sanitize_handler_metadata(handler_metadata):
    """Remove sensitive information like cloud storage credentials to return as response"""
    return_metadata = handler_metadata.copy()
    if "cloud_specific_details" in return_metadata.get("cloud_details", {}):
        for key in ("access_key", "secret_key", "token"):
            return_metadata["cloud_details"]["cloud_specific_details"].pop(key, None)
    return_metadata.pop("client_secret", None)
    return return_metadata


def get_all_pending_jobs():
    """Returns a list of all jobs with status of Pending, Running, or Canceling"""
    jobs = []

    if not os.path.exists(tao_root):
        return jobs
    orgs = [d for d in os.listdir(tao_root) if d != base_exp_uuid]
    for org in orgs:
        org_path = os.path.join(tao_root, org)
        experiments = os.listdir(os.path.join(org_path, 'experiments')) if os.path.exists(os.path.join(org_path, 'experiments')) else []
        for experiment_id in experiments:
            handler_metadata = get_handler_metadata_with_jobs(org, experiment_id)
            metadata_jobs = handler_metadata.get("jobs", [])
            for job in metadata_jobs:
                status = job.get("status", "Unknown")
                user_id = handler_metadata.get("user_id", None)
                network = handler_metadata.get("network_arch", None)
                isAutoML = handler_metadata.get("automl_settings", {}).get("automl_enabled", False)
                if status in ("Running", "Pending", "Canceling"):
                    job["user_id"] = user_id
                    job["kind"] = "experiment"
                    job["network"] = network
                    job["org_name"] = org
                    job["is_automl"] = isAutoML
                    job["handler_id"] = experiment_id
                    jobs.append(job)

        datasets = os.listdir(os.path.join(org_path, 'datasets')) if os.path.exists(os.path.join(org_path, 'datasets')) else []
        for dataset_id in datasets:
            handler_metadata = get_handler_metadata_with_jobs(org, dataset_id)
            metadata_jobs = handler_metadata.get("jobs", [])
            for job in metadata_jobs:
                status = job.get("status", "Unknown")
                user_id = handler_metadata.get("user_id", None)
                network = handler_metadata.get("type", None)
                isAutoML = False
                if status in ("Running", "Pending", "Canceling"):
                    job["user_id"] = user_id
                    job["kind"] = "dataset"
                    job["network"] = network
                    job["org_name"] = org
                    job["is_automl"] = isAutoML
                    job["handler_id"] = dataset_id
                    jobs.append(job)

    return jobs
