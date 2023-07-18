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

"""API handler modules"""
import os
import sys
import json
import uuid
import glob
import shutil
import threading
import datetime
from handlers.automl_handler import AutoMLHandler

from handlers import stateless_handlers
from handlers.stateless_handlers import check_read_access, check_write_access
from handlers.utilities import Code, read_network_config, run_system_command, download_ptm, VALID_DSTYPES, VALID_NETWORKS, AUTOML_DISABLED_NETWORKS
from handlers.chaining import create_job_contexts, infer_action_from_job
from handlers.ds_upload import DS_UPLOAD_TO_FUNCTIONS
from job_utils import executor as jobDriver
from job_utils.workflow_driver import on_new_job, on_delete_job
from specs_utils import csv_to_json_schema


# Helpers
def resolve_existence(user_id, kind, handler_id):
    """Return whether metadata.json exists or not"""
    if kind not in ["dataset", "model"]:
        return False
    metadata_path = os.path.join(stateless_handlers.get_root(), user_id, kind + "s", handler_id, "metadata.json")
    return os.path.exists(metadata_path)


def resolve_job_existence(user_id, kind, handler_id, job_id):
    """Return whether job_id.json exists in jobs_metadata folder or not"""
    if kind not in ["dataset", "model"]:
        return False
    metadata_path = os.path.join(stateless_handlers.get_root(), user_id, kind + "s", handler_id, "jobs_metadata", job_id + ".json")
    return os.path.exists(metadata_path)


def resolve_root(user_id, kind, handler_id):
    """Returns handler root"""
    return os.path.join(stateless_handlers.get_root(), user_id, kind + "s", handler_id)


def resolve_metadata(user_id, kind, handler_id):
    """Reads metadata.json and return it's contents"""
    metadata_path = os.path.join(resolve_root(user_id, kind, handler_id), "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata
    return {}


def resolve_metadata_with_jobs(user_id, kind, handler_id):
    """Reads job_id.json in jobs_metadata folder and return it's contents"""
    metadata = resolve_metadata(user_id, kind, handler_id)
    if metadata:
        metadata["jobs"] = []
        job_metadatas_root = resolve_root(user_id, kind, handler_id) + "/jobs_metadata/"
        for json_file in glob.glob(job_metadatas_root + "*.json"):
            metadata["jobs"].append(stateless_handlers.load_json_data(json_file))
        return metadata
    return {}


def get_user_models(user_id):
    """Returns a list of models that are available for the given user_id"""
    user_root = stateless_handlers.get_root() + f"{user_id}/models/"
    models, ret_lst = [], []
    if os.path.isdir(user_root):
        models = os.listdir(user_root)
    for model in models:
        if resolve_existence(user_id, "model", model):
            ret_lst.append(model)
    return ret_lst


def get_user_datasets(user_id):
    """Returns a list of datasets that are available for the given user_id"""
    user_root = stateless_handlers.get_root() + f"{user_id}/datasets/"
    datasets, ret_lst = [], []
    if os.path.isdir(user_root):
        datasets = os.listdir(user_root)
    for dataset in datasets:
        if resolve_existence(user_id, "dataset", dataset):
            ret_lst.append(dataset)
    return ret_lst


def write_handler_metadata(user_id, kind, handler_id, metadata):
    """Writes metadata.json with the contents of metadata variable passed"""
    metadata_file = os.path.join(resolve_root(user_id, kind, handler_id), "metadata.json")
    with open(metadata_file, "w", encoding='utf-8') as f:
        f.write(json.dumps(metadata, indent=4))


def get_dataset_actions(ds_type, ds_format):
    """Reads the dataset's network config and returns the valid actions of the given dataset type and format"""
    actions_default = read_network_config(ds_type)["api_params"]["actions"]

    # Define all anamolous formats where actions are not same as ones listed in the network config
    TYPE_FORMAT_ACTIONS_MAP = {("object_detection", "raw"): [],
                               ("object_detection", "coco_raw"): [],
                               ("instance_segmentation", "raw"): [],
                               ("instance_segmentation", "coco_raw"): [],
                               ("semantic_segmentation", "raw"): [],
                               ("semantic_segmentation", "unet"): []
                               }

    actions_override = TYPE_FORMAT_ACTIONS_MAP.get((ds_type, ds_format), actions_default)
    return actions_override


def nested_update(source, additions):
    """Merge one dictionary(additions) into another(source)"""
    if not isinstance(additions, dict):
        return source
    for key, value in additions.items():
        if isinstance(value, dict):
            source[key] = nested_update(source[key], value)
        else:
            source[key] = value
    return source


def is_job_automl(user_id, model_id, job_id):
    """Returns if the job is automl-based job or not"""
    try:
        root = resolve_root(user_id, "model", model_id)
        jobdir = os.path.join(root, job_id)
        automl_signature = os.path.join(jobdir, "controller.log")
        return os.path.exists(automl_signature)
    except:
        return False


def is_request_automl(user_id, handler_id, parent_job_id, actions, kind):
    """Returns if the job requested is automl based train or not"""
    handler_metadata = resolve_metadata(user_id, kind, handler_id)
    if handler_metadata.get("automl_enabled", False) and actions == ["train"]:
        return True
    return False


# validate workspace
def validate_workspace():
    """Checks if the workspace directory is in the current structure or not"""
    root = stateless_handlers.get_root()

    # populate pre-existing models
    user_dirs = os.listdir(root)
    for user_id in user_dirs:

        user_dir = os.path.join(root, user_id)

        # sanity check #1
        if not os.path.isdir(user_dir):  # not a directory
            print("Random files exist!! Wrong workspace structure, must only have user IDs", file=sys.stderr)
            continue

        # sanity check #2
        try:  # not a valid user id
            uuid.UUID(user_id)
            print("Directory not corresponding to a UUID", file=sys.stderr)
        except:
            continue

        # get pre-existing datasets
        user_datasets_path = os.path.join(user_dir, "datasets")

        dir_contents = []
        if os.path.isdir(user_datasets_path):
            dir_contents = os.listdir(user_datasets_path)

        # NOTE: Assumes pre-existing datasets have data already uploaded.
        for content in dir_contents:
            metadata_path = user_datasets_path + "/" + content + "/metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding='utf-8') as f:
                    meta_data = json.load(f)
                dataset_id = meta_data.get("id", None)
                if dataset_id is None:
                    continue
                print(f"Found dataset: {dataset_id}", file=sys.stderr)
                # Assert we have logs, specs, jobs_metadata folders. If not create all those...
                stateless_handlers.make_root_dirs(user_id, "datasets", dataset_id)

        # get pre-existing models
        user_models_path = os.path.join(user_dir, "models")

        dir_contents = []
        if os.path.isdir(user_models_path):
            dir_contents = os.listdir(user_models_path)
        for content in dir_contents:
            metadata_path = user_models_path + "/" + content + "/metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding='utf-8') as f:
                    meta_data = json.load(f)
                model_id = meta_data.get("id", None)
                if model_id is None:
                    continue
                print(f"Found model: {model_id}", file=sys.stderr)
                # Assert we have logs, specs, jobs_metadata folders. If not create all those...
                stateless_handlers.make_root_dirs(user_id, "models", model_id)


def load_metadata_json(json_file):
    """Loads the json file provided"""
    with open(json_file, "r", encoding='utf-8') as f:
        metadata = json.load(f)

    return metadata


class AppHandler:
    """
    App Handler class
    - Static class
    """

    # Dataset API
    @staticmethod
    def list_datasets(user_id):
        """
        user_id: str, uuid
        Returns:
        list(dict) - list of datasets accessible by user where each element is metadata of a dataset
        """
        # Collect all metadatas
        metadatas = []
        for dataset_id in list(set(get_user_datasets(user_id) + stateless_handlers.get_public_datasets())):
            metadatas.append(stateless_handlers.get_handler_metadata(dataset_id))
        return metadatas

    # Create dataset
    @staticmethod
    def create_dataset(user_id, request_dict):
        """
        user_id: str, uuid
        request_dict: dict following DatasetReqSchema
            - type is required
            - format is required
        Returns:
        - 201 with metadata of created dataset if successful
        - 400 if dataset type and format not given
        """
        # Create a dataset ID and its root
        dataset_id = str(uuid.uuid4())

        # Gather type,format fields from request
        ds_type = request_dict.get("type", None)
        ds_format = request_dict.get("format", None)
        # Perform basic checks - valid type and format?
        if ds_type not in VALID_DSTYPES:
            msg = "Invalid dataset type"
            return Code(400, {}, msg)
        if ds_format not in read_network_config(ds_type)["api_params"]["formats"]:
            msg = "Incompatible dataset format and type"
            return Code(400, {}, msg)

        if request_dict.get("public", False):
            stateless_handlers.add_public_dataset(dataset_id)

        dataset_actions = get_dataset_actions(ds_type, ds_format)
        # Create metadata dict and create some initial folders
        metadata = {"id": dataset_id,
                    "created_on": datetime.datetime.now().isoformat(),
                    "last_modified": datetime.datetime.now().isoformat(),
                    "name": request_dict.get("name", "My Dataset"),
                    "description": request_dict.get("description", "My TAO Dataset"),
                    "version": request_dict.get("version", "1.0.0"),
                    "logo": request_dict.get("logo", "https://www.nvidia.com"),
                    "type": ds_type,
                    "format": ds_format,
                    "actions": dataset_actions
                    }
        stateless_handlers.make_root_dirs(user_id, "datasets", dataset_id)
        write_handler_metadata(user_id, "dataset", dataset_id, metadata)

        # Read this metadata from saved file...
        return_metadata = resolve_metadata_with_jobs(user_id, "dataset", dataset_id)
        ret_Code = Code(201, return_metadata, "Dataset created")
        return ret_Code

    # Update existing dataset for user based on request dict
    @staticmethod
    def update_dataset(user_id, dataset_id, request_dict):
        """
        user_id: str, uuid
        dataset_id: str, uuid
        request_dict: dict following DatasetReqSchema
            - type is required
            - format is required
        Returns:
        Code object
        - 200 with metadata of updated dataset if successful
        - 404 if dataset not found / user cannot access
        - 400 if invalid update
        """
        if not resolve_existence(user_id, "dataset", dataset_id):
            return Code(404, {}, "Dataset not found")

        if not check_write_access(user_id, dataset_id):
            return Code(404, {}, "Dataset not available")
        if request_dict.get("public", None):
            if request_dict["public"]:
                stateless_handlers.add_public_dataset(dataset_id)
            else:
                stateless_handlers.remove_public_dataset(dataset_id)
        metadata = resolve_metadata(user_id, "dataset", dataset_id)
        for key in request_dict.keys():

            # Cannot process the update, so return 400
            if key in ["type", "format"]:
                if request_dict[key] != metadata.get(key):
                    msg = f"Cannot change dataset {key}"
                    return Code(400, {}, msg)

            if key in ["name", "description", "version", "logo"]:
                requested_value = request_dict[key]
                if requested_value is not None:
                    metadata[key] = requested_value
                    metadata["last_modified"] = datetime.datetime.now().isoformat()
        write_handler_metadata(user_id, "dataset", dataset_id, metadata)
        # Read this metadata from saved file...
        return_metadata = resolve_metadata_with_jobs(user_id, "dataset", dataset_id)
        ret_Code = Code(200, return_metadata, "Dataset updated")
        return ret_Code

    # Retrieve existing dataset for user based on request dict
    @staticmethod
    def retrieve_dataset(user_id, dataset_id):
        """
        user_id: str, uuid
        dataset_id: str, uuid
        Returns:
        Code object
        - 200 with metadata of retrieved dataset if successful
        - 404 if dataset not found / user cannot access
        """
        if not resolve_existence(user_id, "dataset", dataset_id):
            return Code(404, {}, "Dataset not found")

        if not check_read_access(user_id, dataset_id):
            return Code(404, {}, "Dataset not found")
        return_metadata = resolve_metadata_with_jobs(user_id, "dataset", dataset_id)
        return Code(200, return_metadata, "Dataset retrieved")

    # Delete a user's dataset
    @staticmethod
    def delete_dataset(user_id, dataset_id):
        """
        user_id: str, uuid
        dataset_id: str, uuid
        Returns:
        Code object
        - 200 with metadata of deleted dataset if successful
        - 404 if dataset not found / user cannot access
        - 400 if dataset has running jobs / being used by a model and hence cannot be deleted
        """
        if not resolve_existence(user_id, "dataset", dataset_id):
            return Code(404, {}, "Dataset not found")

        if dataset_id not in get_user_datasets(user_id):
            return Code(404, {}, "Dataset cannot be deleted")

        # If dataset is being used by user's models.
        metadata_file_pattern = stateless_handlers.get_root() + f"{user_id}/models/**/metadata.json"
        metadata_files = glob.glob(metadata_file_pattern)
        for metadata_file in metadata_files:
            metadata = load_metadata_json(metadata_file)
            train_datasets = metadata.get("train_datasets", [])
            if type(train_datasets) != list:
                train_datasets = [train_datasets]
            if dataset_id in metadata.get("train_datasets", []) + \
                [metadata.get("eval_dataset", None),
                 metadata.get("inference_dataset", None),
                 metadata.get("calibration_dataset", None)]:
                return Code(400, {}, "Dataset in use")
        # Check if any job running
        return_metadata = resolve_metadata_with_jobs(user_id, "dataset", dataset_id)
        for job in return_metadata["jobs"]:
            if job["status"] == "Running":
                return Code(400, {}, "Dataset in use")

        # Check if dataset is public, then someone could be running it
        if return_metadata.get("public", False):
            return Code(400, {}, "Dataset is Public. Cannot delete")

        # Check if dataset is read only, if yes, cannot delete
        if return_metadata.get("read_only", False):
            return Code(400, {}, "Dataset is read only. Cannot delete")
        # Remove metadata file to signify deletion
        os.remove(stateless_handlers.get_handler_metadata_file(dataset_id))
        # Remove the whole folder as a Daemon...
        deletion_command = f"rm -rf {stateless_handlers.get_handler_root(dataset_id)}"
        delete_thread = threading.Thread(target=run_system_command, args=(deletion_command,))
        delete_thread.start()
        return Code(200, return_metadata, "Dataset deleted")

    @staticmethod
    def upload_dataset(user_id, dataset_id, file_tgz):
        """
        user_id: str, uuid
        dataset_id: str, uuid
        file_tgz: Flask request.files["file"]
        Returns:
        Code object
        - 201 with {} if successful
        - 404 if dataset not found / user cannot access
        - 400 if upload validation fails

        """
        if not resolve_existence(user_id, "dataset", dataset_id):
            return Code(404, {}, "Dataset not found")

        if not check_write_access(user_id, dataset_id):
            return Code(404, {}, "Dataset not available")
        # Save tar file at the dataset root
        tar_path = os.path.join(stateless_handlers.get_handler_root(dataset_id), "data.tar.gz")
        file_tgz.save(tar_path)

        metadata = resolve_metadata(user_id, "dataset", dataset_id)
        print("Uploading dataset to server", file=sys.stderr)
        return_Code = DS_UPLOAD_TO_FUNCTIONS[metadata.get("type")](tar_path, metadata)
        print("Uploading complete", file=sys.stderr)
        return return_Code

    # Spec API

    @staticmethod
    def get_spec_schema(user_id, handler_id, action, kind):
        """
        user_id: str, uuid
        handler_id: str, uuid corresponding to model/dataset
        action: str, a valid Action for a dataset
        kind: str, one of ["model","dataset"]
        Returns:
        Code object
        - 200 with spec in a json-schema format
        - 404 if model/dataset not found / user cannot access
        """
        if not resolve_existence(user_id, kind, handler_id):
            return Code(404, {}, "Spec schema not found")

        if not check_read_access(user_id, handler_id):
            return Code(404, {}, "Spec schema not available")

        metadata = resolve_metadata(user_id, kind, handler_id)
        # Action not available
        if action not in metadata.get("actions", []):
            return Code(404, {}, "Action not found")

        # Read csv from spec_utils/specs/<network_name>/action.csv
        # Convert to json schema
        json_schema = {}
        DIR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        network = metadata.get("network_arch", None)
        if not network:
            # Used for data conversion
            network = metadata.get("type", None)
        # Try regular format for CSV_PATH => "<network> - <action>.csv"
        CSV_PATH = os.path.join(DIR_PATH, "specs_utils", "specs", network, f"{network} - {action}.csv")
        if not os.path.exists(CSV_PATH):
            # Try secondary format for CSV_PATH => "<network> - <action>__<dataset-format>.csv"
            fmt = metadata.get("format", "_")
            CSV_PATH = os.path.join(DIR_PATH, "specs_utils", "specs", network, f"{network} - {action}__{fmt}.csv")
            if not os.path.exists(CSV_PATH):
                Code(404, {}, "Default specs do not exist for action")

        inferred_class_names = []
        # If class-wise config is applicable
        if read_network_config(network)["api_params"]["classwise"] == "True":
            # For each train dataset for the model
            metadata = resolve_metadata(user_id, kind, handler_id)
            for train_ds in metadata.get("train_datasets", []):
                # Obtain class list from classes.json
                classes_json = os.path.join(stateless_handlers.get_handler_root(train_ds), "classes.json")
                if not os.path.exists(classes_json):
                    continue
                with open(classes_json, "r", encoding='utf-8') as f:
                    inferred_class_names += json.loads(f.read())  # It is a list
            inferred_class_names = list(set(inferred_class_names))
        json_schema = csv_to_json_schema.convert(CSV_PATH, inferred_class_names)
        return Code(200, json_schema, "Schema retrieved")

    @staticmethod
    def get_spec(user_id, handler_id, action, kind):
        """
        user_id: str, uuid
        handler_id: str, uuid corresponding to model/dataset
        action: str, a valid Action for a dataset
        kind: str, one of ["model","dataset"]
        Returns:
        Code object
        - 200 with spec in a json format
        - 404 if model/dataset not found / user cannot access
        """
        if not resolve_existence(user_id, kind, handler_id):
            return Code(404, {}, "Spec not found")

        if not check_read_access(user_id, handler_id):
            return Code(404, {}, "Spec not available")

        metadata = resolve_metadata(user_id, kind, handler_id)

        # Action not available
        if action not in metadata.get("actions", []):
            return Code(404, {}, "Action not found")
        # read spec from action.json
        action_spec_path = stateless_handlers.get_handler_spec_root(handler_id) + f"/{action}.json"
        if os.path.exists(action_spec_path):
            data = {}
            with open(action_spec_path, mode='r', encoding='utf-8-sig') as f:
                data = json.load(f)
            msg = "Spec retrieved"
            return Code(200, data, msg)
        msg = "Spec not found"
        return Code(404, {}, msg)

    @staticmethod
    def save_spec(user_id, handler_id, action, request_dict, kind):
        """
        user_id: str, uuid
        handler_id: str, uuid corresponding to model/dataset
        action: str, a valid Action for a dataset
        request_dict: specs given by user
        kind: str, one of ["model","dataset"]
        Returns:
        Code object
        - 201 with posted spec in a json format
        - 400 if invalid update
        - 404 if model/dataset not found / user cannot access
        """
        if not resolve_existence(user_id, kind, handler_id):
            return Code(404, {}, "Not found")

        if not check_write_access(user_id, handler_id):
            return Code(404, {}, "Not available")

        metadata = resolve_metadata(user_id, kind, handler_id)

        # Action not available
        if action not in metadata.get("actions", []):
            return Code(404, {}, "Action not found")

        # save the request_dict inside action.json
        action_spec_path = stateless_handlers.get_handler_spec_root(handler_id) + f"/{action}.json"
        metadata["last_modified"] = datetime.datetime.now().isoformat()
        write_handler_metadata(user_id, kind, handler_id, metadata)

        # Get specs schema
        schema_ret = AppHandler.get_spec_schema(user_id, handler_id, action, kind)
        if schema_ret.code != 200:
            msg = "Schema not found"
            return Code(404, {}, msg)

        # Harden and validate specs
        # try:
        #     print(request_dict, "\n", file=sys.stderr)
        #     print(schema, "\n", file=sys.stderr)
        #     hardened_spec = specCheck.harden(request_dict,schema)
        #     print(hardened_spec,  "\n", file=sys.stderr)
        #     failed_spec = specCheck.validate(hardened_spec,schema)
        #     print(failed_spec,  "\n", file=sys.stderr)
        #     if failed_spec:
        #         return Code(404,{},failed_spec)
        #     else:
        #         request_dict = hardened_spec.copy()
        # except Exception as e:
        #     return Code(404,{},str(e))

        with open(action_spec_path, "w", encoding='utf-8') as f:
            request_json_string = json.dumps(request_dict, indent=4)
            f.write(request_json_string)
        msg = "Spec saved"
        return Code(201, request_dict, msg)

    @staticmethod
    def update_spec(user_id, handler_id, action, request_dict, kind):
        """
        user_id: str, uuid
        handler_id: str, uuid corresponding to model/dataset
        action: str, a valid Action for a dataset
        request_dict: specs given by user
        kind: str, one of ["model","dataset"]
        Returns:
        Code object
        - 201 with posted spec in a json format
        - 400 if invalid update
        - 404 if model/dataset not found / user cannot access
        """
        if not resolve_existence(user_id, kind, handler_id):
            return Code(404, {}, "Not found")

        if not check_write_access(user_id, handler_id):
            return Code(404, {}, "Not available")

        metadata = resolve_metadata(user_id, kind, handler_id)

        # Action not available
        if action not in metadata.get("actions", []):
            return Code(404, {}, "Action not found")

        # read spec from action.json
        action_spec_path = stateless_handlers.get_handler_spec_root(handler_id) + f"/{action}.json"
        if os.path.exists(action_spec_path):
            data = {}
            with open(action_spec_path, mode='r', encoding='utf-8-sig') as f:
                data = json.load(f)
            print("Data", data, file=sys.stderr)

            try:
                nested_update(data, request_dict)
                print("Data", data, file=sys.stderr)
                return_code = AppHandler.save_spec(user_id, handler_id, action, data, kind)
                if return_code.code == 201:
                    msg = "Spec retrieved"
                    return Code(200, data, msg)
                msg = "Specs save failed"
                return Code(400, {}, msg)
            except:
                msg = "Specs cannot be updated, check the request"
                return Code(400, {}, msg)

        # if it doesn't exist, error
        else:
            msg = "Spec not found"
            return Code(404, {}, msg)

    # Job API

    @staticmethod
    def job_run(user_id, handler_id, parent_job_id, actions, kind):
        """
        user_id: str, uuid
        handler_id: str, uuid corresponding to model/dataset
        parent_job_id: str, uuid
        actions: list(str), each action corresponds to a valid action
        kind: str, one of ["model","dataset"]
        Returns:
        201 with list(str) where each str is a uuid for job if jobs successfully queued
        404 with [] if dataset/model/parent_job_id/actions not found
        """
        if not resolve_existence(user_id, kind, handler_id):
            return Code(404, [], f"{kind} not found")

        if not check_write_access(user_id, handler_id):
            return Code(404, [], f"{kind} not found")

        if parent_job_id:
            if not resolve_job_existence(user_id, kind, handler_id, parent_job_id):
                return Code(404, [], "job not found")

        if len(actions) == 0:
            return Code(404, [], "action not found")

        handler_metadata = resolve_metadata(user_id, kind, handler_id)

        if "ptm" in handler_metadata.keys():
            ptm_ids = handler_metadata["ptm"]
            for ptm_id in ptm_ids:
                if ptm_id:
                    if stateless_handlers.get_handler_metadata(ptm_id).get("ngc_path", None):
                        download_ptm(ptm_id)
                        # job_run_thread = threading.Thread(target=download_ptm,args=(ptm_id,))
                        # job_run_thread.start()

        if is_request_automl(user_id, handler_id, parent_job_id, actions, kind):
            return AutoMLHandler.start(user_id, handler_id, handler_metadata)
        try:
            job_ids = [str(uuid.uuid4()) for i in actions]
            job_contexts = create_job_contexts(parent_job_id, actions, job_ids, handler_id)
            on_new_job(job_contexts)
            return Code(201, job_ids, "Jobs scheduled")
        except:
            return Code(404, [], "action not found")

    @staticmethod
    def job_list(user_id, handler_id, kind):
        """
        user_id: str, uuid
        handler_id: str, uuid corresponding to model/dataset
        kind: str, one of ["model","dataset"]
        Returns:
        200, list(dict) - each dict follows JobResultSchema if found
        404, [] if not found
        """
        if not resolve_existence(user_id, kind, handler_id):
            return Code(404, [], f"{kind} not found")

        if not check_read_access(user_id, handler_id):
            return Code(404, [], f"{kind} not found")

        return_metadata = resolve_metadata_with_jobs(user_id, kind, handler_id).get("jobs", [])
        return Code(200, return_metadata, "Jobs retrieved")

    @staticmethod
    def job_cancel(user_id, handler_id, job_id, kind):
        """
        user_id: str, uuid
        handler_id: str, uuid corresponding to model/dataset
        job_id: str, uuid corresponding to job to be cancelled
        kind: str, one of ["model","dataset"]
        Returns:
        200, [job_id] - if job can be cancelled
        404, [] if not found
        """
        if not resolve_existence(user_id, kind, handler_id):
            return Code(404, [], f"{kind} not found")

        if not check_write_access(user_id, handler_id):
            return Code(404, [], f"{kind} not found")

        if not resolve_job_existence(user_id, kind, handler_id, job_id):
            return Code(404, [], "job not found")

        if is_job_automl(user_id, handler_id, job_id):
            return AutoMLHandler.stop(user_id, handler_id, job_id)

        # If job is error / done, then cancel is NoOp
        job_metadata = stateless_handlers.get_handler_job_metadata(handler_id, job_id)
        job_status = job_metadata.get("status", "Error")

        if job_status in ["Error", "Done"]:
            return Code(404, [], "incomplete job not found")

        if job_status == "Pending":
            on_delete_job(handler_id, job_id)
            stateless_handlers.update_job_status(handler_id, job_id, status="Error")
            return Code(200, [job_id], "job cancelled")

        if job_status == "Running":
            try:
                # Delete K8s job
                jobDriver.delete(job_id)
                stateless_handlers.update_job_status(handler_id, job_id, status="Error")
                return Code(200, [job_id], "job cancelled")
            except:
                return Code(404, [], "job not found in platform")

        else:
            return Code(404, [], "job status not found")

    @staticmethod
    def job_retrieve(user_id, handler_id, job_id, kind):
        """
        user_id: str, uuid
        handler_id: str, uuid corresponding to model/dataset
        job_id: str, uuid corresponding to job to be retrieved
        kind: str, one of ["model","dataset"]
        Returns:
        200, dict following JobResultSchema - if job found
        404, {} if not found
        """
        if not resolve_existence(user_id, kind, handler_id):
            return Code(404, {}, "Dataset not found")

        if not check_read_access(user_id, handler_id):
            return Code(404, {}, "Dataset not found")

        if not resolve_job_existence(user_id, kind, handler_id, job_id):
            return Code(404, {}, "Job not found")

        if is_job_automl(user_id, handler_id, job_id):
            return AutoMLHandler.retrieve(user_id, handler_id, job_id)
        path = os.path.join(stateless_handlers.get_root(), user_id, kind + "s", handler_id, "jobs_metadata", job_id + ".json")
        job_meta = stateless_handlers.load_json_data(path)
        return Code(200, job_meta, "Job retrieved")

    # Delete job
    @staticmethod
    def job_delete(user_id, handler_id, job_id, kind):
        """
        user_id: str, uuid
        handler_id: str, uuid corresponding to model/dataset
        job_id: str, uuid corresponding to job to be deleted
        kind: str, one of ["model","dataset"]
        Returns:
        200, [job_id] - if job can be deleted
        404, [] if not found
        """
        if not resolve_existence(user_id, kind, handler_id):
            return Code(404, [], f"{kind} not found")

        if not check_write_access(user_id, handler_id):
            return Code(404, [], f"{kind} not found")

        if not resolve_job_existence(user_id, kind, handler_id, job_id):
            return Code(404, [], "job not found")
        try:
            # If job is running, cannot delete
            job_metadata = stateless_handlers.get_handler_job_metadata(handler_id, job_id)
            if job_metadata.get("status", "Error") in ["Running", "Pending"]:
                return Code(400, [], "job cannot be deleted")
            # Delete job metadata
            job_metadata_path = os.path.join(stateless_handlers.get_handler_jobs_metadata_root(handler_id), job_id + ".json")
            if os.path.exists(job_metadata_path):
                os.remove(job_metadata_path)
            # Delete job logs
            job_log_path = os.path.join(stateless_handlers.get_handler_log_root(handler_id), job_id + ".txt")
            if os.path.exists(job_log_path):
                os.remove(job_log_path)
            # Delete the job directory in the background
            deletion_command = "rm -rf " + os.path.join(stateless_handlers.get_handler_root(handler_id), job_id)
            targz_path = os.path.join(stateless_handlers.get_handler_root(handler_id), job_id + ".tar.gz")
            if os.path.exists(targz_path):
                deletion_command += "; rm -rf " + targz_path
            delete_thread = threading.Thread(target=run_system_command, args=(deletion_command,))
            delete_thread.start()
            return Code(200, [job_id], "job deleted")
        except:
            return Code(400, [], "job cannot be deleted")

    # Download model job
    @staticmethod
    def job_download(user_id, handler_id, job_id, kind):
        """
        user_id: str, uuid
        handler_id: str, uuid corresponding to model/dataset
        job_id: str, uuid corresponding to job to be deleted
        kind: str, one of ["model","dataset"]
        Returns:
        200, Path to a tar.gz created from the job directory
        404, None if not found
        """
        if not resolve_existence(user_id, kind, handler_id):
            return Code(404, None, f"{kind} not found")
        if not check_read_access(user_id, handler_id):
            return Code(404, None, f"{kind} not found")
        if not resolve_job_existence(user_id, kind, handler_id, job_id):
            return Code(404, None, "job not found")
        try:
            root = stateless_handlers.get_handler_root(handler_id)
            # Copy job logs from root/logs/<job_id>.txt to root/<job_id>/logs_from_toolkit.txt
            job_log_path = stateless_handlers.get_handler_log_root(handler_id) + f"/{job_id}.txt"
            if os.path.exists(job_log_path):
                shutil.copy(job_log_path, stateless_handlers.get_handler_root(handler_id) + f"/{job_id}/logs_from_toolkit.txt")
            out_tar = os.path.join(root, job_id + ".tar.gz")
            command = f"cd {root} ; tar -zcvf {job_id}.tar.gz {job_id} ; cd -"
            run_system_command(command)
            if os.path.exists(out_tar):
                return Code(200, out_tar, "job deleted")
            return Code(404, None, "job output not found")

        except:
            return Code(404, None, "job output not found")

    # Model API
    @staticmethod
    def list_models(user_id):
        """
        user_id: str, uuid
        Returns:
        list(dict) - list of models accessible by user where each element is metadata of a model
        """
        # Collect all metadatas
        metadatas = []
        for model_id in list(set(get_user_models(user_id) + stateless_handlers.get_public_models())):
            metadatas.append(stateless_handlers.get_handler_metadata(model_id))
        return metadatas

    @staticmethod
    def create_model(user_id, request_dict):
        """
        user_id: str, uuid
        request_dict: dict following ModelReqSchema
            - network_arch is required
            - encryption_key is required (not enforced)
        Returns:
        - 201 with metadata of created model if successful
        - 400 if model type and format not given
        """
        # Create a dataset ID and its root
        model_id = str(uuid.uuid4())

        # Gather type,format fields from request
        mdl_nw = request_dict.get("network_arch", None)
        # Perform basic checks - valid type and format?
        if mdl_nw not in VALID_NETWORKS:
            msg = "Invalid network arch"
            return Code(400, {}, msg)

        if request_dict.get("public", False):
            stateless_handlers.add_public_model(model_id)
        # Create metadata dict and create some initial folders
        # Initially make datasets, ptm None
        metadata = {"id": model_id,
                    "created_on": datetime.datetime.now().isoformat(),
                    "last_modified": datetime.datetime.now().isoformat(),
                    "name": request_dict.get("name", "My Model"),
                    "description": request_dict.get("description", "My TAO Model"),
                    "version": request_dict.get("version", "1.0.0"),
                    "logo": request_dict.get("logo", "https://www.nvidia.com"),
                    "ngc_path": request_dict.get("ngc_path", ""),
                    "encryption_key": request_dict.get("encryption_key", "tlt_encode"),
                    "read_only": request_dict.get("read_only", False),
                    "public": request_dict.get("public", False),
                    "network_arch": mdl_nw,
                    "dataset_type": read_network_config(mdl_nw)["api_params"]["dataset_type"],
                    "actions": read_network_config(mdl_nw)["api_params"]["actions"],
                    "train_datasets": [],
                    "eval_dataset": None,
                    "inference_dataset": None,
                    "additional_id_info": None,
                    "calibration_dataset": None,
                    "ptm": [],
                    "automl_enabled": False,
                    "automl_algorithm": None,
                    "metric": None,
                    "automl_add_hyperparameters": "",
                    "automl_remove_hyperparameters": ""
                    }

        if request_dict.get("automl_enabled", False):
            if mdl_nw not in AUTOML_DISABLED_NETWORKS:
                metadata["automl_enabled"] = True
                metadata["automl_algorithm"] = request_dict.get("automl_algorithm", "Bayesian")
                metadata["metric"] = request_dict.get("metric", "map")
                metadata["automl_add_hyperparameters"] = request_dict.get("automl_add_hyperparameters", "")
                metadata["automl_remove_hyperparameters"] = request_dict.get("automl_remove_hyperparameters", "")
                # AutoML optional params
                if request_dict.get("automl_max_recommendations"):
                    metadata["automl_max_recommendations"] = request_dict.get("automl_max_recommendations")
                if request_dict.get("automl_delete_intermediate_ckpt"):
                    metadata["automl_delete_intermediate_ckpt"] = request_dict.get("automl_delete_intermediate_ckpt")
                if request_dict.get("automl_R"):
                    metadata["automl_R"] = request_dict.get("automl_R")
                if request_dict.get("automl_nu"):
                    metadata["automl_nu"] = request_dict.get("automl_nu")
                if request_dict.get("epoch_multiplier"):
                    metadata["epoch_multiplier"] = request_dict.get("epoch_multiplier")
            else:
                return Code(400, {}, "automl_enabled cannot be True for unsupported network")

        # Update datasets and ptms if given
        for key in ["train_datasets", "eval_dataset", "inference_dataset", "calibration_dataset", "ptm"]:
            if key not in request_dict.keys():
                continue
            value = request_dict[key]
            if stateless_handlers.model_update_handler_attributes(user_id, metadata, key, value):
                metadata[key] = value
            else:
                return Code(400, {}, f"Provided {key} cannot be added")
        # Actual "creation" happens here...
        stateless_handlers.make_root_dirs(user_id, "models", model_id)
        write_handler_metadata(user_id, "model", model_id, metadata)

        # Read this metadata from saved file...
        return_metadata = resolve_metadata_with_jobs(user_id, "model", model_id)
        ret_Code = Code(201, return_metadata, "Model created")
        return ret_Code

    # Update existing model for user based on request dict
    @staticmethod
    def update_model(user_id, model_id, request_dict):
        """
        user_id: str, uuid
        model_id: str, uuid
        request_dict: dict following ModelReqSchema
        Returns:
        - 200 with metadata of updated model if successful
        - 404 if model not found / user cannot access
        - 400 if invalid update / model is read only
        """
        if not resolve_existence(user_id, "model", model_id):
            return Code(400, {}, "Does not exist")

        if not check_write_access(user_id, model_id):
            return Code(400, {}, "Does not exist")

        # if public is set to True => add it to public_models, if it is set to False => take it down
        # if public is not there, do nothing
        if request_dict.get("public", None):
            if request_dict["public"]:
                stateless_handlers.add_public_model(model_id)
            else:
                stateless_handlers.remove_public_model(model_id)

        metadata = resolve_metadata(user_id, "model", model_id)
        for key in request_dict.keys():

            # Cannot process the update, so return 400
            if key in ["network_arch"]:
                if request_dict[key] != metadata.get(key):
                    msg = f"Cannot change model {key}"
                    return Code(400, {}, msg)

            if key in ["name", "description", "version", "logo",
                       "ngc_path", "encryption_key", "read_only", "public"]:
                requested_value = request_dict[key]
                if requested_value is not None:
                    metadata[key] = requested_value
                    metadata["last_modified"] = datetime.datetime.now().isoformat()

            if key in ["train_datasets", "eval_dataset", "inference_dataset", "calibration_dataset", "ptm"]:
                value = request_dict[key]
                if stateless_handlers.model_update_handler_attributes(user_id, metadata, key, value):
                    metadata[key] = value
                else:
                    return Code(400, {}, f"Provided {key} cannot be added")

            if key in ["automl_enabled"]:
                value = request_dict[key]
                # If False, can set. If True, need to check if AutoML is supported
                if value:
                    mdl_nw = metadata.get("network_arch", "")
                    if mdl_nw not in AUTOML_DISABLED_NETWORKS:
                        metadata[key] = True
                        metadata["automl_algorithm"] = request_dict.get("automl_algorithm", "Bayesian")
                        metadata["metric"] = request_dict.get("metric", "map")
                        metadata["automl_add_hyperparameters"] = request_dict.get("automl_add_hyperparameters", "")
                        metadata["automl_remove_hyperparameters"] = request_dict.get("automl_remove_hyperparameters", "")

                        # AutoML optional params
                        if request_dict.get("automl_max_recommendations"):
                            metadata["automl_max_recommendations"] = request_dict.get("automl_max_recommendations")
                        if request_dict.get("automl_delete_intermediate_ckpt"):
                            metadata["automl_delete_intermediate_ckpt"] = request_dict.get("automl_delete_intermediate_ckpt")
                        if request_dict.get("automl_R"):
                            metadata["automl_R"] = request_dict.get("automl_R")
                        if request_dict.get("automl_nu"):
                            metadata["automl_nu"] = request_dict.get("automl_nu")
                        if request_dict.get("epoch_multiplier"):
                            metadata["epoch_multiplier"] = request_dict.get("epoch_multiplier")

                    else:
                        return Code(400, {}, "automl_enabled cannot be True for unsupported network")
                else:
                    metadata[key] = value
        write_handler_metadata(user_id, "model", model_id, metadata)
        # Read this metadata from saved file...
        return_metadata = resolve_metadata_with_jobs(user_id, "model", model_id)
        ret_Code = Code(200, return_metadata, "Model updated")
        return ret_Code

    @staticmethod
    def retrieve_model(user_id, model_id):
        """
        user_id: str, uuid
        model_id: str, uuid

        Returns:
        - 200 with metadata of retrieved model if successful
        - 404 if model not found / user cannot access
        """
        if not resolve_existence(user_id, "model", model_id):
            return Code(404, {}, "Model not found")

        if not check_read_access(user_id, model_id):
            return Code(404, {}, "Model not found")
        return_metadata = resolve_metadata_with_jobs(user_id, "model", model_id)
        return Code(200, return_metadata, "Model retrieved")

    @staticmethod
    def delete_model(user_id, model_id):
        """
        user_id: str, uuid
        model_id: str, uuid
        Returns:
        Code object
        - 200 with metadata of deleted model if successful
        - 404 if model not found / user cannot access
        - 400 if mdoel has running jobs / being used and hence cannot be deleted
        """
        if not resolve_existence(user_id, "model", model_id):
            return Code(404, {}, "Model not found")

        if model_id not in get_user_models(user_id):
            return Code(404, {}, "Model cannot be deleted")

        # If model is being used by user's models.
        metadata_file_pattern = stateless_handlers.get_root() + f"{user_id}/models/**/metadata.json"
        metadata_files = glob.glob(metadata_file_pattern)
        for metadata_file in metadata_files:
            metadata = load_metadata_json(metadata_file)
            if model_id == metadata.get("ptm", None):
                return Code(400, {}, "Model in use as a ptm")
        # Check if any job running
        return_metadata = resolve_metadata_with_jobs(user_id, "model", model_id)
        for job in return_metadata["jobs"]:
            if job["status"] == "Running":
                return Code(400, {}, "Model in use")

        # Check if model is public, then someone could be running it
        if return_metadata.get("public", False):
            return Code(400, {}, "Model is Public. Cannot delete")

        # Check if model is read only, if yes, cannot delete
        if return_metadata.get("read_only", False):
            return Code(400, {}, "Model is read only. Cannot delete")
        # Remove metadata file to signify deletion
        os.remove(stateless_handlers.get_handler_metadata_file(model_id))
        # Remove the whole folder as a Daemon...
        deletion_command = f"rm -rf {stateless_handlers.get_handler_root(model_id)}"
        delete_thread = threading.Thread(target=run_system_command, args=(deletion_command,))
        delete_thread.start()
        return Code(200, return_metadata, "Model deleted")

    @staticmethod
    def resume_model_job(user_id, model_id, job_id, kind):
        """
        user_id: str, uuid
        model_id: str, uuid corresponding to model
        job_id: str, uuid corresponding to a train job
        Returns:
        201 with [job_id] if job resumed and added to queue
        400 with [] if job_id does not correspond to a train action or if it cannot be resumed
        404 with [] if model/job_id not found
        """
        if not resolve_existence(user_id, "model", model_id):
            return Code(404, [], "Model not found")

        if not check_write_access(user_id, model_id):
            return Code(404, [], "Model not found")

        action = infer_action_from_job(model_id, job_id)
        if action != "train":
            return Code(400, [], "Action not train")
        handler_metadata = resolve_metadata(user_id, kind, model_id)

        if is_job_automl(user_id, model_id, job_id):
            return AutoMLHandler.resume(user_id, model_id, job_id, handler_metadata)

        try:
            # Create a job and run it
            job_contexts = create_job_contexts(None, ["train"], [job_id], model_id)
            on_new_job(job_contexts)
            return Code(200, [job_id], "Action resumed")
        except:
            return Code(400, [], "Action cannot be resumed")
