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

"""API handler modules"""
import re
import copy
from datetime import datetime, timezone
import glob
import json
import os
import shutil
import sys
import tarfile
import threading
import time
import traceback
import uuid

from automl.utils import merge_normal_and_automl_job_meta
from constants import (AUTOML_DISABLED_NETWORKS, TENSORBOARD_DISABLED_NETWORKS, TENSORBOARD_EXPERIMENT_LIMIT, VALID_DSTYPES, VALID_MODEL_DOWNLOAD_TYPE, VALID_NETWORKS, TAO_NETWORKS, _DATA_GENERATE_ACTIONS, MONAI_NETWORKS, MEDICAL_CUSTOM_ARCHITECT)
from handlers import ngc_handler, stateless_handlers
from handlers.automl_handler import AutoMLHandler
from handlers.cloud_storage import create_cs_instance
from handlers.ds_upload import DS_UPLOAD_TO_FUNCTIONS
from handlers.encrypt import NVVaultEncryption
# from handlers import nvcf_handler
from handlers.monai.helpers import CapGpuUsage, download_from_url, validate_monai_bundle, CUSTOMIZED_BUNDLE_URL_FILE, CUSTOMIZED_BUNDLE_URL_KEY
from handlers.monai_dataset_handler import MONAI_DATASET_ACTIONS, MonaiDatasetHandler
from handlers.monai_model_handler import MonaiModelHandler
# TODO: force max length of code line to 120 chars
from handlers.stateless_handlers import (check_read_access, check_write_access, get_handler_spec_root, get_root,
                                         infer_action_from_job, is_valid_uuid4, printc,
                                         resolve_existence, resolve_metadata, resolve_root, base_exp_uuid, get_handler_log_root,
                                         get_handler_job_metadata, get_jobs_root, sanitize_handler_metadata, write_handler_metadata)
from handlers.tis_handler import TISHandler
from handlers.tensorboard_handler import TensorboardHandler
from handlers.utilities import (Code, download_base_experiment, download_dataset, get_monai_bundle_path, search_for_base_experiment, get_files_from_cloud,
                                prep_tis_model_repository, resolve_checkpoint_root_and_search, validate_and_update_experiment_metadata, validate_num_gpu)
from handlers.mongo_handler import MongoHandler
from job_utils import executor as jobDriver
from job_utils.workflow_driver import create_job_context, on_delete_job, on_new_job
from job_utils.automl_job_utils import on_delete_automl_job
from specs_utils import csv_to_json_schema
from utils import create_folder_with_permissions, get_admin_api_key, run_system_command, read_network_config, merge_nested_dicts, check_and_convert, safe_load_file, safe_dump_file, log_monitor, DataMonitorLogTypeEnum

# Identify if workflow is on NGC
BACKEND = os.getenv("BACKEND", "local-k8s")


# Helpers
def resolve_job_existence(job_id):
    """Return whether job exists or not"""
    mongo_jobs = MongoHandler("tao", "jobs")
    job = mongo_jobs.find_one({'id': job_id})
    if job:
        return True
    return False


def delete_jobs_for_handler(handler_id, kind):
    """Deletes job metadatas associated with handler_id"""
    mongo_jobs = MongoHandler("tao", "jobs")
    mongo_jobs.delete_many({f"{kind}_id": handler_id})


def resolve_metadata_with_jobs(user_id, org_name, kind, handler_id, jobs_return_type="dictionary"):
    """Reads job_id.json in jobs_metadata folder and return it's contents"""
    if not user_id:
        print("Can't resolve job metadata without user information", file=sys.stderr)
        return {}
    handler_id = "*" if handler_id in ("*", "all") else handler_id
    metadata = {} if handler_id == "*" else resolve_metadata(kind, handler_id)
    if metadata or handler_id == "*":
        metadata["jobs"] = {}
        if jobs_return_type == "list":
            metadata["jobs"] = []
        jobs = stateless_handlers.get_jobs_for_handler(handler_id, kind)
        for job_meta in jobs:
            job_id = job_meta["id"]
            if is_job_automl(user_id, org_name, job_id):
                merge_normal_and_automl_job_meta(user_id, org_name, job_id, job_meta)
            if jobs_return_type == "dictionary":
                metadata["jobs"][job_id] = job_meta
            if jobs_return_type == "list":
                metadata["jobs"].append(job_meta)
        return metadata
    return {}


def get_org_experiments(org_name):
    """Returns a list of experiment IDs that are available for the given org_name"""
    mongo_experiments = MongoHandler("tao", "experiments")
    org_experiments = mongo_experiments.find({'org_name': org_name})
    experiments = []
    for experiment in org_experiments:
        experiment_id = experiment.get('id')
        experiments.append(experiment_id)
    return experiments


def get_org_datasets(org_name):
    """Returns a list of dataset IDs that are available for the given org_name"""
    mongo_datasets = MongoHandler("tao", "datasets")
    org_datasets = mongo_datasets.find({'org_name': org_name})
    datasets = []
    for dataset in org_datasets:
        dataset_id = dataset.get('id')
        datasets.append(dataset_id)
    return datasets


def get_org_workspaces(org_name):
    """Returns a list of workspace IDs that are available in given org_name"""
    mongo_workspaces = MongoHandler("tao", "workspaces")
    org_workspaces = mongo_workspaces.find({'org_name': org_name})
    workspaces = []
    for workspace in org_workspaces:
        workspace_id = workspace.get('id')
        workspaces.append(workspace_id)
    return workspaces


def get_user_experiments(user_id, mongo_users=None):
    """Returns a list of experiments that are available for the user"""
    user = stateless_handlers.get_user(user_id, mongo_users)
    experiments = user.get("experiments", [])
    return experiments


def get_user_datasets(user_id, mongo_users=None):
    """Returns a list of datasets that are available for the user"""
    user = stateless_handlers.get_user(user_id, mongo_users)
    datasets = user.get("datasets", [])
    return datasets


def get_user_workspaces(user_id, mongo_users=None):
    """Returns a list of datasets that are available for the user in given org_name"""
    user = stateless_handlers.get_user(user_id, mongo_users)
    workspaces = user.get("workspaces", [])
    return workspaces


def get_job(job_id):
    """Returns job from DB"""
    mongo_jobs = MongoHandler("tao", "jobs")
    job_query = {'id': job_id}
    job = mongo_jobs.find_one(job_query)
    return job


def get_experiment(experiment_id):
    """Returns experiment from DB"""
    mongo_experiments = MongoHandler("tao", "experiments")
    experiment_query = {'id': experiment_id}
    experiment = mongo_experiments.find_one(experiment_query)
    return experiment


def get_dataset(dataset_id):
    """Returns dataset from DB"""
    mongo_datasets = MongoHandler("tao", "datasets")
    dataset_query = {'id': dataset_id}
    dataset = mongo_datasets.find_one(dataset_query)
    return dataset


def get_workspace(workspace_id):
    """Returns workspace from DB"""
    mongo_workspaces = MongoHandler("tao", "workspaces")
    workspace_query = {'id': workspace_id}
    workspace = mongo_workspaces.find_one(workspace_query)
    return workspace


def create_blob_dataset(org_name, kind, handler_id):
    """Creates a blob dataset"""
    # Make a placeholder for S3 blob dataset
    msg = "Doesn't support the blob dataset for now."
    return Code(400, {}, msg)


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


def nested_update(source, additions, allow_overwrite=True):
    """Merge one dictionary(additions) into another(source)"""
    if not isinstance(additions, dict):
        return source
    for key, value in additions.items():
        if isinstance(value, dict):
            # Initialize key in source if not present
            if key not in source:
                source[key] = {}
            source[key] = nested_update(source[key], value, allow_overwrite=allow_overwrite)
        else:
            source[key] = value if allow_overwrite else source.get(key, value)
    return source


def is_job_automl(user_id, org_name, job_id):
    """Returns if the job is automl-based job or not"""
    try:
        root = stateless_handlers.get_jobs_root(user_id, org_name)
        jobdir = os.path.join(root, job_id)
        automl_signature = os.path.join(jobdir, "controller.log")
        return os.path.exists(automl_signature)
    except:
        print(traceback.format_exc(), file=sys.stderr)
        return False


def is_request_automl(handler_id, action, kind):
    """Returns if the job requested is automl based train or not"""
    handler_metadata = resolve_metadata(kind, handler_id)
    if handler_metadata.get("automl_settings", {}).get("automl_enabled", False) and action == "train":
        return True
    return False


def get_job_logs(log_file_path):
    with open(log_file_path, 'r', encoding="utf-8") as log_file:
        while True:
            log_line = log_file.readline()
            if not log_line:
                break

            yield log_line


class AppHandler:
    """
    App Handler class
    - Static class
    """

    # Workspace API
    @staticmethod
    def list_workspaces(user_id, org_name):
        """
        user_id: str, uuid
        org_name: str
        Returns:
        list(dict) - list of workspaces accessible by user where each element is metadata of a workspace
        """
        # Collect all metadatas
        metadatas = []
        for workspace_id in list(set(get_org_workspaces(org_name))):
            handler_metadata = stateless_handlers.get_handler_metadata(workspace_id, 'workspaces')
            shared_workspace = handler_metadata.get("shared", False)
            if handler_metadata:
                if shared_workspace or handler_metadata.get("user_id") == user_id:
                    metadatas.append(handler_metadata)
            else:
                # Something is wrong. The user metadata has a workspace that doesn't exist in the system.
                contexts = {"user_id": user_id, "org_name": org_name, "handler_id": workspace_id}
                printc("Workspace not found. Skipping.", contexts, file=sys.stderr)
        return metadatas

    @staticmethod
    def retrieve_workspace(org_name, workspace_id):
        """
        org_name: str
        workspace_id: str, uuid
        Returns:
        Code object
        - 200 with metadata of retrieved workspace if successful
        - 404 if workspace not found / user cannot access
        """
        handler_metadata = resolve_metadata("workspace", workspace_id)
        if not handler_metadata:
            return Code(404, {}, "Workspace not found")

        if not check_read_access(org_name, workspace_id, kind="workspaces"):
            return Code(404, {}, "Workspace not found")

        return Code(200, handler_metadata, "Workspace retrieved")

    @staticmethod
    def retrieve_cloud_datasets(org_name, workspace_id, dataset_type, dataset_format, dataset_intention):
        """
        org_name: str
        workspace_id: str, uuid
        dataset_type: str
        dataset_format: str
        dataset_intention: str
        Returns:
        Code object
        - 200 with path of cloud datasets if successfull
        - 404 if workspace not found / user cannot access
        """
        handler_metadata = resolve_metadata("workspace", workspace_id)
        if not handler_metadata:
            return Code(404, {}, "Workspace not found")

        if not check_read_access(org_name, workspace_id, kind="workspaces"):
            return Code(404, {}, "Workspace not found")

        cloud_instance, _ = create_cs_instance(handler_metadata)
        cloud_files, _ = cloud_instance.list_files_in_folder("data")
        suggestions = set([])
        for cloud_file_path in cloud_files:
            cloud_folder = os.path.dirname(cloud_file_path)
            if dataset_type in ("semantic_segmentation", "pose_classification"):
                cloud_folder = os.path.dirname(cloud_folder)
            if dataset_type == "ml_recog":
                index_of_folder = cloud_folder.find("metric_learning_recognition")
                if index_of_folder != "-1":
                    cloud_folder = cloud_folder[0:index_of_folder]
            dataset_handler_metadata = {
                "type": dataset_type,
                "format": dataset_format,
                "use_for": dataset_intention
            }
            is_cloud_dataset_present = DS_UPLOAD_TO_FUNCTIONS[dataset_type](org_name, dataset_handler_metadata, temp_dir=f"/{cloud_folder}", workspace_metadata=handler_metadata)
            if is_cloud_dataset_present:
                suggestions.add(f"/{cloud_folder}")
        suggestions = list(suggestions)
        return_response_data = {"dataset_paths": suggestions}
        if suggestions:
            return Code(200, return_response_data, "Dataset folder path suggestions retrieved")
        return Code(200, return_response_data, "Dataset folder path suggestion couldn't be retrieved")

    @staticmethod
    def create_workspace(user_id, org_name, request_dict):
        """
        org_name: str
        user_id: str, uuid
        request_dict: dict following WorkspaceRspSchema
            - type is required
            - format is required
        Returns:
        - 201 with metadata of created workspace if successful
        - 400 with appropriate error messages
        """
        workspace_id = str(uuid.uuid4())
        # Create metadata dict and create some initial folders
        metadata = {"id": workspace_id,
                    "user_id": user_id,
                    "org_name": org_name,
                    "created_on": datetime.now(tz=timezone.utc),
                    "last_modified": datetime.now(tz=timezone.utc),
                    "name": request_dict.get("name", "My Workspace"),
                    "shared": request_dict.get("shared", False),
                    "version": request_dict.get("version", "1.0.0"),
                    "cloud_type": request_dict.get("cloud_type", ""),
                    "cloud_specific_details": request_dict.get("cloud_specific_details", {}),
                    }

        encrypted_metadata = copy.deepcopy(metadata)

        # Encrypt Cloud details
        if BACKEND in ("BCP", "NVCF") and encrypted_metadata["cloud_specific_details"]:
            cloud_specific_details = encrypted_metadata["cloud_specific_details"]
            if cloud_specific_details:
                config_path = os.getenv("VAULT_SECRET_PATH", None)
                encryption = NVVaultEncryption(config_path)
                for key, value in cloud_specific_details.items():
                    if encryption.check_config()[0]:
                        encrypted_metadata["cloud_specific_details"][key] = encryption.encrypt(value)
                    elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                        return Code(400, {}, "Vault service does not work, can't save cloud workspace")

        print("encrypted_metadata", encrypted_metadata, file=sys.stderr)
        try:
            if encrypted_metadata["cloud_type"] in ("aws", "azure"):
                create_cs_instance(encrypted_metadata)
        except:
            print(traceback.format_exc, file=sys.stderr)
            return Code(400, {}, "Provided cloud credentials are invalid")

        write_handler_metadata(workspace_id, encrypted_metadata, "workspace")
        mongo_users = MongoHandler("tao", "users")
        workspaces = get_user_workspaces(user_id, mongo_users)
        workspaces.append(workspace_id)
        mongo_users.upsert({'id': user_id}, {'id': user_id, 'workspaces': workspaces})

        ret_Code = Code(201, metadata, "Workspace created")
        return ret_Code

    @staticmethod
    def update_workspace(org_name, workspace_id, request_dict):
        """
        org_name: str
        workspace_id: str, uuid
        request_dict: dict following WorkspaceRspSchema
            - type is required
            - format is required
        Returns:
        Code object
        - 200 with metadata of updated workspace if successful
        - 404 if workspace not found / user cannot access
        - 400 if invalid update
        """
        metadata = resolve_metadata("workspace", workspace_id)
        if not metadata:
            return Code(404, {}, "Workspace not found")

        if not check_write_access(org_name, workspace_id, kind="workspaces"):
            return Code(404, {}, "Workspace not available")

        update_keys = request_dict.keys()
        for key in ["name", "version", "cloud_type", "shared"]:
            if key in update_keys:
                requested_value = request_dict[key]
                if requested_value is not None:
                    metadata[key] = requested_value
                    metadata["last_modified"] = datetime.now(tz=timezone.utc)

        encrypted_metadata = copy.deepcopy(metadata)
        if "cloud_specific_details" in request_dict.keys():
            # Encrypt Cloud details
            for key, value in request_dict["cloud_specific_details"]:
                if key == "cloud_type":
                    encrypted_metadata["cloud_type"] = value
                if key == "cloud_specific_details":
                    if BACKEND in ("BCP", "NVCF"):
                        config_path = os.getenv("VAULT_SECRET_PATH", None)
                        encryption = NVVaultEncryption(config_path)
                    elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                        return Code(400, {}, "Vault service does not work, can't save cloud workspace")
                    for cloud_key, cloud_value in request_dict["cloud_specific_details"]:
                        encrypted_metadata["cloud_specific_details"][cloud_key] = cloud_value
                        if BACKEND in ("BCP", "NVCF"):
                            if encryption.check_config()[0]:
                                encrypted_metadata["cloud_specific_details"][cloud_key] = encryption.encrypt(cloud_value)

        if encrypted_metadata["cloud_type"] in ("aws", "azure"):
            try:
                if "cloud_type" in request_dict.keys() or "cloud_specific_details" in request_dict.keys():
                    if encrypted_metadata["cloud_type"] in ("aws", "azure"):
                        create_cs_instance(encrypted_metadata)
            except:
                return Code(400, {}, "Provided cloud credentials are invalid")

        write_handler_metadata(workspace_id, encrypted_metadata, "workspace")
        ret_Code = Code(200, metadata, "Workspace updated")
        return ret_Code

    @staticmethod
    def delete_workspace(org_name, workspace_id):
        """
        org_name: str
        workspace_id: str, uuid
        Returns:
        Code object
        - 200 if successful
        - 404 if user cannot access
        """
        handler_metadata = resolve_metadata("workspace", workspace_id)
        if not handler_metadata:
            return Code(200, {}, "Workspace deleted")
        user_id = handler_metadata.get("user_id")

        if workspace_id not in get_user_workspaces(user_id):
            return Code(404, {}, "Workspace cannot be deleted as it's doesn't belong to user")

        # If workspace is being used by user's experiments or datasets.
        experiments = get_user_experiments(user_id)

        for experiment_id in experiments:
            experiment_metadata = get_experiment(experiment_id)
            experiment_workspace = experiment_metadata.get("workspace", "")
            if experiment_workspace and workspace_id in experiment_workspace:
                return Code(400, {}, f"Experiment {experiment_metadata['id']} ({experiment_metadata['id']}) in use; Delete experiment first")

            train_datasets = experiment_metadata.get("train_datasets", [])
            if not isinstance(train_datasets, list):
                train_datasets = [train_datasets]
            for dataset_id in train_datasets:
                dataset_metadata = get_dataset(dataset_id)
                dataset_workspace = dataset_metadata.get("workspace", "")
                if workspace_id == dataset_workspace:
                    return Code(400, {}, f"Dataset {dataset_metadata['id']} ({dataset_metadata['id']}) in use; Delete dataset first")

            for key in ["eval_dataset", "inference_dataset", "calibration_dataset"]:
                additional_dataset_id = experiment_metadata.get(key)
                if additional_dataset_id:
                    dataset_metadata = get_dataset(additional_dataset_id)
                    dataset_workspace = dataset_metadata.get("workspace", "")
                    if workspace_id == dataset_workspace:
                        return Code(400, {}, f"Dataset {dataset_metadata['id']} ({dataset_metadata['id']}) in use; Delete dataset first")

        mongo_users = MongoHandler("tao", "users")
        user = stateless_handlers.get_user(user_id, mongo_users)
        workspaces = user.get("workspaces", [])
        if workspace_id in workspaces:
            workspaces.remove(workspace_id)
            mongo_users.upsert({'id': user_id}, {'id': user_id, 'workspaces': workspaces})
        mongo_workspaces = MongoHandler("tao", "workspaces")
        mongo_workspaces.delete_one({'id': workspace_id})
        return Code(200, {"message": "Workspace deleted"}, "")

    # Dataset API
    @staticmethod
    def list_datasets(user_id, org_name):
        """
        user_id: str, uuid
        org_name: str
        Returns:
        list(dict) - list of datasets accessible by user where each element is metadata of a dataset
        """
        # Collect all metadatas
        metadatas = []
        for dataset_id in list(set(get_org_datasets(org_name) + stateless_handlers.get_public_datasets())):
            handler_metadata = stateless_handlers.get_handler_metadata(dataset_id, 'datasets')
            shared_dataset = handler_metadata.get("shared", False)
            if handler_metadata:
                if shared_dataset or handler_metadata.get("user_id") == user_id:
                    handler_metadata = sanitize_handler_metadata(handler_metadata)
                    metadatas.append(handler_metadata)
            else:
                # Something is wrong. The user metadata has a dataset that doesn't exist in the system.
                contexts = {"user_id": user_id, "org_name": org_name, "handler_id": dataset_id}
                printc("Dataset not found. Skipping.", contexts, file=sys.stderr)
        return metadatas

    @staticmethod
    def get_dataset_formats(dataset_type):
        """
        dataset_type: str
        Returns:
        list(str) - list of dataset_formats for the given dataset_type
        """
        try:
            dataset_formats = []
            accepted_dataset_intents = []
            api_params = read_network_config(dataset_type).get("api_params", {})
            if api_params:
                if api_params.get("formats", []):
                    dataset_formats += api_params.get("formats", [])
                if api_params.get("accepted_ds_intents", []):
                    accepted_dataset_intents += api_params.get("accepted_ds_intents", [])
            return Code(200, {"dataset_formats": dataset_formats, "accepted_dataset_intents": accepted_dataset_intents}, "")
        except Exception:
            print(traceback.format_exc(), file=sys.stderr)
            return Code(404, [], "Exception caught during getting dataset formats")

    # Create dataset
    @staticmethod
    def create_dataset(user_id, org_name, request_dict, dataset_id=None, from_ui=False):
        """
        org_name: str
        request_dict: dict following DatasetReqSchema
            - type is required
            - format is required
        from_ui: whether a dataset creation call is from UI
        Returns:
        - 201 with metadata of created dataset if successful
        - 400 if dataset type and format not given
        """
        # Gather type,format fields from request
        ds_type = request_dict.get("type", None)
        if ds_type == "ocrnet":
            intention = request_dict.get("use_for", [])
            if not (intention in (["training"], ["evaluation"])):
                return Code(400, {}, "Use_for in dataset metadata is not set ['training'] or ['evaluation']. Please set use_for appropriately")

        ds_format = request_dict.get("format", None)
        # Perform basic checks - valid type and format?
        if ds_type not in VALID_DSTYPES:
            msg = "Invalid dataset type"
            return Code(400, {}, msg)

        # For monai dataset, don't check the dataset type.
        if ds_format not in read_network_config(ds_type)["api_params"]["formats"] and ds_format != "monai":
            msg = "Incompatible dataset format and type"
            return Code(400, {}, msg)

        intention = request_dict.get("use_for", [])
        if ds_format in ("raw", "coco_raw") and intention:
            if intention != ["testing"]:
                msg = "raw or coco_raw's format should be associated with ['testing'] intent"
                return Code(400, {}, msg)

        # Create a dataset ID and its root
        pull = False
        if not dataset_id:
            pull = True
            dataset_id = str(uuid.uuid4())

        if request_dict.get("public", False):
            stateless_handlers.add_public_dataset(dataset_id)

        dataset_actions = get_dataset_actions(ds_type, ds_format) if ds_format != "monai" else MONAI_DATASET_ACTIONS

        # Create metadata dict and create some initial folders
        metadata = {"id": dataset_id,
                    "user_id": user_id,
                    "org_name": org_name,
                    "created_on": datetime.now(tz=timezone.utc),
                    "last_modified": datetime.now(tz=timezone.utc),
                    "name": request_dict.get("name", "My Dataset"),
                    "shared": request_dict.get("shared", False),
                    "description": request_dict.get("description", "My TAO Dataset"),
                    "version": request_dict.get("version", "1.0.0"),
                    "docker_env_vars": request_dict.get("docker_env_vars", {}),
                    "logo": request_dict.get("logo", "https://www.nvidia.com"),
                    "type": ds_type,
                    "format": ds_format,
                    "actions": dataset_actions,
                    "client_url": request_dict.get("client_url", None),
                    "client_id": request_dict.get("client_id", None),
                    "client_secret": request_dict.get("client_secret", None),  # TODO:: Store Secrets in Vault
                    "filters": request_dict.get("filters", None),
                    "status": request_dict.get("status", "starting") if ds_format != "monai" else "pull_complete",
                    "cloud_file_path": request_dict.get("cloud_file_path"),
                    "url": request_dict.get("url"),
                    "workspace": request_dict.get("workspace"),
                    "use_for": intention
                    }

        if metadata.get("url", ""):
            if not metadata.get("url").startswith("https"):
                return Code(400, {}, "Invalid pull URL passed")

        # Encrypt the MLOPs keys
        if BACKEND in ("BCP", "NVCF") and metadata["docker_env_vars"]:
            config_path = os.getenv("VAULT_SECRET_PATH", None)
            encryption = NVVaultEncryption(config_path)
            for key, value in metadata["docker_env_vars"].items():
                if encryption.check_config()[0]:
                    metadata["docker_env_vars"][key] = encryption.encrypt(value)
                elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                    return Code(400, {}, "Vault service does not work, can't enable MLOPs services")

        # Encrypt the client secret if the dataset is a monai dataset and the vault agent has been set.
        if metadata["client_secret"] and ds_format == "monai":
            config_path = os.getenv("VAULT_SECRET_PATH", None)
            encryption = NVVaultEncryption(config_path)
            if encryption.check_config()[0]:
                metadata["client_secret"] = encryption.encrypt(metadata["client_secret"])
            elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                return Code(400, {}, "Cannot create dataset because vault service does not work.")

        # For MONAI dataset only
        if ds_format == "monai":
            client_url = request_dict.get("client_url", None)
            log_content = f"user_id:{user_id}, org_name:{org_name}, from_ui:{from_ui}, dataset_url:{client_url}, action:creation"
            log_monitor(log_type=DataMonitorLogTypeEnum.medical_dataset, log_content=log_content)
            if client_url is None:
                msg = "Must provide a url to create a MONAI dataset."
                return Code(400, {}, msg)

            status, m = MonaiDatasetHandler.status_check(metadata)
            if not status or m:
                return Code(400, {}, m)

        stateless_handlers.make_root_dirs(user_id, org_name, "datasets", dataset_id)
        write_handler_metadata(dataset_id, metadata, "dataset")
        mongo_users = MongoHandler("tao", "users")
        user_query = {'id': user_id}
        datasets = get_user_datasets(user_id, mongo_users)
        datasets.append(dataset_id)
        mongo_users.upsert(user_query, {'id': user_id, 'datasets': datasets})

        # Pull dataset in background if known URL
        if pull:
            job_run_thread = threading.Thread(target=AppHandler.pull_dataset, args=(org_name, dataset_id,))
            job_run_thread.start()

        # Read this metadata from saved file...
        return_metadata = resolve_metadata_with_jobs(user_id, org_name, "dataset", dataset_id)
        return_metadata = sanitize_handler_metadata(return_metadata)
        ret_Code = Code(201, return_metadata, "Dataset created")
        return ret_Code

    @staticmethod
    def create_dataset_dict_from_experiment_metadata(org_name, dataset_id, action, handler_metadata, specs):
        """Create a request dict for output dataset creation from input dataset's metadata"""
        infer_ds = handler_metadata.get("inference_dataset", None)
        if infer_ds:
            dataset_metadata = stateless_handlers.get_handler_metadata(infer_ds, "datasets")
        else:
            dataset_metadata = copy.deepcopy(handler_metadata)
        request_dict = {}
        output_dataset_type = dataset_metadata.get("type")
        output_dataset_format = dataset_metadata.get("format")
        use_for = dataset_metadata.get("use_for")
        request_dict["type"] = output_dataset_type
        request_dict["status"] = dataset_metadata.get("status", "pull_complete")
        request_dict["format"] = output_dataset_format
        request_dict["use_for"] = use_for
        request_dict["workspace"] = dataset_metadata.get("workspace")
        request_dict["cloud_file_path"] = os.path.join("/results/", dataset_id)
        request_dict["name"] = f"{dataset_metadata.get('name')} (created from Data services {action} action)"
        request_dict["shared"] = dataset_metadata.get("shared", False)
        request_dict["use_for"] = dataset_metadata.get("use_for", [])
        request_dict["docker_env_vars"] = dataset_metadata.get("docker_env_vars", {})
        return request_dict

    # Update existing dataset for user based on request dict
    @staticmethod
    def update_dataset(org_name, dataset_id, request_dict):
        """
        org_name: str
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
        metadata = resolve_metadata("dataset", dataset_id)
        if not metadata:
            return Code(404, {}, "Dataset not found")

        if not check_write_access(org_name, dataset_id, kind="datasets"):
            return Code(404, {}, "Dataset not available")
        if request_dict.get("public", None):
            if request_dict["public"]:
                stateless_handlers.add_public_dataset(dataset_id)
            else:
                stateless_handlers.remove_public_dataset(dataset_id)
        user_id = metadata.get("user_id")
        pull = False
        for key in request_dict.keys():

            # Cannot process the update, so return 400
            if key in ["type", "format"]:
                if request_dict[key] != metadata.get(key):
                    msg = f"Cannot change dataset {key}"
                    return Code(400, {}, msg)

            if key in ["name", "description", "version", "logo", "shared"]:
                requested_value = request_dict[key]
                if requested_value is not None:
                    metadata[key] = requested_value
                    metadata["last_modified"] = datetime.now(tz=timezone.utc)

            if key == "cloud_file_path":
                if metadata["status"] not in ("pull_complete", "invalid_pull"):
                    return Code(400, {}, f"Cloud file_path can be updated only when status is pull_complete or invalid_pull, the current status is {metadata['status']}. Try again after sometime")
                pull = True
                metadata["status"] = "starting"
                metadata["cloud_file_path"] = request_dict[key]

            if key == "docker_env_vars":
                # Encrypt the MLOPs keys
                requested_value = request_dict[key]
                if BACKEND in ("BCP", "NVCF"):
                    config_path = os.getenv("VAULT_SECRET_PATH", None)
                    encryption = NVVaultEncryption(config_path)
                    for mlops_key, value in requested_value.items():
                        if encryption.check_config()[0]:
                            metadata["docker_env_vars"][mlops_key] = encryption.encrypt(value)
                        elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                            return Code(400, {}, "Vault service does not work, can't enable MLOPs services")
                else:
                    metadata["docker_env_vars"] = requested_value

        # Pull dataset in background if known URL
        if pull:
            job_run_thread = threading.Thread(target=AppHandler.pull_dataset, args=(org_name, dataset_id,))
            job_run_thread.start()

        write_handler_metadata(dataset_id, metadata, "dataset")
        # Read this metadata from saved file...
        return_metadata = resolve_metadata_with_jobs(user_id, org_name, "dataset", dataset_id)
        return_metadata = sanitize_handler_metadata(return_metadata)
        ret_Code = Code(200, return_metadata, "Dataset updated")
        return ret_Code

    # Retrieve existing dataset for user based on request dict
    @staticmethod
    def retrieve_dataset(org_name, dataset_id):
        """
        org_name: str
        dataset_id: str, uuid
        Returns:
        Code object
        - 200 with metadata of retrieved dataset if successful
        - 404 if dataset not found / user cannot access
        """
        handler_metadata = resolve_metadata("dataset", dataset_id)
        if not handler_metadata:
            return Code(404, {}, "Dataset not found")

        if not check_read_access(org_name, dataset_id, kind="datasets"):
            return Code(404, {}, "Dataset not found")

        user_id = handler_metadata.get("user_id")
        return_metadata = resolve_metadata_with_jobs(user_id, org_name, "dataset", dataset_id)
        return_metadata = sanitize_handler_metadata(return_metadata)
        if return_metadata.get("status") == "invalid_pull":
            return Code(404, return_metadata, "Dataset pulled from cloud doesn't match folder structure required")
        return Code(200, return_metadata, "Dataset retrieved")

    # Delete a user's dataset
    @staticmethod
    def delete_dataset(org_name, dataset_id):
        """
        org_name: str
        dataset_id: str, uuid
        Returns:
        Code object
        - 200 with metadata of deleted dataset if successful
        - 404 if user cannot access
        - 400 if dataset has running jobs / being used by a experiment and hence cannot be deleted
        """
        handler_metadata = resolve_metadata("dataset", dataset_id)
        if not handler_metadata:
            return Code(200, {}, "Dataset deleted")

        if not check_write_access(org_name, dataset_id, kind="datasets"):
            return Code(404, {}, "Dataset not available")

        user_id = handler_metadata.get("user_id")

        # If dataset is being used by user's experiments.
        experiments = get_user_experiments(user_id)
        for experiment_id in experiments:
            metadata = stateless_handlers.get_handler_metadata(experiment_id, "experiment")
            datasets_in_use = set(metadata.get("train_datasets", []))
            for key in ["eval_dataset", "inference_dataset", "calibration_dataset"]:
                additional_dataset_id = metadata.get(key)
                if additional_dataset_id:
                    datasets_in_use.add(additional_dataset_id)
            if dataset_id in datasets_in_use:
                return Code(400, {}, "Dataset in use")

        # Check if any job running
        return_metadata = resolve_metadata_with_jobs(user_id, org_name, "dataset", dataset_id)
        for job in return_metadata["jobs"]:
            if return_metadata["jobs"][job]["status"] == "Running":
                return Code(400, {}, "Dataset in use")

        # Check if dataset is public, then someone could be running it
        if return_metadata.get("public", False):
            return Code(400, {}, "Dataset is Public. Cannot delete")

        # Check if dataset is read only, if yes, cannot delete
        if return_metadata.get("read_only", False):
            return Code(400, {}, "Dataset is read only. Cannot delete")

        mongo_users = MongoHandler("tao", "users")
        datasets = get_user_datasets(user_id, mongo_users)
        if dataset_id in datasets:
            datasets.remove(dataset_id)
        user_query = {'id': user_id}
        mongo_users.upsert(user_query, {'id': user_id, 'datasets': datasets})

        mongo_datasets = MongoHandler("tao", "datasets")
        dataset_query = {'id': dataset_id}
        mongo_datasets.delete_one(dataset_query)
        delete_jobs_for_handler(dataset_id, "dataset")
        # TODO: Delete logs for dataset
        return_metadata = sanitize_handler_metadata(return_metadata)
        return Code(200, return_metadata, "Dataset deleted")

    @staticmethod
    def validate_dataset(org_name, dataset_id, temp_dir=None, file_path=None):
        """
        org_name: str
        dataset_id: str, uuid
        temp_dir: str, path of temporary directory
        file_path: str, path of file/folder
        Returns:
        Code object
        - 201 with {} if successful
        - 404 if dataset not found / user cannot access
        - 400 if upload validation fails

        """
        metadata = resolve_metadata("dataset", dataset_id)
        if not metadata:
            return Code(404, {}, "Dataset not found")

        if not check_write_access(org_name, dataset_id, kind="datasets"):
            return Code(404, {}, "Dataset not available")

        if metadata.get("format") == "monai":
            return Code(404, metadata, "Uploading external data is not supported for MONAI Dataset")

        try:
            metadata["status"] = "in_progress"
            write_handler_metadata(dataset_id, metadata, "dataset")

            def validate_dataset_thread():
                try:
                    valid_datset_structure = DS_UPLOAD_TO_FUNCTIONS[metadata.get("type")](org_name, metadata, temp_dir=temp_dir)
                    shutil.rmtree(temp_dir)
                    metadata["status"] = "pull_complete"
                    if not valid_datset_structure:
                        print("Dataset structure validation failed", metadata, file=sys.stderr)
                        metadata["status"] = "invalid_pull"
                    write_handler_metadata(dataset_id, metadata, "dataset")
                except:
                    metadata["status"] = "invalid_pull"
                    write_handler_metadata(dataset_id, metadata, "dataset")
                    print(traceback.format_exc(), file=sys.stderr)

            thread = threading.Thread(target=validate_dataset_thread)
            thread.start()
            return Code(201, {}, "Server recieved file and upload process started")
        except:
            metadata["status"] = "invalid_pull"
            write_handler_metadata(dataset_id, metadata, "dataset")
            print(traceback.format_exc(), file=sys.stderr)
            return Code(404, [], "Exception caught during upload")

    @staticmethod
    def pull_dataset(org_name, dataset_id):
        """
        org_name: str
        dataset_id: str, uuid
        """
        try:
            temp_dir, file_path = download_dataset(dataset_id)
            AppHandler.validate_dataset(org_name, dataset_id, temp_dir=temp_dir, file_path=file_path)
        except:
            metadata = resolve_metadata("dataset", dataset_id)
            metadata["status"] = "invalid_pull"
            write_handler_metadata(dataset_id, metadata, "dataset")
            print(traceback.format_exc(), file=sys.stderr)

    # Spec API

    @staticmethod
    def get_spec_schema(org_name, handler_id, action, kind):
        """
        org_name: str
        handler_id: str, uuid corresponding to experiment/dataset
        action: str, a valid Action for a dataset
        kind: str, one of ["experiment","dataset"]
        Returns:
        Code object
        - 200 with spec in a json-schema format
        - 404 if experiment/dataset not found / user cannot access
        """
        metadata = resolve_metadata(kind, handler_id)
        if not metadata:
            return Code(404, {}, "Spec schema not found")

        if not check_read_access(org_name, handler_id, kind=kind + "s"):
            return Code(404, {}, "Spec schema not available")

        # Action not available
        if action not in metadata.get("actions", []):
            return Code(404, {}, "Action not found")

        base_experiment_spec = {}
        if metadata.get("base_experiment", []):
            base_experiment_metadatas = stateless_handlers.read_base_experiment_metadata()
            for base_experiment_id in metadata["base_experiment"]:
                base_experiment_metadata = base_experiment_metadatas.get(base_experiment_id)
                if base_experiment_metadata and base_experiment_metadata.get("base_experiment_metadata", {}).get("spec_file_present"):
                    base_experiment_root = stateless_handlers.get_handler_root(base_exp_uuid, "experiments", base_exp_uuid, base_experiment_id)
                    base_experiment_spec_path = search_for_base_experiment(base_experiment_root,
                                                                           network=metadata.get("network_arch", ""),
                                                                           spec=True)
                    if not os.path.exists(base_experiment_spec_path):
                        return Code(404, {}, "Base spec file not present.")
                    base_experiment_spec = safe_load_file(base_experiment_spec_path, file_type="yaml")

        # Read csv from spec_utils/specs/<network_name>/action.csv
        # Convert to json schema
        json_schema = {}

        if kind == "dataset" and metadata.get("format") == "monai":
            json_schema = MonaiDatasetHandler.get_schema(action)
            return Code(200, json_schema, "Schema retrieved")

        if kind == "experiment" and metadata.get("type").lower() == "medical":
            json_schema = MonaiModelHandler.get_schema(action)
            return Code(200, json_schema, "Schema retrieved")

        network = metadata.get("network_arch", None)
        if not network:
            # Used for dataset jobs
            network = metadata.get("type", None)
        microservices_network = network
        microservices_action = action
        if network == "object_detection":
            if action == "annotation_format_convert":
                microservices_network = "annotations"
                microservices_action = "convert"
            if action == "augment":
                microservices_network = "augmentation"
                microservices_action = "generate"
            if action in ("analyze", "validate"):
                microservices_network = "analytics"

        network_config = read_network_config(network)
        if BACKEND in ("local-k8s", "BCP"):
            if microservices_network in TAO_NETWORKS:
                response = jobDriver.create_microservice_and_send_request(api_endpoint="get_schema", network=microservices_network, action=microservices_action, num_gpu=0, org_name=org_name, handler_id=handler_id, handler_kind=kind + "s")
                if response and response.ok:
                    json_schema = response.json()
                    return Code(200, json_schema, "Schema retrieved")
                return Code(404, {}, "Unable to fetch schema from microservices")

        DIR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
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
        if network_config["api_params"]["classwise"] == "True":
            # For each train dataset for the experiment
            metadata = resolve_metadata(kind, handler_id)
            if network == "detectnet_v2" and action == "train":
                if not metadata.get("train_datasets", []):
                    return Code(400, [], "Assign datasets before getting train specs")
            for train_ds in metadata.get("train_datasets", []):
                # Obtain class list from classes.json
                classes_json = os.path.join(stateless_handlers.get_handler_root(org_name, "datasets", train_ds, None), "classes.json")
                if not os.path.exists(classes_json):
                    if network == "detectnet_v2" and action == "train":
                        return Code(400, [], "Classes.json is not part of the dataset. Either provide them or wait for dataset convert to complete before getting specs")
                    continue
                with open(classes_json, "r", encoding='utf-8') as f:
                    inferred_class_names += json.loads(f.read())  # It is a list
            inferred_class_names = list(set(inferred_class_names))
        json_schema = csv_to_json_schema.convert(CSV_PATH, inferred_class_names)

        # elif BACKEND == "NVCF":
        #     json_schema = {}
        #     deployment_string = os.getenv(f'FUNCTION_{NETWORK_CONTAINER_MAPPING[microservices_network]}')
        #     if action == "gen_trt_engine":
        #         deployment_string = os.getenv('FUNCTION_TAO_DEPLOY')
        #     nvcf_response = nvcf_handler.invoke_function(deployment_string=deployment_string, network=microservices_network, action=microservices_action, microservice_action="get_schema")
        #     if nvcf_response.status_code != 200:
        #         if nvcf_response.status_code == 202:
        #             return Code(404, {}, "Schema from NVCF couldn't be obtained in 60 seconds, Retry again")
        #         return Code(nvcf_response.status_code, {}, str(nvcf_response.json()))
        #     json_schema = nvcf_response.json().get("response")

        if "default" in json_schema and base_experiment_spec:
            json_schema["default"] = merge_nested_dicts(json_schema["default"], base_experiment_spec)
        return Code(200, json_schema, "Schema retrieved")

    @staticmethod
    def get_base_experiment_spec_schema(experiment_id, action):
        """
        experiment_id: str, uuid corresponding to the base experiment
        action: str, a valid Action for a dataset
        Returns:
        Code object
        - 200 with spec in a json-schema format
        - 404 if experiment/action not found / user cannot access
        """
        base_experiment_spec = {}
        base_experiment_metadatas = stateless_handlers.read_base_experiment_metadata()
        base_experiment_metadata = base_experiment_metadatas.get(experiment_id)
        if action not in base_experiment_metadata.get("actions", []):
            return Code(404, {}, "Action not found")

        network = base_experiment_metadata.get("network_arch")
        if BACKEND in ("local-k8s", "BCP"):
            if network in TAO_NETWORKS:
                response = jobDriver.create_microservice_and_send_request(api_endpoint="get_schema", network=network, action=action, num_gpu=0)
                if response and response.ok:
                    json_schema = response.json()
                    return Code(200, json_schema, "Schema retrieved")
                return Code(404, {}, "Unable to fetch schema from microservices")

        if base_experiment_metadata and base_experiment_metadata.get("base_experiment_metadata", {}).get("spec_file_present"):
            base_experiment_root = stateless_handlers.get_handler_root(base_exp_uuid, "experiments", base_exp_uuid, experiment_id)
            base_experiment_spec_path = search_for_base_experiment(base_experiment_root,
                                                                   network=base_experiment_metadata.get("network_arch", ""),
                                                                   spec=True)
            if not os.path.exists(base_experiment_spec_path):
                return Code(404, {}, "Base spec file not present.")
            base_experiment_spec = safe_load_file(base_experiment_spec_path, file_type="yaml")

        # Read csv from spec_utils/specs/<network_name>/action.csv
        # Convert to json schema
        json_schema = {}
        DIR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        network = base_experiment_metadata.get("network_arch", None)
        # Try regular format for CSV_PATH => "<network> - <action>.csv"
        CSV_PATH = os.path.join(DIR_PATH, "specs_utils", "specs", network, f"{network} - {action}.csv")
        if not os.path.exists(CSV_PATH):
            Code(404, {}, "Default specs do not exist for action")

        json_schema = csv_to_json_schema.convert(CSV_PATH)
        if "default" in json_schema and base_experiment_spec:
            json_schema["default"] = merge_nested_dicts(json_schema["default"], base_experiment_spec)
        return Code(200, json_schema, "Schema retrieved")

    @staticmethod
    def get_spec_schema_without_handler_id(org_name, network, format, action, train_datasets):
        """
        network: str, valid network architecture name supported
        format: str, valid format of the architecture if necessary
        action: str, a valid Action for a dataset
        train_datasets: list, list of uuid's of train dataset id's
        Returns:
        Code object
        - 200 with spec in a json-schema format
        - 404 if experiment/dataset not found / user cannot access
        """
        # Action not available
        if not network:
            return Code(404, {}, "Pass network name to the request")
        if not action:
            return Code(404, {}, "Pass action name to the request")

        # Read csv from spec_utils/specs/<network_name>/action.csv
        # Convert to json schema
        json_schema = {}
        DIR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        # Try regular format for CSV_PATH => "<network> - <action>.csv"
        CSV_PATH = os.path.join(DIR_PATH, "specs_utils", "specs", network, f"{network} - {action}.csv")
        if not os.path.exists(CSV_PATH):
            # Try secondary format for CSV_PATH => "<network> - <action>__<dataset-format>.csv"
            CSV_PATH = os.path.join(DIR_PATH, "specs_utils", "specs", network, f"{network} - {action}__{format}.csv")
            if not os.path.exists(CSV_PATH):
                Code(404, {}, "Default specs do not exist for action")

        inferred_class_names = []
        # If class-wise config is applicable
        if read_network_config(network)["api_params"]["classwise"] == "True":
            # For each train dataset for the experiment
            for train_ds in train_datasets:
                # Obtain class list from classes.json
                classes_json = os.path.join(stateless_handlers.get_handler_root(org_name, "datasets", train_ds, None), "classes.json")
                if not os.path.exists(classes_json):
                    continue
                with open(classes_json, "r", encoding='utf-8') as f:
                    inferred_class_names += json.loads(f.read())  # It is a list
            inferred_class_names = list(set(inferred_class_names))
        json_schema = csv_to_json_schema.convert(CSV_PATH, inferred_class_names)
        return Code(200, json_schema, "Schema retrieved")

    # Job API

    @staticmethod
    def job_run(org_name, handler_id, parent_job_id, action, kind, specs=None, name=None, description=None, num_gpu=-1, platform=None, from_ui=False):
        """
        org_name: str
        handler_id: str, uuid corresponding to experiment/dataset
        parent_job_id: str, uuid
        action: str
        kind: str, one of ["experiment","dataset"]
        specs: dict spec for each action.
        name: str
        description: str
        from_ui: whether a job call is from UI
        Returns:
        201 with str where str is a uuid for job if job is successfully queued
        400 with [] if action was not executed successfully
        404 with [] if dataset/experiment/parent_job_id/actions not found/standards are not met
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{handler_id} {kind} doesn't exist")

        if not check_write_access(org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{org_name} has no write access to {handler_id}")

        if not action:
            return Code(404, [], "action not sent")

        user_id = handler_metadata.get("user_id")
        if not user_id:
            return Code(404, [], "User ID couldn't be found in the experiment metadata. Try creating the experiment again")

        if kind == "experiment" and handler_metadata.get("type").lower() == "medical":
            if action not in handler_metadata.get("actions", []):
                return Code(404, {}, "Action not found")

            if not isinstance(specs, dict):
                return Code(404, [], f"{specs} must be a dictionary. Received {type(specs)}")

            default_spec = read_network_config(handler_metadata["network_arch"])["spec_params"].get(action, {})
            nested_update(specs, default_spec, allow_overwrite=False)
            network_arch = handler_metadata["network_arch"]
            if "num_gpus" in specs:
                return Code(400, [], "num_gpus is not a valid key in the specs. Use num_gpu instead.")
            num_gpu, err_msg = validate_num_gpu(specs.get("num_gpu", None), action)
            if num_gpu <= 0 and err_msg:
                return Code(400, [], err_msg)
            log_content = f"user_id:{user_id}, org_name:{org_name}, from_ui:{from_ui}, job_type:experiment, network_arch:{network_arch}, action:{action}, num_gpu:{num_gpu}"
            log_monitor(log_type=DataMonitorLogTypeEnum.medical_job, log_content=log_content)
            if action == "inference":
                return MonaiModelHandler.run_inference(org_name, handler_id, handler_metadata, specs)

            # regular async jobs
            if action == "annotation":
                all_metadata = stateless_handlers.get_jobs_for_handler(handler_id, kind)
                if [m for m in all_metadata if m["action"] == "annotation" and m["status"] in ("Running", "Pending")]:
                    return Code(400, [], "There is one running/pending annotation job. Please stop it first.")
                if handler_metadata.get("eval_dataset", None) is None:
                    return Code(404, {}, "Annotation job requires eval dataset in the model metadata.")

        if kind == "dataset" and handler_metadata.get("format") == "monai":
            log_content = f"user_id:{user_id}, org_name:{org_name}, from_ui:{from_ui}, job_type:dataset, action:{action}"
            log_monitor(log_type=DataMonitorLogTypeEnum.medical_job, log_content=log_content)
            return MonaiDatasetHandler.run_job(org_name, handler_id, handler_metadata, action, specs)

        if "base_experiment" in handler_metadata.keys():
            base_experiment_ids = handler_metadata["base_experiment"]
            for base_experiment_id in base_experiment_ids:
                if base_experiment_id:
                    if stateless_handlers.get_base_experiment_metadata(base_experiment_id).get("ngc_path", None):
                        print(f"Starting download of {base_experiment_id}", file=sys.stderr)
                        if handler_metadata["network_arch"] in MONAI_NETWORKS:
                            ptm_download_thread = threading.Thread(target=download_base_experiment, args=(user_id, base_experiment_id,))
                            ptm_download_thread.start()

        try:
            job_id = str(uuid.uuid4())
            if action in _DATA_GENERATE_ACTIONS:  # These actions create a new dataset as part of their actions
                request_dict = AppHandler.create_dataset_dict_from_experiment_metadata(org_name, job_id, action, handler_metadata, specs)
                response = AppHandler.create_dataset(user_id, org_name, request_dict, dataset_id=job_id)
                if response.code not in (200, 201):
                    return response
            create_folder_with_permissions(f"/shared/orgs/{org_name}/users/{user_id}/jobs/{job_id}")
            if specs:
                spec_schema_response = AppHandler.get_spec_schema(org_name, handler_id, action, kind)
                if spec_schema_response.code == 200:
                    spec_schema = spec_schema_response.data
                    default_spec = spec_schema["default"]
                    check_and_convert(specs, default_spec)
                spec_json_path = os.path.join(get_handler_spec_root(user_id, org_name, handler_id), f"{job_id}-{action}-spec.json")
                safe_dump_file(spec_json_path, specs)
            msg = ""
            if is_request_automl(handler_id, action, kind):
                AutoMLHandler.start(user_id, org_name, handler_id, job_id, handler_metadata, name=name)
                msg = "AutoML "
            else:
                job_context = create_job_context(parent_job_id, action, job_id, handler_id, user_id, org_name, kind, handler_metadata=handler_metadata, specs=specs, name=name, description=description, num_gpu=num_gpu, platform=platform)
                on_new_job(job_context)
            return Code(201, job_id, f"{msg}Job scheduled")
        except:
            print(traceback.format_exc(), file=sys.stderr)
            return Code(404, [], "Exception in job_run fn")

    @staticmethod
    def job_get_epoch_numbers(org_name, handler_id, job_id, kind):
        """
        org_name: str
        handler_id: str, uuid corresponding to experiment/dataset
        Returns:
        200, list(str) - list of epoch numbers as strings
        404, [] if not found
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        if not check_write_access(org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        handler_metadata = stateless_handlers.get_handler_metadata(handler_id, kind + "s")
        if not handler_metadata or "user_id" not in handler_metadata:
            return Code(404, [], "job trying to update not found")

        job = get_job(job_id)
        if not job:
            return Code(404, [], "job trying to update not found")
        try:
            job_files, _, _, _ = get_files_from_cloud(handler_metadata, job_id)
            epoch_numbers = []
            for job_file in job_files:
                # Extract numbers before the extension using regex
                match = re.search(r'(\d+)(?=\.(pth|hdf5|tlt)$)', job_file)
                if match:
                    epoch_numbers.append(match.group(1))
            return Code(200, {"data": epoch_numbers}, "Job status updated")
        except Exception:
            print(traceback.format_exc(), file=sys.stderr)
            return Code(404, [], "Exception caught during getting epoch numbers")

    @staticmethod
    def job_status_update(org_name, handler_id, job_id, kind, callback_data):
        """
        org_name: str
        handler_id: str, uuid corresponding to experiment/dataset
        kind: str, one of ["experiment","dataset"]
        Returns:
        200, list(dict) - each dict follows JobResultSchema if found
        404, [] if not found
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        if not check_write_access(org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id", None)
        job = get_job(job_id)
        if not job or not user_id:
            return Code(404, [], "job trying to update not found")

        job_root = os.path.join(stateless_handlers.get_jobs_root(user_id, org_name), job_id)
        if is_job_automl(user_id, org_name, job_id):
            experiment_number = callback_data.get("experiment_number", "0")
            job_root = f"{job_root}/experiment_{experiment_number}/"
        if os.path.exists(job_root):
            status_json_file = os.path.join(job_root, "status.json")
            with open(status_json_file, "a", encoding='utf-8') as file_ptr:
                file_ptr.write(callback_data["status"])
                file_ptr.write("\n")
            os.chmod(status_json_file, 0o666)
        return Code(200, [], "Job status updated")

    @staticmethod
    def job_log_update(org_name, handler_id, job_id, kind, callback_data):
        """
        org_name: str
        handler_id: str, uuid corresponding to experiment/dataset
        kind: str, one of ["experiment","dataset"]
        Returns:
        200, list(dict) - each dict follows JobResultSchema if found
        404, [] if not found
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        if not check_write_access(org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id", None)
        job = get_job(job_id)
        if not job or not user_id:
            return Code(404, [], "job trying to update not found")

        log_file = os.path.join(stateless_handlers.get_handler_log_root(user_id, org_name, handler_id), job_id + ".txt")
        if is_job_automl(user_id, org_name, job_id):
            job_root = os.path.join(stateless_handlers.get_jobs_root(user_id, org_name), job_id)
            experiment_number = callback_data.get("experiment_number", "0")
            log_file = f"{job_root}/experiment_{experiment_number}/log.txt"
        with open(log_file, "a", encoding='utf-8') as file_ptr:
            file_ptr.write(callback_data["log_contents"])
            file_ptr.write("\n")
        return Code(200, [], "Job status updated")

    @staticmethod
    def job_list(user_id, org_name, handler_id, kind):
        """
        org_name: str
        handler_id: str, uuid corresponding to experiment/dataset
        kind: str, one of ["experiment","dataset"]
        Returns:
        200, list(dict) - each dict follows JobResultSchema if found
        404, [] if not found
        """
        if handler_id not in ("*", "all") and not resolve_existence(kind, handler_id):
            return Code(404, [], f"{kind} not found")

        if handler_id not in ("*", "all") and not check_read_access(org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        return_metadata = resolve_metadata_with_jobs(user_id, org_name, kind, handler_id, jobs_return_type="list").get("jobs", [])

        return Code(200, return_metadata, "Jobs retrieved")

    @staticmethod
    def job_cancel(org_name, handler_id, job_id, kind):
        """
        org_name: str
        handler_id: str, uuid corresponding to experiment/dataset
        job_id: str, uuid corresponding to job to be cancelled
        kind: str, one of ["experiment","dataset"]
        Returns:
        200, [job_id] - if job can be cancelled
        404, [] if not found
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        if not check_write_access(org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id")
        job_metadata = stateless_handlers.get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, [], "job trying to cancel not found")

        if is_job_automl(user_id, org_name, job_id):
            stateless_handlers.update_job_status(handler_id, job_id, status="Canceling", kind=kind + "s")
            automl_response = AutoMLHandler.stop(user_id, org_name, handler_id, job_id)
            # Remove any pending jobs from Workflow queue
            try:
                on_delete_automl_job(org_name, handler_id, job_id)
            except:
                return Code(200, {"message": f"job {job_id} cancelled, and no pending recommendations"})
            stateless_handlers.update_job_status(handler_id, job_id, status="Canceled", kind=kind + "s")
            return automl_response

        # If job is error / done, then cancel is NoOp
        job_status = job_metadata.get("status", "Error")

        if job_status in ["Error", "Done", "Canceled", "Canceling", "Pausing", "Paused"]:
            return Code(201, {f"Job {job_id} with current status {job_status} can't be attemped to cancel. Current status should be one of Running, Pending, Resuming"})

        if job_status == "Pending":
            stateless_handlers.update_job_status(handler_id, job_id, status="Canceling", kind=kind + "s")
            on_delete_job(org_name, handler_id, job_id, kind + "s")
            stateless_handlers.update_job_status(handler_id, job_id, status="Canceled", kind=kind + "s")
            return Code(200, {"message": f"Pending job {job_id} cancelled"})

        if job_status == "Running":
            try:
                # Delete K8s job
                specs = job_metadata.get("specs", None)
                use_ngc = not (specs and "cluster" in specs and specs["cluster"] == "local")
                stateless_handlers.update_job_status(handler_id, job_id, status="Canceling", kind=kind + "s")
                jobDriver.delete(job_id, use_ngc=use_ngc)
                k8s_status = jobDriver.status(org_name, handler_id, job_id, kind + "s", use_ngc=use_ngc)
                while k8s_status in ("Done", "Error", "Running", "Pending"):
                    if k8s_status in ("Done", "Error"):
                        break
                    k8s_status = jobDriver.status(org_name, handler_id, job_id, kind + "s", use_ngc=use_ngc)
                    time.sleep(5)
                stateless_handlers.update_job_status(handler_id, job_id, status="Canceled", kind=kind + "s")
                return Code(200, {"message": f"Running job {job_id} cancelled"})
            except:
                print("Cancel traceback", traceback.format_exc(), file=sys.stderr)
                return Code(404, [], "job not found in platform")
        else:
            return Code(404, [], "job status not found")

    @staticmethod
    def job_pause(org_name, handler_id, job_id, kind):
        """
        org_name: str
        handler_id: str, uuid corresponding to experiment/dataset
        job_id: str, uuid corresponding to job to be paused
        kind: str, one of ["experiment","dataset"]
        Returns:
        200, [job_id] - if job can be paused
        404, [] if not found
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        if not check_write_access(org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id")
        job_metadata = stateless_handlers.get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, [], "job trying to pause not found")

        if is_job_automl(user_id, org_name, job_id):
            stateless_handlers.update_job_status(handler_id, job_id, status="Pausing", kind=kind + "s")
            automl_response = AutoMLHandler.stop(user_id, org_name, handler_id, job_id)
            # Remove any pending jobs from Workflow queue
            try:
                on_delete_automl_job(org_name, handler_id, job_id)
            except:
                return Code(200, {"message": f"job {job_id} cancelled, and no pending recommendations"})
            stateless_handlers.update_job_status(handler_id, job_id, status="Paused", kind=kind + "s")
            return automl_response

        job_action = job_metadata.get("action", "")
        if job_action not in ("train", "retrain"):
            return Code(404, [], f"Only train or retrain jobs can be paused. The current action is {job_action}")
        job_status = job_metadata.get("status", "Error")

        # If job is error / done, or one of cancel or pause states then pause is NoOp
        if job_status in ["Error", "Done", "Canceled", "Canceling", "Pausing", "Paused"]:
            return Code(201, {"message": f"Job {job_id} with current status {job_status} can't be attemped to pause. Current status should be one of Running, Pending, Resuming"})

        if job_status == "Pending":
            stateless_handlers.update_job_status(handler_id, job_id, status="Pausing", kind=kind + "s")
            on_delete_job(org_name, handler_id, job_id)
            stateless_handlers.update_job_status(handler_id, job_id, status="Paused", kind=kind + "s")
            return Code(200, {"message": f"Pending job {job_id} paused"})

        if job_status == "Running":
            try:
                # Delete K8s job
                specs = job_metadata.get("specs", None)
                use_ngc = not (specs and "cluster" in specs and specs["cluster"] == "local")
                stateless_handlers.update_job_status(handler_id, job_id, status="Pausing", kind=kind + "s")
                jobDriver.delete(job_id, use_ngc=use_ngc)
                k8s_status = jobDriver.status(org_name, handler_id, job_id, kind + "s", use_ngc=use_ngc)
                while k8s_status in ("Done", "Error", "Running", "Pending"):
                    if k8s_status in ("Done", "Error"):
                        break
                    k8s_status = jobDriver.status(org_name, handler_id, job_id, kind + "s", use_ngc=use_ngc)
                    time.sleep(5)
                stateless_handlers.update_job_status(handler_id, job_id, status="Paused", kind=kind + "s")
                return Code(200, {"message": f"Running job {job_id} paused"})
            except:
                print("Pause traceback", traceback.format_exc(), file=sys.stderr)
                return Code(404, [], "job not found in platform")

        else:
            return Code(404, [], "job status not found")

    @staticmethod
    def all_job_cancel(org_name, handler_id, kind):
        """
        org_name: str
        handler_id: str, uuid corresponding to experiment/dataset
        kind: str, one of ["experiment","dataset"]
        Returns:
        200, if all jobs within experiment can be cancelled
        404, [] if not found
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        if not check_write_access(org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        def cancel_jobs_within_handler(cancel_handler_id, cancel_kind):
            cancel_success = True
            cancel_message = ""
            jobs = stateless_handlers.get_jobs_for_handler(cancel_handler_id, cancel_kind)
            for job_metadata in jobs:
                job_id = job_metadata.get("id")
                job_status = job_metadata.get("status", "Error")
                if job_status not in ["Error", "Done", "Canceled", "Canceling", "Pausing", "Paused"] and job_id:
                    cancel_response = AppHandler.job_cancel(org_name, cancel_handler_id, job_id, cancel_kind)
                    if cancel_response.code != 200:
                        if type(cancel_response.data) is dict and cancel_response.data.get("error_desc", "") != "incomplete job not found":
                            cancel_success = False
                            cancel_message += f"Cancelation for job {job_id} failed due to {str(cancel_response.data)} "
            return cancel_success, cancel_message

        if handler_metadata.get("all_jobs_cancel_status") == "Canceling":
            return Code(201, {"message": "Canceling all jobs is already triggered"})

        try:
            handler_metadata["all_jobs_cancel_status"] = "Canceling"
            write_handler_metadata(handler_id, handler_metadata, kind)

            appended_message = ""
            for train_dataset in handler_metadata.get("train_datasets", []):
                jobs_cancel_sucess, message = cancel_jobs_within_handler(train_dataset, "dataset")
                appended_message += message

            eval_dataset = handler_metadata.get("eval_dataset", None)
            if eval_dataset:
                jobs_cancel_sucess, message = cancel_jobs_within_handler(eval_dataset, "dataset")
                appended_message += message

            inference_dataset = handler_metadata.get("inference_dataset", None)
            if inference_dataset:
                jobs_cancel_sucess, message = cancel_jobs_within_handler(inference_dataset, "dataset")
                appended_message += message

            jobs_cancel_sucess, message = cancel_jobs_within_handler(handler_id, kind)
            appended_message += message

            handler_metadata = resolve_metadata(kind, handler_id)
            if jobs_cancel_sucess:
                handler_metadata["all_jobs_cancel_status"] = "Canceled"
                write_handler_metadata(handler_id, handler_metadata, kind)
                return Code(200, {"message": "All jobs within experiment canceled"})
            handler_metadata["all_jobs_cancel_status"] = "Error"
            write_handler_metadata(handler_id, handler_metadata, kind)
            return Code(404, [], appended_message)
        except:
            handler_metadata["all_jobs_cancel_status"] = "Error"
            write_handler_metadata(handler_id, handler_metadata, kind)
            return Code(404, [], "Runtime exception caught during deleting a job")

    @staticmethod
    def job_retrieve(org_name, handler_id, job_id, kind):
        """
        org_name: str
        handler_id: str, uuid corresponding to experiment/dataset
        job_id: str, uuid corresponding to job to be retrieved
        kind: str, one of ["experiment","dataset"]
        Returns:
        200, dict following JobResultSchema - if job found
        404, {} if not found
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, {}, "Dataset not found")

        if not check_read_access(org_name, handler_id, kind=kind + "s"):
            return Code(404, {}, "Dataset not found")

        user_id = handler_metadata.get("user_id")

        job_meta = get_job(job_id)
        if not job_meta:
            return Code(404, {}, "Job trying to retrieve not found")
        if is_job_automl(user_id, org_name, job_id):
            return AutoMLHandler.retrieve(user_id, org_name, handler_id, job_id)
        return Code(200, job_meta, "Job retrieved")

    @staticmethod
    def publish_model(org_name, team_name, experiment_id, job_id, display_name, description):
        """
        Publish a model with the provided information.
        org_name: str, Organization name
        team_name: str, Team name
        experiment_id: str, uuid, Experiment ID
        job_id: str, uuid, Job ID
        display_name: str, Display name for the model
        description: str, Description of the model
        """
        handler_metadata = resolve_metadata("experiment", experiment_id)
        if not handler_metadata:
            return Code(404, {}, "Experiment not found")

        if not check_read_access(org_name, experiment_id, kind="experiments"):
            return Code(404, {}, "Experiment cant be read")

        user_id = handler_metadata.get("user_id")

        job_metadata = stateless_handlers.get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, {}, "Job trying to retrieve not found")

        job_status = job_metadata.get("status", "Error")
        if job_status not in ("Success", "Done"):
            return Code(404, {}, "Job is not in success or Done state")
        job_action = job_metadata.get("action")
        if job_action not in ("train", "prune", "retrain", "export", "gen_trt_engine"):
            return Code(404, {}, "Publish model is available only for train, prune, retrain, export, gen_trt_engine actions")

        try:
            source_file = resolve_checkpoint_root_and_search(handler_metadata, job_id)
            if not source_file:
                return Code(404, [], "Unable to find a model for the given job")

            # Create NGC model
            ngc_api_key, use_cookie = ngc_handler.get_user_api_key(user_id)
            code, message = ngc_handler.create_model(org_name, team_name, handler_metadata, source_file, ngc_api_key, use_cookie, display_name, description)
            if code not in [200, 201]:
                print("Error while creating NGC model")
                return Code(code, {}, message)

            # Upload model version
            if not ngc_api_key or use_cookie:
                ngc_api_key = get_admin_api_key()
            response_code, response_message = ngc_handler.upload_model(org_name, team_name, handler_metadata, source_file, ngc_api_key, job_id, job_action)
            if "already exists" in response_message:
                response_message = "Version trying to upload already exists, use remove_published_model endpoint to reupload the model"
            return Code(response_code, {}, response_message)
        except:
            print(traceback.format_exc(), file=sys.stderr)
            return Code(404, {}, "Unable to publish model")

    @staticmethod
    def remove_published_model(org_name, team_name, experiment_id, job_id):
        """
        Publish a model with the provided information.
        org_name: str, Organization name
        team_name: str, Team name
        experiment_id: str, uuid, Experiment ID
        job_id: str, uuid, Job ID
        """
        handler_metadata = resolve_metadata("experiment", experiment_id)
        if not handler_metadata:
            return Code(404, {}, "Experiment not found")

        if not check_read_access(org_name, experiment_id, kind="experiments"):
            return Code(404, {}, "Experiment cant be read")

        user_id = handler_metadata.get("user_id")
        job_metadata = stateless_handlers.get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, {}, "Job trying to retrieve not found")

        job_status = job_metadata.get("status", "Error")
        if job_status not in ("Success", "Done"):
            return Code(404, {}, "Job is not in success or Done state")
        job_action = job_metadata.get("action")
        if job_action not in ("train", "prune", "retrain", "export", "gen_trt_engine"):
            return Code(404, {}, "Delete published model is available only for train, prune, retrain, export, gen_trt_engine actions")

        try:
            ngc_api_key, use_cookie = ngc_handler.get_user_api_key(user_id)
            response = ngc_handler.delete_model(org_name, team_name, handler_metadata, ngc_api_key, use_cookie, job_id, job_action)
            if response.ok:
                return Code(response.status_code, {}, "Sucessfully deleted model")
            return Code(response.status_code, {}, "Unable to delete published model")
        except:
            print(traceback.format_exc(), file=sys.stderr)
            return Code(404, {}, "Unable to delete published model")

    # Delete job
    @staticmethod
    def job_delete(org_name, handler_id, job_id, kind):
        """
        org_name: str
        handler_id: str, uuid corresponding to experiment/dataset
        job_id: str, uuid corresponding to job to be deleted
        kind: str, one of ["experiment","dataset"]
        Returns:
        200, [job_id] - if job can be deleted
        404, [] if not found
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        if not check_write_access(org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id")
        job_metadata = stateless_handlers.get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, [], "job trying to delete not found")

        try:
            # If job is running, cannot delete
            if job_metadata.get("status", "Error") in ["Running", "Pending"]:
                return Code(400, [], "job cannot be deleted")
            # Delete job metadata
            mongo_jobs = MongoHandler("tao", "jobs")
            mongo_jobs.delete_one({'id': job_id})
            # Delete job logs
            job_log_path = os.path.join(stateless_handlers.get_handler_log_root(user_id, org_name, handler_id), job_id + ".txt")
            if os.path.exists(job_log_path):
                os.remove(job_log_path)
            return Code(200, [job_id], "job deleted")
        except:
            print(traceback.format_exc(), file=sys.stderr)
            return Code(400, [], "job cannot be deleted")

    # Download experiment job
    @staticmethod
    def job_download(org_name, handler_id, job_id, kind, file_lists=None, best_model=None, latest_model=None, tar_files=True, export_type="tao"):
        """
        org_name: str
        handler_id: str, uuid corresponding to experiment/dataset
        job_id: str, uuid corresponding to job to be downloaded
        kind: str, one of ["experiment","dataset"]
        export_type: str, one of ["monai_bundle", "tao"]
        Returns:
        200, Path to a tar.gz created from the job directory
        404, None if not found
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, None, f"{kind} not found")

        if not check_read_access(org_name, handler_id, kind=kind + "s"):
            return Code(404, None, f"{kind} not found")

        user_id = handler_metadata.get("user_id")
        handler_job_metadata = stateless_handlers.get_handler_job_metadata(job_id)
        if not handler_job_metadata:
            return Code(404, None, "job trying to download not found")

        try:
            if export_type not in VALID_MODEL_DOWNLOAD_TYPE:
                return Code(404, None, f"Export format {export_type} not found.")
            root = stateless_handlers.get_jobs_root(user_id, org_name)
            if export_type == "monai_bundle":
                job_root = os.path.join(root, job_id)
                path_status_code = get_monai_bundle_path(job_root)
                if path_status_code.code != 201:
                    return path_status_code
                bundle_path = path_status_code.data
                command = f"cd {root} ; tar -zcvf {job_id}.tar.gz {job_id}/{bundle_path}; cd -"
                run_system_command(command)
                out_tar = os.path.join(root, job_id + ".tar.gz")
                if os.path.exists(out_tar):
                    return Code(200, out_tar, "job deleted")
                return Code(404, None, "job output not found")

            # Following is for `if export_type == "tao":`
            # Copy job logs from root/logs/<job_id>.txt to root/<job_id>/logs_from_toolkit.txt
            out_tar = os.path.join(root, job_id + ".tar.gz")
            files = [os.path.join(root, job_id)]
            if file_lists or best_model or latest_model:
                files = []
                for file in file_lists:
                    if os.path.exists(os.path.join(root, file)):
                        files.append(os.path.join(root, file))
                action = handler_job_metadata.get("action", "")
                epoch_number_dictionary = handler_metadata.get("checkpoint_epoch_number", {})
                best_checkpoint_epoch_number = epoch_number_dictionary.get(f"best_model_{job_id}", 0)
                latest_checkpoint_epoch_number = epoch_number_dictionary.get(f"latest_model_{job_id}", 0)
                if (not best_model) and latest_model:
                    best_checkpoint_epoch_number = latest_checkpoint_epoch_number
                network = handler_metadata.get("network_arch", "")
                if network in ("classification_pyt", "detectnet_v2", "pointpillars", "segformer", "unet"):
                    format_epoch_number = str(best_checkpoint_epoch_number)
                else:
                    format_epoch_number = f"{best_checkpoint_epoch_number:03}"
                if best_model or latest_model:
                    job_root = os.path.join(root, job_id)
                    if handler_metadata.get("automl_settings", {}).get("automl_enabled") is True and action == "train":
                        job_root = os.path.join(job_root, "best_model")
                    find_trained_tlt = glob.glob(f"{job_root}/*{format_epoch_number}.tlt") + glob.glob(f"{job_root}/train/*{format_epoch_number}.tlt") + glob.glob(f"{job_root}/weights/*{format_epoch_number}.tlt")
                    find_trained_pth = glob.glob(f"{job_root}/*{format_epoch_number}.pth") + glob.glob(f"{job_root}/train/*{format_epoch_number}.pth") + glob.glob(f"{job_root}/weights/*{format_epoch_number}.pth")
                    find_trained_hdf5 = glob.glob(f"{job_root}/*{format_epoch_number}.hdf5") + glob.glob(f"{job_root}/train/*{format_epoch_number}.hdf5") + glob.glob(f"{job_root}/weights/*{format_epoch_number}.hdf5")
                    if find_trained_tlt:
                        files.append(find_trained_tlt[0])
                    if find_trained_pth:
                        files.append(find_trained_pth[0])
                    if find_trained_hdf5:
                        files.append(find_trained_hdf5[0])
                if not files:
                    return Code(404, None, "Atleast one of the requested files not present")

            if files == [os.path.join(root, job_id)]:
                print("Entire job folder already cached", file=sys.stderr)
                if os.path.exists(out_tar):
                    return Code(200, out_tar, "entire job folder downloaded")
                return Code(404, None, "Caching of download failed hence job not downloaded")

            files = list(set(files))
            if tar_files or (not tar_files and len(files) > 1):
                def get_files_recursively(directory):
                    return [file for file in glob.glob(os.path.join(directory, '**'), recursive=True) if os.path.isfile(file) and not file.endswith(".lock")]
                all_files = []
                for file in files:
                    if os.path.isdir(file):
                        all_files.extend(get_files_recursively(file))
                    elif os.path.isfile(file):
                        all_files.append(file)

                out_tar = out_tar.replace(".tar.gz", str(uuid.uuid4()) + ".tar.gz")  # Appending UUID to not overwrite the tar file created at end of job complete
                with tarfile.open(out_tar, "w:gz") as tar:
                    for file_path in all_files:
                        tar.add(file_path, arcname=file_path.replace(root, "", 1))
                return Code(200, out_tar, "selective files of job downloaded")

            if files and os.path.exists(os.path.join(root, files[0])):
                return Code(200, os.path.join(root, files[0]), "single file of job downloaded")
            return Code(404, None, "job output not found")

        except:
            print(traceback.format_exc(), file=sys.stderr)
            return Code(404, None, "job output not found")

    @staticmethod
    def job_list_files(org_name, handler_id, job_id, retrieve_logs, retrieve_specs, kind):
        """
        org_name: str
        handler_id: str, uuid corresponding to experiment/dataset
        job_id: str, uuid corresponding to job for which the files need to be listed
        kind: str, one of ["experiment","dataset"]
        Returns:
        200, list(str) - list of file paths wtih respect to the job
        404, None if no files are found
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        if not check_read_access(org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id")
        job = get_job(job_id)
        if not job:
            return Code(404, None, "job trying to view not found")

        files = stateless_handlers.get_job_files(user_id, org_name, handler_id, job_id, retrieve_logs, retrieve_specs)
        if files:
            return Code(200, files, "Job files retrieved")
        return Code(200, files, "No downloadable files for this job is found")

    # Get realtime job logs
    @staticmethod
    def get_job_logs(org_name, handler_id, job_id, kind, automl_experiment_index=None):
        """Returns real time job logs"""
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, {}, f"{kind} not found.")

        # Get log file path
        # Normal action log is saved at /orgs/<org_name>/users/<user_id>/logs/<job_id>.txt
        # AutoML train  log is saved at /orgs/<org_name>/users/<user_id>/jobs/<job_id>/experiment_<recommendation_index>/log.txt
        user_id = handler_metadata.get("user_id")
        log_file_path = os.path.join(get_handler_log_root(user_id, org_name, handler_id), str(job_id) + ".txt")
        job_metadata = get_handler_job_metadata(job_id)
        if handler_metadata.get("automl_settings", {}).get("automl_enabled", False) and job_metadata.get("action") == "train":
            root = os.path.join(get_jobs_root(user_id, org_name), job_id)
            automl_index = safe_load_file(root + "/current_rec.json")
            if automl_experiment_index is not None:
                automl_index = int(automl_experiment_index)
            log_file_path = os.path.join(root, f"experiment_{automl_index}", "log.txt")

        # File not present - Use detailed message or job status
        if not os.path.exists(log_file_path):
            detailed_result_msg = job_metadata.get("result", {}).get("detailed_status", {}).get("message", "")
            if detailed_result_msg:
                return Code(200, detailed_result_msg)

            if handler_metadata.get("automl_settings", {}).get("automl_enabled", False) and job_metadata.get("action") == "train":
                if handler_metadata.get("status") in ["Canceled", "Canceling"]:
                    return Code(200, "AutoML training has been canceled.")
                if handler_metadata.get("status") in ["Paused", "Pausing"]:
                    return Code(200, "AutoML training has been paused.")
                if handler_metadata.get("status") == "Resuming":
                    return Code(200, "AutoML training is resuming.")
                if handler_metadata.get("status") == "Running":
                    return Code(200, "Generating new recommendation for AutoML experiment.")
            return Code(404, {}, "Logs for the job are not available yet.")
        return Code(200, get_job_logs(log_file_path))

    # Experiment API
    @staticmethod
    def list_experiments(user_id, org_name, user_only=False):
        """
        user_id: str, uuid
        org_name: str
        Returns:
        list(dict) - list of experiments accessible by user where each element is metadata of a experiment
        """
        # Collect all metadatas
        metadatas = []
        for experiment_id in list(set(get_org_experiments(org_name))):
            handler_metadata = stateless_handlers.get_handler_metadata(experiment_id, "experiments")
            shared_experiment = handler_metadata.get("shared", False)
            if handler_metadata:
                if shared_experiment or handler_metadata.get("user_id") == user_id:
                    handler_metadata = sanitize_handler_metadata(handler_metadata)
                    metadatas.append(handler_metadata)
        if not user_only:
            public_experiments_metadata = stateless_handlers.get_public_experiments()
            metadatas += public_experiments_metadata
        return metadatas

    @staticmethod
    def list_base_experiments():
        """
        Returns:
        list(dict) - list of base experiments accessible by user where each element is metadata of a experiment
        """
        # Collect all metadatas
        metadatas = []
        public_experiments_metadata = stateless_handlers.get_public_experiments()
        metadatas += public_experiments_metadata
        return metadatas

    @staticmethod
    def create_experiment(user_id, org_name, request_dict, experiment_id=None, from_ui=False):
        """
        org_name: str
        request_dict: dict following ExperimentReqSchema
            - network_arch is required
            - encryption_key is required (not enforced)
        from_ui: whether an experiment creation call is from UI
        Returns:
        - 201 with metadata of created experiment if successful
        - 400 if experiment type and format not given
        """
        # Create a dataset ID and its root
        experiment_id = experiment_id or str(uuid.uuid4())

        # Gather type,format fields from request
        mdl_nw = request_dict.get("network_arch", None)
        # Perform basic checks - valid type and format?
        if mdl_nw not in VALID_NETWORKS:
            msg = "Invalid network arch"
            return Code(400, {}, msg)

        if request_dict.get("public", False):
            stateless_handlers.add_public_experiment(experiment_id)

        mdl_type = request_dict.get("type", "vision")
        if str(mdl_nw).startswith("monai_"):
            mdl_type = "medical"

        # Create metadata dict and create some initial folders
        # Initially make datasets, base_experiment None
        metadata = {"id": experiment_id,
                    "user_id": user_id,
                    "org_name": org_name,
                    "created_on": datetime.now(tz=timezone.utc),
                    "last_modified": datetime.now(tz=timezone.utc),
                    "name": request_dict.get("name", "My Experiment"),
                    "shared": request_dict.get("shared", False),
                    "description": request_dict.get("description", "My Experiments"),
                    "version": request_dict.get("version", "1.0.0"),
                    "logo": request_dict.get("logo", "https://www.nvidia.com"),
                    "ngc_path": request_dict.get("ngc_path", ""),
                    "encryption_key": request_dict.get("encryption_key", "tlt_encode"),
                    "read_only": request_dict.get("read_only", False),
                    "public": request_dict.get("public", False),
                    "network_arch": mdl_nw,
                    "type": mdl_type,
                    "dataset_type": read_network_config(mdl_nw)["api_params"]["dataset_type"],
                    "dataset_formats": read_network_config(mdl_nw)["api_params"].get("formats", read_network_config(read_network_config(mdl_nw)["api_params"]["dataset_type"]).get("api_params", {}).get("formats", None)),
                    "accepted_dataset_intents": read_network_config(mdl_nw)["api_params"].get("accepted_ds_intents", []),
                    "actions": read_network_config(mdl_nw)["api_params"]["actions"],
                    "docker_env_vars": request_dict.get("docker_env_vars", {}),
                    "train_datasets": [],
                    "eval_dataset": None,
                    "inference_dataset": None,
                    "additional_id_info": None,
                    "checkpoint_choose_method": request_dict.get("checkpoint_choose_method", "best_model"),
                    "checkpoint_epoch_number": request_dict.get("checkpoint_epoch_number", {}),
                    "calibration_dataset": None,
                    "base_experiment": [],
                    "automl_settings": request_dict.get("automl_settings", {}),
                    "metric": request_dict.get("metric", "kpi"),
                    "realtime_infer": False,
                    "realtime_infer_support": False,
                    "realtime_infer_endpoint": None,
                    "realtime_infer_model_name": None,
                    "realtime_infer_request_timeout": request_dict.get("realtime_infer_request_timeout", 60),
                    "model_params": request_dict.get("model_params", {}),
                    "tensorboard_enabled": request_dict.get("tensorboard_enabled", False),
                    "workspace": request_dict.get("workspace", None),
                    "experiment_actions": request_dict.get('experiment_actions', []),
                    "tags": list(set(map(lambda x: x.lower(), request_dict.get("tags", [])))),
                    }

        if metadata.get("automl_settings", {}).get("automl_enabled") and mdl_nw in AUTOML_DISABLED_NETWORKS:
            return Code(400, {}, "automl_enabled cannot be True for unsupported network")

        if metadata.get("automl_settings", {}).get("automl_enabled") and metadata.get("tensorboard_enabled", False):
            return Code(400, {}, "Tensorboard not yet supported for AutoML experiments")
        if mdl_nw in TAO_NETWORKS and (not metadata.get("workspace")):
            return Code(400, {}, "Workspace must be provided for experiment creation")

        # Encrypt the MLOPs keys
        if BACKEND in ("BCP", "NVCF") and metadata["docker_env_vars"]:
            config_path = os.getenv("VAULT_SECRET_PATH", None)
            encryption = NVVaultEncryption(config_path)
            for key, value in metadata["docker_env_vars"].items():
                if encryption.check_config()[0]:
                    metadata["docker_env_vars"][key] = encryption.encrypt(value)
                elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                    return Code(400, {}, "Vault service does not work, can't enable MLOPs services")

        # Update datasets and base_experiments if given.
        # "realtime_infer" will be checked later, since in some cases (in MEDICAL_CUSTOM_ARCHITECT), need to prepare base_experiment first
        metadata, error_code = validate_and_update_experiment_metadata(org_name, request_dict, metadata, ["train_datasets", "eval_dataset", "inference_dataset", "calibration_dataset", "base_experiment"])
        if error_code:
            return error_code

        def clean_on_error(experiment_id=experiment_id):
            handle_dir = os.path.join(get_root(), org_name, "experiments", experiment_id)
            if os.path.exists(handle_dir):
                shutil.rmtree(handle_dir, ignore_errors=True)

        if mdl_type == "medical":
            is_custom_bundle = metadata["network_arch"] in MEDICAL_CUSTOM_ARCHITECT
            is_auto3seg_inference = metadata["network_arch"] == "monai_automl_generated"
            no_ptm = (metadata["base_experiment"] is None) or (len(metadata["base_experiment"]) == 0)
            if no_ptm and is_custom_bundle:
                # If base_experiment is not provided, then we will need to create a model to host the files downloaded from NGC.
                # This is a temporary solution until we have a better way to handle this.
                bundle_url = request_dict.get("bundle_url", None)
                if bundle_url is None:
                    return Code(400, {}, "Either `bundle_url` or `ngc_path` needs to be defined for MONAI Custom Model.")
                base_experiment_id = str(uuid.uuid4())
                ptm_metadata = metadata.copy()
                ptm_metadata["id"] = base_experiment_id
                ptm_metadata["name"] = "base_experiment_" + metadata["name"]
                ptm_metadata["description"] = " PTM auto-generated. " + metadata["description"]
                ptm_metadata["train_datasets"] = []
                ptm_metadata["eval_dataset"] = None
                ptm_metadata["inference_dataset"] = None
                # since "realtime_infer" is not updated by update_metadata, specify it from request_dict here first for download.
                ptm_metadata["realtime_infer"] = request_dict.get("realtime_infer", False)
                ptm_metadata["realtime_infer_support"] = ptm_metadata["realtime_infer"]
                stateless_handlers.make_root_dirs(user_id, org_name, "experiments", base_experiment_id)
                write_handler_metadata(base_experiment_id, ptm_metadata, "experiment")
                # Download it from the provided url
                download_from_url(bundle_url, base_experiment_id)

                # The base_experiment is downloaded, now we need to make sure it is correct.
                bundle_checks = []
                if ptm_metadata["realtime_infer"]:
                    bundle_checks.append("infer")
                ptm_file = validate_monai_bundle(base_experiment_id, checks=bundle_checks)
                if (ptm_file is None) or (not os.path.isdir(ptm_file)):
                    clean_on_error(experiment_id=base_experiment_id)
                    return Code(400, {}, "Failed to download base experiment, or the provided bundle does not follow MONAI bundle format.")

                ptm_metadata["base_experiment_pull_complete"] = "pull_complete"
                write_handler_metadata(base_experiment_id, ptm_metadata, "experiment")
                bundle_url_path = os.path.join(resolve_root(org_name, "experiment", base_experiment_id), CUSTOMIZED_BUNDLE_URL_FILE)
                safe_dump_file(bundle_url_path, {CUSTOMIZED_BUNDLE_URL_KEY: bundle_url})
                metadata["base_experiment"] = [base_experiment_id]
            elif no_ptm and is_auto3seg_inference:
                bundle_url = request_dict.get("bundle_url", None)
                if bundle_url is None:
                    return Code(400, {}, "Either `bundle_url` or `ngc_path` needs to be defined for MONAI Custom Model.")
                bundle_url_path = os.path.join(resolve_root(org_name, "experiment", experiment_id), CUSTOMIZED_BUNDLE_URL_FILE)
                os.makedirs(resolve_root(org_name, "experiment", experiment_id), exist_ok=True)
                safe_dump_file(bundle_url_path, {CUSTOMIZED_BUNDLE_URL_KEY: bundle_url})

            network_arch = metadata.get("network_arch", None)
            log_content = f"user_id:{user_id}, org_name:{org_name}, from_ui:{from_ui}, network_arch:{network_arch}, action:creation, no_ptm:{no_ptm}"
            log_monitor(log_type=DataMonitorLogTypeEnum.medical_experiment, log_content=log_content)

        # check "realtime_infer"
        metadata, error_code = validate_and_update_experiment_metadata(org_name, request_dict, metadata, ["realtime_infer"])
        if error_code:
            return error_code

        if mdl_type == "medical" and metadata["realtime_infer"]:
            base_experiment_id = metadata["base_experiment"][0]
            model_params = metadata["model_params"]
            job_id = None
            if metadata["network_arch"] not in MEDICAL_CUSTOM_ARCHITECT:
                # Need to download the base_experiment to set up the TIS for realtime infer
                # Customizd model already has the base_experiment downloaded in the previous step thus skip here
                if stateless_handlers.get_base_experiment_metadata(base_experiment_id).get("ngc_path", None):
                    download_base_experiment(user_id, base_experiment_id)
                else:
                    additional_id_info = request_dict.get("additional_id_info", None)
                    job_id = additional_id_info if additional_id_info and is_valid_uuid4(additional_id_info) else None
                    if not job_id:
                        return Code(400, {}, f"Non-NGC base_experiment {base_experiment_id} needs job_id in the request for path location")
            success, model_name, msg, bundle_metadata = prep_tis_model_repository(model_params, base_experiment_id, org_name, user_id, experiment_id, job_id=job_id)
            if not success:
                clean_on_error(experiment_id=experiment_id)
                return Code(400, {}, msg)

            # If Labels not supplied then pick from Model Bundle
            if model_params.get("labels", None) is None:
                _, pred = next(iter(bundle_metadata["network_data_format"]["outputs"].items()))
                labels = {k: v.lower() for k, v in pred.get("channel_def", {}).items() if v.lower() != "background"}
                model_params["labels"] = labels
            else:
                labels = model_params.get("labels")
                if len(labels) == 0:
                    return Code(400, {}, "Labels cannot be empty")

            # get replicas data to determine if create multiple replicas
            replicas = model_params.get("replicas", 1)
            success, msg = CapGpuUsage.schedule(org_name, replicas)
            if not success:
                return Code(400, {}, msg)
            response = TISHandler.start(org_name, experiment_id, metadata, model_name, replicas)
            if response.code != 201:
                TISHandler.stop(experiment_id, metadata)
                CapGpuUsage.release_used(org_name, replicas)
                clean_on_error(experiment_id=experiment_id)
                return response
            metadata["realtime_infer_endpoint"] = response.data["pod_ip"]
            metadata["realtime_infer_model_name"] = model_name
            metadata["realtime_infer_support"] = True

        # Create Tensorboard deployment if enabled
        if metadata.get("tensorboard_enabled", False):
            if mdl_nw in TENSORBOARD_DISABLED_NETWORKS:
                clean_on_error(experiment_id)
                return Code(400, {}, f"Network {mdl_nw} not supported for Tensorboard")
            if TensorboardHandler.check_user_metadata(user_id):
                response = TensorboardHandler.start(org_name, experiment_id, user_id)
                if response.code != 201:
                    TensorboardHandler.stop(experiment_id, user_id)
                    clean_on_error(experiment_id)
                    return response
            else:
                clean_on_error(experiment_id)
                return Code(400, {}, f"Maximum of {TENSORBOARD_EXPERIMENT_LIMIT} Tensorboard Experiments allowed per user. Disable or delete a Tensorboard Experiment and try again.")

        # Actual "creation" happens here...
        stateless_handlers.make_root_dirs(user_id, org_name, "experiments", experiment_id)
        write_handler_metadata(experiment_id, metadata, "experiment")

        mongo_users = MongoHandler("tao", "users")
        experiments = get_user_experiments(user_id, mongo_users)
        experiments.append(experiment_id)
        if mdl_type == 'medical':
            for base_exp in metadata.get("base_experiment", []):
                experiments.append(base_exp)
        mongo_users.upsert({'id': user_id}, {'id': user_id, 'experiments': experiments})

        experiment_actions = request_dict.get('experiment_actions', [])
        retry_experiment_id = request_dict.get('retry_experiment_id', None)
        error_response = None
        if retry_experiment_id:
            error_response = AppHandler.retry_experiment(org_name, user_id, retry_experiment_id, experiment_id, from_ui)
        elif experiment_actions:
            error_response = AppHandler.retry_experiment_actions(org_name, experiment_id, experiment_actions, from_ui)
        if error_response:
            clean_on_error(experiment_id)
            return error_response

        # Read this metadata from saved file...
        return_metadata = resolve_metadata_with_jobs(user_id, org_name, "experiment", experiment_id)
        return_metadata = sanitize_handler_metadata(return_metadata)
        ret_Code = Code(201, return_metadata, "Experiment created")

        # TODO: may need to call "monai_triton_client" with dummy request to accelerate
        return ret_Code

    # Retry existing experiment
    @staticmethod
    def retry_experiment(org_name, user_id, retry_experiment_id, new_experiment_id, from_ui):
        handler_metadata = stateless_handlers.get_handler_metadata_with_jobs(retry_experiment_id, "experiments")
        handler_jobs = handler_metadata.get("jobs", [])
        job_map = {}
        for job in handler_jobs:
            job_id = job.get('id')
            job_action = job.get('action')
            if job_id and job_action:
                print(f"Loading existing specs from job {job_id}", file=sys.stderr)
                spec_json_path = os.path.join(get_handler_spec_root(user_id, org_name, retry_experiment_id), f"{job_id}-{job_action}-spec.json")
                specs = safe_load_file(spec_json_path)
                name = job.get('name')
                description = job.get('description')
                retry_parent_job_id = job.get('parent_id', None)
                parent_job_id = None
                if retry_parent_job_id:
                    retry_parent_job = get_job(retry_parent_job_id)
                    parent_action = retry_parent_job.get('action')
                    parent_job_id = job_map.get(parent_action, None)
                response = AppHandler.job_run(org_name=org_name, handler_id=new_experiment_id, parent_job_id=parent_job_id, action=job_action, kind='experiment', specs=specs, name=name, description=description, from_ui=from_ui)
                if response.code == 201:
                    job_id = response.data
                    job_map[job_action] = job_id
                    print(f"Created {job_action} job with id {job_id} for experiment {new_experiment_id}", file=sys.stderr)
                else:
                    return response
        return None

    # Retry experiment actions
    @staticmethod
    def retry_experiment_actions(org_name, experiment_id, experiment_actions, from_ui):
        raw_actions = []
        action_lookup = {}
        for action_dict in experiment_actions:
            action = action_dict.get('action')
            specs = action_dict.get('specs', {})
            name = action_dict.get('name')
            description = action_dict.get('description')
            num_gpu = action_dict.get('num_gpu', -1)
            platform = action_dict.get('platform', None)
            action_data = {'specs': specs, 'name': name, 'description': description, 'num_gpu': num_gpu, 'platform': platform}
            if action:
                raw_actions.append(action)
                action_lookup[action] = action_data

        if raw_actions and action_lookup:
            job_mapping = stateless_handlers.validate_chained_actions(raw_actions)
            if not job_mapping:
                return Code(400, {}, "Invalid workflow chaining")

            job_action_to_id = {}
            for mapping in job_mapping:
                child_action = mapping.get('child')
                parent_action = mapping.get('parent', None)
                if child_action in action_lookup:
                    lookup_data = action_lookup[child_action]
                    specs = {}
                    if not specs and not lookup_data.get('specs', {}):
                        specs_response = AppHandler.get_spec_schema(org_name, experiment_id, child_action, 'experiment')
                        if specs_response.code == 200:
                            spec_schema = specs_response.data
                            specs = spec_schema["default"]
                            print("Retrieved specs from DNN: ", specs, file=sys.stderr)
                        else:
                            return specs_response
                    else:
                        specs = action_lookup[child_action].get('specs', {})
                    name = lookup_data.get('name')
                    description = lookup_data.get('description')
                    num_gpu = lookup_data.get('num_gpu', -1)
                    platform = lookup_data.get('platform', None)
                    parent_job_id = job_action_to_id.get(parent_action, None)
                    response = AppHandler.job_run(org_name=org_name, handler_id=experiment_id, parent_job_id=parent_job_id, action=child_action, kind='experiment', specs=specs, name=name, description=description, num_gpu=num_gpu, platform=platform, from_ui=from_ui)
                    if response.code == 201:
                        job_id = response.data
                        print(f"Created {child_action} job with id {job_id} for experiment {experiment_id}", file=sys.stderr)
                        job_action_to_id[child_action] = job_id
                    else:
                        return response
        return None

    # Update existing experiment for user based on request dict
    @staticmethod
    def update_experiment(org_name, experiment_id, request_dict):
        """
        org_name: str
        experiment_id: str, uuid
        request_dict: dict following ExperimentReqSchema
        Returns:
        - 200 with metadata of updated experiment if successful
        - 404 if experiment not found / user cannot access
        - 400 if invalid update / experiment is read only
        """
        metadata = resolve_metadata("experiment", experiment_id)
        if not metadata:
            return Code(400, {}, "Experiment does not exist")

        if not check_write_access(org_name, experiment_id, kind="experiments"):
            return Code(400, {}, "User doesn't have write access to experiment")

        # if public is set to True => add it to public_experiments, if it is set to False => take it down
        # if public is not there, do nothing
        if request_dict.get("public", None):
            if request_dict["public"]:
                stateless_handlers.add_public_experiment(experiment_id)
            else:
                stateless_handlers.remove_public_experiment(experiment_id)

        user_id = metadata.get("user_id")
        for key in request_dict.keys():

            # Cannot process the update, so return 400
            if key in ["network_arch", "experiment_params", "base_experiment_metadata"]:
                if request_dict[key] != metadata.get(key):
                    msg = f"Cannot change experiment {key}"
                    return Code(400, {}, msg)

            if (key == "realtime_infer") and (request_dict[key] != metadata.get(key)):
                if request_dict[key] is False:
                    response = TISHandler.stop(experiment_id, metadata)
                    replicas = metadata.get("model_params", {}).get("replicas", 1)
                    CapGpuUsage.release_used(org_name, replicas)
                    if response.code != 201:
                        return response
                    metadata[key] = False
                else:
                    return Code(400, {}, f"Can only change {key} from True to False.")

            if key in ["name", "description", "version", "logo",
                       "ngc_path", "encryption_key", "read_only",
                       "metric", "public", "shared", "tags"]:
                requested_value = request_dict[key]
                if requested_value is not None:
                    metadata[key] = requested_value
                    metadata["last_modified"] = datetime.now(tz=timezone.utc)
                    if key == "tags":
                        metadata[key] = list(set(map(lambda x: x.lower(), requested_value)))

            if key == "docker_env_vars":
                # Encrypt the MLOPs keys
                requested_value = request_dict[key]
                if BACKEND in ("BCP", "NVCF"):
                    config_path = os.getenv("VAULT_SECRET_PATH", None)
                    encryption = NVVaultEncryption(config_path)
                    for mlops_key, value in requested_value.items():
                        if encryption.check_config()[0]:
                            metadata["docker_env_vars"][mlops_key] = encryption.encrypt(value)
                        elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                            return Code(400, {}, "Vault service does not work, can't enable MLOPs services")
                else:
                    metadata["docker_env_vars"] = requested_value

            metadata, error_code = validate_and_update_experiment_metadata(org_name, request_dict, metadata, ["train_datasets", "eval_dataset", "inference_dataset", "calibration_dataset", "base_experiment", "checkpoint_choose_method", "checkpoint_epoch_number"])
            if error_code:
                return error_code

            automl_enabled = metadata.get("automl_settings", {}).get("automl_enabled", False)
            tensorboard_enabled = metadata.get("tensorboard_enabled", False)
            if key == "automl_settings":
                value = request_dict[key]
                automl_enabled = value.get('automl_enabled', False)
                # If False, can set. If True, need to check if AutoML is supported
                if value:
                    mdl_nw = metadata.get("network_arch", "")
                    if tensorboard_enabled and automl_enabled:
                        return Code(400, {}, "automl_enabled cannot be True for Tensorboard experiment")
                    if mdl_nw not in AUTOML_DISABLED_NETWORKS:
                        metadata[key] = request_dict.get(key, {})
                    else:
                        return Code(400, {}, "automl_enabled cannot be True for unsupported network")
                else:
                    metadata[key] = value

            if key == "tensorboard_enabled":
                value = request_dict[key]
                mdl_nw = metadata.get("network_arch", "")
                if not tensorboard_enabled and value:  # Enable Tensorboard
                    if mdl_nw in TENSORBOARD_DISABLED_NETWORKS:
                        return Code(400, {}, f"Network {mdl_nw} not supported for Tensorboard")
                    if automl_enabled:
                        return Code(400, {}, "AutoML not supported yet for Tensorboard")
                    if TensorboardHandler.check_user_metadata(user_id):
                        response = TensorboardHandler.start(org_name, experiment_id, user_id)
                        if response.code != 201:
                            TensorboardHandler.stop(experiment_id, user_id)
                            return response
                    else:
                        return Code(400, {}, f"Maximum of {TENSORBOARD_EXPERIMENT_LIMIT} Tensorboard Experiments allowed per user. Disable or delete a Tensorboard Experiment and try again.")
                elif tensorboard_enabled and not value:  # Disable Tensorboard
                    response = TensorboardHandler.stop(org_name, experiment_id, user_id)
                    if response.code != 200:
                        return response
                metadata[key] = value

        write_handler_metadata(experiment_id, metadata, "experiment")
        # Read this metadata from saved file...
        return_metadata = resolve_metadata_with_jobs(user_id, org_name, "experiment", experiment_id)
        return_metadata = sanitize_handler_metadata(return_metadata)
        ret_Code = Code(200, return_metadata, "Experiment updated")
        return ret_Code

    @staticmethod
    def retrieve_experiment(org_name, experiment_id):
        """
        org_name: str
        experiment_id: str, uuid

        Returns:
        - 200 with metadata of retrieved experiment if successful
        - 404 if experiment not found / user cannot access
        """
        handler_metadata = resolve_metadata("experiment", experiment_id)
        if experiment_id not in ("*", "all") and not handler_metadata:
            return Code(404, {}, "Experiment not found")

        if experiment_id not in ("*", "all") and not check_read_access(org_name, experiment_id, kind="experiments"):
            return Code(404, {}, "Experiment not found")

        user_id = handler_metadata.get("user_id")
        return_metadata = resolve_metadata_with_jobs(user_id, org_name, "experiment", experiment_id)
        return_metadata = sanitize_handler_metadata(return_metadata)
        return Code(200, return_metadata, "Experiment retrieved")

    @staticmethod
    def delete_experiment(org_name, experiment_id):
        """
        org_name: str
        experiment_id: str, uuid
        Returns:
        Code object
        - 200 with metadata of deleted experiment if successful
        - 404 if user cannot access
        - 400 if experiment cannot be deleted b/c (1) pending/running jobs (2) public (3) read-only (4) TIS stop fails
        """
        handler_metadata = resolve_metadata("experiment", experiment_id)
        if not handler_metadata:
            return Code(200, {}, "Experiment deleted")
        if not check_write_access(org_name, experiment_id, kind="experiments"):
            return Code(404, {}, "User doesn't have write access to experiment")

        user_id = handler_metadata.get("user_id")
        # If experiment is being used by user's experiments.
        experiments = get_user_experiments(user_id)

        if experiment_id not in experiments:
            return Code(404, {}, "Experiment cannot be deleted")

        for handler_id in experiments:
            metadata = get_experiment(handler_id)
            if experiment_id == metadata.get("base_experiment", None):
                return Code(400, {}, "Experiment in use as a base_experiment")

        return_metadata = resolve_metadata_with_jobs(user_id, org_name, "experiment", experiment_id)
        for job in return_metadata["jobs"]:
            if return_metadata["jobs"][job]["status"] in ("Pending", "Running"):
                return Code(400, {}, "Experiment in use")

        # Check if experiment is public, then someone could be running it
        if return_metadata.get("public", False):
            return Code(400, {}, "Experiment is Public. Cannot delete")

        # Check if experiment is read only, if yes, cannot delete
        if return_metadata.get("read_only", False):
            return Code(400, {}, "Experiment is read only. Cannot delete")

        # Check if the experiment is being used by a realtime infer job
        if return_metadata.get("realtime_infer", False):
            response = TISHandler.stop(experiment_id, return_metadata)
            replicas = metadata.get("model_params", {}).get("replicas", 1)
            CapGpuUsage.release_used(org_name, replicas)
            if response is not None and response.code != 201:
                return response

        if handler_metadata.get("tensorboard_enabled", False):
            response = TensorboardHandler.stop(org_name, experiment_id, user_id)
            if response.code != 200:
                return response

        if experiment_id in experiments:
            experiments.remove(experiment_id)
            mongo_users = MongoHandler("tao", "users")
            mongo_users.upsert({'id': user_id}, {'id': user_id, 'experiments': experiments})

        delete_jobs_for_handler(experiment_id, "experiment")
        mongo_experiments = MongoHandler("tao", "experiments")
        mongo_experiments.delete_one({'id': experiment_id})
        return_metadata = sanitize_handler_metadata(return_metadata)
        return Code(200, return_metadata, "Experiment deleted")

    @staticmethod
    def resume_experiment_job(org_name, experiment_id, job_id, kind, parent_job_id=None, specs=None, name=None, description=None, num_gpu=-1, platform=None):
        """
        org_name: str
        experiment_id: str, uuid corresponding to experiment
        job_id: str, uuid corresponding to a train job
        Returns:
        201 with [job_id] if job resumed and added to queue
        400 with [] if job_id does not correspond to a train action or if it cannot be resumed
        404 with [] if experiment/job_id not found
        """
        handler_metadata = resolve_metadata(kind, experiment_id)
        if not handler_metadata:
            return Code(404, [], "Experiment not found")

        if not check_write_access(org_name, experiment_id, kind="experiments"):
            return Code(404, [], "Experiment not found")

        job_metadata = stateless_handlers.get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, None, "job trying to resume not found")
        action = job_metadata.get("action", "")
        action = infer_action_from_job(experiment_id, job_id)
        status = job_metadata.get("status", "")
        if status != "Paused":
            return Code(400, [], f"Job status should be paused, not {status}")
        if action not in ("train", "retrain"):
            return Code(400, [], f"Action should be train, retrain, not {action}")

        user_id = handler_metadata.get("user_id")
        if not user_id:
            return Code(404, [], "User ID couldn't be found in the experiment metadata. Try creating the experiment again")

        msg = ""
        try:
            stateless_handlers.update_job_status(experiment_id, job_id, status="Resuming", kind=kind + "s")
            if is_job_automl(user_id, org_name, job_id):
                msg = "AutoML "
                AutoMLHandler.resume(user_id, org_name, experiment_id, job_id, handler_metadata, name=name)
            else:
                # Create a job and run it
                if not specs:
                    print("Loading existing specs from paused job", file=sys.stderr)
                    spec_json_path = os.path.join(get_handler_spec_root(user_id, org_name, experiment_id), f"{job_id}-{action}-spec.json")
                    specs = safe_load_file(spec_json_path)
                if not parent_job_id:
                    print("Loading existing parent_job_id from paused job", file=sys.stderr)
                    parent_job_id = handler_metadata.get('parent_job_id', None)
                job_context = create_job_context(parent_job_id, "train", job_id, experiment_id, user_id, org_name, kind, handler_metadata=handler_metadata, specs=specs, name=name, description=description, num_gpu=num_gpu, platform=platform)
                on_new_job(job_context)
            return Code(200, {"message": f"{msg}Action for job {job_id} resumed"})
        except:
            print(traceback.format_exc(), file=sys.stderr)
            return Code(400, [], "Action cannot be resumed")

    @staticmethod
    def automl_details(org_name, experiment_id, job_id):
        """Compiles meaningful results of an AutoML run"""
        try:
            handler_metadata = resolve_metadata("experiment", experiment_id)
            user_id = handler_metadata.get("user_id")
            root = stateless_handlers.get_jobs_root(user_id, org_name)
            jobdir = os.path.join(root, job_id)
            automl_controller_data = safe_load_file(os.path.join(jobdir, "controller.json"))

            automl_interpretable_result = {}

            # Get current experiment id
            current_rec = safe_load_file(os.path.join(jobdir, "current_rec.json"))
            if not current_rec:
                current_rec = 0
            automl_interpretable_result["current_experiment_id"] = current_rec

            # Get per experiment result and status
            automl_interpretable_result["experiments"] = {}
            for experiment_details in automl_controller_data:
                automl_interpretable_result["metric"] = experiment_details.get("metric")
                exp_id = experiment_details.get("id")
                automl_interpretable_result["experiments"][exp_id] = {}
                automl_interpretable_result["experiments"][exp_id]["result"] = experiment_details.get("result")
                automl_interpretable_result["experiments"][exp_id]["status"] = experiment_details.get("status")

            # Get the best experiment id
            if os.path.exists(os.path.join(jobdir, "best_model")):
                rec_files = glob.glob(os.path.join(jobdir, "best_model", "recommendation*.yaml"))
                if rec_files:
                    experiment_name = os.path.splitext(os.path.basename(rec_files[0]))[0]
                    experiment_id = experiment_name.split("_")[1]
                    automl_interpretable_result["best_experiment_id"] = int(experiment_id)
            return Code(200, automl_interpretable_result, "AutoML results compiled")
        except:
            print(traceback.format_exc(), file=sys.stderr)
            return Code(400, [], "Error in constructing AutoML results")
