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
import datetime
import glob
import json
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import traceback
import uuid

from handlers import ngc_handler, stateless_handlers
from handlers.automl_handler import AutoMLHandler
from handlers.ds_upload import DS_CHANGE_PERMISSIONS, _extract_images, write_dir_contents
from handlers.encrypt import NVVaultEncryption
from handlers.medical.dicom_web_client import DicomWebClient
from handlers.medical.helpers import CapGpuUsage, download_from_url, validate_medical_bundle
from handlers.medical_dataset_handler import MEDICAL_DATASET_ACTIONS, MedicalDatasetHandler
from handlers.medical_model_handler import MedicalModelHandler
# TODO: force max length of code line to 120 chars
from handlers.stateless_handlers import (check_read_access, check_write_access, get_handler_spec_root, get_root,
                                         infer_action_from_job, is_valid_uuid4, list_all_job_metadata,
                                         resolve_existence, resolve_metadata, resolve_root, safe_dump_file,
                                         safe_load_file)
from handlers.tis_handler import TISHandler
from handlers.utilities import (AUTOML_DISABLED_NETWORKS, VALID_DSTYPES, VALID_MODEL_DOWNLOAD_TYPE, VALID_NETWORKS,
                                Code, download_base_experiment, download_dataset, get_medical_bundle_path,
                                prep_tis_model_repository, read_network_config)
from job_utils import executor as jobDriver
from job_utils.workflow_driver import create_job_context, on_delete_job, on_new_job
from specs_utils import csv_to_json_schema
from utils.utils import create_folder_with_permissions, run_system_command
from werkzeug.datastructures import FileStorage

# Identify if workflow is on NGC
NGC_RUNNER = os.getenv("NGC_RUNNER", "False")


# Helpers
def resolve_job_existence(user_id, kind, handler_id, job_id):
    """Return whether job_id.json exists in jobs_metadata folder or not"""
    if kind not in ["dataset", "experiment"]:
        return False
    metadata_path = os.path.join(stateless_handlers.get_root(), user_id, kind + "s", handler_id, "jobs_metadata", job_id + ".json")
    return os.path.exists(metadata_path)


def resolve_metadata_with_jobs(user_id, kind, handler_id):
    """Reads job_id.json in jobs_metadata folder and return it's contents"""
    handler_id = "*" if handler_id in ("*", "all") else handler_id
    metadata = {} if handler_id == "*" else resolve_metadata(user_id, kind, handler_id)
    if metadata or handler_id == "*":
        metadata["jobs"] = []
        job_metadatas_root = resolve_root(user_id, kind, handler_id) + "/jobs_metadata/"
        for json_file in glob.glob(job_metadatas_root + "*.json"):
            job_meta = stateless_handlers.safe_load_file(json_file)
            metadata["jobs"].append(job_meta)
        return metadata
    return {}


def get_user_experiments(user_id):
    """Returns a list of experiments that are available for the given user_id"""
    user_root = stateless_handlers.get_root() + f"{user_id}/experiments/"
    experiments, ret_lst = [], []
    if os.path.isdir(user_root):
        experiments = os.listdir(user_root)
    for experiment in experiments:
        if resolve_existence(user_id, "experiment", experiment):
            ret_lst.append(experiment)
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
    safe_dump_file(metadata_file, metadata)


def save_dicom_web_manifest(user_id, kind, handler_id, ds_location, ds_access_key, ds_filters):
    """Writes manifest.json with the contents pulled from the dicom web server."""
    manifest_file = os.path.join(resolve_root(user_id, kind, handler_id), "manifest.json")
    # TODO Need to discuss how to pass the user_id for dicom web server
    dicom_client = DicomWebClient(ds_location, "dlmed", ds_access_key)
    ret_code = dicom_client.create_dataset_manifest_file(manifest_file, dicom_filter=ds_filters)
    return ret_code


def create_blob_dataset(user_id, kind, handler_id):
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


def is_job_automl(user_id, experiment_id, job_id):
    """Returns if the job is automl-based job or not"""
    try:
        root = resolve_root(user_id, "experiment", experiment_id)
        jobdir = os.path.join(root, job_id)
        automl_signature = os.path.join(jobdir, "controller.log")
        return os.path.exists(automl_signature)
    except:
        print(traceback.format_exc(), file=sys.stderr)
        return False


def is_request_automl(user_id, handler_id, parent_job_id, action, kind):
    """Returns if the job requested is automl based train or not"""
    handler_metadata = resolve_metadata(user_id, kind, handler_id)
    if handler_metadata.get("automl_enabled", False) and action == "train":
        return True
    return False


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
            metadatas.append(stateless_handlers.get_handler_metadata(user_id, dataset_id))
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
        ds_pull = request_dict.get("pull", None)
        # Perform basic checks - valid type and format?
        if ds_type not in VALID_DSTYPES:
            msg = "Invalid dataset type"
            return Code(400, {}, msg)

        # For medical dataset, don't check the dataset type.
        if ds_format not in read_network_config(ds_type)["api_params"]["formats"] and ds_format != "medical":
            msg = "Incompatible dataset format and type"
            return Code(400, {}, msg)
        if ds_pull and not ds_pull.startswith("http"):
            msg = "Invalid pull URL"
            return Code(400, {}, msg)

        if request_dict.get("public", False):
            stateless_handlers.add_public_dataset(dataset_id)

        dataset_actions = get_dataset_actions(ds_type, ds_format) if ds_format != "medical" else MEDICAL_DATASET_ACTIONS

        # Create metadata dict and create some initial folders
        metadata = {"id": dataset_id,
                    "created_on": datetime.datetime.now().isoformat(),
                    "last_modified": datetime.datetime.now().isoformat(),
                    "name": request_dict.get("name", "My Dataset"),
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
                    "status": "not_present",
                    "pull": ds_pull
                    }
        # Encrypt the client secret if the dataset is a medical dataset and the vault agent has been set.
        if metadata["client_secret"] and ds_format == "medical":
            config_path = os.getenv("VAULT_SECRET_PATH", None)
            encryption = NVVaultEncryption(config_path)
            if encryption.check_config()[0]:
                metadata["client_secret"] = encryption.encrypt(metadata["client_secret"])
            elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                return Code(400, {}, "Cannot create dataset because vault service does not work.")

        # For MEDICAL dataset only
        if ds_format == "medical":
            client_url = request_dict.get("client_url", None)
            if client_url is None:
                msg = "Must provide a url to create a MEDICAL dataset."
                return Code(400, {}, msg)

            status, m = MedicalDatasetHandler.status_check(metadata)
            if not status:
                return Code(400, {}, f"Failed to connect to given URL {client_url} because of: \n {m}")

        if NGC_RUNNER == "True":
            ngc_handler.create_workspace(user_id, "datasets", dataset_id)
        stateless_handlers.make_root_dirs(user_id, "datasets", dataset_id)
        write_handler_metadata(user_id, "dataset", dataset_id, metadata)

        # Pull dataset in background if known URL
        if ds_pull:
            job_run_thread = threading.Thread(target=AppHandler.pull_dataset, args=(user_id, dataset_id,))
            job_run_thread.start()

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

            if key in ["name", "description", "version", "logo", "docker_env_vars"]:
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
        - 400 if dataset has running jobs / being used by a experiment and hence cannot be deleted
        """
        if not resolve_existence(user_id, "dataset", dataset_id):
            return Code(404, {}, "Dataset not found")

        if dataset_id not in get_user_datasets(user_id):
            return Code(404, {}, "Dataset cannot be deleted")

        # If dataset is being used by user's experiments.
        metadata_file_pattern = stateless_handlers.get_root() + f"{user_id}/experiments/**/metadata.json"
        metadata_files = glob.glob(metadata_file_pattern)
        for metadata_file in metadata_files:
            metadata = safe_load_file(metadata_file)
            train_datasets = metadata.get("train_datasets", [])
            if isinstance(train_datasets, list):
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

        # Validate if dataset exists
        meta_file = stateless_handlers.get_handler_metadata_file(user_id, dataset_id, "datasets")
        if not os.path.exists(meta_file):
            return Code(404, {}, "Dataset is already deleted.")

        # Remove metadata file to signify deletion
        os.remove(meta_file)
        dataset_root = stateless_handlers.get_handler_root(user_id, 'datasets', dataset_id, None)
        dataset_workspace_root = stateless_handlers.get_handler_root(user_id, 'datasets', dataset_id, None, ngc_runner_fetch=True)
        if NGC_RUNNER == "True":
            ngc_handler.delete_workspace(user_id, dataset_id)

        deletion_command = f"rm -rf {dataset_root} {dataset_workspace_root}"
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

        metadata = resolve_metadata(user_id, "dataset", dataset_id)
        if metadata.get("format") == "medical":
            return Code(404, metadata, "Uploading external data is not supported for MEDICAL Dataset")

        try:
            metadata["status"] = "in_progress"
            write_handler_metadata(user_id, "dataset", dataset_id, metadata)
            # Save tar file at the dataset root
            handler_root = stateless_handlers.get_handler_root(user_id, "datasets", dataset_id, None, ngc_runner_fetch=True)
            root = f"{handler_root}"
            if NGC_RUNNER == "True":
                temp_dir = tempfile.TemporaryDirectory().name  # pylint: disable=R1732
                root = f"{temp_dir}/{handler_root}"
            create_folder_with_permissions(root)
            tar_path = f"{root}/{str(uuid.uuid4())}.tar"
            file_tgz.save(tar_path)

            def upload_dataset_thread():
                print("Uploading dataset to server", file=sys.stderr)

                print("Extracting images from data tarball file", file=sys.stderr)
                _extract_images(tar_path, root)
                if metadata.get("type") == "semantic_segmentation":
                    write_dir_contents(os.path.join(root, "images"), os.path.join(root, "images.txt"))
                    if metadata.get("format") == "unet":
                        write_dir_contents(os.path.join(root, "masks"), os.path.join(root, "masks.txt"))
                if NGC_RUNNER == "True":
                    result = ngc_handler.upload_to_ngc_workspace(dataset_id, root, "/")
                    if result.stdout:
                        print("Workspace upload results", result.stdout.decode("utf-8"), file=sys.stderr)  # Decode and print the standard error
                    if result.stderr:
                        print("Workspace upload error", result.stderr.decode("utf-8"), file=sys.stderr)  # Decode and print the standard error
                    shutil.rmtree(temp_dir)
                    assert not result.stderr

                print("Extraction complete", file=sys.stderr)

                if metadata.get("type", "") in DS_CHANGE_PERMISSIONS.keys():
                    print("Changing permissions of necessary files", file=sys.stderr)
                    permissions_changed = DS_CHANGE_PERMISSIONS[metadata.get("type")](user_id, metadata)
                    print("Permission changes complete", file=sys.stderr)
                    if not permissions_changed:
                        print("Error while changing permissions of dataset files", file=sys.stderr)

                metadata["status"] = "present"
                write_handler_metadata(user_id, "dataset", dataset_id, metadata)

            thread = threading.Thread(target=upload_dataset_thread)
            thread.start()
            return Code(201, {}, "Server recieved file and upload process started")
        except:
            print(traceback.format_exc(), file=sys.stderr)
            return Code(404, [], "Exception caught during upload")

    @staticmethod
    def pull_dataset(user_id, dataset_id):
        """
        user_id: str, uuid
        dataset_id: str, uuid
        """
        file_path = download_dataset(user_id, dataset_id)
        with open(file_path, 'rb') as f:
            file_tgz = FileStorage(f)
            AppHandler.upload_dataset(user_id, dataset_id, file_tgz)

    # Spec API

    @staticmethod
    def get_spec_schema(user_id, handler_id, action, kind):
        """
        user_id: str, uuid
        handler_id: str, uuid corresponding to experiment/dataset
        action: str, a valid Action for a dataset
        kind: str, one of ["experiment","dataset"]
        Returns:
        Code object
        - 200 with spec in a json-schema format
        - 404 if experiment/dataset not found / user cannot access
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

        if kind == "dataset" and metadata.get("format") == "medical":
            json_schema = MedicalDatasetHandler.get_schema(action)
            return Code(200, json_schema, "Schema retrieved")

        if kind == "experiment" and metadata.get("type").lower() == "medical":
            json_schema = MedicalModelHandler.get_schema(action)
            return Code(200, json_schema, "Schema retrieved")

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
            # For each train dataset for the experiment
            metadata = resolve_metadata(user_id, kind, handler_id)
            if network == "detectnet_v2" and action == "train":
                if not metadata.get("train_datasets", []):
                    return Code(400, [], "Assign datasets before getting train specs")
            for train_ds in metadata.get("train_datasets", []):
                # Obtain class list from classes.json
                classes_json = os.path.join(stateless_handlers.get_handler_root(user_id, "datasets", train_ds, None, ngc_runner_fetch=True), "classes.json")
                if not os.path.exists(classes_json):
                    if network == "detectnet_v2" and action == "train":
                        return Code(400, [], "Classes.json is not part of the dataset. Either provide them or wait for dataset convert to complete before getting specs")
                    continue
                with open(classes_json, "r", encoding='utf-8') as f:
                    inferred_class_names += json.loads(f.read())  # It is a list
            inferred_class_names = list(set(inferred_class_names))
        json_schema = csv_to_json_schema.convert(CSV_PATH, inferred_class_names)
        return Code(200, json_schema, "Schema retrieved")

    @staticmethod
    def get_spec_schema_without_handler_id(user_id, network, format, action, train_datasets):
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
                classes_json = os.path.join(stateless_handlers.get_handler_root(user_id, "datasets", train_ds, None, ngc_runner_fetch=True), "classes.json")
                if not os.path.exists(classes_json):
                    continue
                with open(classes_json, "r", encoding='utf-8') as f:
                    inferred_class_names += json.loads(f.read())  # It is a list
            inferred_class_names = list(set(inferred_class_names))
        json_schema = csv_to_json_schema.convert(CSV_PATH, inferred_class_names)
        return Code(200, json_schema, "Schema retrieved")

    # Job API

    @staticmethod
    def job_run(user_id, handler_id, parent_job_id, action, kind, specs=None, name=None, description=None):
        """
        user_id: str, uuid
        handler_id: str, uuid corresponding to experiment/dataset
        parent_job_id: str, uuid
        action: str
        kind: str, one of ["experiment","dataset"]
        specs: dict spec for each action.
        name: str
        description: str
        Returns:
        201 with str where str is a uuid for job if job is successfully queued
        400 with [] if action was not executed successfully
        404 with [] if dataset/experiment/parent_job_id/actions not found/standards are not met
        """
        if not resolve_existence(user_id, kind, handler_id):
            return Code(404, [], f"{handler_id} {kind} doesn't exist")

        if not check_write_access(user_id, handler_id):
            return Code(404, [], f"{user_id} has no write access to {handler_id}")

        if not action:
            return Code(404, [], "action not sent")

        handler_metadata = resolve_metadata(user_id, kind, handler_id)

        if kind == "experiment" and handler_metadata.get("type").lower() == "medical":
            if action not in handler_metadata.get("actions", []):
                return Code(404, {}, "Action not found")

            if not isinstance(specs, dict):
                return Code(404, [], f"{specs} must be a dictionary. Received {type(specs)}")

            default_spec = read_network_config(handler_metadata["network_arch"])["spec_params"].get(action, {})
            nested_update(specs, default_spec, allow_overwrite=False)

            if action == "inference" and handler_metadata.get("network_arch") != "medical_segmentation":
                return MedicalModelHandler.run_inference(user_id, handler_id, handler_metadata, specs)

            # regular async jobs
            if action == "annotation":
                all_metadata = list_all_job_metadata(user_id, handler_id)
                if [m for m in all_metadata if m["action"] == "annotation" and m["status"] in ("Running", "Pending")]:
                    return Code(400, [], "There is one running or pending annotation/continual learning job in this experiment. Please stop it first.")
                if handler_metadata.get("eval_dataset", None) is None:
                    return Code(404, {}, "Annotation job requires eval dataset in the model metadata.")

        if kind == "dataset" and handler_metadata.get("format") == "medical":
            return MedicalDatasetHandler.run_job(user_id, handler_id, handler_metadata, action, specs)

        if "base_experiment" in handler_metadata.keys():
            base_experiment_ids = handler_metadata["base_experiment"]
            for base_experiment_id in base_experiment_ids:
                if base_experiment_id:
                    if stateless_handlers.get_base_experiment_metadata(base_experiment_id).get("ngc_path", None):
                        print(f"Starting download of {base_experiment_id}", file=sys.stderr)
                        ptm_download_thread = threading.Thread(target=download_base_experiment, args=(base_experiment_id,))
                        ptm_download_thread.start()

        try:
            job_id = str(uuid.uuid4())
            create_folder_with_permissions(f"/shared/users/{user_id}/{kind}s/{handler_id}/{job_id}")
            if specs:
                spec_json_path = os.path.join(get_handler_spec_root(user_id, handler_id), f"{job_id}-{action}-spec.json")
                safe_dump_file(spec_json_path, specs)
            msg = ""
            if is_request_automl(user_id, handler_id, parent_job_id, action, kind):
                AutoMLHandler.start(user_id, handler_id, job_id, handler_metadata)
                msg = "AutoML "
            else:
                job_context = create_job_context(parent_job_id, action, job_id, handler_id, user_id, kind, specs=specs, name=name, description=description)
                on_new_job(job_context)
            return Code(201, job_id, f"{msg}Job scheduled")
        except:
            print(traceback.format_exc(), file=sys.stderr)
            return Code(404, [], "Exception in job_run fn")

    @staticmethod
    def job_list(user_id, handler_id, kind):
        """
        user_id: str, uuid
        handler_id: str, uuid corresponding to experiment/dataset
        kind: str, one of ["experiment","dataset"]
        Returns:
        200, list(dict) - each dict follows JobResultSchema if found
        404, [] if not found
        """
        if handler_id not in ("*", "all") and not resolve_existence(user_id, kind, handler_id):
            return Code(404, [], f"{kind} not found")

        if handler_id not in ("*", "all") and not check_read_access(user_id, handler_id):
            return Code(404, [], f"{kind} not found")

        return_metadata = resolve_metadata_with_jobs(user_id, kind, handler_id).get("jobs", [])
        return Code(200, return_metadata, "Jobs retrieved")

    @staticmethod
    def job_cancel(user_id, handler_id, job_id, kind):
        """
        user_id: str, uuid
        handler_id: str, uuid corresponding to experiment/dataset
        job_id: str, uuid corresponding to job to be cancelled
        kind: str, one of ["experiment","dataset"]
        Returns:
        200, [job_id] - if job can be cancelled
        404, [] if not found
        """
        if not resolve_existence(user_id, kind, handler_id):
            return Code(404, [], f"{kind} not found")

        if not check_write_access(user_id, handler_id):
            return Code(404, [], f"{kind} not found")

        if not resolve_job_existence(user_id, kind, handler_id, job_id):
            return Code(404, [], "job trying to cancel not found")

        if is_job_automl(user_id, handler_id, job_id):
            stateless_handlers.update_job_status(user_id, handler_id, job_id, status="Canceled")
            return AutoMLHandler.stop(user_id, handler_id, job_id)

        # If job is error / done, then cancel is NoOp
        job_metadata = stateless_handlers.get_handler_job_metadata(user_id, handler_id, job_id)
        job_status = job_metadata.get("status", "Error")

        if job_status in ["Error", "Done", "Canceled"]:
            return Code(404, [], "incomplete job not found")

        if job_status == "Pending":
            on_delete_job(user_id, handler_id, job_id)
            stateless_handlers.update_job_status(user_id, handler_id, job_id, status="Canceled")
            return Code(200, job_id, "Pending job cancelled")

        if job_status == "Running":
            try:
                # Delete K8s job
                specs = job_metadata.get("specs", None)
                use_ngc = not (specs and "cluster" in specs and specs["cluster"] == "local")
                jobDriver.delete(job_id, use_ngc=use_ngc)
                stateless_handlers.update_job_status(user_id, handler_id, job_id, status="Canceled")
                return Code(200, job_id, "Running job cancelled")
            except:
                print(traceback.format_exc(), file=sys.stderr)
                return Code(404, [], "job not found in platform")

        else:
            return Code(404, [], "job status not found")

    @staticmethod
    def job_retrieve(user_id, handler_id, job_id, kind):
        """
        user_id: str, uuid
        handler_id: str, uuid corresponding to experiment/dataset
        job_id: str, uuid corresponding to job to be retrieved
        kind: str, one of ["experiment","dataset"]
        Returns:
        200, dict following JobResultSchema - if job found
        404, {} if not found
        """
        if not resolve_existence(user_id, kind, handler_id):
            return Code(404, {}, "Dataset not found")

        if not check_read_access(user_id, handler_id):
            return Code(404, {}, "Dataset not found")

        if not resolve_job_existence(user_id, kind, handler_id, job_id):
            return Code(404, {}, "Job trying to retrieve not found")

        if is_job_automl(user_id, handler_id, job_id):
            return AutoMLHandler.retrieve(user_id, handler_id, job_id)
        path = os.path.join(stateless_handlers.get_root(), user_id, kind + "s", handler_id, "jobs_metadata", job_id + ".json")
        job_meta = stateless_handlers.safe_load_file(path)
        return Code(200, job_meta, "Job retrieved")

    # Delete job
    @staticmethod
    def job_delete(user_id, handler_id, job_id, kind):
        """
        user_id: str, uuid
        handler_id: str, uuid corresponding to experiment/dataset
        job_id: str, uuid corresponding to job to be deleted
        kind: str, one of ["experiment","dataset"]
        Returns:
        200, [job_id] - if job can be deleted
        404, [] if not found
        """
        if not resolve_existence(user_id, kind, handler_id):
            return Code(404, [], f"{kind} not found")

        if not check_write_access(user_id, handler_id):
            return Code(404, [], f"{kind} not found")

        if not resolve_job_existence(user_id, kind, handler_id, job_id):
            return Code(404, [], "job trying to delete not found")
        try:
            # If job is running, cannot delete
            job_metadata = stateless_handlers.get_handler_job_metadata(user_id, handler_id, job_id)
            if job_metadata.get("status", "Error") in ["Running", "Pending"]:
                return Code(400, [], "job cannot be deleted")
            # Delete job metadata
            job_metadata_path = os.path.join(stateless_handlers.get_handler_jobs_metadata_root(user_id, handler_id), job_id + ".json")
            if os.path.exists(job_metadata_path):
                os.remove(job_metadata_path)
            # Delete job logs
            job_log_path = os.path.join(stateless_handlers.get_handler_log_root(user_id, handler_id), job_id + ".txt")
            if os.path.exists(job_log_path):
                os.remove(job_log_path)
            # Delete the job directory in the background
            job_handler_root = stateless_handlers.get_handler_root(user_id, kind + "s", handler_id, None)
            job_handler_workspace_root = stateless_handlers.get_handler_root(user_id, kind + "s", handler_id, None, ngc_runner_fetch=True)
            deletion_command = f"rm -rf {os.path.join(job_handler_root, job_id)} {os.path.join(job_handler_workspace_root, job_id)}"
            targz_path = os.path.join(job_handler_root, job_id + ".tar.gz")
            if os.path.exists(targz_path):
                deletion_command += f"; rm -rf {job_handler_root}/{job_id}*tar.gz"
            delete_thread = threading.Thread(target=run_system_command, args=(deletion_command,))
            delete_thread.start()
            return Code(200, [job_id], "job deleted")
        except:
            print(traceback.format_exc(), file=sys.stderr)
            return Code(400, [], "job cannot be deleted")

    # Download experiment job
    @staticmethod
    def job_download(user_id, handler_id, job_id, kind, file_lists=None, best_model=None, latest_model=None, tar_files=True, export_type="tao"):
        """
        user_id: str, uuid
        handler_id: str, uuid corresponding to experiment/dataset
        job_id: str, uuid corresponding to job to be downloaded
        kind: str, one of ["experiment","dataset"]
        export_type: str, one of ["medical_bundle", "tao"]
        Returns:
        200, Path to a tar.gz created from the job directory
        404, None if not found
        """
        if not resolve_existence(user_id, kind, handler_id):
            return Code(404, None, f"{kind} not found")
        if not check_read_access(user_id, handler_id):
            return Code(404, None, f"{kind} not found")
        if not resolve_job_existence(user_id, kind, handler_id, job_id):
            return Code(404, None, "job trying to download not found")
        try:
            if export_type not in VALID_MODEL_DOWNLOAD_TYPE:
                return Code(404, None, f"Export format {export_type} not found.")
            root = stateless_handlers.get_handler_root(user_id, kind + "s", handler_id, None)
            workspace_root = stateless_handlers.get_handler_root(user_id, kind + "s", handler_id, None, ngc_runner_fetch=True)
            if export_type == "medical_bundle":
                local_job_root = os.path.join(root, job_id)
                ngc_job_root = os.path.join(workspace_root, job_id)
                local_path_status = get_medical_bundle_path(local_job_root)
                ngc_path_status = get_medical_bundle_path(ngc_job_root)
                if local_path_status.code != 201 and ngc_path_status.code != 201:
                    return local_path_status
                bundle_path = local_path_status.data if local_path_status.code == 201 else ngc_path_status.data
                valid_root = root if local_path_status.code == 201 else workspace_root
                out_tar = os.path.join(root, job_id + ".tar.gz")
                command = f"cd {valid_root} ; tar -zcvf {out_tar} {job_id}/{bundle_path}; cd -"
                run_system_command(command)
                if os.path.exists(out_tar):
                    return Code(200, out_tar, "job downloaded")
                return Code(404, None, "job output not found")

            # Following is for `if export_type == "tao":`
            # Copy job logs from root/logs/<job_id>.txt to root/<job_id>/logs_from_toolkit.txt
            out_tar = os.path.join(root, job_id + ".tar.gz")
            files = [os.path.join(root, job_id), os.path.join(workspace_root, job_id)]
            if file_lists or best_model or latest_model:
                files = []
                for file in file_lists:
                    if os.path.exists(os.path.join(root, file)):
                        files.append(os.path.join(root, file))
                    if os.path.exists(os.path.join(workspace_root, file)):
                        files.append(os.path.join(workspace_root, file))
                handler_metadata = stateless_handlers.get_handler_metadata(user_id, handler_id)
                handler_job_metadata = stateless_handlers.get_handler_job_metadata(user_id, handler_id, job_id)
                action = handler_job_metadata.get("action", "")
                epoch_number_dictionary = handler_metadata.get("checkpoint_epoch_number", {})
                best_checkpoint_epoch_number = epoch_number_dictionary.get(f"best_model_{job_id}", 0)
                latest_checkpoint_epoch_number = epoch_number_dictionary.get(f"latest_model_{job_id}", 0)
                if (not best_model) and latest_model:
                    best_checkpoint_epoch_number = latest_checkpoint_epoch_number
                network = handler_metadata.get("network_arch", "")
                if network in ("bpnet", "classification_pyt", "detectnet_v2", "fpenet", "pointpillars", "efficientdet_tf1", "faster_rcnn", "mask_rcnn", "segformer", "unet"):
                    format_epoch_number = str(best_checkpoint_epoch_number)
                else:
                    format_epoch_number = f"{best_checkpoint_epoch_number:03}"
                if best_model or latest_model:
                    job_root = os.path.join(workspace_root, job_id)
                    if handler_metadata.get("automl_enabled") is True and action == "train":
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

            if files == [os.path.join(root, job_id), os.path.join(workspace_root, job_id)]:
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
                        tar.add(file_path, arcname=file_path.replace(root, "", 1).replace(workspace_root, "", 1))
                return Code(200, out_tar, "selective files of job downloaded")

            if files and os.path.exists(os.path.join(root, files[0])):
                return Code(200, os.path.join(root, files[0]), "single file of job downloaded")
            if files and os.path.exists(os.path.join(workspace_root, files[0])):
                return Code(200, os.path.join(workspace_root, files[0]), "single file of job downloaded")
            return Code(404, None, "job output not found")

        except:
            print(traceback.format_exc(), file=sys.stderr)
            return Code(404, None, "job output not found")

    @staticmethod
    def job_list_files(user_id, handler_id, job_id, retrieve_logs, retrieve_specs, kind):
        """
        user_id: str, uuid
        handler_id: str, uuid corresponding to experiment/dataset
        job_id: str, uuid corresponding to job for which the files need to be listed
        kind: str, one of ["experiment","dataset"]
        Returns:
        200, list(str) - list of file paths wtih respect to the job
        404, None if no files are found
        """
        if not resolve_existence(user_id, kind, handler_id):
            return Code(404, [], f"{kind} not found")

        if not check_read_access(user_id, handler_id):
            return Code(404, [], f"{kind} not found")

        pvc_files = stateless_handlers.get_job_files(user_id, handler_id, job_id, retrieve_logs, retrieve_specs)
        workspace_files = []
        if NGC_RUNNER == "True":
            workspace_files = stateless_handlers.get_job_files(user_id, handler_id, job_id, retrieve_logs, retrieve_specs, ngc_runner_fetch=True)
        files = pvc_files + workspace_files
        if files:
            return Code(200, files, "Job files retrieved")
        return Code(200, files, "No downloadable files for this job is found")

    # Experiment API
    @staticmethod
    def list_experiments(user_id, user_only=False):
        """
        user_id: str, uuid
        Returns:
        list(dict) - list of experiments accessible by user where each element is metadata of a experiment
        """
        # Collect all metadatas
        metadatas = []
        experiments = get_user_experiments(user_id)
        for experiment_id in list(set(experiments)):
            metadatas.append(stateless_handlers.get_handler_metadata(user_id, experiment_id))
        if not user_only:
            public_experiments_metadata = stateless_handlers.get_public_experiments()
            metadatas += public_experiments_metadata
        return metadatas

    @staticmethod
    def create_experiment(user_id, request_dict, experiment_id=None):
        """
        user_id: str, uuid
        request_dict: dict following ExperimentReqSchema
            - network_arch is required
            - encryption_key is required (not enforced)
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
        if str(mdl_nw).startswith("medical_"):
            mdl_type = "medical"

        # Create metadata dict and create some initial folders
        # Initially make datasets, base_experiment None
        metadata = {"id": experiment_id,
                    "created_on": datetime.datetime.now().isoformat(),
                    "last_modified": datetime.datetime.now().isoformat(),
                    "name": request_dict.get("name", "My Experiment"),
                    "description": request_dict.get("description", "My Experiments"),
                    "version": request_dict.get("version", "1.0.0"),
                    "logo": request_dict.get("logo", "https://www.nvidia.com"),
                    "ngc_path": request_dict.get("ngc_path", ""),
                    "is_ptm_backbone": request_dict.get("is_ptm_backbone", True),
                    "encryption_key": request_dict.get("encryption_key", "tlt_encode"),
                    "read_only": request_dict.get("read_only", False),
                    "public": request_dict.get("public", False),
                    "network_arch": mdl_nw,
                    "type": mdl_type,
                    "dataset_type": read_network_config(mdl_nw)["api_params"]["dataset_type"],
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
                    "automl_enabled": False,
                    "automl_algorithm": None,
                    "metric": None,
                    "realtime_infer": False,
                    "realtime_infer_support": False,
                    "realtime_infer_endpoint": None,
                    "realtime_infer_model_name": None,
                    "realtime_infer_request_timeout": request_dict.get("realtime_infer_request_timeout", 60),
                    "model_params": request_dict.get("model_params", {}),
                    "automl_add_hyperparameters": "[]",
                    "automl_remove_hyperparameters": "[]"
                    }

        if request_dict.get("automl_enabled", False):
            if mdl_nw not in AUTOML_DISABLED_NETWORKS:
                metadata["automl_enabled"] = True
                metadata["automl_algorithm"] = request_dict.get("automl_algorithm", "bayesian")
                metadata["metric"] = request_dict.get("metric", "map")
                metadata["automl_add_hyperparameters"] = request_dict.get("automl_add_hyperparameters", "[]")
                metadata["automl_remove_hyperparameters"] = request_dict.get("automl_remove_hyperparameters", "[]")
                # AutoML optional params
                if request_dict.get("automl_max_recommendations"):
                    metadata["automl_max_recommendations"] = request_dict.get("automl_max_recommendations")
                if request_dict.get("automl_delete_intermediate_ckpt"):
                    metadata["automl_delete_intermediate_ckpt"] = request_dict.get("automl_delete_intermediate_ckpt")
                if request_dict.get("override_automl_disabled_params"):
                    metadata["override_automl_disabled_params"] = request_dict.get("override_automl_disabled_params")
                if request_dict.get("automl_R"):
                    metadata["automl_R"] = request_dict.get("automl_R")
                if request_dict.get("automl_nu"):
                    metadata["automl_nu"] = request_dict.get("automl_nu")
                if request_dict.get("epoch_multiplier"):
                    metadata["epoch_multiplier"] = request_dict.get("epoch_multiplier")
            else:
                return Code(400, {}, "automl_enabled cannot be True for unsupported network")

        # Update datasets and base_experiments if given
        for key in ["train_datasets", "eval_dataset", "inference_dataset", "calibration_dataset", "base_experiment", "realtime_infer"]:
            if key not in request_dict.keys():
                continue
            value = request_dict[key]
            if stateless_handlers.experiment_update_handler_attributes(user_id, metadata, key, value):
                metadata[key] = value
            else:
                return Code(400, {}, f"Provided {key} cannot be added")

        def clean_on_error(experiment_id=experiment_id):
            handle_dir = os.path.join(get_root(), user_id, "experiments", experiment_id)
            if os.path.exists(handle_dir):
                shutil.rmtree(handle_dir, ignore_errors=True)
            if NGC_RUNNER == "True":
                ngc_handler_dir = handle_dir.replace(get_root(), get_root(ngc_runner_fetch=True))
                if os.path.exists(ngc_handler_dir):
                    shutil.rmtree(ngc_handler_dir, ignore_errors=True)

        if mdl_type == "medical" and metadata["network_arch"] == "medical_custom":
            no_ptm = (metadata["base_experiment"] is None) or (len(metadata["base_experiment"]) == 0)
            if no_ptm:
                # If base_experiment is not provided, then we will need to create a model to host the files downloaded from NGC.
                # This is a temporary solution until we have a better way to handle this.
                bundle_url = request_dict.get("bundle_url", None)
                if bundle_url is None:
                    return Code(400, {}, "Either `bundle_url` or `ngc_path` needs to be defined for MEDICAL Custom Model.")
                base_experiment_id = str(uuid.uuid4())
                ptm_metadata = metadata.copy()
                ptm_metadata["id"] = base_experiment_id
                ptm_metadata["name"] = "base_experiment_" + metadata["name"]
                ptm_metadata["description"] = " PTM auto-generated. " + metadata["description"]
                ptm_metadata["train_datasets"] = []
                ptm_metadata["eval_dataset"] = None
                ptm_metadata["inference_dataset"] = None
                ptm_metadata["realtime_infer"] = False
                stateless_handlers.make_root_dirs(user_id, "experiments", base_experiment_id)
                write_handler_metadata(user_id, "experiment", base_experiment_id, ptm_metadata)
                # Download it from the provided url
                download_from_url(bundle_url, base_experiment_id)

                # the base_experiment is downloaded, now we need to make sure it is correct.
                ptm_file = validate_medical_bundle(base_experiment_id)

                if (ptm_file is None) or (not os.path.isdir(ptm_file)):
                    clean_on_error(experiment_id=base_experiment_id)
                    return Code(400, {}, "Failed to download base experiment, or the provided bundle does not follow MEDICAL bundle format.")
                if NGC_RUNNER == "True":
                    ngc_handler.create_workspace(user_id, "experiments", base_experiment_id)
                    temp_dir = tempfile.mkdtemp()
                    shutil.move(ptm_file, temp_dir)
                    workspace_id = ngc_handler.get_workspace_id(user_id, base_experiment_id)
                    if workspace_id is None:
                        clean_on_error(experiment_id=base_experiment_id)
                        shutil.rmtree(temp_dir)
                        print("Failed to get workspace id just created.", file=sys.stderr)
                        return Code(400, {}, "Failed to create workspace.")
                    result = ngc_handler.upload_to_ngc_workspace(workspace_id, temp_dir, "/")
                    shutil.rmtree(temp_dir)
                    if result.stdout:
                        print("Workspace upload results", result.stdout.decode("utf-8"), file=sys.stderr)  # Decode and print the standard error
                    if result.stderr:
                        print("Workspace upload error", result.stderr.decode("utf-8"), file=sys.stderr)  # Decode and print the standard error
                        ngc_handler.delete_workspace(user_id,  base_experiment_id)
                        clean_on_error(experiment_id=base_experiment_id)
                        print("Failed to upload workspace id", file=sys.stderr)
                        return Code(400, {}, "Failed to create workspace.")

                ptm_metadata["base_experiment_pull_complete"] = "present"
                write_handler_metadata(user_id, "experiment", base_experiment_id, ptm_metadata)
                metadata["base_experiment"] = [base_experiment_id]
            else:
                base_experiment_id = metadata["base_experiment"][0] if isinstance(metadata["base_experiment"], list) else metadata["base_experiment"]
                ptm_file = validate_medical_bundle(base_experiment_id, ngc_runner_fetch=True)
                if (ptm_file is None) or (not os.path.isdir(ptm_file)):
                    return Code(400, {}, "Provided base_experiment is not a MEDICAL bundle.")

        if mdl_type == "medical" and metadata["realtime_infer"]:
            base_experiment_id = metadata["base_experiment"][0]
            model_params = metadata["model_params"]
            job_id = None
            # Need to download the base_experiment to set up the TIS for realtime infer
            if stateless_handlers.get_base_experiment_metadata(base_experiment_id).get("ngc_path", None):
                download_base_experiment(base_experiment_id)
            else:
                additional_id_info = request_dict.get("additional_id_info", None)
                job_id = additional_id_info if is_valid_uuid4(additional_id_info) else None
                if not job_id:
                    return Code(400, {}, f"Non-NGC base_experiment {base_experiment_id} needs job_id in the request for path location")
            success, model_name, msg, bundle_metadata = prep_tis_model_repository(model_params, base_experiment_id, user_id, experiment_id, job_id=job_id)
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
            success, msg = CapGpuUsage.schedule(user_id, replicas)
            if not success:
                return Code(400, {}, msg)
            response = TISHandler.start(user_id, experiment_id, metadata, replicas)
            if response.code != 201:
                TISHandler.stop(experiment_id, metadata)
                CapGpuUsage.release_used(user_id, replicas)
                clean_on_error(experiment_id=experiment_id)
                return response
            metadata["realtime_infer_endpoint"] = response.data["pod_ip"]
            metadata["realtime_infer_model_name"] = model_name
            metadata["realtime_infer_support"] = True

        # Actual "creation" happens here...
        if NGC_RUNNER == "True":
            ngc_handler.create_workspace(user_id, "experiments", experiment_id)
        stateless_handlers.make_root_dirs(user_id, "experiments", experiment_id)
        write_handler_metadata(user_id, "experiment", experiment_id, metadata)

        # Read this metadata from saved file...
        return_metadata = resolve_metadata_with_jobs(user_id, "experiment", experiment_id)
        ret_Code = Code(201, return_metadata, "Experiment created")

        # TODO: may need to call "medical_triton_client" with dummy request to accelerate
        return ret_Code

    # Update existing experiment for user based on request dict
    @staticmethod
    def update_experiment(user_id, experiment_id, request_dict):
        """
        user_id: str, uuid
        experiment_id: str, uuid
        request_dict: dict following ExperimentReqSchema
        Returns:
        - 200 with metadata of updated experiment if successful
        - 404 if experiment not found / user cannot access
        - 400 if invalid update / experiment is read only
        """
        if not resolve_existence(user_id, "experiment", experiment_id):
            return Code(400, {}, "Experiment does not exist")

        if not check_write_access(user_id, experiment_id):
            return Code(400, {}, "User doesn't have write access to experiment")

        # if public is set to True => add it to public_experiments, if it is set to False => take it down
        # if public is not there, do nothing
        if request_dict.get("public", None):
            if request_dict["public"]:
                stateless_handlers.add_public_experiment(experiment_id)
            else:
                stateless_handlers.remove_public_experiment(experiment_id)

        metadata = resolve_metadata(user_id, "experiment", experiment_id)
        for key in request_dict.keys():

            # Cannot process the update, so return 400
            if key in ["network_arch", "experiment_params"]:
                if request_dict[key] != metadata.get(key):
                    msg = f"Cannot change experiment {key}"
                    return Code(400, {}, msg)

            if (key == "realtime_infer") and (request_dict[key] != metadata.get(key)):
                if request_dict[key] is False:
                    response = TISHandler.stop(experiment_id, metadata)
                    replicas = metadata.get("model_params", {}).get("replicas", 1)
                    CapGpuUsage.release_used(user_id, replicas)
                    if response.code != 201:
                        return response
                    metadata[key] = False
                else:
                    return Code(400, {}, f"Can only change {key} from True to False.")

            if key in ["name", "description", "version", "logo",
                       "ngc_path", "encryption_key", "read_only",
                       "metric", "public", "docker_env_vars", "is_ptm_backbone"]:
                requested_value = request_dict[key]
                if requested_value is not None:
                    metadata[key] = requested_value
                    metadata["last_modified"] = datetime.datetime.now().isoformat()

            if key in ["train_datasets", "eval_dataset", "inference_dataset", "calibration_dataset", "base_experiment", "checkpoint_choose_method", "checkpoint_epoch_number"]:
                value = request_dict[key]
                if stateless_handlers.experiment_update_handler_attributes(user_id, metadata, key, value):
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
                        metadata["automl_algorithm"] = request_dict.get("automl_algorithm", "bayesian")
                        metadata["metric"] = request_dict.get("metric", "map")
                        metadata["automl_add_hyperparameters"] = request_dict.get("automl_add_hyperparameters", "[]")
                        metadata["automl_remove_hyperparameters"] = request_dict.get("automl_remove_hyperparameters", "[]")

                        # AutoML optional params
                        if request_dict.get("automl_max_recommendations"):
                            metadata["automl_max_recommendations"] = request_dict.get("automl_max_recommendations")
                        if request_dict.get("automl_delete_intermediate_ckpt"):
                            metadata["automl_delete_intermediate_ckpt"] = request_dict.get("automl_delete_intermediate_ckpt")
                        if request_dict.get("override_automl_disabled_params"):
                            metadata["override_automl_disabled_params"] = request_dict.get("override_automl_disabled_params")
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
        write_handler_metadata(user_id, "experiment", experiment_id, metadata)
        # Read this metadata from saved file...
        return_metadata = resolve_metadata_with_jobs(user_id, "experiment", experiment_id)
        ret_Code = Code(200, return_metadata, "Experiment updated")
        return ret_Code

    @staticmethod
    def retrieve_experiment(user_id, experiment_id):
        """
        user_id: str, uuid
        experiment_id: str, uuid

        Returns:
        - 200 with metadata of retrieved experiment if successful
        - 404 if experiment not found / user cannot access
        """
        if experiment_id not in ("*", "all") and not resolve_existence(user_id, "experiment", experiment_id):
            return Code(404, {}, "Experiment not found")

        if experiment_id not in ("*", "all") and not check_read_access(user_id, experiment_id):
            return Code(404, {}, "Experiment not found")
        return_metadata = resolve_metadata_with_jobs(user_id, "experiment", experiment_id)
        return Code(200, return_metadata, "Experiment retrieved")

    @staticmethod
    def delete_experiment(user_id, experiment_id):
        """
        user_id: str, uuid
        experiment_id: str, uuid
        Returns:
        Code object
        - 200 with metadata of deleted experiment if successful
        - 404 if experiment not found / user cannot access
        - 400 if experiment cannot be deleted b/c (1) running jobs (2) public (3) read-only (4) TIS stop fails
        """
        if not resolve_existence(user_id, "experiment", experiment_id):
            return Code(404, {}, "Experiment not found")

        if experiment_id not in get_user_experiments(user_id):
            return Code(404, {}, "Experiment cannot be deleted")

        # If experiment is being used by user's experiments.
        metadata_file_pattern = stateless_handlers.get_root() + f"{user_id}/experiments/**/metadata.json"
        metadata_files = glob.glob(metadata_file_pattern)
        for metadata_file in metadata_files:
            metadata = safe_load_file(metadata_file)
            if experiment_id == metadata.get("base_experiment", None):
                return Code(400, {}, "Experiment in use as a base_experiment")
        # Check if any job running
        return_metadata = resolve_metadata_with_jobs(user_id, "experiment", experiment_id)
        for job in return_metadata["jobs"]:
            if job["status"] == "Running":
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
            CapGpuUsage.release_used(user_id, replicas)
            if response is not None and response.code != 201:
                return response

        experiment_root = stateless_handlers.get_handler_root(user_id, "experiments", experiment_id, None)
        experiment_workspace_root = stateless_handlers.get_handler_root(user_id, "experiments", experiment_id, None, ngc_runner_fetch=True)
        experiment_meta_file = stateless_handlers.get_handler_metadata_file(user_id, experiment_id, "experiments")
        if not os.path.exists(experiment_meta_file):
            return Code(404, {}, "Experiment is already deleted.")

        print(f"Removing experiment (meta): {experiment_meta_file}", file=sys.stderr)
        os.unlink(experiment_meta_file)

        # Remove the workspace if running on ngc
        print(f"Removing experiment folder: {experiment_root}", file=sys.stderr)
        if NGC_RUNNER == "True":
            ngc_handler.delete_workspace(user_id, experiment_id)

        deletion_command = f"rm -rf {experiment_root} {experiment_workspace_root}"
        delete_thread = threading.Thread(target=run_system_command, args=(deletion_command,))
        delete_thread.start()
        return Code(200, return_metadata, "Experiment deleted")

    @staticmethod
    def resume_experiment_job(user_id, experiment_id, job_id, parent_job_id, kind, specs=None):
        """
        user_id: str, uuid
        experiment_id: str, uuid corresponding to experiment
        job_id: str, uuid corresponding to a train job
        Returns:
        201 with [job_id] if job resumed and added to queue
        400 with [] if job_id does not correspond to a train action or if it cannot be resumed
        404 with [] if experiment/job_id not found
        """
        if not resolve_existence(user_id, "experiment", experiment_id):
            return Code(404, [], "Experiment not found")

        if not check_write_access(user_id, experiment_id):
            return Code(404, [], "Experiment not found")

        action = infer_action_from_job(user_id, experiment_id, job_id)
        if action != "train":
            return Code(400, [], "Action not train")
        handler_metadata = resolve_metadata(user_id, kind, experiment_id)

        msg = ""
        try:
            if is_job_automl(user_id, experiment_id, job_id):
                msg = "AutoML "
                AutoMLHandler.resume(user_id, experiment_id, job_id, handler_metadata)
            else:
                # Create a job and run it
                job_context = create_job_context(parent_job_id, "train", job_id, experiment_id, user_id, kind, specs=specs)
                on_new_job(job_context)
            return Code(200, job_id, f"{msg}Action resumed")
        except:
            print(traceback.format_exc(), file=sys.stderr)
            return Code(400, [], "Action cannot be resumed")
