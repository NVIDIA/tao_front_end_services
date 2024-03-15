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

"""MEDICAL Dataset Handler module."""
import base64
import os
import pathlib
import random
import re
import sys
import json
import tempfile
import shutil
from threading import Thread

import requests
import validators
from filelock import FileLock
from handlers import ngc_handler, stateless_handlers
from handlers.encrypt import NVVaultEncryption
from handlers.medical.dataset.cache import LocalCache
from handlers.medical.dataset.dicom import DicomEndpoint
from handlers.medical.dataset.object_storage import ObjectStorageEndpoint
from handlers.utilities import Code
from handlers.stateless_handlers import resolve_existence, resolve_root, resolve_metadata, get_default_lock_file_path
from handlers.medical.helpers import ImageLabelRecord
from utils.utils import create_folder_with_permissions

MEDICAL_DATASET_ACTIONS = [
    "nextimage",
    "cacheimage",
    "notify",
]

dataset_cache_handles: dict[str, LocalCache] = {}


def download_dataset_to_ngc(endpoint, dataset_id):
    """
    Download the given dataset to ngc workspce using ngc handler.
    """
    temp_dir = tempfile.TemporaryDirectory().name  # pylint: disable=R1732
    create_folder_with_permissions(temp_dir)
    if isinstance(endpoint, ObjectStorageEndpoint):
        endpoint.download_all_objects(temp_dir)
    else:
        raise ValueError("DICOM dataset is not supported in NGC runner.")
    result = ngc_handler.upload_to_ngc_workspace(dataset_id, temp_dir, "/")
    if result.stdout:
        print(f"Workspace {dataset_id} upload results", result.stdout.decode("utf-8"), file=sys.stderr)  # Decode and print the standard error
    if result.stderr:
        print(f"Workspace {dataset_id} upload error", result.stderr.decode("utf-8"), file=sys.stderr)  # Decode and print the standard error
    shutil.rmtree(temp_dir)
    return result


def get_filename_from_cd(url, cd):
    """Get filename from content-disposition"""
    if not cd:
        if url.find('/'):
            return url.rsplit('/', 1)[1]
        return None

    fname = re.findall('filename=(.+)', cd)
    if len(fname) == 0:
        return None
    return fname[0]


def update_manifest_path(root_path, manifest_filename, max_path=1, label_key=["label"]):
    """Update the path in manifest.json to be absolute path"""
    manifest_path = os.path.join(root_path, manifest_filename)
    if not os.path.exists(manifest_path):
        return []

    # Load manifest.json
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if "data" not in manifest:
        return []

    if "label_key" in manifest:
        manifest_label_key = manifest["label_key"]
        if isinstance(manifest_label_key, list) and manifest_label_key:
            label_key = manifest["label_key"]
        elif isinstance(manifest_label_key, str):
            label_key = [manifest_label_key]

    # Update paths and handle label keys
    updated_data = []
    for entry in manifest["data"]:
        if label_key:
            # Check if all of the label keys exist and are non-empty
            if not all(key in entry and entry[key] for key in label_key):
                continue  # Skip entry if none of the label keys are present or are empty

        for key, value in entry.items():
            if isinstance(value, dict) and "path" in value:
                # Update path values
                for i, path in enumerate(value["path"]):
                    value["path"][i] = os.path.join(root_path, path)
                if max_path == 1:
                    entry[key] = value["path"][0]
                else:
                    entry[key] = value["path"][:max_path]

        updated_data.append(entry)

    # Update manifest with filtered and updated data
    manifest["data"] = updated_data
    return manifest


class MedicalDatasetHandler:
    """MEDICAL Dataset Handler class."""

    @staticmethod
    def get_data_from_manifest(user_id, dataset_id, workspace_dir, max_path=1):
        """Get "data" from manifest.json and format image/label path to be absolute path"""
        if not dataset_id:
            return []

        # load dataset metadata
        dataset_metadata = resolve_metadata(user_id, "dataset", dataset_id)
        if not dataset_metadata:
            return []

        # download data from the remote to mounted ngc workspace
        ep = MedicalDatasetHandler.endpoint(dataset_metadata)
        if isinstance(ep, ObjectStorageEndpoint):
            root_path = os.path.join(workspace_dir, "datasets", dataset_id)
            ep.download_all_objects(root_path)
        else:
            return []

        # find the path to manifest.json
        manifest_path = os.path.join(root_path, "manifest.json")
        if not os.path.exists(manifest_path):
            return []

        # load manifest.json
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if "data" not in manifest:
            return []

        # update manifest
        manifest = update_manifest_path(root_path, "manifest.json", max_path=max_path)

        # return only the data field of the manifest
        return manifest["data"] if "data" in manifest else []

    @staticmethod
    def prepare_datalist(user_id, dataset_id, ngc_runner_fetch=True):
        """Prepare the datalist for the dataset"""
        if not dataset_id:
            return []

        dataset_metadata = resolve_metadata(user_id, "dataset", dataset_id)
        if not dataset_metadata:
            return []

        ep = MedicalDatasetHandler.endpoint(dataset_metadata)
        if os.getenv("NGC_RUNNER", "") == "True" and ngc_runner_fetch:
            # check endpoint type
            if not isinstance(ep, ObjectStorageEndpoint):
                raise ValueError("DICOM dataset is not supported in NGC runner.")

            # download dataset
            download_path = os.path.join(stateless_handlers.get_root(ngc_runner_fetch=True), user_id, "datasets", dataset_id)
            download_res = download_dataset_to_ngc(ep, dataset_id)
            if download_res.stderr:
                raise TimeoutError("Cannot upload dataset to cloud.")
            manifest = update_manifest_path(download_path, "manifest.json")
            return manifest["data"] if "data" in manifest else []
        images = ep.get_labeled_images()
        ds = []
        for image in images:
            x = MedicalDatasetHandler.from_cache(user_id, dataset_id, image)
            if x.code != 201:
                print(f"Failed to fetch Image from cache: {image}", file=sys.stderr)
                continue

            data_entry = {'image': x.data["image"]}
            # count the number of string labels
            num_str_labels = 0
            for label in images[image]["labels"]:
                if isinstance(label, str):
                    # label is a string, which is the label id
                    num_str_labels += 1
                    # TODO: need to handle the case where there are multiple string labels
                    if num_str_labels > 1:
                        print(f"Multiple string labels (id) are not supported: {images[image]['labels']}", file=sys.stderr)
                        continue
                    y = MedicalDatasetHandler.from_cache(user_id, dataset_id, label)
                    if y.code != 201:
                        print(f"Failed to fetch Label from cache: {label}; Image: {image}", file=sys.stderr)
                        continue
                    data_entry.update({"label": y.data["image"]})
                elif isinstance(label, dict):
                    # label is a dict, which is the label metadata
                    data_entry.update(label)
            ds.append(data_entry)
        return ds

    @staticmethod
    def action_next_image(user_id, dataset_id, dataset_metadata, action_spec):
        """Get the next image from the dataset"""
        ep = MedicalDatasetHandler.endpoint(dataset_metadata)
        unlabeled = ep.get_unlabeled_images()
        if not unlabeled:
            return Code(404, msg="No Unlabeled Images Found")

        image = random.choices(list(unlabeled.keys()))[0]
        ret_info = {
            "image": image,
            "meta": unlabeled[image]
        }
        ret_info["meta"].update(ep.get_info(image))

        MedicalDatasetHandler.action_cache_image(user_id, dataset_id, dataset_metadata, {"image": image})
        return Code(201, ret_info, "Got the sample.")

    @staticmethod
    def action_cache_image(user_id, dataset_id, dataset_metadata, action_spec, background=True):
        """Cache the image to local cache"""
        if not resolve_existence(user_id, 'dataset', dataset_id):
            return Code(400, {}, f"Dataset Not Exists: {dataset_id}")

        # always cache images locally
        ds_root = os.path.join(stateless_handlers.get_root(ngc_runner_fetch=False), user_id, "datasets", dataset_id)
        cache_path = os.path.join(ds_root, "cache")
        cache_store = dataset_cache_handles.get(dataset_id)
        if not cache_store:
            cache_store = LocalCache(store_path=cache_path)
            dataset_cache_handles[user_id] = cache_store

        image = action_spec.get("image")
        if not image:
            return Code(400, {}, "Invalid Input.  Spec is missing `image`")

        cache_id = image
        cache_info = cache_store.get_cache(cache_id)
        if cache_info is None:
            def download_image():
                print(f'Downloading [{image}] in Background ({background})', file=sys.stderr)
                ep = MedicalDatasetHandler.endpoint(dataset_metadata)
                image_file = os.path.join(cache_path, cache_id, "image")
                save_file = ep.download(image, image_file)
                return cache_store.add_cache(cache_id, save_file, expiry=action_spec.get("ttl", 3600))[1]

            if background:
                thread = Thread(target=download_image)
                thread.start()
            else:
                cache_info = download_image()

        return Code(201, cache_info.to_json() if cache_info else None, f"Caching Image: {image}")

    @staticmethod
    def action_notify(user_id, handler_id, spec):
        """Notify the dataset handler with the label info"""
        controller_path = os.path.join(resolve_root(user_id, "dataset", handler_id), "notify_record.json")
        controller_lock = get_default_lock_file_path(controller_path)
        with FileLock(controller_lock, mode=0o666):
            image_recorder = ImageLabelRecord(controller_path)
            status, msg = image_recorder.process_data(spec)
            if not status:
                return Code(404, {}, msg)
            image_recorder.export(controller_path)

        return Code(201, {}, "Notification Received")

    @staticmethod
    def run_job(user_id, handler_id, handler_metadata, action, spec):
        """Run the job for the handler"""
        spec = spec if spec else {}
        if action == "nextimage":
            ret_code = MedicalDatasetHandler.action_next_image(user_id, handler_id, handler_metadata, spec)
        elif action == "cacheimage":
            ret_code = MedicalDatasetHandler.action_cache_image(user_id, handler_id, handler_metadata, spec)
            ret_code.data = {}
        elif action == "notify":
            ret_code = MedicalDatasetHandler.action_notify(user_id, handler_id, spec)
        else:
            return Code(404, {}, f"Cannot execute action {action}")

        return ret_code

    @staticmethod
    def cache_image_non_ds(user_id, image, ttl=3600):
        """Cache the image to local cache"""
        if not image:
            return Code(400, {}, "Invalid Input.  Spec is missing `image`")

        image_url = image.strip()
        if not validators.url(image_url):
            return Code(400, {}, "Invalid Image URL. Only URL is accepted")

        ds_root = os.path.join(stateless_handlers.get_root(), user_id)
        if not os.path.exists(ds_root):
            return Code(400, {}, f"User {user_id} Not Exists")

        handler_id = 'non_ds'
        cache_path = os.path.join(ds_root, "cache")
        cache_store = dataset_cache_handles.get(handler_id)
        if not cache_store:
            cache_store = LocalCache(store_path=cache_path)
            dataset_cache_handles[user_id] = cache_store

        cache_id = str(base64.encodestring(image_url))
        cache_info = cache_store.get_cache(cache_id)
        if cache_info is None:
            r = requests.get(image_url, allow_redirects=True)
            image_file = get_filename_from_cd(image_url, r.headers.get('content-disposition'))
            if not image_file:
                print(f"Failed to cache {image_url};  Can't determine filename", file=sys.stderr)
                return Code(400, cache_info, f"Failed to determine Caching Image type: {image}")

            with open(image_file, 'wb') as fp:
                fp.write(r.content)

            image_file = os.path.join(cache_path, cache_id, "image")
            file_ext = pathlib.Path(image_file).suffix
            uncompress = file_ext in [".zip", ".tar", ".gztar", ".bztar", ".xztar"]
            _, cache_info = cache_store.add_cache(cache_id, image_file, expiry=ttl, uncompress=uncompress)

        return Code(201, cache_info.to_json() if cache_info else None, f"Caching Image: {image}")

    @staticmethod
    def from_cache(user_id, dataset_id, image, ttl=3600):
        """Get the image from cache"""
        if dataset_id:
            dataset_metadata = resolve_metadata(user_id, 'dataset', dataset_id)
            return MedicalDatasetHandler.action_cache_image(
                user_id,
                dataset_id,
                dataset_metadata,
                {"image": image, "ttl": 3600},
                background=False,
            )
        return MedicalDatasetHandler.cache_image_non_ds(user_id, image, ttl=ttl)

    @staticmethod
    def clean_cache():
        """Clean the expired cache"""
        root_dir = stateless_handlers.get_root()
        if not os.path.exists(root_dir):
            return

        cache_paths = []
        for user_id in os.listdir(root_dir):
            if not os.path.isdir(os.path.join(root_dir, user_id)):
                continue

            path = os.path.join(root_dir, user_id, "cache")
            if os.path.exists(path) and os.path.isdir(path):
                cache_paths.append(path)

            ds_root = os.path.join(root_dir, user_id, "datasets")
            if not os.path.exists(ds_root) or not os.path.isdir(ds_root):
                continue

            for dataset_id in os.listdir(ds_root):
                path = os.path.join(ds_root, dataset_id, "cache")
            if os.path.exists(path) and os.path.isdir(path):
                cache_paths.append(path)

        for path in cache_paths:
            # print(f"Trying to remove Expired Cache from: {path}")  # redundant, only show user_id + dataset_id
            cache_store = LocalCache(store_path=path)
            cache_store.remove_expired()

    @staticmethod
    def get_schema(action):
        """Get the schema for the action"""
        if action == "nextimage":
            return {
                "limit": 1
            }
        if action == "cacheimage":
            return {
                "image": "",
                "ttl": 3600
            }
        if action == "notify":
            return {
                "added": [{
                    "image": "",
                    "label": "",
                }],
                "updated": [],
                "removed": [],
            }
        return {}

    @staticmethod
    def endpoint(metadata):
        """Get the DICOM endpoint by using the configs in the metadata"""
        # TODO Need to implement a function to tell the server automatically. No need for MVP.

        client_id = metadata.get("client_id", None)
        client_secret = metadata.get("client_secret", None)
        filters = metadata.get("filters", None)
        client_url = metadata.get("client_url", None)
        config_path = os.getenv("VAULT_SECRET_PATH", None)
        encryption = NVVaultEncryption(config_path)
        if encryption.check_config()[0]:
            client_secret = encryption.decrypt(client_secret)
        elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
            return None

        dicom_endpoint = DicomEndpoint(
            url=client_url,
            client_id=client_id,
            client_secret=client_secret,
            filters=filters,
        )

        obj_storage_endpoint = ObjectStorageEndpoint(
            url=client_url,
            client_id=client_id,
            client_secret=client_secret,
            filters=filters,
        )

        if obj_storage_endpoint.status_check()[0]:
            return obj_storage_endpoint

        if dicom_endpoint.status_check()[0]:
            return dicom_endpoint

        return None

    @staticmethod
    def status_check(metadata):
        """Check if a MEDICAL dataset is valid."""
        ep = MedicalDatasetHandler.endpoint(metadata)
        if ep is not None:
            return ep.status_check()

        return False, f"Cannot resolve the dataset with metadata:\n {metadata}"
