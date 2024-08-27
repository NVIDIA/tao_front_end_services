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
Helper classes, functions, constants

Classes:
- Code
- JobContext
- StatusParser

Functions:
- load_json_spec
- search_for_base_experiment
- search_for_dataset
- get_dataset_download_command
- get_model_results_path
- write_nested_dict
- build_cli_command

Constants:
- VALID_DSTYPES
- VALID_NETWORKS
- IS_SPEC_NEEDED

"""
import os
import re
import sys
import copy
import glob
import json
import math
import uuid
import shutil
import requests
import tempfile
import traceback
import subprocess
from datetime import datetime, timezone, timedelta

from constants import _ITER_MODELS, CONTINUOUS_STATUS_KEYS, _PYT_TAO_NETWORKS, STATUS_JSON_MISMATCH_WITH_CHECKPOINT_EPOCH, STATUS_JSON_MISMATCH_WITH_CHECKPOINT_EPOCH_TMP, NETWORK_METRIC_MAPPING, TAO_NETWORKS, _TF2_NETWORKS, MISSING_EPOCH_FORMAT_NETWORKS
from handlers.cloud_storage import create_cs_instance
from handlers.encrypt import NVVaultEncryption
from handlers.stateless_handlers import get_handler_metadata, get_handler_job_metadata, get_handler_root, get_jobs_root, get_base_experiment_path, get_root, get_latest_ver_folder, get_base_experiment_metadata, write_job_metadata, update_base_experiment_metadata, resolve_metadata, write_handler_metadata, experiment_update_handler_attributes, update_handler_with_jobs_info, get_workspace_string_identifier, BACKEND
from handlers.monai.template_python import TEMPLATE_TIS_MODEL, TEMPLATE_TIS_CONFIG, TEMPLATE_CONTINUAL_LEARNING
from handlers.ngc_handler import validate_ptm_download, download_ngc_model
from handlers.monai.helpers import find_matching_bundle_dir
from utils import create_folder_with_permissions, safe_load_file


# Helper Classes
class TAOResponse:
    """Helper class for API response"""

    def __init__(self, code, data):
        """Initialize TAOResponse helper class"""
        self.code = code
        self.data = data
        self.attachment_key = None


def Code(code, data={}, msg=""):
    """Wraps TAOResponse and returns appropriate responses"""
    if code in [200, 201]:
        return TAOResponse(code, data)

    if code in [400, 404]:
        error_data = {"error_desc": msg, "error_code": code}
        return TAOResponse(code, error_data)

    error_data = {"error_desc": msg, "error_code": code}
    return TAOResponse(404, error_data)


class JobContext:
    """Class for holding job related information"""

    # Initialize Job Related fields
    # Contains API related parameters
    # ActionPipeline interacts with Toolkit and uses this JobContext
    def __init__(self, job_id, parent_id, network, action, handler_id, user_id, org_name, kind, created_on=None, specs=None, name=None, description=None, num_gpu=-1, platform=None):
        """Initialize JobContext class"""
        # Non-state variables
        self.id = job_id
        self.parent_id = parent_id
        self.network = network
        self.action = action
        self.handler_id = handler_id
        self.user_id = user_id
        self.org_name = org_name
        self.kind = kind
        self.created_on = created_on
        if not self.created_on:
            self.created_on = datetime.now(tz=timezone.utc)

        # State variables
        self.last_modified = datetime.now(tz=timezone.utc)
        self.status = "Pending"  # Starts off like this
        self.result = {}
        self.specs = specs
        # validate and update num_gpu
        if specs is not None:
            self.specs["num_gpu"] = validate_num_gpu(specs.get("num_gpu"), action)[0]
        self.name = name
        self.description = description
        self.num_gpu = num_gpu
        self.platform = platform

        self.write()

    def write(self):
        """Write the schema dict to jobs_metadata/job_id.json file"""
        # Create a job metadata
        write_job_metadata(self.id, self.schema())
        update_handler_with_jobs_info(self.schema(), self.handler_id, self.id, self.kind + "s")

    def __repr__(self):
        """Returns the schema dict"""
        return self.schema().__repr__()

    # ModelHandler / DatasetHandler interacts with this function
    def schema(self):
        """Creates schema dict based on the member variables"""
        _schema = {  # Cannot modify
            "name": self.name,
            "description": self.description,
            "id": self.id,
            "org_name": self.org_name,
            "parent_id": self.parent_id,
            "action": self.action,
            "created_on": self.created_on,
            "specs": self.specs,
            f"{self.kind}_id": self.handler_id,
            # Can modify
            "last_modified": self.last_modified,
            "status": self.status,
            "result": self.result}
        return _schema


class StatusParser:
    """Class for parsing status.json"""

    def __init__(self, status_file, network, results_dir, first_epoch_number=-1):
        """Intialize StatusParser class"""
        self.status_file = status_file
        self.network = network
        self.results_dir = results_dir
        self.cur_line = 0
        # Initialize results
        self.results = {}
        # Logging fields
        self.results["date"] = ""
        self.results["time"] = ""
        self.results["status"] = ""
        self.results["message"] = ""
        # Categorical
        self.results["categorical"] = {}
        # KPI
        self.results["kpi"] = {}
        # Graphical
        self.results["graphical"] = {}
        # Key metric for continual learning
        self.results["key_metric"] = 0.0

        self.last_seen_epoch = 0
        self.best_epoch_number = 0
        self.latest_epoch_number = 0
        self.first_epoch_number = first_epoch_number
        #
        self.gr_dict_cache = []

    def _update_first_epoch_number(self, epoch_number):
        if self.first_epoch_number == -1:
            self.first_epoch_number = epoch_number

    def _update_categorical(self, status_dict):
        """Update categorical key of status line"""
        if "epoch" in status_dict:
            self.last_seen_epoch = status_dict["epoch"]
        if "cur_iter" in status_dict and self.network in _ITER_MODELS:
            self.last_seen_epoch = status_dict["cur_iter"]

        # Categorical
        if "categorical" in status_dict:
            cat_dict = status_dict["categorical"]
            if type(cat_dict) is not dict:
                return
            for _, value_dict in cat_dict.items():
                if type(value_dict) is not dict:
                    return
            self.results["categorical"].update(cat_dict)

    def _process_object_count_kpi(self, kpi_dict):
        "Process object count KPI from DS Analyze action"
        index_to_object = kpi_dict.get('index', {})
        count_num = kpi_dict.get('count_num', {})
        percent = kpi_dict.get('percent', {})
        self.results['kpi']['object_count_index'] = {"values": {}}
        self.results['kpi']['object_count_num'] = {"values": {}}
        self.results['kpi']['object_count_percent'] = {"values": {}}
        for key, value in index_to_object.items():
            self.results['kpi']['object_count_index']['values'][str(key)] = value
        for key, value in count_num.items():
            self.results['kpi']['object_count_num']['values'][str(key)] = value
        for key, value in percent.items():
            self.results['kpi']['object_count_percent']['values'][str(key)] = value

    def _process_bbox_area_kpi(self, kpi_dict):
        """Process Bounding Box Area KPI from DS Analyze action"""
        type_to_object = kpi_dict.get('type', {})
        mean = kpi_dict.get('mean', {})
        self.results['kpi']['bbox_area_type'] = {"values": {}}
        self.results['kpi']['bbox_area_mean'] = {"values": {}}
        for key, value in type_to_object.items():
            self.results['kpi']['bbox_area_type']['values'][str(key)] = value
        for key, value in mean.items():
            self.results['kpi']['bbox_area_mean']['values'][str(key)] = value

    def _update_kpi(self, status_dict):
        """Update kpi key of status line"""
        if "epoch" in status_dict:
            self.last_seen_epoch = status_dict["epoch"]
            if "kpi" in status_dict:
                self._update_first_epoch_number(status_dict["epoch"])
        if "cur_iter" in status_dict and self.network in _ITER_MODELS:
            self.last_seen_epoch = status_dict["cur_iter"]
            if "kpi" in status_dict:
                self._update_first_epoch_number(status_dict["cur_iter"])
        if "mode" in status_dict and status_dict["mode"] == "train":
            return

        if "kpi" in status_dict:
            kpi_dict = status_dict["kpi"]
            if type(kpi_dict) is not dict:
                return
            analyze_type = kpi_dict.get('analyze_type', '')
            if analyze_type == 'object_count':  # DS Analyze KPI
                self._process_object_count_kpi(kpi_dict)
            elif analyze_type == 'bbox_area':
                self._process_bbox_area_kpi(kpi_dict)
            else:
                for key, value in kpi_dict.items():
                    if type(value) is dict:
                        # Process it differently
                        float_value = StatusParser.force_float(value.get("value", None))
                    else:
                        float_value = StatusParser.force_float(value)
                    # Simple append to "values" if the list exists
                    if key in self.results["kpi"]:
                        if float_value is not None:
                            if self.last_seen_epoch not in self.results["kpi"][key]["values"].keys():
                                self.results["kpi"][key]["values"][str(self.last_seen_epoch)] = float_value
                    else:
                        if float_value is not None:
                            self.results["kpi"][key] = {"values": {str(self.last_seen_epoch): float_value}}

    def _update_graphical(self, status_dict):
        """Update graphical key of status line"""
        if "epoch" in status_dict:
            self.last_seen_epoch = status_dict["epoch"]
        if "cur_iter" in status_dict and self.network in _ITER_MODELS:
            self.last_seen_epoch = status_dict["cur_iter"]

        if "graphical" in status_dict:
            gr_dict = status_dict["graphical"]
            # If the exact same dict was seen before, skip (an artefact of how status logger is written)
            if gr_dict in self.gr_dict_cache:
                return
            self.gr_dict_cache.append(gr_dict)
            if type(gr_dict) is not dict:
                return
            for key, value in gr_dict.items():
                plot_helper_dict = {}
                if type(value) is dict:
                    # Process it differently
                    float_value = StatusParser.force_float(value.get("value", None))
                    # Store x_min, x_max, etc... if given
                    for plot_helper_key in ["x_min", "x_max", "y_min", "y_max", "units"]:
                        if value.get(plot_helper_key):
                            plot_helper_dict[plot_helper_key] = value.get(plot_helper_key)
                else:
                    float_value = StatusParser.force_float(value)
                # Simple append to "values" if the list exists
                if key in self.results["graphical"]:
                    if float_value is not None:
                        if self.last_seen_epoch not in self.results["graphical"][key]["values"].keys():
                            self.results["graphical"][key]["values"][str(self.last_seen_epoch)] = float_value
                else:
                    if float_value is not None:
                        self.results["graphical"][key] = {"values": {str(self.last_seen_epoch): float_value}}

                if key in self.results["graphical"]:
                    # Put together x_min, x_max, y_min, y_max
                    graph_key_vals = self.results["graphical"][key]["values"]
                    self.results["graphical"][key].update({"x_min": 0,
                                                           "x_max": len(graph_key_vals),
                                                           "y_min": 0,
                                                           "y_max": StatusParser.force_max([val for key, val in graph_key_vals.items()]),
                                                           "units": None})
                    # If given in value, then update x_min, x_max, etc...
                    self.results["graphical"][key].update(plot_helper_dict)

    @staticmethod
    def force_float(value):
        """Convert str to float"""
        try:
            if (type(value) is str and value.lower() in ["nan", "infinity", "inf"]) or (type(value) is float and (math.isnan(value) or value == float('inf') or value == float('-inf'))):
                return None
            return float(value)
        except:
            return None

    @staticmethod
    def force_min(values):
        """Return min elements in the list"""
        values_no_none = [val for val in values if val is not None]
        if values_no_none != []:
            return min(values_no_none)
        return 0

    @staticmethod
    def force_max(values):
        """Return max elements in the list"""
        values_no_none = [val for val in values if val is not None]
        if values_no_none != []:
            return max(values_no_none)
        return 1e10

    def post_process_results(self, total_epochs=0, eta="", last_seen_epoch=0, automl=False):
        """Post process the status.json contents to be compatible with defined schema's in app.py"""
        # Copy the results
        processed_results = {}
        # Detailed results
        processed_results["detailed_status"] = {}
        for key in ["date", "time", "status", "message"]:
            processed_results["detailed_status"][key] = self.results[key]
        # Categorical
        processed_results["categorical"] = []
        for key, value_dict in self.results["categorical"].items():
            value_dict_unwrapped = [{"category": cat, "value": StatusParser.force_float(val)} for cat, val in value_dict.items()]
            processed_results["categorical"].append({"metric": key, "category_wise_values": value_dict_unwrapped})

        # KPI and Graphical
        for result_type in ("kpi", "graphical"):
            processed_results[result_type] = []
            for key, value_dict in self.results[result_type].items():
                dict_schema = {"metric": key}
                dict_schema.update(value_dict)
                processed_results[result_type].append(dict_schema)

        # Continuous remain the same
        for key in CONTINUOUS_STATUS_KEYS:
            processed_results[key] = self.results.get(key, None)
            if automl:
                if key == "epoch":
                    processed_results["automl_experiment_epoch"] = processed_results[key]
                if key == "max_epoch":
                    processed_results["automl_experiment_max_epoch"] = processed_results[key]
        processed_results["starting_epoch"] = int(self.first_epoch_number)
        processed_results["max_epoch"] = int(total_epochs)
        processed_results["epoch"] = int(self.last_seen_epoch)
        if automl and eta != "":
            processed_results["epoch"] = int(last_seen_epoch)
            if type(eta) is float:
                eta = str(timedelta(seconds=eta))
            processed_results["eta"] = str(eta)
        return processed_results

    def update_results(self, total_epochs=0, eta="", last_seen_epoch=0, automl=False):
        """Update results in status.json"""
        if not os.path.exists(self.status_file):
            # Try to find out status.json
            sjsons = glob.glob(self.results_dir + "/**/status.json", recursive=True)
            if sjsons:
                self.status_file = sjsons[0]
        # Read all the status lines in status.json till now
        good_statuses = []
        if os.path.exists(self.status_file):
            with open(self.status_file, "r", encoding='utf-8') as f:
                lines_to_process = f.readlines()[self.cur_line:]
                for line in lines_to_process:
                    try:
                        status_dict = json.loads(str(line))
                        good_statuses.append(status_dict)
                    except:
                        continue
                    self.cur_line += 1

        for status_dict in good_statuses:
            # Logging fields
            for key in ["date", "time", "status", "message"]:
                if key in status_dict:
                    self.results[key] = status_dict[key]

            # Categorical
            self._update_categorical(status_dict)

            # KPI
            self._update_kpi(status_dict)

            # Graphical
            self._update_graphical(status_dict)

            # Continuous
            for key in status_dict:
                if key in CONTINUOUS_STATUS_KEYS:
                    # verbosity is an additional status.json variable API does not process
                    self.results[key] = status_dict[key]

        return self.post_process_results(total_epochs, eta, last_seen_epoch, automl)

    def trim_list(self, metric_list, automl_algorithm, brain_epoch_number):
        """Retains only the tuples whose epoch numbers are <= required epochs"""
        trimmed_list = []
        for tuple_var in metric_list:
            epoch, value = (int(tuple_var[0]), tuple_var[1])
            if epoch >= 0:
                if automl_algorithm in ("bayesian", "b", ""):
                    if self.network in (_PYT_TAO_NETWORKS - set(["pointpillars", "segformer", "ml_recog"])):
                        if epoch < brain_epoch_number:  # epoch number in checkpoint starts from 0 or models whose validation logs are generated before the training logs
                            trimmed_list.append((epoch, value))
                    else:
                        trimmed_list.append((epoch, value))
                elif (self.network in _TF2_NETWORKS or self.network in ("segformer", "classification_pyt", "ml_recog")) and epoch <= brain_epoch_number:
                    trimmed_list.append((epoch, value))
                elif epoch < brain_epoch_number:
                    trimmed_list.append((epoch, value))
        return trimmed_list

    def read_metric(self, results, metric="loss", automl_algorithm="", automl_root="", brain_epoch_number=0):
        """
        Parses the status parser object and returns the metric of interest
        result: value from status_parser.update_results()
        returns: the metric requested in normalized float
        """
        metric_value = 0.0
        try:
            for result_type in ("graphical", "kpi"):
                for log in results[result_type]:
                    if metric == "kpi":
                        criterion = NETWORK_METRIC_MAPPING[self.network]
                    else:
                        criterion = metric
                    reverse_sort = True
                    if metric == "loss" or criterion in ("loss", "evaluation_cost "):
                        reverse_sort = False

                    if log["metric"] == criterion:
                        if log["values"]:
                            values_to_search = self.trim_list(metric_list=log["values"].items(), automl_algorithm=automl_algorithm, brain_epoch_number=brain_epoch_number)
                            if automl_algorithm in ("hyperband", "h"):
                                brain_dict = safe_load_file(automl_root + "/brain.json")
                                if (len(brain_dict.get("ni", [float('-inf')])[str(brain_dict.get("bracket", 0))]) != (brain_dict.get("sh_iter", float('inf')) + 1)):
                                    self.best_epoch_number, metric_value = values_to_search[-1]
                                else:
                                    self.best_epoch_number, metric_value = sorted(sorted(values_to_search, key=lambda x: x[0], reverse=False), key=lambda x: x[1], reverse=reverse_sort)[0]
                            else:
                                self.best_epoch_number, metric_value = sorted(sorted(values_to_search, key=lambda x: x[0], reverse=True), key=lambda x: x[1], reverse=reverse_sort)[0]
                            self.latest_epoch_number, _ = sorted(values_to_search, key=lambda x: x[0], reverse=True)[0]
                            metric_value = float(metric_value)
                            break
        except Exception:
            # Something went wrong inside...
            print(traceback.format_exc(), file=sys.stderr)
            print("Requested metric not found, defaulting to 0.0", file=sys.stderr)
            if (metric == "kpi" and NETWORK_METRIC_MAPPING[self.network] in ("loss", "evaluation_cost ")) or (metric in ("loss", "evaluation_cost ")):
                metric_value = 0.0
            else:
                metric_value = float('inf')

        if self.network in STATUS_JSON_MISMATCH_WITH_CHECKPOINT_EPOCH:
            self.best_epoch_number += 1
            self.latest_epoch_number += 1
        if self.network in STATUS_JSON_MISMATCH_WITH_CHECKPOINT_EPOCH_TMP:
            self.best_epoch_number -= 1
            self.latest_epoch_number -= 1
        print(f"Metric returned is {metric_value} at best epoch/iter {self.best_epoch_number} while latest epoch/iter is {self.latest_epoch_number}", file=sys.stderr)
        return metric_value + 1e-07, self.best_epoch_number, self.latest_epoch_number


def load_json_spec(spec_json_path):
    """Load json and delete version key if present in the csv specs"""
    try:
        spec = safe_load_file(spec_json_path)
        if spec.get("version"):
            del spec["version"]
        return spec
    except:
        return {}


def search_for_dataset(root):
    """Return path of the dataset file"""
    datasets = glob.glob(root + "/*.tar.gz", recursive=False) + glob.glob(root + "/*.tgz", recursive=False) + glob.glob(root + "/*.tar", recursive=False)

    if datasets:
        dataset_path = datasets[0]  # pick one arbitrarily
        return dataset_path
    return None


def search_for_base_experiment(root, network="", spec=False):
    """Return path of the Base-experiment file for MonAI or spec file for TAO under the Base-experiment root folder"""
    if spec:
        artifacts = glob.glob(root + "/**/*experiment.yaml", recursive=True)
    else:
        artifacts = glob.glob(root + "/**/*.tlt", recursive=True) + glob.glob(root + "/**/*.hdf5", recursive=True) + glob.glob(root + "/**/*.pth", recursive=True) + glob.glob(root + "/**/*.pth.tar", recursive=True) + glob.glob(root + "/**/*.pt", recursive=True)
    if artifacts:
        artifact_path = artifacts[0]  # pick one arbitrarily
        return artifact_path
    return None


def get_dataset_download_command(dataset_metadata):
    """Frames a wget and untar commands to download the dataset"""
    workspace_id = dataset_metadata.get("workspace")
    workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
    meta_data = copy.deepcopy(workspace_metadata)

    cloud_type = meta_data.get("cloud_type")
    cloud_specific_details = meta_data.get("cloud_specific_details")

    if cloud_specific_details:
        cs_instance, cloud_specific_details = create_cs_instance(meta_data)

    cloud_file_path = dataset_metadata.get("cloud_file_path")

    cloud_download_url = dataset_metadata.get("url", "")
    if not cloud_type:
        cloud_type = "self_hosted"
        if "huggingface" in cloud_download_url:
            cloud_type = "huggingface"

    temp_dir = tempfile.TemporaryDirectory().name  # pylint: disable=R1732
    create_folder_with_permissions(temp_dir)

    # if pull url, then download the dataset into some place inside root
    cmnd = ""
    if cloud_type == "self_hosted":
        cmnd = f"until wget --timeout=1 --tries=1 --retry-connrefused --no-verbose --directory-prefix={temp_dir}/ {cloud_download_url}; do sleep 10; done"
    elif cloud_type in ("aws", "azure"):
        if cloud_file_path.startswith("/"):
            cloud_file_path = cloud_file_path[1:]
        print("Downloading to", os.path.join(temp_dir, cloud_file_path), file=sys.stderr)
        cs_instance.download_folder(cloud_file_path, temp_dir)
    elif cloud_type == "huggingface":
        if cloud_specific_details:
            hf_token = cloud_specific_details.get("token", "")
            match = re.match(r"https://huggingface.co/datasets/([^/]+)/", cloud_download_url)
            username = ""
            if match:
                username = match.group(1)
            cmnd = f"git clone https://{username}:{hf_token}@{cloud_download_url.replace('https://', '')} {temp_dir}"
        else:
            cmnd = f"git clone {cloud_download_url} {temp_dir}"
    # run and wait till it finishes / run in background
    if cmnd:
        print(f"Executing command: {cmnd}", file=sys.stderr)
    return cmnd, temp_dir


def download_dataset(handler_dataset):
    """Calls wget and untar"""
    if handler_dataset is None:
        return None, None
    tar_file_path = None
    metadata = resolve_metadata("dataset", handler_dataset)
    status = metadata.get("status")
    temp_dir = ""
    if status == "starting":
        metadata["status"] = "in_progress"
        write_handler_metadata(handler_dataset, metadata, "datasets")

        dataset_download_command, temp_dir = get_dataset_download_command(metadata)  # this will not be None since we check this earlier
        if dataset_download_command:
            if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                # In dev setting, we don't need to set HOME
                result = subprocess.run(['/bin/bash', '-c', dataset_download_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            else:
                result = subprocess.run(['/bin/bash', '-c', 'HOME=/var/www/ && ' + dataset_download_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if result.stdout:
                print("Dataset pull stdout", result.stdout.decode("utf-8"), file=sys.stderr)
            if result.stderr:
                print("Dataset pull stderr", result.stderr.decode("utf-8"), file=sys.stderr)

        tar_file_path = search_for_dataset(temp_dir)
        if not tar_file_path:  # If dataset downloaded is of folder type
            tar_file_path = temp_dir
        metadata["status"] = "pull_complete"
        write_handler_metadata(handler_dataset, metadata, "datasets")
    return temp_dir, tar_file_path


def validate_and_update_experiment_metadata(org_name, request_dict, meta_data, key_list):
    """
    Update experiment metadata with given key_list from request_dict if present.
    If checks fail, return metadata and error code.
    """
    for key in key_list:
        if key in request_dict.keys():
            value = request_dict[key]
            if experiment_update_handler_attributes(org_name, meta_data, key, value):
                meta_data[key] = value
            else:
                return meta_data, Code(400, {}, f"Provided {key} cannot be added")
    return meta_data, None


def validate_and_update_base_experiment_metadata(base_experiment_file, base_experiment_id, meta_data):
    """Checks downloaded file hash and updates status in metadata"""
    sha256_digest = meta_data.get("sha256_digest", "")
    print(f"File {base_experiment_file} already exists, validating", file=sys.stderr)
    sha256_digest_matched = validate_ptm_download(base_experiment_file, sha256_digest)
    msg = "complete" if sha256_digest_matched else "in-complete"
    print(f"Download of {base_experiment_id} is {msg}", file=sys.stderr)
    meta_data["base_experiment_pull_complete"] = "pull_complete" if sha256_digest_matched else "starting"
    update_base_experiment_metadata(base_experiment_id, meta_data)


def download_base_experiment(user_id, base_experiment_id, spec=False):
    """Uses NGC API to download experiment spec files for the base experiment"""
    if base_experiment_id is None:
        return
    base_experiment_root = get_base_experiment_path(base_experiment_id)
    meta_data = get_base_experiment_metadata(base_experiment_id)
    network = meta_data.get("network_arch", "")
    base_experiment_file = search_for_base_experiment(base_experiment_root, network=network, spec=spec)
    sha256_digest = meta_data.get("sha256_digest", "")
    is_tao_network = network in TAO_NETWORKS
    print("PTM metadata", meta_data, file=sys.stderr)

    spec_file_present = meta_data.get("base_experiment_metadata", {}).get("spec_file_present")
    if not spec_file_present and is_tao_network:
        meta_data["base_experiment_pull_complete"] = "pull_complete"
        update_base_experiment_metadata(base_experiment_id, meta_data)

    elif base_experiment_file is None and meta_data.get("base_experiment_pull_complete", "") != "in_progress":
        print("File doesn't exist, downloading", file=sys.stderr)
        # check if metadata exists
        if not meta_data:
            return
        meta_data["base_experiment_pull_complete"] = "in_progress"
        update_base_experiment_metadata(base_experiment_id, meta_data)
        ngc_path = meta_data.get("ngc_path", "")
        sha256_digest_matched = download_ngc_model(user_id, is_tao_network, base_experiment_id, ngc_path, base_experiment_root, sha256_digest)
        # if prc failed => then ptm_file is None and we proceed without a ptm (because if ptm does not exist in ngc, it must not be loaded!)
        base_experiment_file = search_for_base_experiment(base_experiment_root, network=network)
        msg = "complete" if sha256_digest_matched else "in-complete"
        print(f"Download of {base_experiment_id} is {msg}", file=sys.stderr)
        meta_data["base_experiment_pull_complete"] = "pull_complete" if sha256_digest_matched else "starting"
        update_base_experiment_metadata(base_experiment_id, meta_data)

    elif base_experiment_file and meta_data.get("base_experiment_pull_complete", "") != "in_progress":
        validate_and_update_base_experiment_metadata(base_experiment_file, base_experiment_id, meta_data)
        print(f"File {base_experiment_file} already exists, validating", file=sys.stderr)
        sha256_digest_matched = validate_ptm_download(base_experiment_file, sha256_digest)
        msg = "complete" if sha256_digest_matched else "in-complete"
        print(f"Download of {base_experiment_id} is {msg}", file=sys.stderr)
        meta_data["base_experiment_pull_complete"] = "pull_complete" if sha256_digest_matched else "starting"
        update_base_experiment_metadata(base_experiment_id, meta_data)
    print("Base Experiment is totally/partially downloaded to", base_experiment_file, file=sys.stderr)
    if base_experiment_file and os.path.exists(base_experiment_file):
        print(f"Current_time : {datetime.now(tz=timezone.utc)}, File modified time: {os.path.getmtime(base_experiment_file)}", file=sys.stderr)


def write_nested_dict(dictionary, key_dotted, value):
    """Merge 2 dicitonaries"""
    ptr = dictionary
    keys = key_dotted.split(".")
    for key in keys[:-1]:
        # This applies to the classwise_config case that save the configs with list
        if type(ptr) is not dict:
            temp = {}
            for ptr_dic in ptr:
                temp.update(ptr_dic)
            ptr = temp
        ptr = ptr.setdefault(key, {})
    ptr[keys[-1]] = value


def write_nested_dict_if_exists(target_dict, nested_key, source_dict, key):
    """Merge 2 dicitonaries if given key exists in the source dictionary"""
    if key in source_dict:
        write_nested_dict(target_dict, nested_key, source_dict[key])
    # if key is not there, no update


def read_nested_dict(dictionary, flattened_key):
    """Returns the value of a flattened key separated by dots"""
    for key in flattened_key.split("."):
        value = dictionary[key]
        dictionary = value
    return value


def build_cli_command(config_data):
    """Generate cli command from the values of config_data"""
    # data is a dict
    # cmnd generates --<field_name> <value> for all key,value in data
    # Usage: To generate detectnet_v2 train --<> <> --<> <>,
    # The part after detectnet_v2 train is generated by this
    cmnd = ""
    for key, value in config_data.items():
        assert (type(value) is not dict)
        assert (type(value) is not list)
        if type(value) is bool:
            if value:
                cmnd += f"--{key} "
        else:
            cmnd += f"--{key}={value} "
    return cmnd


def get_flatten_specs(dict_spec, flat_specs, parent=""):
    """Flatten nested dictionary"""
    for key, value in dict_spec.items():
        if isinstance(value, dict):
            get_flatten_specs(value, flat_specs, parent + key + ".")
        else:
            flat_key = parent + key
            flat_specs[flat_key] = value


def get_train_spec(job_context, handler_root):
    """Read and return the train spec"""
    train_spec_path = os.path.join(handler_root, f"{job_context.id}-train-spec.json")
    spec = load_json_spec(train_spec_path)
    return spec


def get_total_epochs(job_context, handler_root, automl=False, automl_experiment_id=None):
    """Get the epoch/iter number from job_context.id-train-spec.json"""
    spec = {}
    if automl:
        json_spec_path = os.path.join(handler_root, job_context.id, f"recommendation_{automl_experiment_id}.json")
        spec = safe_load_file(json_spec_path)
    if not spec:
        spec = get_train_spec(job_context, handler_root)
    max_epoch = 100.0
    for key1 in spec:
        if key1 in ("training_config", "train_config", "train"):
            for key2 in spec[key1]:
                if key2 in ("num_epochs", "epochs", "n_epochs", "max_iters"):
                    max_epoch = int(spec[key1][key2])
                elif key2 in ("train_config"):
                    for key3 in spec[key1][key2]:
                        if key3 == "runner":
                            for key4 in spec[key1][key2][key3]:
                                if key4 == "max_epochs":
                                    max_epoch = int(spec[key1][key2][key3][key4])
        elif key1 in ("num_epochs"):
            max_epoch = int(spec[key1])

    return max_epoch


# NOTE Deprecated function, but still used until classwise config is needed
# for any network - this follows dnv2 schema.
def process_classwise_config(data):
    """Modifies data to re-organize the classwise config into respective collections
    Args: data: spec with classwise_config
    Return: data
    """
    if "classwise_config" not in data:
        return data
    if type(data["classwise_config"]) is not list:
        data["classwise_config"] = [data["classwise_config"]]

    # see if top level conf names exist, if not create it => ideally we want all of these names to exist
    for conf_name in ["bbox_rasterizer_config", "postprocessing_config", "cost_function_config", "evaluation_config"]:
        if data.get(conf_name) is None:
            data[conf_name] = {}
    data["bbox_rasterizer_config"]["target_class_config"] = []
    data["postprocessing_config"]["target_class_config"] = []
    data["cost_function_config"]["target_classes"] = []
    data["evaluation_config"]["minimum_detection_ground_truth_overlap"] = []
    data["evaluation_config"]["evaluation_box_config"] = []

    for class_name in data["classwise_config"]:

        bbox_dict = {"key": class_name["key"], "value:": class_name["value"]["bbox_rasterizer_config"]}
        data["bbox_rasterizer_config"]["target_class_config"].append(bbox_dict)

        post_dict = {"key": class_name["key"], "value:": class_name["postprocessing_config"]}
        data["postprocessing_config"]["target_class_config"].append(post_dict)

        cost_dict = {"name": class_name["key"]}
        cost_dict.update(class_name["value"]["cost_function_config"])
        data["cost_function_config"]["target_classes"].append(cost_dict)

        eval_dict_det = {"key": class_name["key"], "value": class_name["value"]["evaluation_config"]["minimum_detection_ground_truth_overlap"]}
        data["evaluation_config"]["minimum_detection_ground_truth_overlap"].append(eval_dict_det)
        eval_dict_conf = {"key": class_name["key"], "value": class_name["value"]["evaluation_config"]["evaluation_box_config"]}
        data["evaluation_config"]["evaluation_box_config"].append(eval_dict_conf)

    del data["classwise_config"]
    return data


def _check_gpu_conditions(field_name, field_value):
    if not field_value:
        raise ValueError("GPU related value not set")
    available_gpus = int(os.getenv("NUM_GPU_PER_NODE", "0"))
    if field_name in ("gpus", "num_gpus"):
        if int(field_value) < 0:
            raise ValueError("GPU related value requested is negative")
        if int(field_value) > available_gpus:
            raise ValueError(f"GPUs requested count of {field_value} is greater than gpus made available during deployment {available_gpus}")
    if field_name in ("gpu_ids", "gpu_id"):
        available_gpu_ids = set(range(0, available_gpus))
        requested_gpu_ids = set(field_value)
        if not requested_gpu_ids.issubset(available_gpu_ids):
            raise ValueError(f"GPU ids requested is {str(requested_gpu_ids)} but available gpu ids are {str(available_gpu_ids)}")


def get_num_gpus_from_spec(spec, action, default=0):
    """Validate the gpus requested"""
    if not isinstance(spec, dict):
        return default
    field_value = 0
    field_name = ""
    for gpu_param_name in ("gpus", "num_gpus", "gpu_ids", "gpu_id"):
        if gpu_param_name in spec.keys():
            field_name = gpu_param_name
            field_value = spec[gpu_param_name]
            if field_value != 0:
                _check_gpu_conditions(field_name, field_value)
        if action in spec and gpu_param_name in spec[action]:
            field_name = gpu_param_name
            field_value = spec[action][gpu_param_name]
            if field_value != 0:
                _check_gpu_conditions(field_name, field_value)

    if field_name in ("gpus", "num_gpus"):
        return int(field_value)
    if field_name in ("gpu_ids", "gpu_id"):
        if type(field_value) is int:
            return 1
        return len(set(field_value))
    if action in ("train", "evaluate", "prune", "retrain", "export", "gen_trt_engine", "inference", "generate", "augment"):
        return 1
    return default


def validate_num_gpu(num_gpu: int | None, action: str):
    """Validate the requested number of GPUs and return the validated number of GPUs.

    Args:
        num_gpu (str | None): Number of GPUs.
        action (str): Action to be performed.

    Returns:
        int: Validated number of GPUs.
        str: Error message indicating why validation fails.
    """
    # No gpu if num_gpu is not provided
    if num_gpu is None or num_gpu == 0:
        return 0, ""  # No GPU is requested. No need to validate further.
    # Convert num_gpu to int if it is a string
    if not isinstance(num_gpu, int):
        try:
            num_gpu = int(num_gpu)
        except ValueError:
            return 0, f"Requested number of GPUs ({num_gpu}) is not a valid number."
    # Check if num_gpu is a valid number
    if num_gpu < -1:
        return 0, f"Requested number of GPUs ({num_gpu}) is invalid negative number."

    # Get maximum available number of GPUs
    if BACKEND in ("BCP", "NVCF"):
        max_num_gpu = 8
    else:
        num_gpu_per_node = os.getenv("NUM_GPU_PER_NODE")
        if num_gpu_per_node is None:
            return 0, "NUM_GPU_PER_NODE is not set in the environment. Assuming no GPU is available!"
        max_num_gpu = int(num_gpu_per_node)

    # Use all maximum number of GPUs if num_gpu is -1
    if num_gpu == -1:
        return max_num_gpu, f"Requested number of GPUs is -1. Using all maximum number of GPUs ({max_num_gpu})."

    # Limit number of GPUs to the available number of GPUs
    if num_gpu > max_num_gpu:
        return 0, f"Requested number of GPUs ({num_gpu}) is larger than available number of GPUs ({max_num_gpu}). "

    # Use single GPU for actions not supporting multi-GPU
    multi_gpu_supported_actions = ["train", "retrain", "finetune", "auto3dseg", "inference"]  # disable `batchinfer`
    if action not in multi_gpu_supported_actions:
        if num_gpu > 1:
            return 0, f"Multi-GPU is not supported for {action}."

    return num_gpu, ""


def validate_uuid(dataset_id=None, job_id=None, experiment_id=None, workspace_id=None):
    """Validate possible UUIDs"""
    if dataset_id:
        try:
            uuid.UUID(dataset_id)
        except:
            return "Dataset ID passed is not a valid UUID"
    if job_id:
        try:
            uuid.UUID(job_id)
        except:
            return "Job ID passed is not a valid UUID"
    if experiment_id:
        try:
            uuid.UUID(experiment_id)
        except:
            return "Experiment ID passed is not a valid UUID"
    if workspace_id:
        try:
            uuid.UUID(workspace_id)
        except:
            return "Workspace ID passed is not a valid UUID"
    return ""


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


def add_workspace_to_cloud_metadata(workspace_metadata, cloud_metadata):
    """Add microservices needed cloud info to cloud_metadata"""
    cloud_type = workspace_metadata.get('cloud_type', '')

    # AWS, AZURE
    bucket_name = workspace_metadata.get('cloud_specific_details', {}).get('cloud_bucket_name', '')
    access_key = workspace_metadata.get('cloud_specific_details', {}).get('access_key', '')
    secret_key = workspace_metadata.get('cloud_specific_details', {}).get('secret_key', '')
    cloud_region = workspace_metadata.get('cloud_specific_details', {}).get('cloud_region', '')
    cloud_type = workspace_metadata.get("cloud_type")
    if cloud_type not in cloud_metadata:
        cloud_metadata[cloud_type] = {}
    cloud_metadata[cloud_type][bucket_name] = {
        "cloud_region": cloud_region,
        "access_key": access_key,
        "secret_key": secret_key,
    }


def get_cloud_metadata(workspace_ids, cloud_metadata):
    """For each workspace_id provided, fetch the necessary cloud info"""
    workspace_ids = list(set(workspace_ids))
    for workspace_id in workspace_ids:
        workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
        decrypt_handler_metadata(workspace_metadata)
        add_workspace_to_cloud_metadata(workspace_metadata, cloud_metadata)


def send_microservice_request(api_endpoint, network, action, ngc_api_key="", cloud_metadata={}, specs={}, job_id="", tao_api_admin_key="", tao_api_base_url="", tao_api_status_callback_url="", tao_api_ui_cookie="", use_ngc_staging="", automl_experiment_number=""):
    """Make a requests call to the microservice pod"""

    if not tao_api_base_url:
        tao_api_base_url = "https://nvidia.com"
    if not tao_api_status_callback_url:
        tao_api_status_callback_url = "https://nvidia.com"

    if action == "retrain":
        action = "train"

    request_metadata = {"api_endpoint": api_endpoint,
                        "neural_network_name": network,
                        "action_name": action,
                        "ngc_api_key": ngc_api_key,
                        "storage": cloud_metadata,
                        "specs": specs,
                        "job_id": job_id,
                        "tao_api_admin_key": tao_api_admin_key,
                        "tao_api_base_url": tao_api_base_url,
                        "tao_api_status_callback_url": tao_api_status_callback_url,
                        "tao_api_ui_cookie": tao_api_ui_cookie,
                        "use_ngc_staging": use_ngc_staging,
                        "automl_experiment_number": automl_experiment_number,
                        "hosted_service_interaction": "True"
                        }
    base_url = f"http://flask-service-{job_id}.default.svc.cluster.local:8000"
    data = json.dumps(request_metadata)
    endpoint = f"{base_url}/api/v1/nvcf"
    response = requests.post(endpoint, data=data)
    return response


def sanitize_metadata(metadata):
    """Convert metadata datetime objects to strings. MongoDB natively supports datetime objects. However, we pass a dict string as env variable to DNN container.
    DNN Container uses ast.literal_eval to safely reconstruct this string into a dict. However, ast.literal_eval doesn't support datetime objects, so we convert
    datetime objects to string here before passing the dict as a string in the job env variables.
    """
    if 'last_modified' in metadata and isinstance(metadata['last_modified'], datetime):
        date_string = metadata['last_modified'].isoformat()
        metadata['last_modified'] = date_string

    if 'created_on' in metadata and isinstance(metadata['created_on'], datetime):
        date_string = metadata['created_on'].isoformat()
        metadata['created_on'] = date_string

    metadata.pop('_id', None)


def latest_model(files, delimiters="_", epoch_number="000", extensions=[".tlt", ".hdf5", ".pth"]):
    """Returns the latest generated model file based on epoch number"""
    cur_best = 0
    best_model = None
    for file in files:
        _, file_extension = os.path.splitext(file)
        if file_extension not in extensions:
            continue
        model_name = file
        for extension in extensions:
            model_name = re.sub(f"{extension}$", "", model_name)
        delimiters_list = delimiters.split(",")
        if len(delimiters_list) > 1:
            delimiters_list = delimiters_list[0:-1]
        for delimiter in delimiters_list:
            epoch_num = model_name.split(delimiter)[-1]
            model_name = epoch_num
        if len(delimiters) > 1:
            epoch_num = model_name.split(delimiters[-1])[0]
        try:
            epoch_num = int(epoch_num)
        except:
            epoch_num = 0
        if epoch_num >= cur_best:
            cur_best = epoch_num
            best_model = file
    checkpoint_name = None
    if best_model:
        checkpoint_name = f"/{best_model}"
    return checkpoint_name


def filter_files(files, regex_pattern=""):
    """Filter file list based on regex provided"""
    if not regex_pattern:
        regex_pattern = r'^(?!.*lightning_logs).*\.(pth|tlt|hdf5)$'
    checkpoints = [path for path in files if re.match(regex_pattern, path)]
    return checkpoints


def filter_file_objects(file_objects, regex_pattern=""):
    if not regex_pattern:
        regex_pattern = r'.*\.(pth|tlt|hdf5)$'
    filtered_objects = [file_object for file_object in file_objects if re.match(regex_pattern, file_object.name)]
    return filtered_objects


def format_checkpoints_path(checkpoints):
    """Add formatting to the checkpoint name"""
    checkpoint_name = None
    if checkpoints:
        checkpoint_name = f"/{checkpoints[0]}"
    return checkpoint_name


def from_epoch_number(files, delimiters="", epoch_number="000"):
    """Based on the epoch number string passed, returns the path of the checkpoint. If a checkpoint with the epoch info is not present, raises an exception"""
    regex_pattern = fr'^(?!.*lightning_logs).*{epoch_number}\.(pth|tlt|hdf5)$'
    checkpoints = filter_files(files, regex_pattern)
    checkpoint_name = format_checkpoints_path(checkpoints)
    return checkpoint_name


def _get_result_file_path(checkpoint_function, files, format_epoch_number):
    result_file = checkpoint_function(files, delimiters="_", epoch_number=format_epoch_number)
    return result_file


def get_file_list_from_cloud_storage(workspace_metadata, res_root):
    """Return files present in res_root in cloud storage"""
    cs_instance, _ = create_cs_instance(workspace_metadata)
    files, _ = cs_instance.list_files_in_folder(res_root[1:])
    return files


def format_epoch(network, epoch_number):
    """Based on the network returns the epoch number formatted"""
    if network in MISSING_EPOCH_FORMAT_NETWORKS:
        format_epoch_number = str(epoch_number)
    else:
        format_epoch_number = f"{epoch_number:03}"
    return format_epoch_number


def search_for_checkpoint(handler_metadata, job_id, res_root, files, checkpoint_choose_method):
    """Based onf the choice of choosing checkpoint, handle different function calls and return the path found"""
    network = handler_metadata.get("network_arch")
    epoch_number_dictionary = handler_metadata.get("checkpoint_epoch_number", {})
    epoch_number = epoch_number_dictionary.get(f"{checkpoint_choose_method}_{job_id}", 0)

    if checkpoint_choose_method == "latest_model" or "/best_model" in res_root:
        checkpoint_function = latest_model
    elif checkpoint_choose_method in ("best_model", "from_epoch_number"):
        checkpoint_function = from_epoch_number
    else:
        raise ValueError(f"Chosen method to pick checkpoint not valid: {checkpoint_choose_method}")

    format_epoch_number = format_epoch(network, epoch_number)
    result_file = _get_result_file_path(checkpoint_function=checkpoint_function, files=files, format_epoch_number=format_epoch_number)
    if (not result_file) and (checkpoint_choose_method in ("best_model", "from_epoch_number")):
        print("Couldn't find the epoch number requested or the checkpointed associated with the best metric value, defaulting to latest_model", file=sys.stderr)
        checkpoint_function = latest_model
        result_file = _get_result_file_path(checkpoint_function=checkpoint_function, files=files, format_epoch_number=format_epoch_number)

    return result_file


def get_files_from_cloud(handler_metadata, job_id):
    if job_id is None:
        return None

    action = get_handler_job_metadata(job_id).get("action")
    res_root = os.path.join("/results", str(job_id))
    workspace_id = handler_metadata.get("workspace")
    workspace_metadata = resolve_metadata("workspace", workspace_id)
    files = get_file_list_from_cloud_storage(workspace_metadata, res_root)
    return files, action, res_root, workspace_id


def resolve_checkpoint_root_and_search(handler_metadata, job_id):
    """Returns path of the model based on the action of the job"""
    if job_id is None:
        return None

    files, action, res_root, workspace_id = get_files_from_cloud(handler_metadata, job_id)

    if action == "retrain":
        action = "train"

    if action == "train":
        checkpoint_choose_method = handler_metadata.get("checkpoint_choose_method", "best_model")
        result_file = search_for_checkpoint(handler_metadata=handler_metadata, job_id=job_id, res_root=res_root, files=files, checkpoint_choose_method=checkpoint_choose_method)

    elif action == "prune":
        result_file = filter_files(files)
        result_file = format_checkpoints_path(result_file)

    elif action == "export":
        regex_pattern = r'.*\.(onnx|uff)$'
        result_file = filter_files(files, regex_pattern=regex_pattern)
        result_file = format_checkpoints_path(result_file)

    elif action in ("trtexec", "gen_trt_engine"):
        regex_pattern = r'.*\.(engine)$'
        result_file = filter_files(files, regex_pattern=regex_pattern)
        result_file = format_checkpoints_path(result_file)
    else:
        result_file = None

    if result_file:
        workspace_identifier = get_workspace_string_identifier(workspace_id, workspace_cache={})
        result_file = f"{workspace_identifier}{result_file}"

    return result_file


def get_model_results_path(handler_metadata, job_id):
    """Return the model file for the job context and handler metadata passes"""
    print("\nget_model_results_path\n", file=sys.stderr)
    return resolve_checkpoint_root_and_search(handler_metadata, job_id)


def get_model_bundle_root(org_name, experiment_id):
    """Returns the path to the model bundle directory"""
    model_root = os.path.join(get_root(), org_name, "experiments", experiment_id)
    return os.path.join(model_root, "bundle")


def get_model_name(bundle_name):
    """Remove the version number from the bundle name to get the model name for TIS"""
    return re.sub(r"_v\d+\.\d+\.\d+", "", bundle_name)


def copy_bundle_base_experiment2model(base_experiment_id, org_name, user_id, experiment_id, patterns=[r'(.+?)_v\d+\.\d+\.\d+'], job_id=None):
    """
    Copies the pre-trained model to the model directory. Except the pre-trained weights, the whole bundle will be copied.
    - If the base_experiment is from NGC, then the directory will be under {base_exp_uuid}/experiments/{base_exp_uuid}/<experiment_id>:
        ├── metadata.json
        └── spleen_deepedit_annotation_v1.2.3 （not a real version)
            ├── configs
            ├── docs
            ├── LICENSE
            └── models

    - If the base_experiment is from a previously-trained model, the job_id has to be provided to locate the model.
      The orgs/<org_name>/users/<user_id>/<job_id> directory will be:
        ├── status.json
        └── spleen_deepedit_annotation_v1.2.3 （not a real version)
            ├── configs
            ├── docs
            ├── LICENSE
            └── models

    Args:
        base_experiment_id (str): The PTM ID
        org_name (str): The user ID
        experiment_id (str): The model ID
        patterns (list): The regex pattern to match the PTM name
        job_id (str): The job ID

    Returns:
        bool: True if the PTM is copied successfully, False otherwise
        str: message why it failed, or the name of the bundle if successful
    """
    # base_experiment_root = get_handler_root(base_exp_uuid, "experiments", base_exp_uuid, base_experiment_id)
    # This change is based on:
    # https://nvidia.slack.com/archives/C0696NHU7A7/p1704767995516689
    base_experiment_root = get_base_experiment_path(base_experiment_id, create_if_not_exist=False)
    base_experiment_root = base_experiment_root if job_id is None else os.path.join(base_experiment_root, str(job_id))
    if not os.path.isdir(base_experiment_root):
        # the base experiment is not a directory only if job_id is provided
        if job_id:
            base_experiment_root = os.path.join(get_jobs_root(user_id, org_name), str(job_id))
        else:
            base_experiment_root = get_handler_root(org_name=org_name, kind="experiments", handler_id=base_experiment_id)

    if not os.path.isdir(base_experiment_root):
        return False, None, "PTM not found"

    # Find an item that matches one of the patterns
    # e.g. spleen_deepedit_annotation_v1.2.3 （not a real version)
    matching_item = find_matching_bundle_dir(base_experiment_root, patterns)
    if not matching_item:
        return False, None, "No matching item found"

    src = os.path.join(base_experiment_root, matching_item)
    # dst_root = /shared/orgs/<org_name>/experiments/<experiment_id>/bundle/<tis_model_name>
    dst_root = os.path.join(get_model_bundle_root(org_name, experiment_id), get_model_name(matching_item))
    dst = os.path.join(dst_root, matching_item)

    # Ensure destination directory (bundle dir inside tis model dir) exists and is empty
    os.makedirs(dst_root, exist_ok=True)
    if os.path.exists(dst):
        shutil.rmtree(dst)

    # Copy bundle to the model directory
    shutil.copytree(src, dst)

    return True, matching_item, "Copy succeeded"


def ensure_script_format(config, prefix="", postfix=""):
    """Format a list to a string that can be used in the script file"""
    if isinstance(config, list):
        config_file_str = ', '.join(f'"{prefix}{x}{postfix}"' for x in config)  # If it's a list, join the items
        config_file_str = f"[{config_file_str}]"
    elif isinstance(config, str):
        config_file_str = f'"{prefix}{config}{postfix}"'
    else:
        # None or dict
        config_file_str = f"{config}"
    return config_file_str


def generate_tis_model_script(bundle_name, model_params):
    """Generate the Triton Inference Server model script"""
    override = model_params.get("override", {})
    image_key = model_params.get("image_key", "image")
    output_postfix = override.get("output_postfix", "seg")
    output_ext = override.get("output_ext", ".nrrd")
    output_dtype = override.get("output_dtype", "uint8")
    # can be customized by the user
    override["output_postfix"] = output_postfix
    override["output_ext"] = output_ext
    override["output_dtype"] = output_dtype

    # no need and not open for customization
    output_dir = "inference_results"
    override["separate_folder"] = False
    override["output_dir"] = output_dir
    override["dataset#data"] = [{}]

    return TEMPLATE_TIS_MODEL.format(
        bundle_name=bundle_name,
        image_key=image_key,
        override=override,
        output_dir=output_dir,
    )


def generate_config_pbtxt(tis_model):
    """Generate the Triton Inference Server config.pbtxt"""
    return TEMPLATE_TIS_CONFIG.format(tis_model=tis_model)


def _check_tis_model_params(model_params):
    """
    check the format of model_params
    """
    if not isinstance(model_params, dict):
        return False, f"model_params should be a dict, got {type(model_params)}."
    if "override" in model_params:
        override = model_params["override"]
        if isinstance(override, dict):
            return validate_monai_bundle_params(override)
        return False, f"Value of key `override` in model_params should be a dict, got {type(override)}."

    return True, ""


def get_monai_bundle_path(src_root):
    """
    Get the first folder path that has a monai bundle.
    """
    src_root_contents = os.listdir(src_root)
    for content in src_root_contents:
        content_path = os.path.join(src_root, content)
        if os.path.isdir(content_path):
            monai_bundle_content = ["configs", "models", "docs", "LICENSE"]
            monai_bundle_paths = [os.path.join(content_path, x) for x in monai_bundle_content]
            paths_exist = [os.path.exists(x) for x in monai_bundle_paths]
            if not all(paths_exist):
                continue
            return Code(201, content, "Got path!")
    return Code(404, {}, "Cannot export monai bundle.")


def generate_bundle_requirements_file(generate_dir, bundle_metadata):
    """Generate the requirements.txt file for the bundle."""
    libs = []
    restrict_libs = ["monai", "torch", "torchvision", "numpy", "pytorch-ignite"]

    if "optional_packages_version" in bundle_metadata.keys():
        optional_dict = bundle_metadata["optional_packages_version"]
        for name, version in optional_dict.items():
            if name not in restrict_libs:
                libs.append(f"{name}=={version}")

    if len(libs) > 0:
        requirements_file_name = "requirements.txt"
        with open(os.path.join(generate_dir, requirements_file_name), "w", encoding="utf-8") as f:
            for line in libs:
                f.write(f"{line}\n")


def prep_tis_model_repository(model_params, base_experiment_id, org_name, user_id, experiment_id, patterns=[r'(.+?)_v\d+\.\d+\.\d+'], job_id=None, update_model=False):
    """Prepare the model repository for Triton Inference Server"""
    # Copy the PTM to the model directory
    success, bundle_name, msg = copy_bundle_base_experiment2model(base_experiment_id, org_name, user_id, experiment_id, patterns, job_id)
    if not success:
        # should return 4 values to unify with successful return
        return False, None, msg, None
    check_result, check_msg = _check_tis_model_params(model_params)
    if not check_result:
        return False, None, check_msg, None

    # Generate {model_repository}/{tis_model}/{model_version}/model.py
    model_script = generate_tis_model_script(bundle_name, model_params)
    tis_model = get_model_name(bundle_name)
    tis_model_path = os.path.join(get_model_bundle_root(org_name, experiment_id), tis_model)
    os.makedirs(tis_model_path, exist_ok=True)
    lastest_ver = get_latest_ver_folder(tis_model_path)
    # produce a new version subfolder
    model_dir = os.path.join(tis_model_path, str(lastest_ver + 1))
    os.makedirs(model_dir, exist_ok=True)
    model_script_path = os.path.join(model_dir, "model.py")
    with open(model_script_path, "w", encoding="utf-8") as f:
        f.write(model_script)

    model_bundle_configs_dir = os.path.join(os.path.dirname(model_dir), bundle_name, "configs")
    model_bundle_config_metadata_json = os.path.join(model_bundle_configs_dir, "metadata.json")
    model_bundle_metadata = {}
    if os.path.exists(model_bundle_config_metadata_json):
        with open(model_bundle_config_metadata_json, encoding="utf-8") as fp:
            model_bundle_metadata = json.load(fp)

    # no need to re-produce config.pbtxt if just update the model
    if not update_model:
        # Generate {model_repository}/{tis_model}/config.pbtxt
        config_pbtxt = generate_config_pbtxt(tis_model)
        config_pbtxt_path = os.path.join(get_model_bundle_root(org_name, experiment_id), tis_model, "config.pbtxt")
        with open(config_pbtxt_path, "w", encoding="utf-8") as f:
            f.write(config_pbtxt)

    # prepare requirements file
    generate_bundle_requirements_file(tis_model_path, model_bundle_metadata)

    return True, tis_model, "Triton Inference Server model repository prepared successfully", model_bundle_metadata


def validate_monai_bundle_params(model_params):
    """Validate the the param are internal model params withheld from user"""
    if not isinstance(model_params, dict):
        return False, f"model_param override should be a dict, got {type(model_params)}."
    for key in ["bundle_root", "workflow_type", "ckpt_dir", "finetune_model_path"]:
        if key in model_params:
            return False, f"Override {key} in model_params is not allowed."
    for key in ["num_gpu", "cluster"]:
        if key in model_params:
            model_params.pop(key)
    return True, ""


def generate_cl_script(notify_record, job_context, handler_root, logfile, logs_from_toolkit):
    """Generate the continual learning script"""
    job_context_dict = {
        "id": job_context.id,
        "parent_id": job_context.parent_id,
        "network": job_context.network,
        "action": job_context.action,
        "handler_id": job_context.handler_id,
        "user_id": job_context.user_id,
        "org_name": job_context.org_name,
        "kind": job_context.kind,
        "created_on": job_context.created_on,
        "last_modified": job_context.last_modified,
        "specs": job_context.specs,
    }
    return TEMPLATE_CONTINUAL_LEARNING.format(
        notify_record=notify_record,
        job_context_dict=job_context_dict,
        handler_root=handler_root,
        logfile=logfile,
        logs_from_toolkit=logs_from_toolkit,
    )
