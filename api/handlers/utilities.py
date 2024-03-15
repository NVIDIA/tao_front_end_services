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
- read_network_config

Constants:
- VALID_DSTYPES
- VALID_NETWORKS
- IS_SPEC_NEEDED

"""
import os
import glob
import datetime
import json
import shutil
import subprocess
import sys
import re
import uuid
import tarfile
import math
import traceback
import hashlib

from handlers.stateless_handlers import get_handler_job_metadata, get_handler_root, get_base_experiment_path, get_root, get_latest_ver_folder, get_base_experiment_metadata, write_job_metadata, safe_load_file, update_base_experiment_metadata, get_handler_kind, update_job_tar_stats
from handlers.medical.template_python import TEMPLATE_MB_TRAIN, TEMPLATE_TIS_MODEL, TEMPLATE_TIS_CONFIG, TEMPLATE_DICOM_SEG_CONVERTER, TEMPLATE_CONTINUAL_LEARNING
from handlers import ngc_handler
from handlers.medical.helpers import find_matching_bundle_dir


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
    def __init__(self, job_id, parent_id, network, action, handler_id, user_id, kind, created_on=None, specs=None, local_cluster_job_with_ngc_workspaces=False, name=None, description=None):
        """Initialize JobContext class"""
        # Non-state variables
        self.id = job_id
        self.parent_id = parent_id
        self.network = network
        self.action = action
        self.handler_id = handler_id
        self.user_id = user_id
        self.kind = kind
        self.created_on = created_on
        if not self.created_on:
            self.created_on = datetime.datetime.now().isoformat()

        # State variables
        self.last_modified = datetime.datetime.now().isoformat()
        self.status = "Pending"  # Starts off like this
        self.result = {}
        self.specs = specs
        self.local_cluster_job_with_ngc_workspaces = local_cluster_job_with_ngc_workspaces
        self.name = name
        self.description = description

        if self.local_cluster_job_with_ngc_workspaces and os.getenv("NGC_RUNNER", "") == "True":
            ngc_handler.mount_ngc_workspace(self.user_id, self.handler_id)

        self.write()

    def write(self):
        """Write the schema dict to jobs_metadata/job_id.json file"""
        # Create a job metadata
        write_job_metadata(self.user_id, self.handler_id, self.id, self.schema())

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

    def __init__(self, status_file, network, results_dir):
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
        #
        self.gr_dict_cache = []

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

    def _update_kpi(self, status_dict):
        """Update kpi key of status line"""
        if "epoch" in status_dict:
            self.last_seen_epoch = status_dict["epoch"]
        if "cur_iter" in status_dict and self.network in _ITER_MODELS:
            self.last_seen_epoch = status_dict["cur_iter"]
        if "mode" in status_dict and status_dict["mode"] == "train":
            return

        if "kpi" in status_dict:
            kpi_dict = status_dict["kpi"]
            if type(kpi_dict) is not dict:
                return

            for key, value in kpi_dict.items():
                if type(value) is dict:
                    # Process it differently
                    float_value = StatusParser.force_float(value.get("value", None))
                else:
                    float_value = StatusParser.force_float(value)
                # Simple append to "values" if the list exists
                if key in self.results["kpi"]:
                    # Metric info is present in duplicate lines for these network
                    if self.network in ("efficientdet_tf1"):
                        if "epoch" in status_dict and float_value:
                            float_value = None
                    if float_value is not None:
                        if self.last_seen_epoch not in self.results["kpi"][key]["values"].keys():
                            self.results["kpi"][key]["values"][self.last_seen_epoch] = float_value
                else:
                    if float_value is not None:
                        self.results["kpi"][key] = {"values": {self.last_seen_epoch: float_value}}

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
                            self.results["graphical"][key]["values"][self.last_seen_epoch] = float_value
                else:
                    if float_value is not None:
                        self.results["graphical"][key] = {"values": {self.last_seen_epoch: float_value}}

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

    def post_process_results(self):
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
        return processed_results

    def update_results(self):
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

        return self.post_process_results()

    def trim_list(self, metric_list, automl_algorithm, brain_epoch_number):
        """Retains only the tuples whose epoch numbers are <= required epochs"""
        trimmed_list = []
        for tuple_var in metric_list:
            if tuple_var[0] >= 0:
                if automl_algorithm in ("bayesian", "b", ""):
                    if self.network in (_PYT_TAO_NETWORKS - set(["pointpillars", "segformer"])):
                        if tuple_var[0] < brain_epoch_number:  # epoch number in checkpoint starts from 0 or models whose validation logs are generated before the training logs
                            trimmed_list.append(tuple_var)
                    else:
                        trimmed_list.append(tuple_var)
                elif (self.network in STATUS_JSON_MISMATCH_WITH_CHECKPOINT_EPOCH and tuple_var[0] < brain_epoch_number) or (self.network not in STATUS_JSON_MISMATCH_WITH_CHECKPOINT_EPOCH and tuple_var[0] <= brain_epoch_number):
                    trimmed_list.append(tuple_var)
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
                        criterion = network_metric_mapping[self.network]
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
                                    self.best_epoch_number, metric_value = sorted(sorted(values_to_search, key=lambda x: x[0], reverse=True), key=lambda x: x[1], reverse=reverse_sort)[0]
                            else:
                                self.best_epoch_number, metric_value = sorted(sorted(values_to_search, key=lambda x: x[0], reverse=True), key=lambda x: x[1], reverse=reverse_sort)[0]
                                self.latest_epoch_number, _ = sorted(values_to_search, key=lambda x: x[0], reverse=True)[0]
                            metric_value = float(metric_value)
                            break
        except Exception:
            # Something went wrong inside...
            print(traceback.format_exc(), file=sys.stderr)
            print("Requested metric not found, defaulting to 0.0", file=sys.stderr)
            if (metric == "kpi" and network_metric_mapping[self.network] in ("loss", "evaluation_cost ")) or (metric in ("loss", "evaluation_cost ")):
                metric_value = 0.0
            else:
                metric_value = float('inf')

        if self.network in STATUS_JSON_MISMATCH_WITH_CHECKPOINT_EPOCH:
            self.best_epoch_number += 1
            self.latest_epoch_number += 1
        elif automl_algorithm in ("hyperband", "h"):
            if self.network in _PYT_TAO_NETWORKS - _ITER_MODELS:  # epoch number in checkpoint starts from 0 or models whose validation logs are generated before the training logs
                self.best_epoch_number -= 1
        print(f"Metric returned is {metric_value} at best epoch/iter {self.best_epoch_number} while latest epoch/iter is {self.latest_epoch_number}", file=sys.stderr)
        return metric_value + 1e-07, self.best_epoch_number, self.latest_epoch_number


# Helper Functions
def archive_job_results(workspace_root, job_id, job_files=[]):
    """Add job files to existing tar if not present already"""
    if not job_files:
        job_files = glob.glob(f"{workspace_root}/{job_id}/**", recursive=True)
    job_tar = f"{workspace_root}/{job_id}.tar.gz"

    root = ""
    if workspace_root.startswith("/users"):
        root = f"/shared/{workspace_root}"
        if not job_files:
            job_files += glob.glob(f"/shared/{workspace_root}/{job_id}/**", recursive=True)
        job_tar = f"{root}/{job_id}.tar.gz"

    with tarfile.open(job_tar, 'w:gz') as tar:
        existing_files = tar.getnames()
        for file_path in job_files:
            if file_path not in existing_files and os.path.exists(file_path) and os.path.isfile(file_path) and not file_path.endswith(".lock"):
                tar.add(file_path, arcname=file_path.replace(root, "", 1).replace(workspace_root, "", 1))


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


def search_for_base_experiment(root, extension="tlt", network=""):
    """Return path of the PTM file under the PTM root folder"""
    # from root, return model
    # if return is None, that means not hdf5 or tlt inside the folder
    # search for hdf5 / tlt /pth

    # EfficientDet tf2 PTM is a not a single file
    if network in ["classification_tf2", "efficientdet_tf2"]:
        pretrained_root_folder_map = {"classification_tf2": "pretrained_classification_tf2_vefficientnet_b0",
                                      "efficientdet_tf2": "pretrained_efficientdet_tf2_vefficientnet_b0"}
        if len(glob.glob(root + "/**/*")) > 0:
            return os.path.join(root, pretrained_root_folder_map[network])
        return None
    models = glob.glob(root + "/**/*.tlt", recursive=True) + glob.glob(root + "/**/*.hdf5", recursive=True) + glob.glob(root + "/**/*.pth", recursive=True) + glob.glob(root + "/**/*.pth.tar", recursive=True) + glob.glob(root + "/**/*.pt", recursive=True)
    # TODO: remove after next nvaie release, Varun and Subha
    if network == "classification_pyt":
        models += glob.glob(root + "/**/*.ckpt", recursive=True)

    # if .tlt exists
    if models:
        model_path = models[0]  # pick one arbitrarily
        return model_path
    # if no .tlt exists
    return None


def get_dataset_download_command(root):
    """Frames a wget and untar commands to download the dataset"""
    # check if metadata exists
    metadata = glob.glob(root + "/metadata.json")
    if not metadata:
        return None
    metadata = metadata[0]
    # read metadata 'pull' url
    with open(metadata, "r", encoding='utf-8') as f:
        meta_data = json.load(f)
    pull_url = meta_data.get("pull", "")

    # if no pull url
    if not pull_url:
        return None
    # if pull url, then download the dataset into some place inside root
    cmnd = f"TMPDIR=$(mktemp -d) && until wget --timeout=1 --tries=1 --retry-connrefused --no-verbose --directory-prefix=$TMPDIR/ {pull_url}; do sleep 10; done && chmod -R 777 $TMPDIR && cp -r $TMPDIR/* {root}/ && rm -rf $TMP_DIR"
    # run and wait till it finishes / run in background
    print("Executing WGET command: ", cmnd, file=sys.stderr)
    return cmnd


def generate_job_tar_stats(user_id, handler_id, job_id, handler_root):
    """Update file size and sha digest value"""
    if handler_root.startswith("/users"):
        handler_root = f"/shared/{handler_root}"
    job_tar_path = f"{handler_root}/{job_id}.tar.gz"

    sha256 = hashlib.sha256()
    with open(job_tar_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)
    sha_digest = sha256.hexdigest()

    file_size = os.path.getsize(job_tar_path)

    tar_stats = {"sha256_digest": sha_digest, "file_size": file_size}
    update_job_tar_stats(user_id, handler_id, job_id, tar_stats)


def download_dataset(user_id, handler_dataset):
    """Calls wget and untar"""
    if handler_dataset is None:
        return None
    dataset_root = get_handler_root(user_id, "datasets", handler_dataset, None)
    dataset_file = search_for_dataset(dataset_root)
    if dataset_file is None:
        dataset_download_command = get_dataset_download_command(dataset_root)  # this will not be None since we check this earlier
        if dataset_download_command:
            if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                # In dev setting, we don't need to set HOME
                result = subprocess.run(['/bin/bash', '-c', dataset_download_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            else:
                result = subprocess.run(['/bin/bash', '-c', 'HOME=/var/www/ && ' + dataset_download_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if result.stdout:
                print("dataset download stdout", result.stdout.decode("utf-8"), file=sys.stderr)
            if result.stderr:
                error_message = result.stderr.decode("utf-8")
                print("dataset download stderr", error_message, file=sys.stderr)
        dataset_file = search_for_dataset(dataset_root)
    return dataset_file


def validate_and_update_base_experiment_metadata(base_experiment_file, base_experiment_id, meta_data):
    """Checks downloaded file hash and updates status in metadata"""
    sha256_digest = meta_data.get("sha256_digest", "")
    print(f"File {base_experiment_file} already exists, validating", file=sys.stderr)
    sha256_digest_matched = ngc_handler.validate_ptm_download(base_experiment_file, sha256_digest)
    msg = "complete" if sha256_digest_matched else "in-complete"
    print(f"Download of {base_experiment_id} is {msg}", file=sys.stderr)
    meta_data["base_experiment_pull_complete"] = "present" if sha256_digest_matched else "not_present"
    update_base_experiment_metadata(base_experiment_id, meta_data)


def download_base_experiment(base_experiment_id):
    """Calls the ngc model download command and removes the unnecessary files for some models containing multiple model files"""
    if base_experiment_id is None:
        return
    base_experiment_root = get_base_experiment_path(base_experiment_id)
    base_experiment_file = search_for_base_experiment(base_experiment_root)
    meta_data = get_base_experiment_metadata(base_experiment_id)
    sha256_digest = meta_data.get("sha256_digest", "")
    print("PTM metadata", meta_data, file=sys.stderr)
    if base_experiment_file is None and meta_data.get("base_experiment_pull_complete", "") != "in_progress":
        print("File doesn't exist, downloading", file=sys.stderr)
        # check if metadata exists
        if not meta_data:
            return
        meta_data["base_experiment_pull_complete"] = "in_progress"
        update_base_experiment_metadata(base_experiment_id, meta_data)
        ngc_path = meta_data.get("ngc_path", "")
        network_arch = meta_data.get("network_arch", "")
        additional_id_info = meta_data.get("additional_id_info", "")

        sha256_digest_matched = ngc_handler.download_ngc_model(base_experiment_id, ngc_path, base_experiment_root, sha256_digest)
        # if prc failed => then ptm_file is None and we proceed without a ptm (because if ptm does not exist in ngc, it must not be loaded!)
        base_experiment_file = search_for_base_experiment(base_experiment_root, network=network_arch)
        msg = "complete" if sha256_digest_matched else "in-complete"
        print(f"Download of {base_experiment_id} is {msg}", file=sys.stderr)
        meta_data["base_experiment_pull_complete"] = "present" if sha256_digest_matched else "not_present"
        update_base_experiment_metadata(base_experiment_id, meta_data)

        if network_arch == "lprnet":
            if additional_id_info == "us":
                os.system(f"rm {base_experiment_root}/lprnet_vtrainable_v1.0/*ch_*")
            elif additional_id_info == "ch":
                os.system(f"rm {base_experiment_root}/lprnet_vtrainable_v1.0/*us_*")
        elif network_arch == "action_recognition":
            additional_id_info_list = additional_id_info.split(",")
            if len(additional_id_info_list) == 1:
                if additional_id_info_list[0] == "3d":
                    os.system(f"rm {base_experiment_root}/actionrecognitionnet_vtrainable_v1.0/*_2d_*")
                elif additional_id_info_list[0] == "2d":
                    os.system(f"rm {base_experiment_root}/actionrecognitionnet_vtrainable_v1.0/*_3d_*")
            if len(additional_id_info_list) == 2:
                for ind_additional_id_info in additional_id_info_list:
                    if ind_additional_id_info == "a100":
                        os.system(f"rm {base_experiment_root}/actionrecognitionnet_vtrainable_v2.0/*xavier*")
                    elif ind_additional_id_info == "xavier":
                        os.system(f"rm {base_experiment_root}/actionrecognitionnet_vtrainable_v2.0/*a100*")
                    if ind_additional_id_info == "3d":
                        os.system(f"rm {base_experiment_root}/actionrecognitionnet_vtrainable_v2.0/*_2d_*")
                    elif ind_additional_id_info == "2d":
                        os.system(f"rm {base_experiment_root}/actionrecognitionnet_vtrainable_v2.0/*_3d_*")
    elif base_experiment_file and meta_data.get("base_experiment_pull_complete", "") != "in_progress":
        validate_and_update_base_experiment_metadata(base_experiment_file, base_experiment_id, meta_data)
        print(f"File {base_experiment_file} already exists, validating", file=sys.stderr)
        sha256_digest_matched = ngc_handler.validate_ptm_download(base_experiment_file, sha256_digest)
        msg = "complete" if sha256_digest_matched else "in-complete"
        print(f"Download of {base_experiment_id} is {msg}", file=sys.stderr)
        meta_data["base_experiment_pull_complete"] = "present" if sha256_digest_matched else "not_present"
        update_base_experiment_metadata(base_experiment_id, meta_data)
    print("Base Experiment is totally/partially downloaded to", base_experiment_file, file=sys.stderr)
    if base_experiment_file and os.path.exists(base_experiment_file):
        print(f"Current_time : {datetime.datetime.utcnow()}, File modified time: {os.path.getmtime(base_experiment_file)}", file=sys.stderr)


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


def build_cli_command(config_data, spec_data=None):
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


def read_network_config(network):
    """Reads the network handler json config file"""
    # CLONE EXISTS AT pretrained_models.py
    _dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config_json_path = os.path.join(_dir_path, "handlers", "network_configs", f"{network}.config.json")
    cli_config = {}
    with open(config_json_path, mode='r', encoding='utf-8-sig') as f:
        cli_config = json.load(f)
    return cli_config


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
    train_spec_path = os.path.join(handler_root, "specs", f"{job_context.id}-train-spec.json")
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


def validate_gpu_param_value(spec):
    """Validate the gpus requested"""
    for gpu_param_name in ("gpus", "num_gpus", "gpu_ids", "gpu_id"):
        if gpu_param_name in spec.keys():
            field_name = gpu_param_name
            field_value = spec[gpu_param_name]
            _check_gpu_conditions(field_name, field_value)
        if "train" in spec.keys() and gpu_param_name in spec["train"].keys():
            field_name = gpu_param_name
            field_value = spec["train"][gpu_param_name]
            _check_gpu_conditions(field_name, field_value)


def validate_uuid(user_id=None, dataset_id=None, job_id=None, experiment_id=None):
    """Validate possible UUIDs"""
    if user_id:
        try:
            uuid.UUID(user_id)
        except:
            return "User ID passed is not a valid UUID"
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
    return ""


def latest_model(folder, delimiters="_", epoch_number="000", extensions=[".tlt", ".hdf5", ".pth"]):
    """Returns the latest generated model file based on epoch number"""
    cur_best = 0
    best_model = "model.tlt"
    files = os.listdir(folder)
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
    return os.path.join(folder, best_model)


def from_epoch_number(folder, delimiters="", epoch_number="000"):
    """Based on the epoch number string passed, returns the path of the checkpoint. If a checkpoint with the epoch info is not present, raises an exception"""
    find_trained_tlt = glob.glob(f"{folder}/*{epoch_number}.tlt") + glob.glob(f"{folder}/train/*{epoch_number}.tlt") + glob.glob(f"{folder}/weights/*{epoch_number}.tlt")
    find_trained_hdf5 = glob.glob(f"{folder}/*{epoch_number}.hdf5") + glob.glob(f"{folder}/train/*{epoch_number}.hdf5") + glob.glob(f"{folder}/weights/*{epoch_number}.hdf5")
    find_trained_pth = glob.glob(f"{folder}/*{epoch_number}.pth") + glob.glob(f"{folder}/train/*{epoch_number}.pth") + glob.glob(f"{folder}/weights/*{epoch_number}.pth")
    checkpoints = find_trained_tlt + find_trained_hdf5 + find_trained_pth
    if not checkpoints:
        print(f"No checkpoints associated with the epoch number {epoch_number} was found", file=sys.stderr)
        return None
    return checkpoints[0]


def _get_result_file_path(network, checkpoint_function, res_root, format_epoch_number):
    if network in ("classification_tf1", "classification_tf2", "classification_pyt", "efficientdet_tf2", "faster_rcnn", "multitask_classification", "dssd", "ssd", "retinanet", "yolo_v3", "yolo_v4", "yolo_v4_tiny", "segformer", "pointpillars"):
        result_file = checkpoint_function(res_root, delimiters="_", epoch_number=format_epoch_number)
    elif network in ("detectnet_v2", "lprnet", "efficientdet_tf1", "mask_rcnn", "unet", "bpnet", "fpenet"):
        result_file = checkpoint_function(res_root, delimiters="-", epoch_number=format_epoch_number)
    elif network in _PYT_CV_NETWORKS:
        result_file = checkpoint_function(res_root, delimiters="=", epoch_number=format_epoch_number)
    else:
        result_file = None
    return result_file


def _get_model_results_path(handler_metadata, job_id, res_root):
    network = handler_metadata.get("network_arch")
    checkpoint_choose_method = handler_metadata.get("checkpoint_choose_method", "best_model")
    epoch_number_dictionary = handler_metadata.get("checkpoint_epoch_number", {})
    epoch_number = epoch_number_dictionary.get(f"{checkpoint_choose_method}_{job_id}", 0)

    if checkpoint_choose_method == "latest_model" or "/best_model" in res_root:
        checkpoint_function = latest_model
    elif checkpoint_choose_method in ("best_model", "from_epoch_number"):
        checkpoint_function = from_epoch_number
    else:
        raise ValueError(f"Chosen method to pick checkpoint not valid: {checkpoint_choose_method}")

    if network in MISSING_EPOCH_FORMAT_NETWORKS:
        format_epoch_number = str(epoch_number)
    else:
        format_epoch_number = f"{epoch_number:03}"

    result_file = _get_result_file_path(network=network, checkpoint_function=checkpoint_function, res_root=res_root, format_epoch_number=format_epoch_number)
    if (not result_file) and (checkpoint_choose_method in ("best_model", "from_epoch_number")):
        print("Couldn't find the epoch number requested or the checkpointed associated with the best metric value, defaulting to latest_model", file=sys.stderr)
        checkpoint_function = latest_model
        result_file = _get_result_file_path(network=network, checkpoint_function=checkpoint_function, res_root=res_root, format_epoch_number=format_epoch_number)

    return result_file


def get_model_results_path(job_context, handler_metadata, job_id):
    """Returns path of the model based on the action of the job"""
    if job_id is None:
        return None

    handler_id = handler_metadata.get("id")
    kind = get_handler_kind(handler_metadata)
    root = get_handler_root(job_context.user_id, kind, handler_id, None, ngc_runner_fetch=True)

    action = get_handler_job_metadata(job_context.user_id, handler_id, job_id).get("action")
    automl_path = ""
    if handler_metadata.get("automl_enabled") is True and action == "train":
        automl_path = "best_model"

    if action == "retrain":
        action = "train"

    if action == "train":
        res_root = os.path.join(root, str(job_id), automl_path)
        if os.path.exists(res_root + "/weights") and len(os.listdir(res_root + "/weights")) > 0:
            res_root = os.path.join(res_root, "weights")
        if os.path.exists(os.path.join(res_root, action)):
            res_root = os.path.join(res_root, action)
        if os.path.exists(res_root):
            result_file = _get_model_results_path(handler_metadata=handler_metadata, job_id=job_id, res_root=res_root)
        else:
            result_file = None

    elif action == "prune":
        result_file = (glob.glob(f"{os.path.join(root, str(job_id))}/**/*.tlt", recursive=True) + glob.glob(f"{os.path.join(root, str(job_id))}/**/*.hdf5", recursive=True) + glob.glob(f"{os.path.join(root, str(job_id))}/**/*.pth", recursive=True))[0]

    elif action == "export":
        result_file = (glob.glob(f"{os.path.join(root, str(job_id))}/**/*.onnx", recursive=True) + glob.glob(f"{os.path.join(root, str(job_id))}/**/*.uff", recursive=True))[0]

    elif action in ("trtexec", "gen_trt_engine"):
        result_file = os.path.join(root, str(job_id), "model.engine")
        if not os.path.exists(result_file):
            result_file = os.path.join(root, str(job_id), action, "model.engine")
    else:
        result_file = None

    return result_file


def get_model_bundle_root(user_id, experiment_id, ngc_runner_fetch=True):
    """Returns the path to the model bundle directory"""
    model_root = os.path.join(get_root(ngc_runner_fetch=ngc_runner_fetch), user_id, "experiments", experiment_id)
    return os.path.join(model_root, "bundle")


def get_model_name(bundle_name):
    """Remove the version number from the bundle name to get the model name for TIS"""
    return re.sub(r"_v\d+\.\d+\.\d+", "", bundle_name)


def copy_bundle_base_experiment2model(base_experiment_id, user_id, experiment_id, patterns=[r'(.+?)_v\d+\.\d+\.\d+'], job_id=None):
    """
    Copies the pre-trained model to the model directory. Except the pre-trained weights, the whole bundle will be copied.
    - If the base_experiment is from NGC, then the directory will be under {admin_uuid}/experiments/<experiment_id>:
        ├── metadata.json
        └── spleen_deepedit_annotation_v1.2.3 （not a real version)
            ├── configs
            ├── docs
            ├── LICENSE
            └── models

    - If the base_experiment is from a previously-trained model, the job_id has to be provided to locate the model.
      The <user_id>/experiments/<experiment_id>/<job_id> directory will be:
        ├── status.json
        └── spleen_deepedit_annotation_v1.2.3 （not a real version)
            ├── configs
            ├── docs
            ├── LICENSE
            └── models

    Args:
        base_experiment_id (str): The PTM ID
        user_id (str): The user ID
        experiment_id (str): The model ID
        patterns (list): The regex pattern to match the PTM name
        job_id (str): The job ID

    Returns:
        bool: True if the PTM is copied successfully, False otherwise
        str: message why it failed, or the name of the bundle if successful
    """
    # base_experiment_root = get_handler_root(admin_uuid, "experiments", admin_uuid, base_experiment_id)
    # This change is based on:
    # https://nvidia.slack.com/archives/C0696NHU7A7/p1704767995516689
    base_experiment_root = get_base_experiment_path(base_experiment_id, create_if_not_exist=False)
    base_experiment_root = base_experiment_root if job_id is None else os.path.join(base_experiment_root, str(job_id))
    if not os.path.isdir(base_experiment_root):
        # the base experiment is not a directory only if job_id is provided
        base_experiment_root = get_handler_root(user_id=user_id, kind="experiments", handler_id=base_experiment_id, ngc_runner_fetch=False)
        base_experiment_root = os.path.join(base_experiment_root, str(job_id))

    if not os.path.isdir(base_experiment_root):
        return False, None, "PTM not found"

    # Find an item that matches one of the patterns
    # e.g. spleen_deepedit_annotation_v1.2.3 （not a real version)
    matching_item = find_matching_bundle_dir(base_experiment_root, patterns)
    if not matching_item:
        return False, None, "No matching item found"

    src = os.path.join(base_experiment_root, matching_item)
    # dst_root = /shared/users/<user_id>/experiments/<experiment_id>/bundle/<tis_model_name>
    dst_root = os.path.join(get_model_bundle_root(user_id, experiment_id, ngc_runner_fetch=False), get_model_name(matching_item))
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
    output_ext = override.get("output_ext", ".nrrd")
    output_dtype = override.get("output_dtype", "uint8")
    # can be customized by the user
    override["output_ext"] = output_ext
    override["output_dtype"] = output_dtype

    # no need and not open for customization
    output_dir = "inference_results"
    output_postfix = "seg"
    override["output_postfix"] = output_postfix
    override["separate_folder"] = False
    override["output_dir"] = output_dir
    override["dataset#data"] = [{}]

    return TEMPLATE_TIS_MODEL.format(
        bundle_name=bundle_name,
        output_ext=output_ext,
        output_postfix=output_postfix,
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
            return validate_medical_bundle_params(override)
        return False, f"Value of key `override` in model_params should be a dict, got {type(override)}."

    return True, ""


def get_medical_bundle_path(src_root):
    """
    Get the first folder path that has a medical bundle.
    """
    src_root_contents = os.listdir(src_root)
    for content in src_root_contents:
        content_path = os.path.join(src_root, content)
        if os.path.isdir(content_path):
            medical_bundle_content = ["configs", "models", "docs", "LICENSE"]
            medical_bundle_paths = [os.path.join(content_path, x) for x in medical_bundle_content]
            paths_exist = [os.path.exists(x) for x in medical_bundle_paths]
            if not all(paths_exist):
                continue
            return Code(201, content, "Got path!")
    return Code(404, {}, "Cannot export medical bundle.")


def prep_tis_model_repository(model_params, base_experiment_id, user_id, experiment_id, patterns=[r'(.+?)_v\d+\.\d+\.\d+'], job_id=None, update_model=False):
    """Prepare the model repository for Triton Inference Server"""
    # Copy the PTM to the model directory
    success, bundle_name, msg = copy_bundle_base_experiment2model(base_experiment_id, user_id, experiment_id, patterns, job_id)
    if not success:
        # should return 4 values to unify with successful return
        return False, None, msg, None
    check_result, check_msg = _check_tis_model_params(model_params)
    if not check_result:
        return False, None, check_msg, None

    # Generate {model_repository}/{tis_model}/{model_version}/model.py
    model_script = generate_tis_model_script(bundle_name, model_params)
    tis_model = get_model_name(bundle_name)
    tis_model_path = os.path.join(get_model_bundle_root(user_id, experiment_id, ngc_runner_fetch=False), tis_model)
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
        config_pbtxt_path = os.path.join(get_model_bundle_root(user_id, experiment_id, ngc_runner_fetch=False), tis_model, "config.pbtxt")
        with open(config_pbtxt_path, "w", encoding="utf-8") as f:
            f.write(config_pbtxt)

    return True, tis_model, "Triton Inference Server model repository prepared successfully", model_bundle_metadata


def validate_medical_bundle_params(model_params):
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


def generate_bundle_train_script(job_root, bundle_name, status_file, override=None):
    """Generate the bundle train script"""
    bundle_root = os.path.join(job_root, bundle_name)
    path_prefix = bundle_root + "/"
    override = override if override is not None else {}
    config_file_str = ensure_script_format(override.pop("config_file", "configs/train.json"), prefix=path_prefix)
    logging_file_str = ensure_script_format(override.pop("logging_file", "configs/logging.conf"), prefix=path_prefix)
    meta_file_str = ensure_script_format(override.pop("meta_file", "configs/metadata.json"), prefix=path_prefix)
    return TEMPLATE_MB_TRAIN.format(
        bundle_root=bundle_root,
        status_file=status_file,
        override=override,
        config_file_str=config_file_str,
        logging_file_str=logging_file_str,
        meta_file_str=meta_file_str,
    )


def generate_dicom_segmentation_convert_script(config, labels):
    """Generate the dicom segmentation convert script"""
    train_datalist_path = config.get("train#dataset#data", "train_datalist.json")[1:]  # trim % from the beginning
    valid_datalist_path = config.get("validate#dataset#data", "validate_datalist.json")[1:]

    return TEMPLATE_DICOM_SEG_CONVERTER.format(labels=labels, train_datalist_path=train_datalist_path, valid_datalist_path=valid_datalist_path)


def generate_cl_script(notify_record, job_context, handler_root, logfile):
    """Generate the continual learning script"""
    job_context_dict = {
        "id": job_context.id,
        "parent_id": job_context.parent_id,
        "network": job_context.network,
        "action": job_context.action,
        "handler_id": job_context.handler_id,
        "user_id": job_context.user_id,
        "kind": job_context.kind,
        "created_on": job_context.created_on,
        "last_modified": job_context.last_modified,
        "specs": job_context.specs,
    }
    return TEMPLATE_CONTINUAL_LEARNING.format(
        notify_record=notify_record,
        job_context_dict=job_context_dict,
        handler_root=handler_root,
        logfile=logfile
    )


# Helper constants
_OD_NETWORKS = set(["detectnet_v2", "faster_rcnn", "yolo_v3", "yolo_v4", "yolo_v4_tiny", "dssd", "ssd", "retinanet", "efficientdet_tf1", "efficientdet_tf2", "deformable_detr", "dino"])
_PURPOSE_BUILT_MODELS = set(["action_recognition", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pose_classification", "re_identification", "centerpose", "visual_changenet"])

_TF1_NETWORKS = set(["detectnet_v2", "faster_rcnn", "yolo_v4", "yolo_v4_tiny", "yolo_v3", "dssd", "ssd", "retinanet", "unet", "mask_rcnn", "lprnet", "classification_tf1", "efficientdet_tf1", "multitask_classification", "bpnet", "fpenet"])
_TF2_NETWORKS = set(["classification_tf2", "efficientdet_tf2"])
_PYT_TAO_NETWORKS = set(["action_recognition", "deformable_detr", "dino", "mal", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pointpillars", "pose_classification", "re_identification", "centerpose", "segformer", "visual_changenet"])
_PYT_PLAYGROUND_NETWORKS = set(["classification_pyt"])
_PYT_CV_NETWORKS = _PYT_TAO_NETWORKS | _PYT_PLAYGROUND_NETWORKS

VALID_DSTYPES = ("object_detection", "semantic_segmentation", "image_classification",
                 "instance_segmentation", "character_recognition",  # CV
                 "bpnet", "fpenet",  # DRIVEIX
                 "action_recognition", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pointpillars", "pose_classification", "re_identification", "centerpose", "visual_changenet")  # PYT CV MODELS
VALID_NETWORKS = ("detectnet_v2", "faster_rcnn", "yolo_v4", "yolo_v4_tiny", "yolo_v3", "dssd", "ssd", "retinanet",
                  "unet", "mask_rcnn", "lprnet", "classification_tf1", "classification_tf2", "efficientdet_tf1", "efficientdet_tf2", "multitask_classification",
                  "bpnet", "fpenet",  # DRIVEIX
                  "medical_vista3d", "medical_segmentation", "medical_annotation", "medical_classification", "medical_detection", "medical_automl", "medical_custom",  # MEDICAL
                  "action_recognition", "classification_pyt", "mal", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pointpillars", "pose_classification", "re_identification", "centerpose", "visual_changenet", "deformable_detr", "dino", "segformer",  # PYT CV MODELS
                  "annotations", "analytics", "augmentation", "auto_label")  # Data_Service tasks.
NO_SPEC_ACTIONS_MODEL = ("evaluate", "retrain", "inference", "inference_seq", "inference_trt")  # Actions with **optional** specs
NO_PTM_MODELS = set([])  # These networks don't have a pretrained model that can be downloaded from ngc model registry
_ITER_MODELS = set(["segformer"])  # These networks operate on iterations instead of epochs

BACKBONE_AND_FULL_MODEL_PTM_SUPPORTING_NETWORKS = set(["dino", "classification_pyt"])  # These networks have fields in their config file which has both backbone only loading weights as well as full architecture loading; ex: model.pretrained_backbone_path and train.pretrained_model_path in dino

AUTOML_DISABLED_NETWORKS = ["mal"]  # These networks can't support AutoML
NO_VAL_METRICS_DURING_TRAINING_NETWORKS = set(["bpnet", "multitask_classification", "unet"])  # These networks can't support writing validation metrics at regular intervals during training, only at end of training they run evaluation
MISSING_EPOCH_FORMAT_NETWORKS = set(["bpnet", "classification_pyt", "detectnet_v2", "fpenet", "pointpillars", "efficientdet_tf1", "faster_rcnn", "mask_rcnn", "segformer", "unet"])  # These networks have the epoch/iter number not following a format; ex: 1.pth instead of 001.pth
STATUS_JSON_MISMATCH_WITH_CHECKPOINT_EPOCH = set(["pointpillars", "detectnet_v2", "bpnet", "fpenet"])  # status json epoch number is 1 less than epoch number generated in checkppoint file

MEDICAL_DATASET_DEFAULT_SPECS = {
    "next_image_strategy": "sequential",
    "cache_image_url": "",
    "cache_force": False,
    "notify_study_urls": [],
    "notify_image_urls": [],
    "notify_label_urls": [],
}

VALID_MODEL_DOWNLOAD_TYPE = ("medical_bundle", "tao")
MEDICAL_NETWORK_ARCHITECT = ["medical_vista3d", "medical_segmentation", "medical_annotation", "medical_classification", "medical_detection", "medical_automl", "medical_custom"]
CACHE_TIME_OUT = 60 * 60  # cache timeout period in second
LAST_ACCESS_TIME_OUT = 60  # last access timeout period in second
network_metric_mapping = {"action_recognition": "val_acc",
                          "bpnet": "loss",
                          "centerpose": "val_3DIoU",
                          "classification_pyt": "accuracy_top-1",
                          "classification_tf1": "validation_accuracy",
                          "classification_tf2": "val_accuracy",
                          "deformable_detr": "val_mAP50",
                          "detectnet_v2": "mean average precision",
                          "dino": "val_mAP50",
                          "dssd": "mean average precision",
                          "efficientdet_tf1": "AP50",
                          "efficientdet_tf2": "AP50",
                          "faster_rcnn": "mean average precision",
                          "fpenet": "evaluation_cost ",
                          "lprnet": "validation_accuracy",
                          "ml_recog": "val Precision at Rank 1",
                          "multitask_classification": "mean accuracy",
                          "mask_rcnn": "mask_AP",
                          "ocdnet": "hmean",
                          "ocrnet": "val_acc",
                          "optical_inspection": "val_acc",
                          "pointpillars": "loss",
                          "pose_classification": "val_acc",
                          "re_identification": "cmc_rank_1",
                          "retinanet": "mean average precision",
                          "ssd": "mean average precision",
                          "segformer": "Mean IOU",
                          "unet": "loss",
                          "yolo_v3": "mean average precision",
                          "yolo_v4": "mean average precision",
                          "yolo_v4_tiny": "mean average precision",
                          "visual_changenet": "val_acc"}

CONTINUOUS_STATUS_KEYS = ["cur_iter", "epoch", "max_epoch", "eta", "time_per_epoch", "time_per_iter", "key_metric"]
