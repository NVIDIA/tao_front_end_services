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
Helper classes, functions, constants

Classes:
- Code
- JobContext
- StatusParser

Functions:
- run_system_command
- load_json_spec
- search_for_ptm
- search_for_dataset
- get_admin_api_key
- get_request_with_retry
- download_ngc_model
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
import io
import glob
import datetime
import json
import subprocess
import sys
import re
import uuid
import math
import traceback
import base64
import requests
import zipfile
from kubernetes import client, config

from handlers.stateless_handlers import get_handler_root, get_handler_job_metadata


NUM_OF_RETRY = 3


# Helper Classes
class TAOResponse:
    """Helper class for API reponse"""

    def __init__(self, code, data):
        """Initialize TAOResponse helper class"""
        self.code = code
        self.data = data


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
    def __init__(self, job_id, parent_id, network, action, handler_id, created_on=None, specs=None):
        """Initialize JobContext class"""
        # Non-state variables
        self.id = job_id
        self.parent_id = parent_id
        self.network = network
        self.action = action
        self.handler_id = handler_id
        self.created_on = created_on
        if not self.created_on:
            self.created_on = datetime.datetime.now().isoformat()

        # State variables
        self.last_modified = datetime.datetime.now().isoformat()
        self.status = "Pending"  # Starts off like this
        self.result = {}
        self.specs = specs

        self.write()

    def write(self):
        """Write the schema dict to jobs_metadata/job_id.json file"""
        # Create a job metadata
        job_metadata_file = get_handler_root(self.handler_id) + f"/jobs_metadata/{self.id}.json"
        with open(job_metadata_file, "w", encoding='utf-8') as f:
            f.write(json.dumps(self.schema(), indent=4))

    def __repr__(self):
        """Returns the schema dict"""
        return self.schema().__repr__()

    # ModelHandler / DatasetHandler interacts with this function
    def schema(self):
        """Creates schema dict based on the member variables"""
        _schema = {  # Cannot modify
            "id": self.id,
            "parent_id": self.parent_id,
            "action": self.action,
            "created_on": self.created_on,
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
        for key in ["cur_iter", "epoch", "max_epoch", "eta", "time_per_epoch", "time_per_iter"]:
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
                if key in ["cur_iter", "epoch", "max_epoch", "eta", "time_per_epoch", "time_per_iter"]:
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
                elif (self.network in ("pointpillars", "detectnet_v2", "bpnet", "fpenet") and tuple_var[0] < brain_epoch_number) or (self.network not in ("pointpillars", "detectnet_v2", "bpnet", "fpenet") and tuple_var[0] <= brain_epoch_number):
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
                                with open(automl_root + "/brain.json", 'r', encoding='utf-8') as u:
                                    brain_dict = json.loads(u.read())
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

        if self.network in ("pointpillars", "detectnet_v2", "bpnet", "fpenet"):  # status json epoch number is 1 less than epoch number generated in checkppoint file
            self.best_epoch_number += 1
            self.latest_epoch_number += 1
        elif automl_algorithm in ("hyperband", "h"):
            if self.network in _PURPOSE_BUILT_MODELS or self.network in ("deformable_detr", "dino"):  # epoch number in checkpoint starts from 0 or models whose validation logs are generated before the training logs
                self.best_epoch_number -= 1
        print(f"Metric returned is {metric_value} at best epoch/iter {self.best_epoch_number} while latest epoch/iter is {self.latest_epoch_number}", file=sys.stderr)
        return metric_value + 1e-07, self.best_epoch_number, self.latest_epoch_number


# Helper Functions
def load_json_spec(spec_json_path):
    """Load json and delete version key if present in the csv specs"""
    try:
        spec = {}
        with open(spec_json_path, mode='r', encoding='utf-8-sig') as f:
            spec = json.load(f)
            if spec.get("version"):
                del spec["version"]
        return spec
    except:
        return {}


def run_system_command(command):
    """
    Run a linux command - similar to os.system().
    Waits till process ends.
    """
    subprocess.run(['/bin/bash', '-c', command], stdout=subprocess.PIPE, check=False)
    return 0


def search_for_dataset(root):
    """Return path of the dataset file"""
    datasets = glob.glob(root + "/*.tar.gz", recursive=False) + glob.glob(root + "/*.tgz", recursive=False) + glob.glob(root + "/*.tar", recursive=False)

    if datasets:
        dataset_path = datasets[0]  # pick one arbitrarily
        return dataset_path
    return None


def search_for_ptm(root, extension="tlt", network=""):
    """Return path of the PTM file under the PTM root folder"""
    # from root, return model
    # if return is None, that means not hdf5 or tlt inside the folder
    # search for hdf5 / tlt

    # EfficientDet tf2 PTM is a not a single file
    if network in ["classification_tf2", "efficientdet_tf2"]:
        pretrained_root_folder_map = {"classification_tf2": "pretrained_classification_tf2_vefficientnet_b0",
                                      "efficientdet_tf2": "pretrained_efficientdet_tf2_vefficientnet_b0"}
        if len(glob.glob(root + "/**/*")) > 0:
            return os.path.join(root, pretrained_root_folder_map[network])
        return None
    models = glob.glob(root + "/**/*.tlt", recursive=True) + glob.glob(root + "/**/*.hdf5", recursive=True) + glob.glob(root + "/**/*.pth", recursive=True) + glob.glob(root + "/**/*.pth.tar", recursive=True)
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


def download_dataset(handler_dataset):
    """Calls wget and untar"""
    if handler_dataset is None:
        return None
    dataset_root = get_handler_root(handler_dataset)
    dataset_file = search_for_dataset(dataset_root)
    if dataset_file is None:
        dataset_download_command = get_dataset_download_command(dataset_root)  # this will not be None since we check this earlier
        if dataset_download_command:
            subprocess.run(['/bin/bash', '-c', 'HOME=/var/www/ && ' + dataset_download_command], stdout=subprocess.PIPE, check=False)
        dataset_file = search_for_dataset(dataset_root)
    return dataset_file


def get_admin_api_key():
    """Get api key from k8s secret"""
    try:
        if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
            config.load_kube_config()
        else:
            config.load_incluster_config()
        image_pull_secret = os.getenv('IMAGEPULLSECRET', default='imagepullsecret')
        secret = client.CoreV1Api().read_namespaced_secret(image_pull_secret, "default")

        encoded_key = base64.b64decode(next(iter(secret.data.values())))
        api_key = json.loads(encoded_key)["auths"]["nvcr.io"]["password"]

        return api_key
    except Exception as e:
        print(f"Failed to obtain api key from k8s: {e}", file=sys.stderr)
        return ""


def get_request_with_retry(url, headers, retry=0):
    """API GET Request with 3 retries"""
    r = requests.get(url, headers=headers)
    if not r.ok:
        if retry < NUM_OF_RETRY:
            print(f"Retrying {retry} time(s) to GET {url}.", file=sys.stderr)
            return get_request_with_retry(url, headers, retry + 1)
        print(f"Request to GET {url} failed.", file=sys.stderr)
    return r


def download_ngc_model(ngc_path, ptm_root):
    """Download NGC models with admin secret"""
    if ngc_path == "":
        print("Invalid ngc path.", file=sys.stderr)
        return

    ngc_configs = ngc_path.split('/')
    org = ngc_configs[0]
    model, version = ngc_configs[-1].split(':')
    team = ""
    if len(ngc_configs) == 3:
        team = ngc_configs[1]

    # Get access token using k8s admin secret
    api_key = get_admin_api_key()
    if not api_key:
        return

    url = 'https://authn.nvidia.com/token?service=ngc'
    headers = {'Accept': 'application/json', 'Authorization': 'ApiKey ' + api_key}
    response = get_request_with_retry(url, headers=headers)
    if not response.ok:
        return

    token = response.json()["token"]
    headers = {"Authorization": f"Bearer {token} "}
    url_substring = ""
    if team:
        url_substring = f"team/{team}"
    base_url = "https://api.ngc.nvidia.com"
    endpoint = f"v2/org/{org}/{url_substring}/models/{model}/versions/{version}/zip".replace("//", "/")
    url = f"{base_url}/{endpoint}"
    response = get_request_with_retry(url, headers=headers)
    if not response.ok:
        return

    dest_path = f"{ptm_root}/{model}_v{version}"
    os.makedirs(dest_path, exist_ok=True)
    os.chmod(dest_path, 0o777)

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(dest_path)

    print(f"PTM {model}:{version} downloaded successfully.", file=sys.stderr)


def download_ptm(handler_ptm):
    """Calls the ngc model download command and removes the unnecessary files for some models containing multiple model files"""
    if handler_ptm is None:
        return None
    ptm_root = get_handler_root(handler_ptm)
    ptm_file = search_for_ptm(ptm_root)
    if ptm_file is None:
        # check if metadata exists
        metadata = glob.glob(ptm_root + "/metadata.json")
        if not metadata:
            return None
        metadata = metadata[0]
        # read metadata ngc_path
        with open(metadata, "r", encoding='utf-8') as f:
            meta_data = json.load(f)
        ngc_path = meta_data.get("ngc_path", "")
        network_arch = meta_data.get("network_arch", "")
        additional_id_info = meta_data.get("additional_id_info", "")

        download_ngc_model(ngc_path, ptm_root)
        # if prc failed => then ptm_file is None and we proceed without a ptm (because if ptm does not exist in ngc, it must not be loaded!)
        ptm_file = search_for_ptm(ptm_root, network=network_arch)

        if network_arch == "lprnet":
            if additional_id_info == "us":
                os.system(f"rm {ptm_root}/lprnet_vtrainable_v1.0/*ch_*")
            elif additional_id_info == "ch":
                os.system(f"rm {ptm_root}/lprnet_vtrainable_v1.0/*us_*")
        elif network_arch == "action_recognition":
            additional_id_info_list = additional_id_info.split(",")
            if len(additional_id_info_list) == 1:
                if additional_id_info_list[0] == "3d":
                    os.system(f"rm {ptm_root}/actionrecognitionnet_vtrainable_v1.0/*_2d_*")
                elif additional_id_info_list[0] == "2d":
                    os.system(f"rm {ptm_root}/actionrecognitionnet_vtrainable_v1.0/*_3d_*")
            if len(additional_id_info_list) == 2:
                for ind_additional_id_info in additional_id_info_list:
                    if ind_additional_id_info == "a100":
                        os.system(f"rm {ptm_root}/actionrecognitionnet_vtrainable_v2.0/*xavier*")
                    elif ind_additional_id_info == "xavier":
                        os.system(f"rm {ptm_root}/actionrecognitionnet_vtrainable_v2.0/*a100*")
                    if ind_additional_id_info == "3d":
                        os.system(f"rm {ptm_root}/actionrecognitionnet_vtrainable_v2.0/*_2d_*")
                    elif ind_additional_id_info == "2d":
                        os.system(f"rm {ptm_root}/actionrecognitionnet_vtrainable_v2.0/*_3d_*")
        return ptm_file
    return ptm_file


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


def get_train_spec(handler_root):
    """Read and return the train.json spec"""
    spec = load_json_spec(handler_root + "/../specs/train.json")
    return spec


def get_total_epochs(handler_root, automl=False, automl_experiment_id=None):
    """Get the epoch/iter number from train.json"""
    spec = {}
    if automl:
        json_spec_path = os.path.join(handler_root, f"recommendation_{automl_experiment_id}.json")
        if os.path.exists(json_spec_path):
            with open(json_spec_path, "r", encoding='utf-8') as f:
                spec = json.load(f)
    if not spec:
        spec = get_train_spec(handler_root)
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


def validate_uuid(user_id=None, dataset_id=None, job_id=None, model_id=None):
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
    if model_id:
        try:
            uuid.UUID(model_id)
        except:
            return "Model ID passed is not a valid UUID"
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

    if network in ("bpnet", "classification_pyt", "detectnet_v2", "fpenet", "pointpillars", "efficientdet_tf1", "faster_rcnn", "mask_rcnn", "segformer", "unet"):
        format_epoch_number = str(epoch_number)
    else:
        format_epoch_number = f"{epoch_number:03}"

    result_file = _get_result_file_path(network=network, checkpoint_function=checkpoint_function, res_root=res_root, format_epoch_number=format_epoch_number)
    if (not result_file) and (checkpoint_choose_method in ("best_model", "from_epoch_number")):
        print("Couldn't find the epoch number requested or the checkpointed associated with the best metric value, defaulting to latest_model", file=sys.stderr)
        checkpoint_function = latest_model
        result_file = _get_result_file_path(network=network, checkpoint_function=checkpoint_function, res_root=res_root, format_epoch_number=format_epoch_number)

    return result_file


def get_model_results_path(handler_metadata, job_id):
    """Returns path of the model based on the action of the job"""
    if job_id is None:
        return None

    handler_id = handler_metadata.get("id")
    root = get_handler_root(handler_id)

    action = get_handler_job_metadata(handler_id, job_id).get("action")
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
                  "action_recognition", "classification_pyt", "mal", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pointpillars", "pose_classification", "re_identification", "centerpose", "visual_changenet", "deformable_detr", "dino", "segformer",  # PYT CV MODELS
                  "annotations", "analytics", "augmentation", "auto_label")  # Data_Service tasks.
NO_SPEC_ACTIONS_MODEL = ("evaluate", "retrain", "inference", "inference_seq", "inference_trt")  # Actions with **optional** specs
NO_PTM_MODELS = set([])
_ITER_MODELS = ("segformer")

AUTOML_DISABLED_NETWORKS = ["mal"]
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
