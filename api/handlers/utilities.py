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
- get_ngc_download_command
- get_model_results_path
- write_nested_dict
- search_list_for_best_model
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
import subprocess
import sys
import re
import uuid
import math

from handlers.stateless_handlers import get_handler_root, get_handler_job_metadata


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
    def __init__(self, job_id, parent_id, network, action, handler_id, created_on=None):
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

        self.last_seen_epoch = -1
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
            if type(cat_dict) != dict:
                return
            for _, value_dict in cat_dict.items():
                if type(value_dict) != dict:
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
            if type(kpi_dict) != dict:
                return

            for key, value in kpi_dict.items():
                if type(value) == dict:
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
            if type(gr_dict) != dict:
                return
            for key, value in gr_dict.items():
                plot_helper_dict = {}
                if type(value) == dict:
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
                    if key == "mean average precision":
                        # Mean average precision info is present in duplicate lines for these network
                        if self.network in ("dssd", "retinanet", "ssd", "yolo_v3", "yolo_v4", "yolo_v4_tiny"):
                            if "epoch" in status_dict and float_value:
                                float_value = None
                        if float_value is not None:
                            if self.last_seen_epoch not in self.results["graphical"][key]["values"].keys():
                                self.results["graphical"][key]["values"][self.last_seen_epoch] = float_value
                    else:
                        if self.last_seen_epoch not in self.results["graphical"][key]["values"].keys():
                            self.results["graphical"][key]["values"][self.last_seen_epoch] = float_value
                else:
                    if (key != "mean average precision") or (key == "mean average precision" and float_value):
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
            if (type(value) == str and value.lower() in ["nan", "infinity", "inf"]) or (type(value) == float and (math.isnan(value) or value == float('inf') or value == float('-inf'))):
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

    # if .tlt exists
    if models:
        model_path = models[0]  # pick one arbitrarily
        return model_path
    # if no .tlt exists
    return None


def get_ngc_download_command(root):
    """Frames a ngc command to download the PTM's from ngc"""
    # check if metadata exists
    metadata = glob.glob(root + "/metadata.json")
    if not metadata:
        return None
    metadata = metadata[0]
    # read metadata ngc_path
    with open(metadata, "r", encoding='utf-8') as f:
        meta_data = json.load(f)
    ngc_path = meta_data.get("ngc_path", "")
    network_arch = meta_data.get("network_arch", "")
    additional_id_info = meta_data.get("additional_id_info", "")

    # if no ngc patk
    if ngc_path == "":
        return None
    # if ngc path, then download the model into some place inside root and then return a path to hdf5 / tlt
    cmnd = f"TMPDIR=$(mktemp -d) && ngc registry model download-version --dest $TMPDIR/ {ngc_path} && chmod -R 777 $TMPDIR && cp -r $TMPDIR/* {root}/ && rm -rf $TMP_DIR"
    # run and wait till it finishes / run in background
    print("Executing NGC command: ", cmnd, file=sys.stderr)
    return cmnd, network_arch, additional_id_info


def download_ptm(handler_ptm):
    """Calls the ngc model download command and removes the unnecessary files for some models containing multiple model files"""
    if handler_ptm is None:
        return None
    ptm_root = get_handler_root(handler_ptm)
    ptm_file = search_for_ptm(ptm_root)
    if ptm_file is None:
        ptm_download_command, network_arch, additional_id_info = get_ngc_download_command(ptm_root)  # this will not be None since we check this earlier
        subprocess.run(['/bin/bash', '-c', 'HOME=/var/www/ && ' + ptm_download_command], stdout=subprocess.PIPE, check=False)
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
        if type(ptr) != dict:
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
        assert (type(value) != dict)
        assert (type(value) != list)
        if type(value) == bool:
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


# NOTE Deprecated function, but still used until classwise config is needed
# for any network - this follows dnv2 schema.
def process_classwise_config(data):
    """Modifies data to re-organize the classwise config into respective collections
    Args: data: spec with classwise_config
    Return: data
    """
    if "classwise_config" not in data:
        return data
    if type(data["classwise_config"]) != list:
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


def latest_model(files, delimiters="_", extensions=[".tlt", ".hdf5", ".pth"]):
    """Returns the latest generated model file based on epoch number"""
    cur_best = 0
    best_model = "model.tlt"
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
    return best_model


def search_list_for_best_model(files):
    """Returns the latest model based on anticipated model name using regex"""
    cur_best = 0
    best_model = "model.tlt"
    for file_name in files:
        # Patterns to look for
        # detectnet_v2, unet: model.tlt
        # frcnn: model.epoch.<>.tlt
        # Define regex rules for a potential model file
        model_name_regex = re.compile(r'(trained-)?(finetuned-)?model(-)?(.epoch)?([0-9]+)?.tlt')
        # Identify if current file is a model file
        model_name_out = model_name_regex.search(file_name)
        # If it is a valid file, proceed. Else continue the loop
        if model_name_out:
            # Try to extract the integer epoch string from the model file name
            model_pattern = model_name_out.group()
            model_stamp_regex = re.compile(r'[\d]+')
            model_stamp_out = model_stamp_regex.search(model_pattern)
            # Get the epoch number. If model_stamp_out is None, it means the file name is model.tlt, etc...
            model_number = 0
            if model_stamp_out:
                model_number = int(model_stamp_out.group())
            # If model's epoch better than current best, make that the best model
            # Useful when output directory has checkpoints
            if model_number >= cur_best:
                cur_best = model_number
                best_model = model_pattern
    return best_model


def get_model_results_path(handler_metadata, job_id):
    """Returns path of the model based on the action of the job"""
    if job_id is None:
        return None

    network = handler_metadata.get("network_arch")
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
            # If epoch number is baked into tlt output as <yada_yada>_<epoch_number>.tlt
            if network in ("classification_tf1", "classification_tf2", "classification_pyt", "efficientdet_tf2", "faster_rcnn", "multitask_classification", "dssd", "ssd", "retinanet", "yolo_v3", "yolo_v4", "yolo_v4_tiny", "segformer", "pointpillars"):
                result_file = res_root + "/" + latest_model(os.listdir(res_root))
            # If it follows model.tlt pattern with epoch number
            elif network in ("detectnet_v2", "lprnet", "efficientdet_tf1", "mask_rcnn", "unet", "bpnet", "fpenet"):
                result_file = res_root + "/" + latest_model(os.listdir(res_root), delimiters="-")
            elif network in _PYT_CV_NETWORKS:
                result_file = res_root + "/" + latest_model(os.listdir(res_root), delimiters="=")
            else:
                result_file = res_root + "/" + search_list_for_best_model(os.listdir(res_root))
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
_PURPOSE_BUILT_MODELS = set(["action_recognition", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pose_classification", "re_identification"])

_TF1_NETWORKS = set(["detectnet_v2", "faster_rcnn", "yolo_v4", "yolo_v4_tiny", "yolo_v3", "dssd", "ssd", "retinanet", "unet", "mask_rcnn", "lprnet", "classification_tf1", "efficientdet_tf1", "multitask_classification", "bpnet", "fpenet"])
_TF2_NETWORKS = set(["classification_tf2", "efficientdet_tf2"])
_PYT_TAO_NETWORKS = set(["action_recognition", "deformable_detr", "dino", "mal", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pointpillars", "pose_classification", "re_identification", "segformer"])
_PYT_PLAYGROUND_NETWORKS = set(["classification_pyt"])
_PYT_CV_NETWORKS = _PYT_TAO_NETWORKS | _PYT_PLAYGROUND_NETWORKS

VALID_DSTYPES = ("object_detection", "semantic_segmentation", "image_classification",
                 "instance_segmentation", "character_recognition",  # CV
                 "bpnet", "fpenet",  # DRIVEIX
                 "action_recognition", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pointpillars", "pose_classification", "re_identification")  # PYT CV MODELS
VALID_NETWORKS = ("detectnet_v2", "faster_rcnn", "yolo_v4", "yolo_v4_tiny", "yolo_v3", "dssd", "ssd", "retinanet",
                  "unet", "mask_rcnn", "lprnet", "classification_tf1", "classification_tf2", "efficientdet_tf1", "efficientdet_tf2", "multitask_classification",
                  "bpnet", "fpenet",  # DRIVEIX
                  "action_recognition", "classification_pyt", "mal", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pointpillars", "pose_classification", "re_identification", "deformable_detr", "dino", "segformer",  # PYT CV MODELS
                  "annotations", "analytics", "augmentation", "auto_label")  # Data_Service tasks.
NO_SPEC_ACTIONS_MODEL = ("evaluate", "retrain", "inference", "inference_seq", "inference_trt")  # Actions with **optional** specs
NO_PTM_MODELS = set([])
_ITER_MODELS = ("segformer")

AUTOML_DISABLED_NETWORKS = ["mal"]
