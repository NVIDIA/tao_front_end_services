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

"""Pipeline construction for all model actions"""
import os
import json
import threading
import time
import uuid
import copy
import sys
import traceback
import yaml

from job_utils import executor as jobDriver
from handlers.docker_images import DOCKER_IMAGE_MAPPER
from handlers.infer_params import CLI_CONFIG_TO_FUNCTIONS
from handlers.infer_data_sources import DS_CONFIG_TO_FUNCTIONS
from handlers.stateless_handlers import get_handler_root, get_handler_spec_root, get_handler_log_root, get_handler_job_metadata, get_handler_metadata, update_job_results, update_job_status, get_toolkit_status, load_json_data
from handlers.utilities import StatusParser, build_cli_command, write_nested_dict, read_nested_dict, read_network_config, load_json_spec, search_for_ptm, process_classwise_config, validate_gpu_param_value, NO_SPEC_ACTIONS_MODEL, _OD_NETWORKS, _TF1_NETWORKS, AUTOML_DISABLED_NETWORKS
from automl.utils import delete_lingering_checkpoints, wait_for_job_completion
from specs_utils import json_to_kitti, json_to_yaml

SPEC_BACKEND_TO_FUNCTIONS = {"protobuf": json_to_kitti.kitti, "yaml": json_to_yaml.yml}


class ActionPipeline:
    """
    ActionPipeline - Train, Evaluate, Retrain, Prune, Export, Gen_trt_engine (Model),
    TBD: DatasetConvert (for OD networks), Augment

    To spawn a job by handling all dependencies, monitor and close a job end-to-end
    - Inputs:
        - JobContext: To communicate with the Model / Dataset handler
        - Requires Handler & AppHandler to run()
    - Processes spec requirements
        - Prepares specs (generate_specs step)
            - dataset config (defined for each network's train, evaluate, retrain)
            - ptm config (for train, evaluate)
            - parent model and load graph (for retrain)
            - CLI paramters (for all actions)
            - Classwise configs (for all applicable train, evaluate, retrain) => currently for OD networks only
            - Converts json to spec backend

        - Prepare command (generate_run_command)
            - Generate run command
        - Infers image from config.json and platform information (if applicable) and sends it to K8s
        - Interacts with status.json parser (ETA: TBD) and communicated to Handlers through JobContext
        - Supports delete job
        - Supports resume for train
    - Exposed functions:
        - run():
        - delete():
        - resume(): Same as run()
    - Internal functions:
        - parse_status():
        - generate_config(): Assumes <action>.json exists
        - generate_run_command():
    - Helper functions():
        - __init__():
        - _read_api_params()
    """

    def __init__(self, job_context):
        """Initialize the ActionPipeline class"""
        # Job Context - bridge between Action and Handler
        self.job_context = job_context
        # Get some handler related data
        self.job_id = self.job_context.id
        self.network = self.job_context.network
        self.network_config = read_network_config(self.network)
        self.api_params = self._read_api_params()
        self.image = DOCKER_IMAGE_MAPPER[self.api_params.get("image", "")]
        self.handler_metadata = get_handler_metadata(self.job_context.handler_id)
        self.handler_spec_root = get_handler_spec_root(self.job_context.handler_id)
        self.handler_root = get_handler_root(self.job_context.handler_id)
        self.handler_log_root = get_handler_log_root(self.job_context.handler_id)
        self.handler_id = self.job_context.handler_id
        self.tao_deploy_actions = False
        self.action_suffix = ""
        self.parent_job_action = get_handler_job_metadata(self.handler_id, self.job_context.parent_id).get("action")
        if self.job_context.action in ("gen_trt_engine", "trtexec") or (self.parent_job_action in ("gen_trt_engine", "trtexec") and self.network != "bpnet"):
            self.tao_deploy_actions = True
            if self.job_context.action in ("evaluate", "inference") and self.job_context.network in _TF1_NETWORKS:
                self.action_suffix = "_tao_deploy"
        # This will be run inside a thread
        self.thread = None

        # Parameters to launch a job and monitor status
        self.job_name = str(self.job_context.id)

        self.spec = {}
        self.config = {}
        self.platform = None

        self.run_command = ""
        self.status_file = None

    def _read_api_params(self):
        """Read network config json file and return api_params key"""
        return self.network_config.get("api_params", {})

    def generate_config(self):
        """Generate config for this action; Actions may override"""
        return {}, {}

    def generate_run_command(self):
        """Generate run command for this action; Actions may override"""
        return "", None, None

    def post_run(self):
        """Run & modify internal variables after toolkit job is done; Actions may override"""
        return

    def run(self):
        """Calls necessary setup functions; calls job creation; monitors and update status of the job"""
        # Set up
        self.thread = threading.current_thread()
        try:
            # Generate config
            self.spec, self.config = self.generate_config()
            # Generate run command
            self.run_command, self.status_file, outdir = self.generate_run_command()
            # Pipe logs into logfile: <output_dir>/logs_from_toolkit.txt
            if not outdir:
                outdir = CLI_CONFIG_TO_FUNCTIONS["output_dir"](self.job_context, self.handler_metadata)
            logfile = os.path.join(self.handler_log_root, str(self.job_context.id) + ".txt")
            # Pipe stdout and stderr to logfile
            self.run_command += f" > {logfile} 2>&1 >> {logfile}"
            # After command runs, make sure subdirs permission allows anyone to enter and delete
            self.run_command += f"; find {outdir} -type d | xargs chmod 777"
            # After command runs, make sure artifact files permission allows anyone to delete
            self.run_command += f"; find {outdir} -type f | xargs chmod 666"
            # Optionally, pipe self.run_command into a log file
            print(self.run_command, self.status_file, file=sys.stderr)
            # Set up StatusParser
            status_parser = StatusParser(str(self.status_file), self.job_context.network, outdir)
            # Get image
            # If current or parent action is gen_trt_engine or trtexec, then it'a a tao-deploy container action
            if self.tao_deploy_actions:
                self.image = DOCKER_IMAGE_MAPPER["tao-deploy"]
            # Default image for dataset convert for OD networks is tlt-tf1, so override that
            elif self.job_context.action == "convert_efficientdet_tf2":
                self.image = DOCKER_IMAGE_MAPPER["tlt-tf2"]
            print(self.image, file=sys.stderr)

            # Convert self.spec to a backend and post it into a <self.handler_spec_root><job_id>.txt file
            if self.spec:
                if self.api_params["spec_backend"] == "json":
                    kitti_out = self.spec
                    kitti_out = json.dumps(kitti_out, indent=4)
                elif self.job_context.action == "convert_efficientdet_tf2":
                    kitti_out = SPEC_BACKEND_TO_FUNCTIONS["yaml"](self.spec)
                else:
                    kitti_out = SPEC_BACKEND_TO_FUNCTIONS[self.api_params["spec_backend"]](self.spec)
                # store as kitti
                action_spec_path_kitti = CLI_CONFIG_TO_FUNCTIONS["experiment_spec"](self.job_context, self.handler_metadata)
                with open(action_spec_path_kitti, "w", encoding='utf-8') as f:
                    f.write(kitti_out)

            # Submit to K8s
            # Platform is None, but might be updated in self.generate_config() or self.generate_run_command()
            # If platform is indeed None, jobDriver.create would take care of it.
            num_gpu = -1
            if self.job_context.action not in ['train', 'retrain', 'finetune']:
                num_gpu = 1
            jobDriver.create(self.job_name, self.image, self.run_command, num_gpu=num_gpu, accelerator=self.platform)
            print("Job created", self.job_name, file=sys.stderr)
            # Poll every 5 seconds
            k8s_status = jobDriver.status(self.job_name)
            while k8s_status in ["Done", "Error", "Running", "Pending", "Creating"]:
                time.sleep(5)
                # If Done, try running self.post_run()
                if k8s_status == "Done":
                    update_job_status(self.handler_id, self.job_id, status="Running")
                    # Retrieve status one last time!
                    new_results = status_parser.update_results()
                    update_job_results(self.handler_id, self.job_id, result=new_results)
                    try:
                        print("Post running", file=sys.stderr)
                        # If post run is done, make it done
                        self.post_run()
                        update_job_status(self.handler_id, self.job_id, status="Done")
                        break
                    except:
                        # If post run fails, call it Error
                        update_job_status(self.handler_id, self.job_id, status="Error")
                        break
                # If running in K8s, update results to job_context
                elif k8s_status == "Running":
                    update_job_status(self.handler_id, self.job_id, status="Running")
                    # Update results
                    new_results = status_parser.update_results()
                    update_job_results(self.handler_id, self.job_id, result=new_results)

                # Pending is if we have queueing systems down the road
                elif k8s_status == "Pending":
                    continue

                # Creating is if moebius-cloud job is in process of creating batch job
                elif k8s_status == "Creating":
                    # need to get current status and make sure its going from creating to running
                    # till now moebius-job manager would have created batchjob
                    k8s_status = jobDriver.status(self.job_name)
                    continue

                # If the job never submitted or errored out!
                else:
                    update_job_status(self.handler_id, self.job_id, status="Error")
                    break

                k8s_status = jobDriver.status(self.job_name)

            toolkit_status = get_toolkit_status(self.handler_id, self.job_id)
            print(f"Toolkit status for {self.job_id} is {toolkit_status}", file=sys.stderr)
            if toolkit_status != "SUCCESS" and self.job_context.action != "trtexec":
                update_job_status(self.handler_id, self.job_id, status="Error")

            final_status = get_handler_job_metadata(self.handler_id, self.job_id).get("status", "Error")
            print(f"Job Done: {self.job_name} Final status: {final_status}", file=sys.stderr)
            with open(logfile, "a", encoding='utf-8') as f:
                f.write("\nEOF\n")
            return

        except Exception:
            # Something went wrong inside...
            print(traceback.format_exc(), file=sys.stderr)
            print(f"Job {self.job_name} did not start", file=sys.stderr)
            update_job_status(self.handler_id, self.job_id, status="Error")
            update_job_results(self.handler_id, self.job_id, result={"detailed_status": {"message": "Error due to unmet dependencies"}})
            return


class CLIPipeline(ActionPipeline):
    """CLIPipeline for actions involve only cli params"""

    def __init__(self, job_context):
        """Initialize the CLIPipeline class"""
        super().__init__(job_context)

        self.network = job_context.network
        self.action = job_context.action
        # Handle anomalies in network action names
        if self.action == "retrain":
            self.action = "train"
        if self.action == "kmeans":
            self.network = "yolo_v3"
        if self.action == "augment":
            self.network = ""
        if self.network == "instance_segmentation" and self.action == "convert":
            self.network = "mask_rcnn"
            self.action = "dataset_convert"
        if self.network == "object_detection" and self.action == "convert":
            self.network = "detectnet_v2"
            self.action = "dataset_convert"
        if self.network == "object_detection" and "efficientdet" in self.action:
            self.network = self.action.replace("convert_", "")
            self.action = "dataset_convert"
        if self.network == "object_detection" and self.action == "convert_and_index":
            self.network = "ssd"
            self.action = "dataset_convert"

    def generate_config(self):
        """Generate config dictionary"""
        # Get some variables
        action = self.job_context.action
        # User stored CLI param in a json file
        spec_json_path = os.path.join(self.handler_spec_root, action + ".json")
        if os.path.exists(spec_json_path):
            config = load_json_spec(spec_json_path)
        else:
            config = {}
        network = self.job_context.network
        # Get CLI params from config json
        network_config = read_network_config(network)
        if action in network_config["cli_params"].keys():
            for field_name, inference_fn in network_config["cli_params"][f"{action}{self.action_suffix}"].items():
                field_value = CLI_CONFIG_TO_FUNCTIONS[inference_fn](self.job_context, self.handler_metadata)
                if field_value:
                    config[field_name] = field_value
        return {}, config

    def generate_run_command(self):
        """Generate run command"""
        overriden_output_dir = None
        if self.action == "dataset_convert":
            if self.network not in ("bpnet", "efficientdet_tf2", "fpenet", "ocrnet"):
                self.config["results_dir"] = CLI_CONFIG_TO_FUNCTIONS["output_dir"](self.job_context, self.handler_metadata)
            if self.network in ("efficientdet_tf1", "mask_rcnn"):
                self.config["output_dir"] = os.path.join(self.handler_root, "tfrecords")
                self.config["image_dir"] = os.path.join(self.handler_root, self.spec["dataset_convert"]["image_dir"])
                self.config["annotations_file"] = os.path.join(self.handler_root, self.spec["dataset_convert"]["annotations_file"])
                self.config["num_shards"] = self.spec["dataset_convert"]["num_shards"]
                self.config["tag"] = self.spec["dataset_convert"]["tag"]
                if self.network == "mask_rcnn":
                    self.config["include_masks"] = True
            elif self.network == "bpnet":
                if self.config["mode"] == "train":
                    self.config["output_filename"] = os.path.join(self.handler_root, "train")
                elif self.config["mode"] == "test":
                    self.config["output_filename"] = os.path.join(self.handler_root, "val")
                self.config["generate_masks"] = True
            elif self.network == "efficientdet_tf2":
                self.config["experiment_spec"] = CLI_CONFIG_TO_FUNCTIONS["experiment_spec"](self.job_context, self.handler_metadata)
            elif self.network in _OD_NETWORKS:
                self.config["output_filename"] = os.path.join(self.handler_root, "tfrecords/tfrecords")
                self.config["verbose"] = True
                self.config["dataset_export_spec"] = CLI_CONFIG_TO_FUNCTIONS["experiment_spec"](self.job_context, self.handler_metadata)

        if self.action == "inference":
            if self.network == "bpnet":
                self.config["dump_visualizations"] = True

        params_to_cli = build_cli_command(self.config, self.spec)
        run_command = f"{self.network} {self.action} {params_to_cli}"
        if self.action == "trtexec":
            run_command = f"{self.action} {params_to_cli}"

        status_file = os.path.join(self.handler_root, self.job_name, "status.json")
        if self.action == "dataset_convert" and self.network == "ocrnet":
            ds = self.handler_metadata.get("id")
            root = get_handler_root(ds)
            sub_folder = "train"
            if "test" in os.listdir(root):
                sub_folder = "test"
            status_file = f"{root}/{sub_folder}/lmdb/status.json"
            overriden_output_dir = os.path.dirname(status_file)
        return run_command, status_file, overriden_output_dir


# Specs are modified as well => Train, Evaluate, Retrain Actions
class TrainVal(CLIPipeline):
    """Class for model actions which involves both spec file as well as cli params"""

    def generate_config(self):
        """Generates spec and cli params
        Returns:
        spec: contains the network's spec file parameters
        config: contains cli params
        """
        network = self.job_context.network
        action = self.job_context.action
        # Infer CLI params
        config = {}
        network_config = read_network_config(network)
        if action in network_config["cli_params"].keys():
            for field_name, inference_fn in network_config["cli_params"][f"{action}{self.action_suffix}"].items():
                field_value = CLI_CONFIG_TO_FUNCTIONS[inference_fn](self.job_context, self.handler_metadata)
                if field_value:
                    config[field_name] = field_value

        # Read spec from <action>.json for train, resume train, evaluate, retrain. If not there, use train.json
        spec_json_path = os.path.join(self.handler_spec_root, action + ".json")
        if not os.path.exists(spec_json_path):
            if action in NO_SPEC_ACTIONS_MODEL:
                spec_json_path = os.path.join(self.handler_spec_root, action + "train.json")
        spec = load_json_spec(spec_json_path)
        if "experiment_spec_file" in network_config["cli_params"][f"{action}{self.action_suffix}"].keys() and network_config["cli_params"][f"{action}{self.action_suffix}"]["experiment_spec_file"] == "parent_spec_copied":
            spec_path = config["experiment_spec_file"]
            with open(spec_path, "r", encoding='utf-8') as spec_file:
                parent_spec = yaml.safe_load(spec_file)
            if action in parent_spec.keys() and action in spec.keys():
                parent_spec[action] = spec[action]
            if "dataset" in parent_spec.keys() and "dataset" in spec.keys():
                parent_spec["dataset"] = spec["dataset"]
            spec = parent_spec

        # Take .json file, read in spec params, infer spec params
        if action in network_config["spec_params"].keys():
            for field_name, inference_fn in network_config["spec_params"][action].items():
                field_value = CLI_CONFIG_TO_FUNCTIONS[inference_fn](self.job_context, self.handler_metadata)
                if field_value:
                    write_nested_dict(spec, field_name, field_value)
        validate_gpu_param_value(spec)

        # Move CLI params from spec to config
        spec_keys_all = copy.deepcopy(list(spec.keys()))  # Since we will be popping the value out, spec would change @ each iteration
        for field_name in spec_keys_all:
            cnd1 = field_name in network_config["cli_params"][action].keys()
            cnd2 = network_config["cli_params"][f"{action}{self.action_suffix}"].get(field_name, None) == "from_csv"
            cnd3 = type(spec[field_name]) in [str, float, int, bool]
            if cnd1 and cnd2 and cnd3:
                config[field_name] = spec.pop(field_name)
        print("Loaded specs", file=sys.stderr)

        # Infer dataset config
        spec = DS_CONFIG_TO_FUNCTIONS[network](spec, self.job_context, self.handler_metadata)
        print("Loaded dataset", file=sys.stderr)

        # Add classwise config
        classwise = self.api_params["classwise"] == "True"
        if classwise:
            spec = process_classwise_config(spec)

        return spec, config

    def post_run(self):
        """Carry's out functions after the job is executed"""
        # If efficientdet_tf1 copy pruned model so that evaluate can access via parent relation
        action = self.job_context.action
        if self.network in ("efficientdet_tf1", "efficientdet_tf2", "classification_tf2", "ocdnet") and action == "retrain":
            inference_fn = "parent_model"
            pruned_model_path = CLI_CONFIG_TO_FUNCTIONS[inference_fn](self.job_context, self.handler_metadata)
            _, file_extension = os.path.splitext(pruned_model_path)
            print(f"Copying pruned model {pruned_model_path} after retrain to {self.handler_root}/{self.job_id}/pruned_model{file_extension}\n", file=sys.stderr)
            os.system(f"cp {pruned_model_path} {self.handler_root}/{self.job_id}/pruned_model{file_extension}")


class ODConvert(CLIPipeline):
    """Class for Object detection networks which requires tfrecords conversion"""

    # def __init__(self,job_context):
    #     super().__init__(job_context)

    def generate_config(self):
        """Modify the spec parameters necessary for object detection convert and return the modified dictionary"""
        # Read json
        spec_json_path = os.path.join(self.handler_spec_root, f'{self.job_context.action}.json')
        spec = load_json_spec(spec_json_path)
        config = {}

        if self.network == "efficientdet_tf2":
            assert self.handler_metadata.get("format") == "coco"
            if spec.get("dataset_convert") is None:
                spec["dataset_convert"] = {}
            spec["dataset_convert"]["image_dir"] = os.path.join(self.handler_root, "images")
            spec["dataset_convert"]["annotations_file"] = os.path.join(self.handler_root, "annotations.json")
            spec["dataset_convert"]["results_dir"] = CLI_CONFIG_TO_FUNCTIONS["output_dir"](self.job_context, self.handler_metadata)

        # We don’t pass in the spec file to dataset convert process
        # to efficientdet_tf1/mask-rcnn, hence we need to set the configs here.
        # TODO: Have a common theme for all networks
        elif self.network not in ("efficientdet_tf1", "mask_rcnn"):
            assert self.handler_metadata.get("format") == "kitti"
            # Add some parameters to spec
            if spec.get("kitti_config") is None:
                spec["kitti_config"] = {}
            spec["kitti_config"]["image_dir_name"] = "images"
            spec["kitti_config"]["label_dir_name"] = "labels"
            spec["kitti_config"]["root_directory_path"] = self.handler_root + "/"
            spec["image_directory_path"] = self.handler_root + "/"
            if spec["kitti_config"].get("kitti_sequence_to_frames_file"):
                lname = spec["kitti_config"].get("kitti_sequence_to_frames_file")
                fullname = os.path.join(self.handler_root, lname)
                if os.path.exists(fullname):
                    spec["kitti_config"]["kitti_sequence_to_frames_file"] = fullname
        return spec, config

    def post_run(self):
        """Carry's out functions after the job is executed"""
        if self.network not in ("efficientdet_tf1", "mask_rcnn"):
            # Get classes information into a file
            categorical = get_handler_job_metadata(self.handler_id, self.job_id).get("result").get("categorical", [])
            classes = ["car", "person"]
            if len(categorical) > 0:
                cwv = categorical[0]["category_wise_values"]
                classes = [cat_val_dict["category"] for cat_val_dict in cwv]
            with open(os.path.join(self.handler_root, "classes.json"), "w", encoding='utf-8') as f:
                f.write(json.dumps(classes))
            # Remove warning file(s) from tfrecords directory
            tfwarning_path = os.path.join(self.handler_root, "tfrecords", "tfrecords_warning.json")
            if os.path.exists(tfwarning_path):
                os.remove(tfwarning_path)
            tfwarning_path_idx = os.path.join(self.handler_root, "tfrecords", "idx-tfrecords_warning.json")
            if os.path.exists(tfwarning_path_idx):
                os.remove(tfwarning_path_idx)


class UNETDatasetConvert(CLIPipeline):
    """Class for Unet's dataset tfrecords conversion"""

    def generate_config(self):
        """Modify the spec parameters necessary for object detection convert and return the modified dictionary"""
        spec_json_path = os.path.join(self.handler_spec_root, "convert.json")
        if os.path.exists(spec_json_path):
            spec = load_json_spec(spec_json_path)
        else:
            spec = {}

        config = {"coco_file": CLI_CONFIG_TO_FUNCTIONS["od_annotations"](self.job_context, self.handler_metadata),
                  "results_dir": os.path.join(self.handler_root, "masks")}
        if spec.get("num_files"):
            config["num_files"] = spec.get("num_files")

        return spec, config

    def generate_run_command(self):
        """Generate run command"""
        network = "unet"
        action = "dataset_convert"
        params_to_cli = build_cli_command(self.config)
        run_command = f"{network} {action} {params_to_cli}"
        status_file = os.path.join(self.handler_root, self.job_name, "status.json")
        return run_command, status_file, None

    def post_run(self):
        """Carry's out functions after the job is executed"""
        masks_dir = os.path.join(self.handler_root, "masks")
        images_dir = os.path.join(self.handler_root, "images")
        # write intersection to masks.txt and images.txt
        masks_txt = os.path.join(self.handler_root, "masks.txt")
        images_txt = os.path.join(self.handler_root, "images.txt")
        with open(images_txt, "w", encoding='utf-8') as im_file, open(masks_txt, "w", encoding='utf-8') as ma_file:
            available_masks = [m.split(".")[0] for m in os.listdir(masks_dir)]
            for image in os.listdir(images_dir):
                im_name = image.split(".")[0]
                if im_name in available_masks:
                    im_file.write(os.path.join(images_dir, image + "\n"))
                    ma_file.write(os.path.join(masks_dir, im_name + ".png\n"))


ODAugment = TrainVal


class Dnv2Inference(CLIPipeline):
    """Class for detectnet_v2 specific changes required during inference"""

    def generate_config(self):
        """Makes necessaary changes to the spec parameters for detectnet v2 inference"""
        network = "detectnet_v2"
        action = "inference"
        # Infer CLI params
        config = {}
        network_config = read_network_config(network)
        if action in network_config["cli_params"].keys():
            for field_name, inference_fn in network_config["cli_params"][f"{action}{self.action_suffix}"].items():
                field_value = CLI_CONFIG_TO_FUNCTIONS[inference_fn](self.job_context, self.handler_metadata)
                if field_value:
                    config[field_name] = field_value

        # Read spec from <action>.json for train, resume train, evaluate, retrain. If not there, use train.json
        spec_json_path = os.path.join(self.handler_spec_root, action + ".json")
        spec = load_json_spec(spec_json_path)  # Dnv2 NEEDS inference spec

        # As per regular TrainVal, do not infer spec params, no need to move spec to cli
        # No need to add dataset configs / classwise configs
        # Instead do the following: if parent is tlt, enter tlt config and parent is trt, enter trt config

        parent_job_id = self.job_context.parent_id
        parent_action = get_handler_job_metadata(self.handler_id, parent_job_id).get("action")  # This should not fail if dependency passed
        if parent_action in ["export", "gen_trt_engine", "trtexec"]:
            key = "inferencer_config.tensorrt_config.trt_engine"
        else:
            key = "inferencer_config.tlt_config.model"
        parent_model = CLI_CONFIG_TO_FUNCTIONS["parent_model_evaluate"](self.job_context, self.handler_metadata)
        if parent_model:
            write_nested_dict(spec, key, parent_model)

        # Move CLI params from spec to config
        spec_keys_all = copy.deepcopy(list(spec.keys()))  # Since we will be popping the value out, spec would change @ each iteration
        for field_name in spec_keys_all:
            cnd1 = field_name in network_config["cli_params"][action].keys()
            cnd2 = network_config["cli_params"][f"{action}{self.action_suffix}"].get(field_name, None) == "from_csv"
            cnd3 = type(spec[field_name]) in [str, float, int, bool]
            if cnd1 and cnd2 and cnd3:
                config[field_name] = spec.pop(field_name)

        return spec, config


class AutoMLPipeline(ActionPipeline):
    """AutoML pipeline which carry's out network specific param changes; generating run commands and creating job for individual experiments"""

    def __init__(self, job_context):
        """Initialize the AutoMLPipeline class"""
        super().__init__(job_context)
        self.job_root = self.handler_root + f"/{self.job_context.id}"
        self.rec_number = self.get_recommendation_number()
        self.expt_root = f"{self.job_root}/experiment_{self.rec_number}"
        self.recs_dict = load_json_data(json_file=f"{self.job_root}/controller.json")
        self.brain_dict = load_json_data(json_file=f"{self.job_root}/brain.json")

        if not os.path.exists(self.expt_root):
            os.makedirs(self.expt_root)

    def add_ptm_dependency(self, recommended_values):
        """Add PTM as a dependency if backbone or num_layers is part of hyperparameter sweep"""
        # See if a ptm is needed (if not searching num_layers / backbone, no PTM), just take default
        ptm_id = None
        if "backbone" in recommended_values.keys() or "num_layers" in recommended_values.keys():
            for dep in self.job_context.dependencies:
                if dep.type == "automl_ptm":
                    ptm_id = dep.name
                    break
        if ptm_id:
            recommended_values["ptm"] = search_for_ptm(get_handler_root(ptm_id))

    def generate_config(self, recommended_values):
        """Generate config for AutoML experiment"""
        spec_json_path = os.path.join(get_handler_spec_root(self.job_context.handler_id), "train.json")
        spec = load_json_spec(spec_json_path)

        epoch_multiplier = self.brain_dict.get("epoch_multiplier", None)
        if epoch_multiplier is not None:
            current_ri = int(self.brain_dict.get("ri", {"0": [float('-inf')]})[str(self.brain_dict.get("bracket", 0))][0])

        for param_type in ("automl_spec_params", "automl_cli_params"):
            for field_name, inference_fn in self.network_config[param_type].items():
                if "automl_" in inference_fn:
                    field_value = CLI_CONFIG_TO_FUNCTIONS[inference_fn](self.job_context, self.handler_metadata, self.job_root, self.rec_number)
                elif "assign_const_value" in inference_fn:
                    if epoch_multiplier:
                        field_value = int(epoch_multiplier * current_ri)
                        if self.network == "mask_rcnn":
                            field_value = int(field_value * (spec["num_examples_per_epoch"] / spec["train_batch_size"]))
                    else:
                        field_value = int(read_nested_dict(spec, field_name))
                        if "assign_const_value," in inference_fn:
                            dependent_parameter_names = inference_fn.split(",")
                            dependent_field_value = int(read_nested_dict(spec, dependent_parameter_names[1]))
                            if len(dependent_parameter_names) == 2:
                                field_value = min(field_value, dependent_field_value)
                            elif len(dependent_parameter_names) == 3:
                                field_value = int(read_nested_dict(spec, dependent_parameter_names[2]))

                    if self.network == "segformer" and "logging_interval" in field_name:
                        field_value -= 1
                else:
                    field_value = CLI_CONFIG_TO_FUNCTIONS[inference_fn](self.job_context, self.handler_metadata)
                if field_value:
                    if param_type == "automl_spec_params":
                        write_nested_dict(spec, field_name, field_value)
                    else:
                        self.config[field_name] = field_value

        spec = DS_CONFIG_TO_FUNCTIONS[self.network](spec, self.job_context, self.handler_metadata)
        print("Loaded AutoML specs", file=sys.stderr)

        for param_name, param_value in recommended_values.items():
            write_nested_dict(spec, param_name, param_value)
        validate_gpu_param_value(spec)

        # Move CLI params from spec to config
        spec_keys_all = copy.deepcopy(list(spec.keys()))  # Since we will be popping the value out, spec would change @ each iteration
        for field_name in spec_keys_all:
            cnd1 = field_name in self.network_config["automl_cli_params"].keys()
            cnd2 = self.network_config["automl_cli_params"].get(field_name, None) == "from_csv"
            cnd3 = type(spec[field_name]) in [str, float, int, bool]
            if cnd1 and cnd2 and cnd3:
                self.config[field_name] = spec.pop(field_name)

        if self.network not in AUTOML_DISABLED_NETWORKS:
            spec = process_classwise_config(spec)

        # Save specs to a yaml/kitti file
        updated_spec_string = SPEC_BACKEND_TO_FUNCTIONS[self.api_params["spec_backend"]](spec)
        action_spec_path = os.path.join(self.job_root, f"recommendation_{self.rec_number}.{self.api_params['spec_backend']}")

        with open(action_spec_path, "w", encoding='utf-8') as f:
            f.write(updated_spec_string)

    def generate_run_command(self):
        """Generate the command to be run inside docker for AutoML experiment"""
        params_to_cli = build_cli_command(self.config)
        run_command = f"{self.network} train {params_to_cli}"
        logfile = os.path.join(self.expt_root, "log.txt")
        run_command += f" > {logfile} 2>&1 >> {logfile}"
        return run_command

    def get_recommendation_number(self):
        """Return the current recommendation number"""
        rec_number = None
        for dep in self.job_context.dependencies:
            if dep.type == "automl":
                rec_number = int(dep.name)
                break
        return rec_number

    def write_json(self, file_path, json_dict):
        """Write a json file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_dict, f,
                      separators=(',', ':'),
                      sort_keys=True,
                      indent=4)

    def run(self):
        """Calls necessary setup functions; calls job creation; update status of the job"""
        try:
            recommended_values = self.recs_dict[self.rec_number].get("specs", {})
            self.add_ptm_dependency(recommended_values)
            self.generate_config(recommended_values)
            run_command = self.generate_run_command()

            # Assign a new job id if not assigned already
            job_id = self.recs_dict[self.rec_number].get("job_id", None)
            if not job_id:
                job_id = str(uuid.uuid4())
                print("New job id being assigned to recommendation", job_id, file=sys.stderr)
                self.recs_dict[self.rec_number]["job_id"] = job_id
                self.write_json(file_path=f"{self.job_root}/controller.json", json_dict=self.recs_dict)

            print(run_command, file=sys.stderr)

            # Wait for existing AutoML jobs to complete
            wait_for_job_completion(job_id)

            delete_lingering_checkpoints(self.recs_dict[self.rec_number].get("best_epoch_number", ""), self.expt_root)
            jobDriver.create(job_id, self.image, run_command, num_gpu=-1)
            print(f"AutoML recommendation with experiment id {self.rec_number} and job id {job_id} submitted", file=sys.stderr)
            k8s_status = jobDriver.status(job_id)
            while k8s_status in ["Done", "Error", "Running", "Pending", "Creating"]:
                time.sleep(5)
                if os.path.exists(os.path.join(self.expt_root, "log.txt")):
                    break
                if k8s_status == "Error":
                    print(f"Relaunching job {job_id}", file=sys.stderr)
                    wait_for_job_completion(job_id)
                    jobDriver.create(job_id, self.image, run_command, num_gpu=-1)
                k8s_status = jobDriver.status(job_id)

            return True

        except Exception:
            print(f"AutoMLpipeline for network {self.network} failed due to exception {traceback.format_exc()}", file=sys.stderr)
            job_id = self.recs_dict[self.rec_number].get("job_id", "")
            print(job_id, file=sys.stderr)

            self.recs_dict[self.rec_number]["status"] = "failure"
            self.write_json(file_path=f"{self.job_root}/controller.json", json_dict=self.recs_dict)

            update_job_status(self.handler_id, self.job_context.id, status="Error")
            jobDriver.delete(self.job_context.id)
            return False


# Each Element can be called with a job_context and returns an ActionPipeline (or its derivative) object
ACTIONS_TO_FUNCTIONS = {"train": TrainVal,
                        "evaluate": TrainVal,
                        "prune": CLIPipeline,
                        "prune_tf2": TrainVal,
                        "prune_with_spec": TrainVal,
                        "retrain": TrainVal,
                        "export": CLIPipeline,
                        "export_with_spec": TrainVal,
                        "export_tf2": TrainVal,
                        "inference": TrainVal,
                        "dnv2inference": Dnv2Inference,
                        "gen_trt_engine": TrainVal,
                        "trtexec": TrainVal,
                        "purpose_built_models_ds_convert": TrainVal,
                        "odconvert": ODConvert,
                        "pyt_odconvert": TrainVal,
                        "unetdatasetconvert": UNETDatasetConvert,
                        "odconvertindex": ODConvert,
                        "odconvertefficientdet_tf1": ODConvert,
                        "odconvertefficientdet_tf2": ODConvert,
                        "odaugment": ODAugment,
                        "data_services": TrainVal
                        }
