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

"""Pipeline construction for all experiment actions"""
import copy
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid

import yaml
from automl.utils import delete_lingering_checkpoints, wait_for_job_completion
from handlers import ngc_handler
from handlers.docker_images import DOCKER_IMAGE_MAPPER, DOCKER_IMAGE_VERSION
from handlers.infer_data_sources import DS_CONFIG_TO_FUNCTIONS
from handlers.infer_params import CLI_CONFIG_TO_FUNCTIONS
from handlers.ngc_handler import load_user_workspace_metadata
# TODO: force max length of characters in a line to be 120
from handlers.stateless_handlers import (admin_uuid, get_base_experiment_path, get_base_experiment_metadata, get_handler_job_metadata,
                                         get_handler_log_root, get_handler_metadata, get_handler_root,
                                         safe_dump_file, safe_load_file, update_job_metadata, update_job_status, write_handler_metadata,
                                         get_handler_spec_root, get_handler_user, get_root, get_toolkit_status, printc)
from handlers.utilities import (_OD_NETWORKS, _TF1_NETWORKS, AUTOML_DISABLED_NETWORKS, NO_SPEC_ACTIONS_MODEL,
                                StatusParser, archive_job_results, build_cli_command, generate_bundle_train_script, generate_cl_script,
                                generate_dicom_segmentation_convert_script, generate_job_tar_stats, get_total_epochs,
                                load_json_spec, process_classwise_config, read_nested_dict, read_network_config,
                                search_for_base_experiment, validate_gpu_param_value, validate_medical_bundle_params,
                                write_nested_dict)
from job_utils import executor as jobDriver
from network_utils.network_constants import ptm_mapper
from specs_utils import json_to_kitti, json_to_yaml
from utils.utils import remove_key_by_flattened_string

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
            - base_experiment config (for train, evaluate)
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
        self.handler_metadata = get_handler_metadata(self.job_context.user_id, self.job_context.handler_id)
        self.handler_spec_root = get_handler_spec_root(self.job_context.user_id, self.job_context.handler_id)
        self.handler_root = get_handler_root(self.job_context.user_id, None, self.job_context.handler_id, None, ngc_runner_fetch=True)
        self.handler_log_root = get_handler_log_root(self.job_context.user_id, self.job_context.handler_id)
        self.handler_id = self.job_context.handler_id
        self.tao_deploy_actions = False
        self.action_suffix = ""
        self.parent_job_action = get_handler_job_metadata(self.job_context.user_id, self.handler_id, self.job_context.parent_id).get("action")
        if self.job_context.action in ("gen_trt_engine", "trtexec") or (self.parent_job_action in ("gen_trt_engine", "trtexec") and self.network != "bpnet"):
            self.tao_deploy_actions = True
            if self.job_context.action in ("evaluate", "inference") and self.job_context.network in _TF1_NETWORKS:
                self.action_suffix = "_tao_deploy"
        self.image = DOCKER_IMAGE_MAPPER[self.api_params.get("image", "")]
        # If current or parent action is gen_trt_engine or trtexec, then it'a a tao-deploy container action
        if self.tao_deploy_actions:
            self.image = DOCKER_IMAGE_MAPPER["tao-deploy"]
        # Default image for dataset convert for OD networks is tlt-tf1, so override that
        elif self.job_context.action == "convert_efficientdet_tf2":
            self.image = DOCKER_IMAGE_MAPPER["tlt-tf2"]
        # Override version of image specific for networks
        if self.network in DOCKER_IMAGE_VERSION.keys():
            self.tao_framework_version, self.tao_model_override_version = DOCKER_IMAGE_VERSION[self.network]
            if self.tao_model_override_version not in self.image:
                self.image = self.image.replace(self.tao_framework_version, self.tao_model_override_version)
        # This will be run inside a thread
        self.thread = None

        # Parameters to launch a job and monitor status
        self.job_name = str(self.job_context.id)

        self.spec = {}
        self.config = {}
        self.platform = None

        self.run_command = ""
        self.status_file = None
        self.logfile = os.path.join(self.handler_log_root, str(self.job_context.id) + ".txt")
        self.ngc_runner = False
        self.workspaces_for_job = []
        if os.getenv("NGC_RUNNER", "") == "True":
            self.ngc_runner = True
            self.workspaces_for_job = ngc_handler.get_workspaces_for_job(self.job_context.user_id, self.job_context.handler_id, self.job_context.parent_id, self.handler_metadata)

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

    def generate_dgx_job_metadata(self, container_run_command, dgx_job_metadata, docker_env_vars):
        """Convert run command generated into format that """
        orgName, teamName, _, aceName = ngc_handler.get_ngc_admin_info()
        dgx_job_metadata["user_id"] = self.job_context.user_id
        dgx_job_metadata["orgName"] = orgName
        dgx_job_metadata["teamName"] = teamName
        dgx_job_metadata["command"] = container_run_command
        dgx_job_metadata["dockerImageName"] = self.image
        dgx_job_metadata["aceName"] = aceName
        dgx_job_metadata["aceInstance"] = "dgxa100.80g.1.norm"
        dgx_job_metadata["runPolicy"] = {}
        dgx_job_metadata["runPolicy"]["preemptClass"] = "RESUMABLE"

        dgx_job_metadata["resultContainerMountPoint"] = "/result"
        if self.job_context.action in ("train", "retrain"):
            dgx_job_metadata["aceInstance"] = "dgxa100.80g.2.norm"

        workspace_mount_list = []
        for workspace in self.workspaces_for_job:
            workspace_dict = {}
            workspace_dict["containerMountPoint"] = workspace['mount_path']
            workspace_dict["id"] = workspace['id']
            workspace_dict["mountMode"] = "RW"
            workspace_mount_list.append(workspace_dict)
        dgx_job_metadata["workspaceMounts"] = workspace_mount_list

        dgx_job_metadata["envs"] = [{"name": "TELEMETRY_OPT_OUT", "value": os.getenv('TELEMETRY_OPT_OUT', default='no')}]
        if docker_env_vars:
            for docker_env_var_key, docker_env_var_value in docker_env_vars.items():
                dgx_job_metadata["envs"].append({"name": docker_env_var_key, "value": docker_env_var_value})

    def handle_multiple_ptm_fields(self):
        """Remove one of end-end or backbone related PTM field based on the Handler metadata info"""
        if self.handler_metadata.get("is_ptm_backbone"):
            parameter_to_remove = ptm_mapper.get("end_to_end", {}).get(self.network)  # if ptm is a backbone remove end_to_end field from config and spec
        else:
            parameter_to_remove = ptm_mapper.get("backbone", {}).get(self.network)  # if ptm is not a backbone remove it field from config and spec
        if parameter_to_remove:
            remove_key_by_flattened_string(self.spec, parameter_to_remove)
            remove_key_by_flattened_string(self.config, parameter_to_remove)

    def detailed_print(self, *args, **kwargs):
        """Prints the details of the job to the console"""
        printc(*args, context=vars(self.job_context), **kwargs)

    def run(self):
        """Calls necessary setup functions; calls job creation; monitors and update status of the job"""
        # Set up
        self.thread = threading.current_thread()
        try:
            # Generate config
            self.spec, self.config = self.generate_config()
            self.handle_multiple_ptm_fields()
            # Generate run command
            self.run_command, self.status_file, outdir = self.generate_run_command()
            # Pipe logs into logfile: <output_dir>/logs_from_toolkit.txt
            if not outdir:
                outdir = CLI_CONFIG_TO_FUNCTIONS["output_dir"](self.job_context, self.handler_metadata)
            # Pipe stdout and stderr to logfile
            self.run_command += f" | tee {self.logfile} 2>&1"
            # After command runs, make sure subdirs permission allows anyone to enter and delete
            self.run_command += f"; find {outdir} -type d | xargs chmod 777"
            # After command runs, make sure artifact files permission allows anyone to delete
            self.run_command += f"; find {outdir} -type f | xargs chmod 666"
            # Optionally, pipe self.run_command into a log file
            self.detailed_print(self.run_command, self.status_file, file=sys.stderr)
            # Set up StatusParser
            status_parser = StatusParser(str(self.status_file), self.job_context.network, outdir)
            self.detailed_print(self.image, file=sys.stderr)

            # Convert self.spec to a backend and post it into a <self.handler_spec_root><job_id>.txt file
            if self.spec:
                file_type = self.api_params["spec_backend"]
                if file_type == "json":
                    kitti_out = self.spec
                    kitti_out = json.dumps(kitti_out)
                elif self.job_context.action == "convert_efficientdet_tf2":
                    file_type = "yaml"
                    kitti_out = SPEC_BACKEND_TO_FUNCTIONS[file_type](self.spec)
                else:
                    kitti_out = SPEC_BACKEND_TO_FUNCTIONS[file_type](self.spec)
                # store as kitti
                action_spec_path_kitti = CLI_CONFIG_TO_FUNCTIONS["experiment_spec"](self.job_context, self.handler_metadata)
                safe_dump_file(action_spec_path_kitti, kitti_out, file_type=file_type)

            # Submit to K8s
            # Platform is None, but might be updated in self.generate_config() or self.generate_run_command()
            # If platform is indeed None, jobDriver.create would take care of it.
            spec = self.job_context.specs
            num_gpu = spec.get("num_gpu", -1) if spec else -1
            if self.job_context.action not in ['train', 'retrain', 'finetune']:
                num_gpu = 1
            docker_env_vars = self.handler_metadata.get("docker_env_vars", {})
            dgx_job_metadata = {}
            if self.ngc_runner:
                self.generate_dgx_job_metadata(self.run_command, dgx_job_metadata, docker_env_vars)
            else:
                dgx_job_metadata = None
            metric = self.handler_metadata.get("metric", "")
            if not metric:
                metric = "loss"
            jobDriver.create(self.job_context.user_id, self.job_name, self.image, self.run_command, num_gpu=num_gpu, accelerator=self.platform, docker_env_vars=docker_env_vars, dgx_job_metadata=dgx_job_metadata)
            self.detailed_print("Job created", self.job_name, file=sys.stderr)
            k8s_status = jobDriver.status(self.job_name, use_ngc=self.ngc_runner)
            while k8s_status in ["Done", "Error", "Running", "Pending"]:
                # If Done, try running self.post_run()
                metadata_status = get_handler_job_metadata(self.job_context.user_id, self.handler_id, self.job_id).get("status", "Error")
                if metadata_status == "Canceled" and k8s_status == "Running":
                    self.detailed_print(f"Canceling job {self.job_id}", file=sys.stderr)
                    jobDriver.delete(self.job_id, use_ngc=self.ngc_runner)
                if k8s_status == "Done":
                    update_job_status(self.job_context.user_id, self.handler_id, self.job_id, status="Running")
                    # Retrieve status one last time!
                    new_results = status_parser.update_results()
                    update_job_metadata(self.job_context.user_id, self.handler_id, self.job_id, metadata_key="result", data=new_results)
                    try:
                        self.detailed_print("Post running", file=sys.stderr)
                        # If post run is done, make it done
                        self.post_run()
                        if self.job_context.action in ['train', 'retrain']:
                            epoch_value = get_total_epochs(self.job_context, self.handler_root)
                            _, best_checkpoint_epoch_number, latest_checkpoint_epoch_number = status_parser.read_metric(results=new_results, metric=metric, brain_epoch_number=epoch_value)
                            self.handler_metadata["checkpoint_epoch_number"][f"best_model_{self.job_name}"] = best_checkpoint_epoch_number
                            self.handler_metadata["checkpoint_epoch_number"][f"latest_model_{self.job_name}"] = latest_checkpoint_epoch_number
                            write_handler_metadata(self.job_context.user_id, self.handler_id, self.handler_metadata)
                        shutil.copy(self.logfile, f"{self.handler_root}/{self.job_id}/logs_from_toolkit.txt")
                        archive_job_results(self.handler_root, self.job_id)
                        generate_job_tar_stats(self.job_context.user_id, self.handler_id, self.job_id, self.handler_root)
                        update_job_status(self.job_context.user_id, self.handler_id, self.job_id, status="Done")
                        break
                    except:
                        # If post run fails, call it Error
                        self.detailed_print(traceback.format_exc(), file=sys.stderr)
                        update_job_status(self.job_context.user_id, self.handler_id, self.job_id, status="Error")
                        break
                # If running in K8s, update results to job_context
                elif k8s_status == "Running":
                    update_job_status(self.job_context.user_id, self.handler_id, self.job_id, status="Running")
                    # Update results
                    new_results = status_parser.update_results()
                    update_job_metadata(self.job_context.user_id, self.handler_id, self.job_id, metadata_key="result", data=new_results)

                # Pending is if we have queueing systems down the road
                elif k8s_status == "Pending":
                    k8s_status = jobDriver.status(self.job_name, use_ngc=self.ngc_runner)
                    continue

                # If the job never submitted or errored out!
                else:
                    update_job_status(self.job_context.user_id, self.handler_id, self.job_id, status="Error")
                    break
                # Poll every 30 seconds
                time.sleep(30)

                k8s_status = jobDriver.status(self.job_name, use_ngc=self.ngc_runner)
            metadata_status = get_handler_job_metadata(self.job_context.user_id, self.handler_id, self.job_id).get("status", "Error")

            toolkit_status = get_toolkit_status(self.job_context.user_id, self.handler_id, self.job_id)
            self.detailed_print(f"Toolkit status for {self.job_id} is {toolkit_status}", file=sys.stderr)
            if metadata_status != "Canceled" and toolkit_status != "SUCCESS" and self.job_context.action != "trtexec":
                update_job_status(self.job_context.user_id, self.handler_id, self.job_id, status="Error")

            self.detailed_print(f"Job Done: {self.job_name} Final status: {metadata_status}", file=sys.stderr)
            with open(self.logfile, "a", encoding='utf-8') as f:
                f.write(f"\n{metadata_status} EOF\n")
            if self.ngc_runner:
                jobDriver.delete(self.job_name)
            return

        except Exception as e:
            # Something went wrong inside...
            self.detailed_print(traceback.format_exc(), file=sys.stderr)
            self.detailed_print(f"Job {self.job_name} did not start", file=sys.stderr)
            with open(self.logfile, "a", encoding='utf-8') as f:
                f.write(f"Error log: \n {e}")
                f.write("\nError EOF\n")
            shutil.copy(self.logfile, f"{self.handler_root}/{self.job_id}/logs_from_toolkit.txt")
            update_job_status(self.job_context.user_id, self.handler_id, self.job_id, status="Error")
            result_dict = {"detailed_status": {"message": "Error due to unmet dependencies"}}
            if isinstance(e, TimeoutError):
                result_dict = {"detailed_status": {"message": "Data downloading from cloud storage failed."}}
            update_job_metadata(self.job_context.user_id, self.handler_id, self.job_id, metadata_key="result", data=result_dict)
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
        spec_json_path = os.path.join(self.handler_spec_root, f"{self.job_context.id}-{self.job_context.action}-spec.json")
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
            root = get_handler_root(self.job_context.user_id, "datasets", ds, None, ngc_runner_fetch=True)
            sub_folder = "train"
            if "test" in os.listdir(root):
                sub_folder = "test"
            status_file = f"{root}/{sub_folder}/lmdb/status.json"
            overriden_output_dir = os.path.dirname(status_file)
        return run_command, status_file, overriden_output_dir


# Specs are modified as well => Train, Evaluate, Retrain Actions
class TrainVal(CLIPipeline):
    """Class for experiment actions which involves both spec file as well as cli params"""

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
        spec_json_path = os.path.join(self.handler_spec_root, f"{self.job_context.id}-{self.job_context.action}-spec.json")
        if not os.path.exists(spec_json_path):
            if action in NO_SPEC_ACTIONS_MODEL:
                spec_json_path = glob.glob(f"{self.handler_spec_root}/**/*train*.json", recursive=True)
                if spec_json_path:
                    spec_json_path = spec_json_path[0]
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
        self.detailed_print("Loaded specs", file=sys.stderr)

        # Infer dataset config
        spec = DS_CONFIG_TO_FUNCTIONS[network](spec, self.job_context, self.handler_metadata)
        self.detailed_print("Loaded dataset", file=sys.stderr)

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
            self.detailed_print(f"Copying pruned model {pruned_model_path} after retrain to {self.handler_root}/{self.job_id}/pruned_model{file_extension}\n", file=sys.stderr)
            os.system(f"cp {pruned_model_path} {self.handler_root}/{self.job_id}/pruned_model{file_extension}")


class ODConvert(CLIPipeline):
    """Class for Object detection networks which requires tfrecords conversion"""

    # def __init__(self,job_context):
    #     super().__init__(job_context)

    def generate_config(self):
        """Modify the spec parameters necessary for object detection convert and return the modified dictionary"""
        # Read json
        spec_json_path = os.path.join(self.handler_spec_root, f"{self.job_context.id}-{self.job_context.action}-spec.json")
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
            categorical = get_handler_job_metadata(self.job_context.user_id, self.handler_id, self.job_id).get("result").get("categorical", [])
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
        spec_json_path = os.path.join(self.handler_spec_root, f"{self.job_context.id}-{action}-spec.json")
        spec = load_json_spec(spec_json_path)  # Dnv2 NEEDS inference spec

        # As per regular TrainVal, do not infer spec params, no need to move spec to cli
        # No need to add dataset configs / classwise configs
        # Instead do the following: if parent is tlt, enter tlt config and parent is trt, enter trt config

        parent_job_id = self.job_context.parent_id
        parent_action = get_handler_job_metadata(self.job_context.user_id, self.handler_id, parent_job_id).get("action")  # This should not fail if dependency passed
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
        self.automl_brain_job_id = self.job_context.id
        self.job_root = self.handler_root + f"/{self.automl_brain_job_id}"
        self.job_metadata_root = self.job_root.replace(get_root(ngc_runner_fetch=True), get_root())
        self.rec_number = self.get_recommendation_number()
        self.expt_root = f"{self.job_root}/experiment_{self.rec_number}"
        self.recs_dict = safe_load_file(f"{self.job_metadata_root}/controller.json")
        self.brain_dict = safe_load_file(f"{self.job_metadata_root}/brain.json")

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
            recommended_values["base_experiment"] = search_for_base_experiment(
                get_handler_root(admin_uuid, "experiments", admin_uuid, ptm_id, ngc_runner_fetch=True)
            )

    def generate_config(self, recommended_values):
        """Generate config for AutoML experiment"""
        spec_json_path = os.path.join(get_handler_spec_root(self.job_context.user_id, self.job_context.handler_id), f"{self.automl_brain_job_id}-train-spec.json")
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
        self.detailed_print("Loaded AutoML specs", file=sys.stderr)

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
        return spec

    def save_recommendation_specs(self):
        """Save recommendation specs to a yaml/kitti file as well as json file"""
        safe_dump_file(os.path.join(self.job_root, f"recommendation_{self.rec_number}.json"), self.spec)
        updated_spec = SPEC_BACKEND_TO_FUNCTIONS[self.api_params["spec_backend"]](self.spec)
        extension = self.api_params['spec_backend']
        action_spec_path = os.path.join(self.job_root, f"recommendation_{self.rec_number}.{extension}")
        safe_dump_file(action_spec_path, updated_spec, file_type=extension)

    def generate_run_command(self):
        """Generate the command to be run inside docker for AutoML experiment"""
        params_to_cli = build_cli_command(self.config)
        run_command = f"{self.network} train {params_to_cli}"
        logfile = os.path.join(self.expt_root, "log.txt")
        run_command += f" | tee {logfile} 2>&1"
        return run_command

    def get_recommendation_number(self):
        """Return the current recommendation number"""
        rec_number = None
        for dep in self.job_context.dependencies:
            if dep.type == "automl":
                rec_number = int(dep.name)
                break
        return rec_number

    def run(self):
        """Calls necessary setup functions; calls job creation; update status of the job"""
        try:
            recommended_values = self.recs_dict[self.rec_number].get("specs", {})
            self.add_ptm_dependency(recommended_values)

            self.spec = self.generate_config(recommended_values)
            self.handle_multiple_ptm_fields()
            self.save_recommendation_specs()
            run_command = self.generate_run_command()

            # Assign a new job id if not assigned already
            job_id = self.recs_dict[self.rec_number].get("job_id", None)
            if not job_id:
                job_id = str(uuid.uuid4())
                self.detailed_print("New job id being assigned to recommendation", job_id, file=sys.stderr)
                self.recs_dict[self.rec_number]["job_id"] = job_id
                safe_dump_file(f"{self.job_metadata_root}/controller.json", self.recs_dict)

            self.detailed_print(run_command, file=sys.stderr)

            # Wait for existing AutoML jobs to complete
            wait_for_job_completion(job_id)

            delete_lingering_checkpoints(self.recs_dict[self.rec_number].get("best_epoch_number", ""), self.expt_root)
            docker_env_vars = self.handler_metadata.get("docker_env_vars", {})
            dgx_job_metadata = {}
            if self.ngc_runner:
                self.generate_dgx_job_metadata(run_command, dgx_job_metadata, docker_env_vars)
            jobDriver.create(self.job_context.user_id, job_id, self.image, run_command, num_gpu=-1, docker_env_vars=docker_env_vars, dgx_job_metadata=dgx_job_metadata)
            self.detailed_print(f"AutoML recommendation with experiment id {self.rec_number} and job id {job_id} submitted", file=sys.stderr)
            k8s_status = jobDriver.status(job_id)
            while k8s_status in ["Done", "Error", "Running", "Pending", "Creating"]:
                time.sleep(5)
                if os.path.exists(os.path.join(self.expt_root, "log.txt")):
                    break
                if k8s_status == "Error":
                    self.detailed_print(f"Relaunching job {job_id}", file=sys.stderr)
                    wait_for_job_completion(job_id)
                    jobDriver.create(self.job_context.user_id, job_id, self.image, run_command, num_gpu=-1, docker_env_vars=docker_env_vars, dgx_job_metadata=dgx_job_metadata)
                k8s_status = jobDriver.status(job_id)

            return True

        except Exception:
            self.detailed_print(f"AutoMLpipeline for network {self.network} failed due to exception {traceback.format_exc()}", file=sys.stderr)
            job_id = self.recs_dict[self.rec_number].get("job_id", "")
            self.detailed_print(job_id, file=sys.stderr)

            self.recs_dict[self.rec_number]["status"] = "failure"
            safe_dump_file(f"{self.job_metadata_root}/controller.json", self.recs_dict)

            update_job_status(self.job_context.user_id, self.handler_id, self.job_context.id, status="Error")
            jobDriver.delete(self.job_context.id)
            return False


class ContinualLearning(ActionPipeline):
    """Class for continual learning specific changes required during annotation action."""

    def __init__(self, job_context):
        """Initialize the ContinualLearning class"""
        super().__init__(job_context)
        self.ngc_runner = False
        # override the ActionPipeline's handler_root
        self.handler_root = get_handler_root(self.job_context.user_id, "experiments", self.job_context.handler_id, None, ngc_runner_fetch=False)
        self.handler_log_root = os.path.join(self.handler_root, "logs")
        self.logfile = os.path.join(self.handler_log_root, str(self.job_context.id) + ".txt")
        self.train_ds = self.handler_metadata["train_datasets"]
        self.image = DOCKER_IMAGE_MAPPER["api"]
        self.job_root = os.path.join(self.handler_root, self.job_context.id)

    def generate_convert_script(self, notify_record):
        """Generate a script to perform continual learning"""
        cl_script = generate_cl_script(notify_record, self.job_context, self.handler_root, self.logfile)
        cl_script_path = os.path.join(self.job_root, "continual_learning.py")
        with open(cl_script_path, "w", encoding="utf-8") as f:
            f.write(cl_script)

        return cl_script_path

    def run(self):
        """Run the continual learning pipeline"""
        # Set up
        self.thread = threading.current_thread()
        try:
            # Find the train dataset path
            if len(self.train_ds) != 1:
                raise ValueError("Continual Learning only supports one train dataset.")
            train_dataset = self.train_ds[0]
            dataset_path = get_handler_root(self.job_context.user_id, "datasets", train_dataset, None, ngc_runner_fetch=False)

            # Track the current training manifest file
            notify_record = os.path.join(dataset_path, "notify_record.json")
            cl_script = self.generate_convert_script(notify_record)

            outdir = self.job_root
            self.detailed_print("Continual Learning started", file=sys.stderr)
            self.run_command = f"python {cl_script} 2>&1 | tee {self.logfile}"
            # After command runs, make sure subdirs permission allows anyone to enter and delete
            self.run_command += f"; find {outdir} -type d | xargs chmod 777"
            # After command runs, make sure artifact files permission allows anyone to delete
            self.run_command += f"; find {outdir} -type f | xargs chmod 666"
            # Optionally, pipe self.run_command into a log file
            self.detailed_print(self.run_command, file=sys.stderr)
            jobDriver.create(self.job_context.user_id, self.job_name, self.image, self.run_command, num_gpu=0)
            self.detailed_print("Job created", self.job_name, file=sys.stderr)
            k8s_status = jobDriver.status(self.job_name, use_ngc=False)
            while k8s_status in ["Running", "Pending"]:
                # Poll every 30 seconds
                time.sleep(30)
                k8s_status = jobDriver.status(self.job_name, use_ngc=False)
            self.detailed_print(f"Job status: {k8s_status}", file=sys.stderr)
            if k8s_status == "Error":
                update_job_status(self.job_context.user_id, self.handler_id, self.job_context.id, status="Error")
            self.detailed_print("Continual Learning finished.", file=sys.stderr)
        except Exception:
            self.detailed_print(f"ContinualLearning for {self.network} failed because {traceback.format_exc()}", file=sys.stderr)
            shutil.copy(self.logfile, f"{self.handler_root}/{self.job_id}/logs_from_toolkit.txt")
            update_job_status(self.job_context.user_id, self.handler_id, self.job_context.id, status="Error")


class BundleTrain(ActionPipeline):
    """Class for MEDICAL bundle specific changes required during training"""

    def __init__(self, job_context):
        """Initialize the BundleTrain class"""
        super().__init__(job_context)
        spec = self.get_spec()
        # override the ActionPipeline's handler_root
        self.handler_root = get_handler_root(self.job_context.user_id, "experiments", self.job_context.handler_id, None, ngc_runner_fetch=True)
        if "cluster" in spec and spec["cluster"] == "local":
            self.ngc_runner = False
            self.handler_root = get_handler_root(self.job_context.user_id, "experiments", self.job_context.handler_id, None, ngc_runner_fetch=False)
        self.handler_log_root = os.path.join(self.handler_root, "logs")
        self.logfile = os.path.join(self.handler_log_root, str(self.job_context.id) + ".txt")
        self.network = job_context.network
        self.action = job_context.action
        self.job_root = os.path.join(self.handler_root, self.job_context.id)
        train_datasets = self.handler_metadata["train_datasets"]
        eval_dataset_id = self.handler_metadata["eval_dataset"]
        self.train_datasets_path = [get_handler_root(self.job_context.user_id, "datasets", dataset_id, None, ngc_runner_fetch=self.ngc_runner) for dataset_id in train_datasets] if train_datasets else []
        self.eval_dataset_path = get_handler_root(self.job_context.user_id, "datasets", eval_dataset_id, None, ngc_runner_fetch=self.ngc_runner) if eval_dataset_id else None
        self.model_script_path = os.path.join(self.job_root, "train.py")

    def get_spec(self):
        """Get spec from spec file or job context"""
        if self.job_context.specs is not None:
            specs = self.job_context.specs.copy()
            spec = specs
        else:
            spec_json_path = os.path.join(self.handler_spec_root, f"{self.job_context.id}-{self.job_context.action}-spec.json")
            if not os.path.exists(spec_json_path):
                self.detailed_print(f"Spec file {spec_json_path} does not exist", file=sys.stderr)
                raise RuntimeError(f"Spec file {spec_json_path} does not exist")
            spec = load_json_spec(spec_json_path)

        for k, v in spec.items():
            if isinstance(v, str) and v == "$default_medical_label_mapping":
                labels = self.handler_metadata.get("model_params", {}).get("labels", None)
                spec[k] = {"default": [[int(i), int(i)] for i in labels]}
                self.detailed_print(f"Using default {k}: {spec[k]}", file=sys.stderr)
        return spec

    def generate_config(self):
        """
        Generate config for medical bundle train
        Returns:
        spec: contains the params for building medical bundle train command
        Notes:
        Similar to TrainVal.generate_config, but we have the following differences:
        - Drop using network config files
        - Drop using the csv spec file
        But we cannot delete the network config and the csv spec file, because they are used by the app_handler, e.g. save_spec and job_run methods.
        """
        spec = self.get_spec()
        success, msg = validate_medical_bundle_params(spec)
        if not success:
            self.detailed_print(msg, file=sys.stderr)
            raise RuntimeError(msg)
        # prepare dataset and update spec
        self.detailed_print("Loaded specs", file=sys.stderr)
        spec = DS_CONFIG_TO_FUNCTIONS[self.network](spec, self.job_context, self.handler_metadata)
        self.detailed_print("Loaded dataset", file=sys.stderr)
        return spec, {}

    def save_model_script(self, model_script):
        """save model script"""
        with open(self.model_script_path, "w", encoding="utf-8") as f:
            f.write(model_script)

    def generate_convert_script(self):
        """
        Generate a script to convert dicom segmentation
        The function temporarily fixes the issue: https://github.com/Project-MEDICAL/MEDICAL/issues/7055
        """
        labels = self.handler_metadata.get("model_params", {}).get("labels", None)
        convert_script = generate_dicom_segmentation_convert_script(self.spec, labels)
        convert_script_path = os.path.join(self.job_root, "convert_labels.py")
        with open(convert_script_path, "w", encoding="utf-8") as f:
            f.write(convert_script)

        return convert_script_path

    def generate_run_command(self):
        """Generate run command"""
        _, bundle_name, overriden_output_dir = CLI_CONFIG_TO_FUNCTIONS["medical_output_dir"](self.job_context, self.handler_metadata)
        status_file = os.path.join(self.handler_root, self.job_name, "status.json")
        dicom_convert = self.spec.pop("dicom_convert", False)
        copy_in_job = self.spec.pop("copy_in_job", None)
        model_script = generate_bundle_train_script(self.job_root, bundle_name, status_file, override=self.spec)
        self.save_model_script(model_script)

        if copy_in_job:
            run_command = copy_in_job + " && "
        else:
            run_command = ""

        if dicom_convert:
            convert_script_path = self.generate_convert_script()
            run_command += f"python {convert_script_path} && python {self.model_script_path}"
        else:
            run_command += f"python {self.model_script_path}"

        run_command = f"({run_command}) 2>&1"

        return run_command, status_file, overriden_output_dir


class Auto3DSegTrain(BundleTrain):
    """MEDICAL Auto3DSeg training action pipeline"""

    def generate_config(self):
        """
        Generate config for auto3dseg train
        """
        # load spec
        if self.job_context.specs is not None:
            specs = self.job_context.specs.copy()
        else:
            spec_json_path = os.path.join(self.handler_spec_root, f"{self.job_context.id}-{self.job_context.action}-spec.json")
            if not os.path.exists(spec_json_path):
                self.detailed_print(f"Spec file {spec_json_path} does not exist", file=sys.stderr)
                raise RuntimeError(f"Spec file {spec_json_path} does not exist")
            specs = load_json_spec(spec_json_path)
        self.detailed_print("Specs are loaded.", file=sys.stderr)

        # create datalist.json
        specs = DS_CONFIG_TO_FUNCTIONS[self.network](specs, self.job_context, self.handler_metadata, self.job_root)
        self.detailed_print("Dataset are loaded.", file=sys.stderr)

        # set work directory for auto3dseg
        specs["work_dir"] = self.job_root

        # get the path to algorithm templates files
        base_experiment_id = self.handler_metadata["base_experiment"][0]
        base_experiment_version = get_base_experiment_metadata(base_experiment_id).get("version")
        templates_path = os.path.join(
            get_base_experiment_path(base_experiment_id),
            f"auto3dseg_v{base_experiment_version}",
            "algorithm_templates",
        )
        # set the path to algorithm templates files
        specs["templates_path_or_url"] = templates_path

        return specs, {}

    def generate_run_command(self):
        """Generate run command"""
        status_file = os.path.join(self.handler_root, self.job_name, "status.json")
        train_script_src = os.path.join(os.path.abspath(os.path.dirname(__file__)), "medical", f"train_{self.action}.py")
        shutil.copyfile(train_script_src, self.model_script_path)
        run_command = f"python {self.model_script_path} '{json.dumps(self.spec)}' {status_file} 2>&1"
        return run_command, status_file, ""

    def post_run(self):
        """Create a new model after the job is executed"""
        from handlers.app_handler import AppHandler

        best_model_path = os.path.join(self.job_root, "best_model")

        # update the hyper_parameters.yaml file to include relative paths
        with open(os.path.join(best_model_path, "configs", "hyper_parameters.yaml"), "r", encoding="utf-8") as f:
            hyper_conf = yaml.safe_load(f)
        hyper_conf["bundle_root"] = "."
        hyper_conf["data_file_base_dir"] = os.path.join("..", "datasets")
        hyper_conf["data_list_file_path"] = os.path.join("..", "datalist.json")
        hyper_conf["infer"]["log_output_file"] = os.path.join("..", "inference.log")
        hyper_conf["infer"]["output_path"] = os.path.join("..", "predictions")
        with open(os.path.join(best_model_path, "configs", "hyper_parameters.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(hyper_conf, f)

        # create a new experiment
        request_dict = {
            "name": self.spec.get("output_experiment_name", "auto3dseg_automl_experiment"),
            "description": self.spec.get("output_experiment_description", "AutoML Generated Segmentation Model based on MEDICAL Auto3DSeg"),
            "type": "medical",
            "network_arch": "medical_segmentation",
            "inference_dataset": self.spec.get("inference_dataset"),
            "realtime_infer": False
        }
        user_id = get_handler_user(self.handler_id)
        ret_code = AppHandler.create_experiment(user_id, request_dict)
        new_model_id = ret_code.data["id"]
        self.detailed_print(f"New model is generated with id: {new_model_id}", file=sys.stderr)

        # copy/upload the best model to the new model
        if self.ngc_runner:
            _, workspaces = load_user_workspace_metadata(user_id)
            new_workspace_id = workspaces[new_model_id]['id']
            copy_command = f"ngc workspace upload --source {best_model_path} --destination bundle {new_workspace_id}"
        else:
            new_model_path = os.path.join(get_handler_root(self.job_context.user_id, kind="experiments", handler_id=new_model_id, ngc_runner_fetch=True), "bundle")
            copy_command = f"cp -rp {best_model_path} {new_model_path}"
        try:
            subprocess.run(copy_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError('Failed to copy the best model to the new model.') from e
        self.detailed_print("The best model is copied/uploaded to the new experiment.", file=sys.stderr)

        # update the status file to include newly generated model id
        status_dict = {
            "date": datetime.date.today().isoformat(),
            "time": datetime.datetime.now().isoformat(),
            "status": "SUCCESS",
            "message": f"A segmentation experiment (id={new_model_id}) is successfully created by MEDICAL AuoML.",
            "model_id": new_model_id,
        }
        with open(self.status_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(status_dict) + "\n")


class MedicalInference(BundleTrain):
    """MEDICAL inference action pipeline"""

    def generate_config(self):
        """
        Generate spec for medical inference and creates "datalist.json"

        Returns:
            spec: contains the params for building medical bundle train command
        """
        # load spec
        if self.job_context.specs is not None:
            specs = self.job_context.specs.copy()
        else:
            spec_json_path = os.path.join(self.handler_spec_root, f"{self.job_context.id}-{self.job_context.action}-spec.json")
            if not os.path.exists(spec_json_path):
                self.detailed_print(f"Spec file {spec_json_path} does not exist", file=sys.stderr)
                raise RuntimeError(f"Spec file {spec_json_path} does not exist")
            specs = load_json_spec(spec_json_path)
        self.detailed_print("Specs are loaded.", file=sys.stderr)

        # create datalist.json
        specs = DS_CONFIG_TO_FUNCTIONS[self.network](specs, self.job_context, self.handler_metadata, self.job_root)
        self.detailed_print("Dataset are loaded.", file=sys.stderr)

        # set work directory for auto3dseg
        specs["work_dir"] = self.job_root

        return specs, {}

    def generate_run_command(self):
        """Generate run command for medical inference"""
        # Check if "infer.py" exists for this models
        src_model_root = os.path.join(
            get_handler_root(self.job_context.user_id, kind="experiments", handler_id=self.handler_metadata["id"], ngc_runner_fetch=True),
            "bundle"
        )
        infer_script_path = os.path.join(src_model_root, "scripts", "infer.py")
        if not os.path.exists(infer_script_path):
            raise RuntimeError(
                f"Currenlty, inference script file (infer.py) is required for MEDICAL inference! "
                f"'{infer_script_path}' does not exist."
            )

        # copy the model into the job root
        model_root = os.path.join(self.job_root, "bundle")
        shutil.copytree(src_model_root, model_root)
        self.detailed_print("Model copied to the job root.", file=sys.stderr)

        # generate the run command
        config_dir = os.path.join(model_root, "configs")
        config_files = ','.join([os.path.join(config_dir, f) for f in os.listdir(config_dir)])
        num_gpu = self.spec.get("num_gpu", 1)
        run_command = f"cd {model_root};"
        if num_gpu <= 1:
            run_command += "python"
        else:
            run_command += f"torchrun --nnodes=1 --nproc_per_node={num_gpu}"
        run_command += f' -m scripts.infer run --config_file {config_files}'

        # create the file to report the job status
        status_file = os.path.join(self.job_root, "status.json")
        status_dict = {
            "date": datetime.date.today().isoformat(),
            "time": datetime.datetime.now().isoformat(),
            "status": "SUCCESS",
            "message": "The data and model are prepared and ready for inference.",
        }
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(status_dict, f)

        return run_command, status_file, self.job_root


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
                        "data_services": TrainVal,
                        "medical_annotation": ContinualLearning,
                        "medical_auto3dseg": Auto3DSegTrain,
                        "medical_train": BundleTrain,
                        "medical_inference": MedicalInference,
                        }
