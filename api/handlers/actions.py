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
import sys
import threading
import time
import traceback
import uuid

import yaml
from automl.utils import delete_lingering_checkpoints, wait_for_job_completion
from constants import (AUTOML_DISABLED_NETWORKS, NO_SPEC_ACTIONS_MODEL, TAO_NETWORKS, _DATA_SERVICES_ACTIONS,
                       MONAI_NETWORKS, MEDICAL_AUTOML_ARCHITECT, MEDICAL_NETWORK_ARCHITECT, MEDICAL_CUSTOM_ARCHITECT, NETWORK_METRIC_MAPPING, NETWORK_CONTAINER_MAPPING)
from dgx_controller import overwrite_job_logs_from_bcp
from handlers.cloud_storage import create_cs_instance
from handlers.ngc_handler import get_user_api_key, bcp_org_name, bcp_team_name, bcp_ace
from handlers.docker_images import DOCKER_IMAGE_MAPPER, DOCKER_IMAGE_VERSION
from handlers.infer_data_sources import DS_CONFIG_TO_FUNCTIONS
from handlers.infer_params import CLI_CONFIG_TO_FUNCTIONS
from handlers.encrypt import NVVaultEncryption
from handlers.medical.helpers import CUSTOMIZED_BUNDLE_URL_FILE, CUSTOMIZED_BUNDLE_URL_KEY, MEDICAL_SERVICE_SCRIPTS
# TODO: force max length of characters in a line to be 120
from handlers.stateless_handlers import (BACKEND, base_exp_uuid, get_base_experiment_metadata, get_handler_job_metadata,
                                         get_handler_log_root, get_handler_metadata, get_jobs_root, get_handler_root,
                                         get_handler_spec_root, get_toolkit_status, printc, resolve_metadata,
                                         update_job_metadata, update_job_status, write_handler_metadata, get_handler_kind)
from handlers.utilities import (StatusParser, build_cli_command, generate_cl_script, get_total_epochs, load_json_spec, process_classwise_config,
                                read_nested_dict, search_for_base_experiment, get_num_gpus_from_spec, validate_medical_bundle_params,
                                write_nested_dict, get_cloud_metadata)
from utils import remove_key_by_flattened_string, read_network_config, find_closest_number, get_admin_api_key, safe_load_file, safe_dump_file
from job_utils import executor as jobDriver
from network_utils.network_constants import ptm_mapper
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
        self.network = self.job_context.network
        self.action = self.job_context.action
        self.network_config = read_network_config(self.network)
        self.api_params = self._read_api_params()
        self.handler_metadata = get_handler_metadata(self.job_context.org_name, self.job_context.handler_id)
        self.workspace_id = self.handler_metadata.get("workspace")
        self.workspace_metadata = get_handler_metadata(self.job_context.org_name, self.workspace_id, "workspaces")
        self.handler_spec_root = get_handler_spec_root(self.job_context.user_id, self.job_context.org_name, self.job_context.handler_id)
        self.handler_root = get_handler_root(self.job_context.org_name, None, self.job_context.handler_id, None)
        self.jobs_root = get_jobs_root(self.job_context.user_id, self.job_context.org_name)
        self.handler_log_root = get_handler_log_root(self.job_context.user_id, self.job_context.org_name, self.job_context.handler_id)
        self.handler_id = self.job_context.handler_id
        self.handler_kind = get_handler_kind(self.handler_metadata)
        self.tao_deploy_actions = False
        self.parent_job_action = get_handler_job_metadata(self.job_context.org_name, self.handler_id, self.job_context.parent_id).get("action")
        if self.job_context.action in ("gen_trt_engine", "trtexec") or self.parent_job_action in ("gen_trt_engine", "trtexec"):
            self.tao_deploy_actions = True
        self.image = DOCKER_IMAGE_MAPPER[self.api_params.get("image", "")]
        if self.job_context.action in _DATA_SERVICES_ACTIONS and not self.network.startswith("medical"):
            self.image = DOCKER_IMAGE_MAPPER["TAO_DS"]
        # If current or parent action is gen_trt_engine or trtexec, then it'a a tao-deploy container action
        if self.tao_deploy_actions:
            self.image = DOCKER_IMAGE_MAPPER["tao-deploy"]
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
        self.platform = self.job_context.platform

        self.run_command = ""
        self.status_file = None
        self.logfile = os.path.join(self.handler_log_root, str(self.job_context.id) + ".txt")
        self.cloud_metadata = {}
        self.cs_instance = None  # initialized in run()
        self.ngc_runner = False
        if BACKEND in ("BCP", "NVCF"):
            self.ngc_runner = True
        self.local_cluster = False
        self.medical_env_variable = {}
        self.num_gpu = self.job_context.specs.get("num_gpu", self.job_context.num_gpu) if self.job_context.specs else self.job_context.num_gpu

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

    def decrypt_docker_env_vars(self, docker_env_vars):
        """Decrypt NvVault encrypted values"""
        if self.ngc_runner:
            config_path = os.getenv("VAULT_SECRET_PATH", None)
            encryption = NVVaultEncryption(config_path)
            for docker_env_var_key, docker_env_var_value in docker_env_vars.items():
                if encryption.check_config()[0]:
                    docker_env_vars[docker_env_var_key] = encryption.decrypt(docker_env_var_value)

    def generate_env_variables(self, job_env_variables, automl_brain_job_id=None, experiment_number=None, automl_exp_job_id=None):
        """Generate env variables required for a job"""
        log_callback_job_id = self.job_context.id
        if automl_exp_job_id:
            log_callback_job_id = automl_exp_job_id

        org_name = self.job_context.org_name
        host_base_url = os.getenv("HOSTBASEURL", "no_url")
        if BACKEND == "local-k8s":
            cluster_ip, cluster_port = jobDriver.get_cluster_ip()
            if cluster_ip and cluster_port:
                host_base_url = f"http://{cluster_ip}:{cluster_port}"

        handler_kind = "experiments"
        if (not self.handler_metadata.get("train_datasets", [])) and self.job_context.network not in ["auto_label", "image"] + MEDICAL_AUTOML_ARCHITECT + MEDICAL_NETWORK_ARCHITECT:
            handler_kind = "datasets"

        status_url = f"{host_base_url}/api/v1/orgs/{org_name}/{handler_kind}/{self.handler_id}/jobs/{self.job_context.id}"
        if automl_brain_job_id:
            status_url = f"{host_base_url}/api/v1/orgs/{org_name}/{handler_kind}/{self.handler_id}/jobs/{automl_brain_job_id}"
            if experiment_number:
                job_env_variables["AUTOML_EXPERIMENT_NUMBER"] = experiment_number

        job_env_variables["TELEMETRY_OPT_OUT"] = os.getenv('TELEMETRY_OPT_OUT', default='no')
        job_env_variables["CLOUD_BASED"] = "True"
        user_api_key, ngc_cookie = get_user_api_key(self.job_context.user_id)
        job_env_variables["TAO_USER_KEY"] = user_api_key
        job_env_variables["TAO_COOKIE_SET"] = str(ngc_cookie)
        job_env_variables["TAO_ADMIN_KEY"] = get_admin_api_key()
        job_env_variables["TAO_API_SERVER"] = host_base_url
        job_env_variables["TAO_API_JOB_ID"] = log_callback_job_id
        job_env_variables["TAO_LOGGING_SERVER_URL"] = status_url
        job_env_variables["USE_NGC_STAGING"] = "True"
        job_env_variables["DEPLOYMENT_MODE"] = os.getenv("DEPLOYMENT_MODE", "PROD")
        if job_env_variables["DEPLOYMENT_MODE"] == "PROD":
            job_env_variables["USE_NGC_STAGING"] = "False"

    def generate_nv_job_metadata(self, container_run_command, nv_job_metadata, job_env_variables):
        """Convert run command generated into format that """
        nv_job_metadata["orgName"] = bcp_org_name
        nv_job_metadata["dockerImageName"] = self.image
        if BACKEND == "BCP":
            nv_job_metadata["user_id"] = self.job_context.user_id
            nv_job_metadata["teamName"] = bcp_team_name
            nv_job_metadata["command"] = container_run_command
            nv_job_metadata["aceName"] = bcp_ace
            nv_job_metadata["aceInstance"] = "dgxa100.80g.1.norm"
            nv_job_metadata["runPolicy"] = {}
            nv_job_metadata["runPolicy"]["preemptClass"] = "RESUMABLE"

            nv_job_metadata["resultContainerMountPoint"] = "/result"
            if self.job_context.action in ("train", "retrain") or (self.job_context.network == "auto_label" and self.job_context.action == "generate"):
                # For backward compatibility, make sure it is using "dgxa100.80g.2.norm" if num_gpu is not explicitly set.
                dgx_type = str(find_closest_number(self.num_gpu, [2, 4, 8])) if self.num_gpu != 1 else "1"
                nv_job_metadata["aceInstance"] = f"dgxa100.80g.{dgx_type}.norm"

            nv_job_metadata["envs"] = []
            for key, value in job_env_variables.items():
                nv_job_metadata["envs"].append({"name": key, "value": value})

        elif BACKEND == "NVCF":
            nv_job_metadata["action"] = self.action
            nv_job_metadata["workspace_ids"] = self.workspace_ids
            nv_job_metadata["deployment_string"] = os.getenv(f'FUNCTION_{NETWORK_CONTAINER_MAPPING[self.network]}')
            if self.tao_deploy_actions:
                nv_job_metadata["deployment_string"] = os.getenv('FUNCTION_TAO_DEPLOY')
            nv_job_metadata["network"] = self.network
            for key, value in job_env_variables.items():
                nv_job_metadata[key] = value

    def get_handler_cloud_details(self):
        """Gather cloud details from various handlers associated for the job"""
        self.workspace_ids = []
        workspace_cache = {}

        def process_metadata(org_name, data_type, dataset_id=None, metadata=None, workspace_cache={}):
            """Process metadata for datasets, workspaces, etc."""
            if not metadata:
                metadata = resolve_metadata(org_name, data_type, dataset_id)
            else:
                metadata = copy.deepcopy(metadata)
            workspace_id = metadata.get("workspace", "")
            if not workspace_id:
                return
            self.workspace_ids.append(workspace_id)

        if self.handler_metadata.get("train_datasets", []):
            for train_ds in self.handler_metadata.get("train_datasets", []):
                process_metadata(self.job_context.org_name, "dataset", dataset_id=train_ds, workspace_cache={})
        elif self.job_context.network not in ["auto_label", "image"] + MEDICAL_AUTOML_ARCHITECT + MEDICAL_NETWORK_ARCHITECT:
            process_metadata(self.job_context.org_name, "dataset", metadata=self.handler_metadata, workspace_cache=workspace_cache)

        eval_ds = self.handler_metadata.get("eval_dataset", None)
        if eval_ds:
            process_metadata(self.job_context.org_name, "dataset", dataset_id=eval_ds, workspace_cache=workspace_cache)

        infer_ds = self.handler_metadata.get("inference_dataset", None)
        if infer_ds:
            process_metadata(self.job_context.org_name, "dataset", dataset_id=infer_ds, workspace_cache=workspace_cache)

        experiment_metadata = copy.deepcopy(self.handler_metadata)
        exp_workspace_id = experiment_metadata.get("workspace")
        self.workspace_ids.append(exp_workspace_id)
        if self.network not in TAO_NETWORKS or (self.network in TAO_NETWORKS and BACKEND == "local-k8s"):
            get_cloud_metadata(self.job_context.org_name, self.workspace_ids, self.cloud_metadata)

    def handle_multiple_ptm_fields(self):
        """Remove one of end-end or backbone related PTM field based on the Handler metadata info"""
        for base_experiment_id in self.handler_metadata.get("base_experiment", []):
            base_experiment_metadata = get_base_experiment_metadata(base_experiment_id)
            if base_experiment_metadata.get("base_experiment_metadata", {}).get("is_backbone"):
                parameter_to_remove = ptm_mapper.get("end_to_end", {}).get(self.network)  # if ptm is a backbone remove end_to_end field from config and spec
            else:
                parameter_to_remove = ptm_mapper.get("backbone", {}).get(self.network)  # if ptm is not a backbone remove it field from config and spec
            if parameter_to_remove:
                remove_key_by_flattened_string(self.spec, parameter_to_remove)
                remove_key_by_flattened_string(self.config, parameter_to_remove)

    def detailed_print(self, *args, **kwargs):
        """Prints the details of the job to the console"""
        printc(*args, context=vars(self.job_context), **kwargs)

    def create_microservice_action_job(self, job_env_variables):
        print("Creating microservices job_action ms pod", file=sys.stderr)
        ngc_api_key, _ = get_user_api_key(self.job_context.user_id)
        response = jobDriver.create_microservice_and_send_request(api_endpoint="post_action",
                                                                  network=self.network,
                                                                  action=self.action,
                                                                  ngc_api_key=ngc_api_key,
                                                                  cloud_metadata=self.cloud_metadata,
                                                                  specs=self.spec,
                                                                  microservice_pod_id=self.job_name,
                                                                  tao_api_admin_key=job_env_variables["TAO_ADMIN_KEY"],
                                                                  tao_api_base_url=job_env_variables["TAO_API_SERVER"],
                                                                  tao_api_status_callback_url=job_env_variables["TAO_LOGGING_SERVER_URL"],
                                                                  tao_api_ui_cookie=job_env_variables["TAO_COOKIE_SET"],
                                                                  use_ngc_staging=job_env_variables["USE_NGC_STAGING"],
                                                                  automl_experiment_number=job_env_variables.get("AUTOML_EXPERIMENT_NUMBER", "0"),
                                                                  num_gpu=self.num_gpu,
                                                                  microservice_container=self.image,
                                                                  org_name=self.job_context.org_name,
                                                                  handler_id=self.handler_id,
                                                                  handler_kind=self.handler_kind)
        if not (response and response.ok):
            jobDriver.delete(self.job_name, use_ngc=self.ngc_runner)
            raise ValueError(f"Not able to submit a microservice job {response}, {response.text}")

    def monitor_job(self):
        """Monitors the job status and updates job metadata"""
        if self.network in MONAI_NETWORKS:
            _, _, outdir = CLI_CONFIG_TO_FUNCTIONS["medical_output_dir"](self.job_context, self.handler_metadata)
            outdir = outdir.rstrip(os.path.sep)
        else:
            _, _, outdir = self.generate_run_command()
        if not outdir:
            outdir = CLI_CONFIG_TO_FUNCTIONS["output_dir"](self.job_context, self.handler_metadata)

        status_parser = StatusParser(str(self.status_file), self.job_context.network, outdir)

        total_epochs = 1
        if self.job_context.action in ['train', 'retrain']:
            total_epochs = get_total_epochs(self.job_context, self.handler_spec_root)

        metric = self.handler_metadata.get("metric", "")
        if not metric:
            metric = NETWORK_METRIC_MAPPING.get(self.network, "loss")

        k8s_status = jobDriver.status(self.job_context.org_name, self.handler_id, self.job_name, self.handler_kind, use_ngc=self.ngc_runner, network=self.network, action=self.action)
        while k8s_status in ["Done", "Error", "Running", "Pending"]:
            # If Done, try running self.post_run()
            metadata_status = get_handler_job_metadata(self.job_context.org_name, self.handler_id, self.job_name, self.handler_kind).get("status", "Error")
            if metadata_status in ("Canceled", "Paused") and k8s_status == "Running":
                self.detailed_print(f"Terminating job {self.job_name}", file=sys.stderr)
                jobDriver.delete(self.job_name, use_ngc=self.ngc_runner)
            if k8s_status == "Done":
                update_job_status(self.job_context.org_name, self.handler_id, self.job_name, status="Running", kind=self.handler_kind)
                # Retrieve status one last time!
                new_results = status_parser.update_results(total_epochs)
                update_job_metadata(self.job_context.org_name, self.handler_id, self.job_name, metadata_key="result", data=new_results, kind=self.handler_kind)
                try:
                    self.detailed_print("Post running", file=sys.stderr)
                    # If post run is done, make it done
                    self.post_run()
                    if self.job_context.action in ['train', 'retrain']:
                        _, best_checkpoint_epoch_number, latest_checkpoint_epoch_number = status_parser.read_metric(results=new_results, metric=metric, brain_epoch_number=total_epochs)
                        self.handler_metadata["checkpoint_epoch_number"][f"best_model_{self.job_name}"] = best_checkpoint_epoch_number
                        self.handler_metadata["checkpoint_epoch_number"][f"latest_model_{self.job_name}"] = latest_checkpoint_epoch_number
                        write_handler_metadata(self.job_context.org_name, self.handler_id, self.handler_metadata, self.handler_kind)
                    if not os.path.exists(f"{self.jobs_root}/{self.job_name}"):
                        os.makedirs(f"{self.jobs_root}/{self.job_name}")
                    if BACKEND == "BCP" and self.network in TAO_NETWORKS:
                        overwrite_job_logs_from_bcp(self.logfile, self.job_name)
                    update_job_status(self.job_context.org_name, self.handler_id, self.job_name, status="Done", kind=self.handler_kind)
                    break
                except:
                    # If post run fails, call it Error
                    self.detailed_print(traceback.format_exc(), file=sys.stderr)
                    update_job_status(self.job_context.org_name, self.handler_id, self.job_name, status="Error", kind=self.handler_kind)
                    break
            # If running in K8s, update results to job_context
            elif k8s_status == "Running":
                update_job_status(self.job_context.org_name, self.handler_id, self.job_name, status="Running", kind=self.handler_kind)
                # Update results
                new_results = status_parser.update_results(total_epochs)
                update_job_metadata(self.job_context.org_name, self.handler_id, self.job_name, metadata_key="result", data=new_results, kind=self.handler_kind)

            # Pending is if we have queueing systems down the road
            elif k8s_status == "Pending":
                k8s_status = jobDriver.status(self.job_context.org_name, self.handler_id, self.job_name, self.handler_kind, use_ngc=self.ngc_runner, network=self.network, action=self.action)
                continue

            # If the job never submitted or errored out!
            else:
                update_job_status(self.job_context.org_name, self.handler_id, self.job_name, status="Error", kind=self.handler_kind)
                break
            # Poll every 30 seconds
            time.sleep(30)

            k8s_status = jobDriver.status(self.job_context.org_name, self.handler_id, self.job_name, self.handler_kind, use_ngc=self.ngc_runner, network=self.network, action=self.action)

        metadata_status = get_handler_job_metadata(self.job_context.org_name, self.handler_id, self.job_name, self.handler_kind).get("status", "Error")

        toolkit_status = get_toolkit_status(self.job_context.org_name, self.handler_id, self.job_name, self.handler_kind)
        self.detailed_print(f"Toolkit status for {self.job_name} is {toolkit_status}", file=sys.stderr)
        if metadata_status not in ("Canceled", "Canceling", "Paused", "Pausing") and toolkit_status != "SUCCESS" and self.job_context.action != "trtexec":
            update_job_status(self.job_context.org_name, self.handler_id, self.job_name, status="Error", kind=self.handler_kind)
            metadata_status = "Error"

        self.detailed_print(f"Job Done: {self.job_name} Final status: {metadata_status}", file=sys.stderr)
        if BACKEND == "BCP" and self.network in TAO_NETWORKS:
            overwrite_job_logs_from_bcp(self.logfile, self.job_name)
        with open(self.logfile, "a", encoding='utf-8') as f:
            f.write(f"\n{metadata_status} EOF\n")
        if self.ngc_runner or (self.network in TAO_NETWORKS and BACKEND == "local-k8s"):
            if metadata_status not in ("Canceled", "Canceling", "Paused", "Pausing"):
                jobDriver.delete(self.job_name)

    def run(self):
        """Calls necessary setup functions and calls job creation"""
        # Set up
        self.thread = threading.current_thread()
        try:
            # Generate config
            self.spec, self.config = self.generate_config()
            self.cs_instance, _ = create_cs_instance(self.workspace_metadata)
            self.handle_multiple_ptm_fields()
            # Populate the cloud metadata for the job
            self.get_handler_cloud_details()
            # Generate run command
            self.run_command, self.status_file, outdir = self.generate_run_command()
            if self.network in TAO_NETWORKS and self.spec:
                self.num_gpu = get_num_gpus_from_spec(self.spec, self.job_context.action, default=self.num_gpu)
            if not outdir:
                outdir = f"/results/{self.job_name}"
            # Pipe stdout and stderr to logfile
            self.run_command += f" 2>&1 | tee /{self.job_name}.txt"
            # After command runs, make sure subdirs permission allows anyone to enter and delete
            self.run_command += f"; find {outdir} -type d | xargs chmod 777"
            # After command runs, make sure artifact files permission allows anyone to delete
            if self.local_cluster:
                outdir = os.path.normpath(outdir) + os.sep  # remove double trailing slashes for filepath
                # Do not change permission for status file in local
                self.run_command += f"; find {outdir} -type f ! -path '{outdir}status.json' | xargs chmod 666"
            else:
                self.run_command += f"; find {outdir} -type f | xargs chmod 666"
            # Optionally, pipe self.run_command into a log file
            self.detailed_print(self.run_command, self.status_file, file=sys.stderr)
            self.detailed_print(self.image, file=sys.stderr)

            nv_job_metadata = {}
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
                nv_job_metadata["spec_file_path"] = action_spec_path_kitti

                # Transfer spec file to cloud
                self.cs_instance.upload_file(action_spec_path_kitti, f"/temp/{self.job_name}.yaml")

            # Submit to K8s
            # Platform is None, but might be updated in self.generate_config() or self.generate_run_command()
            # If platform is indeed None, jobDriver.create would take care of it.
            docker_env_vars = self.handler_metadata.get("docker_env_vars", {})
            self.decrypt_docker_env_vars(docker_env_vars)
            job_env_variables = copy.deepcopy(docker_env_vars)
            # Add environment variables from medical.
            self.generate_env_variables(job_env_variables)
            if self.medical_env_variable:
                job_env_variables.update(self.medical_env_variable)

            # The medical local jobs like training for cl jobs are designed to run on cluster local GPUs.
            if self.ngc_runner:
                self.generate_nv_job_metadata(self.run_command, nv_job_metadata, job_env_variables)
            else:
                nv_job_metadata = None

            if self.network in TAO_NETWORKS and BACKEND == "local-k8s":
                self.create_microservice_action_job(job_env_variables)
            else:
                jobDriver.create(
                    self.job_context.user_id,
                    self.job_context.org_name,
                    self.job_name,
                    self.image,
                    self.run_command,
                    num_gpu=self.num_gpu,
                    accelerator=self.platform,
                    docker_env_vars=job_env_variables,
                    nv_job_metadata=nv_job_metadata,
                    local_cluster=self.local_cluster,
                )
            self.detailed_print("Job created", self.job_name, file=sys.stderr)
            self.monitor_job()
            return

        except Exception as e:
            # Something went wrong inside...
            self.detailed_print(traceback.format_exc(), file=sys.stderr)
            self.detailed_print(f"Job {self.job_name} did not start", file=sys.stderr)
            if BACKEND == "BCP" and self.network in TAO_NETWORKS:
                overwrite_job_logs_from_bcp(self.logfile, self.job_name)
            with open(self.logfile, "a", encoding='utf-8') as f:
                f.write(f"Error log: \n {e}")
                f.write("\nError EOF\n")
            update_job_status(self.job_context.org_name, self.handler_id, self.job_name, status="Error", kind=self.handler_kind)
            result_dict = {"detailed_status": {"message": "Error due to unmet dependencies"}}
            if isinstance(e, TimeoutError):
                result_dict = {"detailed_status": {"message": "Data downloading from cloud storage failed."}}
            update_job_metadata(self.job_context.org_name, self.handler_id, self.job_name, metadata_key="result", data=result_dict, kind=self.handler_kind)
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
        if self.network == "object_detection" and self.action == "convert":
            self.network = "detectnet_v2"
            self.action = "dataset_convert"
        if self.network == "object_detection" and "efficientdet" in self.action:
            self.network = self.action.replace("convert_", "")
            self.action = "dataset_convert"
        if self.network == "object_detection":
            if self.action == "annotation_format_convert":
                self.network = "annotations"
                self.action = "convert"
            if self.action == "auto_labeling":
                self.network = "auto_label"
                self.action = "generate"
            if self.action == "augment":
                self.network = "augmentation"
                self.action = "generate"
            if self.action in ("analyze", "validate"):
                self.network = "analytics"

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
        if action in network_config.get("cli_params", {}).keys():
            for field_name, inference_fn in network_config["cli_params"][action].items():
                field_value = CLI_CONFIG_TO_FUNCTIONS[inference_fn](self.job_context, self.handler_metadata)
                if field_value:
                    config[field_name] = field_value
        return {}, config

    def generate_run_command(self):
        """Generate run command"""
        if self.action == "dataset_convert":
            if self.network not in ("efficientdet_tf2", "ocrnet", "pointpillars"):
                self.config["results_dir"] = CLI_CONFIG_TO_FUNCTIONS["output_dir"](self.job_context, self.handler_metadata)

        params_to_cli = build_cli_command(self.config)
        run_command = f"{self.network} {self.action} {params_to_cli}"
        if self.action == "trtexec":
            run_command = f"{self.action} {params_to_cli}"

        status_file = os.path.join(self.jobs_root, self.job_name, "status.json")
        return run_command, status_file, None


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
        if action in network_config.get("cli_params", {}).keys():
            for field_name, inference_fn in network_config["cli_params"][action].items():
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
        if "cli_params" in network_config and action in network_config["cli_params"] and "experiment_spec_file" in network_config["cli_params"][action].keys() and network_config["cli_params"][action]["experiment_spec_file"] == "parent_spec_copied":
            spec_path = CLI_CONFIG_TO_FUNCTIONS["experiment_spec"](self.job_context, self.handler_metadata)
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

        # Move CLI params from spec to config
        spec_keys_all = copy.deepcopy(list(spec.keys()))  # Since we will be popping the value out, spec would change @ each iteration
        if "cli_params" in network_config and action in network_config["cli_params"]:
            for field_name in spec_keys_all:
                cnd1 = field_name in network_config["cli_params"][action].keys()
                cnd2 = network_config["cli_params"][action].get(field_name, None) == "from_csv"
                cnd3 = type(spec[field_name]) in [str, float, int, bool]
                if cnd1 and cnd2 and cnd3:
                    config[field_name] = spec.pop(field_name)
        self.detailed_print("Loaded specs", file=sys.stderr)

        # Infer dataset config
        spec = DS_CONFIG_TO_FUNCTIONS[self.network](spec, self.job_context, self.handler_metadata)
        self.detailed_print("Loaded dataset", file=sys.stderr)

        # Add classwise config
        classwise = self.api_params["classwise"] == "True"
        if classwise:
            spec = process_classwise_config(spec)

        return spec, config

    def post_run(self):
        """Carry's out functions after the job is executed"""
        # copy pruned model so that evaluate can access via parent relation
        action = self.job_context.action
        if self.network in ("efficientdet_tf2", "classification_tf2", "ocdnet", "ocrnet") and action == "retrain":
            inference_fn = "parent_model"
            pruned_model_path = CLI_CONFIG_TO_FUNCTIONS[inference_fn](self.job_context, self.handler_metadata)
            bucket_name = pruned_model_path.split("//")[1].split("/")[0]
            pruned_model_path = pruned_model_path[pruned_model_path.find(bucket_name) + len(bucket_name):]
            _, file_extension = os.path.splitext(pruned_model_path)
            self.detailed_print(f"Copying pruned model {pruned_model_path} after retrain to /results/{self.job_name}/pruned_model{file_extension}\n", file=sys.stderr)
            self.cs_instance.copy_file(pruned_model_path, f"/results/{self.job_name}/pruned_model{file_extension}")
        if self.job_context.action == "annotation_format_convert":
            if self.spec["data"]["input_format"] == "KITTI":
                self.handler_metadata["format"] = "coco"
            elif self.spec["data"]["input_format"] == "COCO":
                self.handler_metadata["format"] = "kitti"
            write_handler_metadata(self.job_context.org_name, self.handler_id, self.handler_metadata, self.handler_kind)


class AutoMLPipeline(ActionPipeline):
    """AutoML pipeline which carry's out network specific param changes; generating run commands and creating job for individual experiments"""

    def __init__(self, job_context):
        """Initialize the AutoMLPipeline class"""
        super().__init__(job_context)
        self.automl_brain_job_id = self.job_context.id
        self.job_root = os.path.join(get_jobs_root(self.job_context.user_id, self.job_context.org_name), self.automl_brain_job_id)
        self.rec_number = self.get_recommendation_number()
        self.expt_root = f"{self.job_root}/experiment_{self.rec_number}"
        self.recs_dict = safe_load_file(f"{self.job_root}/controller.json")
        self.brain_dict = safe_load_file(f"{self.job_root}/brain.json")
        # Assign a new job id if not assigned already
        self.job_name = self.recs_dict[self.rec_number].get("job_id", None)
        if not self.job_name:
            self.job_name = str(uuid.uuid4())
            self.detailed_print("New job id being assigned to recommendation", self.job_name, file=sys.stderr)
            self.recs_dict[self.rec_number]["job_id"] = self.job_name
            safe_dump_file(f"{self.job_root}/controller.json", self.recs_dict)

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
                get_handler_root(base_exp_uuid, "experiments", base_exp_uuid, ptm_id)
            )

    def generate_config(self, recommended_values):
        """Generate config for AutoML experiment"""
        spec_json_path = os.path.join(get_handler_spec_root(self.job_context.user_id, self.job_context.org_name, self.job_context.handler_id), f"{self.automl_brain_job_id}-train-spec.json")
        spec = load_json_spec(spec_json_path)

        epoch_multiplier = self.brain_dict.get("epoch_multiplier", None)
        if epoch_multiplier is not None:
            current_ri = int(self.brain_dict.get("ri", {"0": [float('-inf')]})[str(self.brain_dict.get("bracket", 0))][0])

        for field_name, inference_fn in self.network_config["automl_spec_params"].items():
            if "automl_" in inference_fn:
                field_value = CLI_CONFIG_TO_FUNCTIONS[inference_fn](self.job_context, self.handler_metadata, self.job_root, self.rec_number, self.job_name)
            elif "assign_const_value" in inference_fn:
                if epoch_multiplier:
                    field_value = int(epoch_multiplier * current_ri)
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
                write_nested_dict(spec, field_name, field_value)

        spec = DS_CONFIG_TO_FUNCTIONS[self.network](spec, self.job_context, self.handler_metadata)
        self.detailed_print("Loaded AutoML specs", file=sys.stderr)

        for param_name, param_value in recommended_values.items():
            write_nested_dict(spec, param_name, param_value)
        self.num_gpu = get_num_gpus_from_spec(spec, "train", default=self.num_gpu)

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
        return action_spec_path

    def generate_run_command(self):
        """Generate the command to be run inside docker for AutoML experiment"""
        params_to_cli = build_cli_command(self.config)
        run_command = f"{self.network} train {params_to_cli}"
        logfile = f'/{self.job_name}.txt'
        run_command += f"  2>&1 | tee {logfile}"
        return run_command

    def get_recommendation_number(self):
        """Return the current recommendation number"""
        rec_number = None
        for dep in self.job_context.dependencies:
            if dep.type == "automl":
                rec_number = int(dep.name)
                break
        return rec_number

    def monitor_job(self, action_spec_path=None, job_env_variables=None, nv_job_metadata=None):
        """Monitors the job status and updates job metadata"""
        if not self.config:
            recommended_values = self.recs_dict[self.rec_number].get("specs", {})
            self.spec = self.generate_config(recommended_values)
            self.handle_multiple_ptm_fields()
            self.get_handler_cloud_details()

        if not job_env_variables:
            docker_env_vars = self.handler_metadata.get("docker_env_vars", {})
            self.decrypt_docker_env_vars(docker_env_vars)
            job_env_variables = copy.deepcopy(docker_env_vars)
            self.generate_env_variables(job_env_variables, automl_brain_job_id=self.automl_brain_job_id, experiment_number=str(self.rec_number))

        run_command = self.generate_run_command()
        if not nv_job_metadata:
            nv_job_metadata = {}
            if self.ngc_runner:
                self.generate_nv_job_metadata(run_command, nv_job_metadata, job_env_variables)
                nv_job_metadata["spec_file_path"] = action_spec_path

        k8s_status = jobDriver.status(self.job_context.org_name, self.handler_id, self.job_name, self.handler_kind, use_ngc=self.ngc_runner, network=self.network, action=self.action)
        while k8s_status in ["Done", "Error", "Running", "Pending", "Creating"]:
            time.sleep(5)
            if os.path.exists(os.path.join(self.expt_root, "status.json")):
                break
            if k8s_status == "Error":
                self.detailed_print(f"Relaunching job {self.job_name}", file=sys.stderr)
                wait_for_job_completion(self.job_name)
                if self.network in TAO_NETWORKS and BACKEND == "local-k8s":
                    self.create_microservice_action_job(job_env_variables)
                else:
                    jobDriver.create(self.job_context.user_id, self.job_context.org_name, self.job_name, self.image, run_command, num_gpu=self.num_gpu, docker_env_vars=job_env_variables, nv_job_metadata=nv_job_metadata)
            k8s_status = jobDriver.status(self.job_context.org_name, self.handler_id, self.job_name, self.handler_kind, use_ngc=self.ngc_runner, network=self.network, action=self.action)

    def run(self):
        """Calls necessary setup functions and calls job creation"""
        try:
            recommended_values = self.recs_dict[self.rec_number].get("specs", {})
            self.cs_instance, _ = create_cs_instance(self.workspace_metadata)
            self.add_ptm_dependency(recommended_values)

            self.spec = self.generate_config(recommended_values)
            self.handle_multiple_ptm_fields()
            self.get_handler_cloud_details()
            action_spec_path = self.save_recommendation_specs()
            run_command = self.generate_run_command()

            self.detailed_print(run_command, file=sys.stderr)

            # Wait for existing AutoML jobs to complete
            wait_for_job_completion(self.job_name)

            delete_lingering_checkpoints(self.recs_dict[self.rec_number].get("best_epoch_number", ""), self.expt_root)
            docker_env_vars = self.handler_metadata.get("docker_env_vars", {})
            self.decrypt_docker_env_vars(docker_env_vars)
            job_env_variables = copy.deepcopy(docker_env_vars)
            self.generate_env_variables(job_env_variables, automl_brain_job_id=self.automl_brain_job_id, experiment_number=str(self.rec_number), automl_exp_job_id=self.job_name)

            nv_job_metadata = {}
            if self.ngc_runner:
                self.generate_nv_job_metadata(run_command, nv_job_metadata, job_env_variables)
                nv_job_metadata["spec_file_path"] = action_spec_path

            if self.network in TAO_NETWORKS and BACKEND == "local-k8s":
                self.create_microservice_action_job(job_env_variables)
            else:
                jobDriver.create(self.job_context.user_id, self.job_context.org_name, self.job_name, self.image, run_command, num_gpu=self.num_gpu, docker_env_vars=job_env_variables, nv_job_metadata=nv_job_metadata)
            self.detailed_print(f"AutoML recommendation with experiment id {self.rec_number} and job id {self.job_name} submitted", file=sys.stderr)
            self.monitor_job(action_spec_path, job_env_variables, nv_job_metadata)

            return True

        except Exception:
            self.detailed_print(f"AutoMLpipeline for network {self.network} failed due to exception {traceback.format_exc()}", file=sys.stderr)
            self.detailed_print(self.job_name, file=sys.stderr)

            self.recs_dict[self.rec_number]["status"] = "failure"
            safe_dump_file(f"{self.job_root}/controller.json", self.recs_dict)

            update_job_status(self.job_context.org_name, self.handler_id, self.job_context.id, status="Error", kind=self.handler_kind)
            jobDriver.delete(self.job_context.id, use_ngc=False)
            return False


class ContinualLearning(ActionPipeline):
    """Class for continual learning specific changes required during annotation action."""

    def __init__(self, job_context):
        """Initialize the ContinualLearning class"""
        super().__init__(job_context)
        self.ngc_runner = False
        # override the ActionPipeline's handler_root
        self.train_ds = self.handler_metadata["train_datasets"]
        self.image = DOCKER_IMAGE_MAPPER["api"]
        self.job_root = os.path.join(self.jobs_root, self.job_context.id)
        self.logs_from_toolkit = f"{self.jobs_root}/{self.job_name}/logs_from_toolkit.txt"

    def generate_convert_script(self, notify_record):
        """Generate a script to perform continual learning"""
        cl_script = generate_cl_script(notify_record, self.job_context, self.handler_root, self.logfile, self.logs_from_toolkit)
        cl_script_path = os.path.join(self.job_root, "continual_learning.py")
        with open(cl_script_path, "w", encoding="utf-8") as f:
            f.write(cl_script)

        return cl_script_path

    def monitor_job(self):
        """Monitors the job status and updates job metadata"""
        k8s_status = jobDriver.status(self.job_context.org_name, self.handler_id, self.job_name, self.handler_kind, use_ngc=False)
        while k8s_status in ["Running", "Pending"]:
            # Poll every 30 seconds
            time.sleep(30)
            k8s_status = jobDriver.status(self.job_context.org_name, self.handler_id, self.job_name, self.handler_kind, use_ngc=False)
        self.detailed_print(f"Job status: {k8s_status}", file=sys.stderr)
        if k8s_status == "Error":
            update_job_status(self.job_context.org_name, self.handler_id, self.job_context.id, status="Error")
        self.detailed_print("Continual Learning finished.", file=sys.stderr)

    def run(self):
        """Run the continual learning pipeline"""
        # Set up
        self.thread = threading.current_thread()
        try:
            # Find the train dataset path
            if len(self.train_ds) != 1:
                raise ValueError("Continual Learning only supports one train dataset.")
            train_dataset = self.train_ds[0]
            dataset_path = get_handler_root(self.job_context.org_name, "datasets", train_dataset, None)

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
            jobDriver.create(self.job_context.user_id, self.job_context.org_name, self.job_name, self.image, self.run_command, num_gpu=0, cl_medical=True)
            self.detailed_print("Job created", self.job_name, file=sys.stderr)
            self.monitor_job()
        except Exception:
            self.detailed_print(f"ContinualLearning for {self.network} failed because {traceback.format_exc()}", file=sys.stderr)
            shutil.copy(self.logfile, self.logs_from_toolkit)
            update_job_status(self.job_context.org_name, self.handler_id, self.job_context.id, status="Error")


class BundleTrain(ActionPipeline):
    """Class for MONAI bundle specific changes required during training"""

    CLOUD_STORAGE_LIB = "cloud_storage"
    UTILS_LIB = "utils"

    def __init__(self, job_context):
        """Initialize the BundleTrain class"""
        super().__init__(job_context)
        spec = self.get_spec()
        # override the ActionPipeline's handler_root
        if "cluster" in spec and spec["cluster"] == "local":
            self.ngc_runner = False  # this will make the executor uses local gpu even if BACKEND = True
            # Though self.local_cluster is kind of duplicate of self.ngc_runner, a wide scope of property,
            # the diff is local_cluster only used to determine the mounting strategy in executor create for medical bundles training.
            self.local_cluster = True
            self.handler_root = get_handler_root(self.job_context.org_name, "experiments", None)
        self.network = job_context.network
        self.action = job_context.action
        self.job_root = os.path.join(self.jobs_root, self.job_context.id)
        self.model_script_path = os.path.join(self.job_root, "train.py")
        self.config_file = "configs/train.json"
        self.bundle_dir = ""

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
        self.detailed_print("Specs are loaded.", file=sys.stderr)
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

    @staticmethod
    def add_prefix(config, prefix):
        """ Add prefix to config string or config list."""
        if isinstance(config, list):
            config_file = [os.path.join(prefix, x) for x in config]  # If it's a list, join the items
        elif isinstance(config, str):
            config_file = os.path.join(prefix, config)
        else:
            raise RuntimeError(f"Do not support config file with type {type(config)}")
        return config_file

    def update_dataset_env_variables(self, datasets_info):
        """Update the environment variables relating to datasets."""
        for dataset_usage in datasets_info:
            secret = datasets_info[dataset_usage].pop("client_secret", "")
            secret_env = datasets_info[dataset_usage]["secret_env"]
            self.medical_env_variable.update({secret_env: secret})

    def get_dicom_convert(self, datasets_info):
        """Check if need to convert the dicom datasets."""
        is_convert_list = []
        for dataset_usage in datasets_info:
            dataset_info = datasets_info[dataset_usage]
            cur_is_dicom = dataset_info["is_dicom"]
            is_convert_list.append(cur_is_dicom)
        is_convert_set = set(is_convert_list)
        if len(is_convert_set) > 1:
            raise RuntimeError("Job does not accept multi type datasets.")

        return is_convert_set.pop()

    def get_ngc_path(self):
        """Get the ngc path of the pretrained model."""
        base_experiment_ids = self.handler_metadata.get("base_experiment", [])
        if len(base_experiment_ids) > 1:
            raise RuntimeError("Bundle train only supports 1 base experiment.")

        meta_data = get_base_experiment_metadata(base_experiment_ids[0])
        ngc_path = meta_data.get("ngc_path", "")
        return ngc_path

    def _get_bundle_url(self, root):
        """Get the url of the customized bundle."""
        bundle_url = ""
        bundle_file = os.path.join(root, CUSTOMIZED_BUNDLE_URL_FILE)
        if os.path.exists(bundle_file):
            bundle_content = safe_load_file(bundle_file)
            bundle_url = bundle_content.get(CUSTOMIZED_BUNDLE_URL_KEY, "")
        return bundle_url

    def get_experiment_cloud_meta(self):
        """Get experiment cloud storage metadata."""
        cloud_type = self.cloud_metadata["experiment"][0]["workspace"]["cloud_type"]
        cloud_specific_details = self.cloud_metadata["experiment"][0]["workspace"]["cloud_specific_details"]
        account_str = "account_name" if cloud_type == "azure" else "access_key"
        account_id = cloud_specific_details[account_str]
        secret_str = "access_key" if cloud_type == "azure" else "secret_key"
        secret = cloud_specific_details[secret_str]

        cloud_meta = {
            "cloud_type": cloud_type,
            "bucket_name": cloud_specific_details["cloud_bucket_name"],
            "region": cloud_specific_details.get("cloud_region", None),
            "access_key": account_id,
            "secret_key": secret,
        }
        return cloud_meta

    def get_job_bundle_info(self, bundle_name):
        """Get the bundle information of the job."""
        bundle_root = os.path.join(self.job_root, bundle_name)
        path_prefix = bundle_root + "/"
        override = self.spec if self.spec else {}
        config_file = self.add_prefix(override.pop("config_file", self.config_file), path_prefix)
        logging_file = self.add_prefix(override.pop("logging_file", "configs/logging.conf"), path_prefix)
        meta_file = self.add_prefix(override.pop("meta_file", "configs/metadata.json"), path_prefix)

        bundle_info = {
            "bundle_root": bundle_root,
            "config_file": config_file,
            "logging_file": logging_file,
            "meta_file": meta_file,
        }
        bundle_info.update(override)
        if self.num_gpu > 1:
            bundle_info["num_gpu"] = self.num_gpu
        return bundle_info

    def generate_datasets_meta(self, datasets_info):
        """Generate the datasets metadata dict."""
        labels = self.handler_metadata.get("model_params", {}).get("labels", None)
        return {"datasets_info": datasets_info, "labels": labels}

    def generate_experiment_meta(self, bundle_name):
        """Generate the experiment metadata dict."""
        bundle_info = self.get_job_bundle_info(bundle_name)
        experiment_cloud_meta = self.get_experiment_cloud_meta()

        experiment_meta = {
            "bundle_info": bundle_info,
            "experiment_cloud_meta": experiment_cloud_meta,
        }
        return experiment_meta

    def generate_job_meta(self, ptm_root, is_customized, src_dir, dst_dir):
        """Generate job metadata dict."""
        # ptm_meta = {"ptm_root": "ptm_root", "bundle_url": "bundle_url", "ngc_path":"ngc_path", "is_customized":"Bool"}
        bundle_url = self._get_bundle_url(ptm_root)
        ngc_path = self.get_ngc_path()
        ptm_meta = {"ptm_root": ptm_root,
                    "bundle_url": bundle_url,
                    "ngc_path": ngc_path,
                    "is_customized": is_customized,
                    }
        copy_meta = {"src_dir": src_dir, "dst_dir": dst_dir}
        return {"ptm_meta": ptm_meta, "copy_meta": copy_meta}

    def generate_run_command(self):
        """Generate run command"""
        ptm_root, bundle_name, overriden_output_dir = CLI_CONFIG_TO_FUNCTIONS["medical_output_dir"](self.job_context, self.handler_metadata)
        copy_needed = bool(ptm_root)
        overriden_output_dir = overriden_output_dir.rstrip(os.path.sep)
        status_file = os.path.join(overriden_output_dir, "status.json")
        datasets_info = self.spec.pop("datasets_info", {})
        self.update_dataset_env_variables(datasets_info)
        self.get_dicom_convert(datasets_info)
        src_dir = os.path.join(ptm_root, bundle_name) if copy_needed else ""
        dst_dir = os.path.join(overriden_output_dir, bundle_name) if copy_needed else ""
        self.bundle_dir = dst_dir
        experiment_json_meta = json.dumps(self.generate_experiment_meta(bundle_name))
        datasets_json_meta = json.dumps(self.generate_datasets_meta(datasets_info))
        is_customized = self.network in MEDICAL_CUSTOM_ARCHITECT
        job_json_meta = json.dumps(self.generate_job_meta(ptm_root, is_customized, src_dir, dst_dir))
        run_command = f"mkdir -p {overriden_output_dir} && mkdir -p {ptm_root} && cd {overriden_output_dir}" + " && "
        run_command += f"cp -r {MEDICAL_SERVICE_SCRIPTS}/* {overriden_output_dir} && "
        run_command += f"python {self.action}.py --experiment_json_meta '{experiment_json_meta}' --datasets_json_meta '{datasets_json_meta}' --job_json_meta '{job_json_meta}'"
        if self.network in MEDICAL_CUSTOM_ARCHITECT:
            bundle_url = self._get_bundle_url(ptm_root)
            if not bundle_url:
                raise RuntimeError("Cannot find the pretrained model url for the customized bundle.")
        elif self.network != "medical_automl_generated":  # medical_automl_generated does not need ngc path nor ptm_root
            ngc_path = self.get_ngc_path()
            if not ngc_path:
                raise RuntimeError("Cannot find the ngc path for the pretrained model.")
        run_command = f"({run_command}) 2>&1"
        return run_command, status_file, overriden_output_dir


class BatchInfer(BundleTrain):
    """Class for MONAI bundle specific changes required during batch inference."""

    def __init__(self, job_context):
        """Initialize the BatchInfer class"""
        super().__init__(job_context)
        self.config_file = "configs/inference.json"

    def get_train_job_meta(self, train_job_id, bundle_name):
        """Get the train job meta data."""
        results_root = get_jobs_root(user_id=self.job_context.user_id, org_name=self.job_context.org_name)
        train_job_path = os.path.join(results_root, train_job_id, bundle_name)
        train_meta = {"train_meta": {"job_path": train_job_path}}
        return train_meta, train_job_path

    def generate_run_command(self):
        """Generate batch inference run command"""
        ptm_root, bundle_name, overriden_output_dir = CLI_CONFIG_TO_FUNCTIONS["medical_output_dir"](self.job_context, self.handler_metadata)
        status_file = os.path.join(overriden_output_dir, "status.json")
        datasets_info = self.spec.pop("datasets_info", {})
        train_job_id = self.spec.pop("train_job_id", "")
        if datasets_info:
            self.update_dataset_env_variables(datasets_info)
            self.get_dicom_convert(datasets_info)
        src_dir = os.path.join(ptm_root, bundle_name)
        dst_dir = os.path.join(overriden_output_dir, bundle_name)
        experiment_json_meta = json.dumps(self.generate_experiment_meta(bundle_name))
        datasets_json_meta = json.dumps(self.generate_datasets_meta(datasets_info))
        is_customized = self.network in MEDICAL_CUSTOM_ARCHITECT
        train_meta = {}
        if train_job_id:
            train_meta, train_job_path = self.get_train_job_meta(train_job_id, bundle_name)
            src_dir = train_job_path

        job_meta = self.generate_job_meta(ptm_root, is_customized, src_dir, dst_dir)
        job_meta.update(train_meta)
        job_json_meta = json.dumps(job_meta)
        run_command = f"mkdir -p {overriden_output_dir} && mkdir -p {ptm_root} && cd {overriden_output_dir}" + " && "
        run_command += f"cp -r {MEDICAL_SERVICE_SCRIPTS}/* {overriden_output_dir} && "
        run_command += f"python batchinfer.py --experiment_json_meta '{experiment_json_meta}' --datasets_json_meta '{datasets_json_meta}' --job_json_meta '{job_json_meta}'"
        if self.network in MEDICAL_CUSTOM_ARCHITECT:
            bundle_url = self._get_bundle_url(ptm_root)
            if not bundle_url:
                raise RuntimeError("Cannot find the pretrained model url for the customized bundle.")
        else:
            ngc_path = self.get_ngc_path()
            if not ngc_path:
                raise RuntimeError("Cannot find the ngc path for the pretrained model.")
        run_command = f"({run_command}) 2>&1"
        return run_command, status_file, overriden_output_dir


class Auto3DSegTrain(BundleTrain):
    """MONAI Auto3DSeg training action pipeline"""

    def generate_experiment_meta(self, bundle_name):
        """Generate the experiment metadata dict."""
        bundle_root = os.path.join(self.job_root, bundle_name)
        experiment_cloud_meta = self.get_experiment_cloud_meta()
        experiment_meta = {
            "input_info": {
                "work_dir": bundle_root,
                "templates_path_or_url": os.path.join(bundle_root, "algorithm_templates"),
                "multi_gpu": self.num_gpu > 1,
                **self.spec,
            },
            "experiment_cloud_meta": experiment_cloud_meta,
        }
        return experiment_meta

    def generate_job_meta(self, ptm_root, is_customized, src_dir, dst_dir):
        """Generate job metadata dict."""
        ngc_path = self.get_ngc_path()
        ptm_meta = {"ptm_root": ptm_root, "ngc_path": ngc_path}
        copy_meta = {"src_dir": src_dir, "dst_dir": dst_dir}
        return {"ptm_meta": ptm_meta, "copy_meta": copy_meta}

    def post_run(self):
        """Create a new model after the job is executed"""
        # set the url of the best model
        best_model_dir = os.path.join(self.bundle_dir, "best_model")
        cloud_folder = best_model_dir.lstrip(os.path.sep)
        # create a new experiment
        from handlers.app_handler import AppHandler
        request_dict = {
            "name": self.spec.get("output_experiment_name", "auto3dseg_automl_experiment"),
            "description": self.spec.get("output_experiment_description", "AutoML Generated Segmentation Model based on MONAI Auto3DSeg"),
            "type": "medical",
            "network_arch": "medical_automl_generated",
            "inference_dataset": self.spec.get("inference_dataset"),
            "realtime_infer": False,
            "bundle_url": cloud_folder,
            "workspace": self.handler_metadata.get("workspace")
        }
        ret_code = AppHandler.create_experiment(self.job_context.user_id, self.job_context.org_name, request_dict)
        new_model_id = ret_code.data["id"]
        self.detailed_print(f"New model is generated with id: {new_model_id}", file=sys.stderr)

        # update the status file to include newly generated model id
        status_dict = {
            "date": datetime.date.today().isoformat(),
            "time": datetime.datetime.now().isoformat(),
            "status": "SUCCESS",
            "message": f"A segmentation experiment (id={new_model_id}) is successfully created by MONAI AuoML.",
            "model_id": new_model_id,
        }
        with open(self.status_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(status_dict) + "\n")

        self.cs_instance.download_folder(cloud_folder, best_model_dir, maintain_src_folder_structure=True)


class Auto3DSegInfer(BundleTrain):
    """MONAI inference action pipeline"""

    def generate_config(self):
        """
        Generate spec for medical inference and creates "datalist.json"

        Returns:
            spec: contains the params for building medical bundle train command
        """
        # load spec
        spec = self.get_spec()

        # create datalist.json
        inference_dataset_id = spec.pop("inference_dataset", None)
        if inference_dataset_id:
            self.handler_metadata["inference_dataset"] = inference_dataset_id
        spec = DS_CONFIG_TO_FUNCTIONS[self.network](spec, self.job_context, self.handler_metadata)
        self.detailed_print("Datasets are loaded.", file=sys.stderr)

        return spec, {}

    def generate_experiment_meta(self, bundle_name):
        """Generate the experiment metadata dict."""
        experiment_cloud_meta = self.get_experiment_cloud_meta()

        experiment_meta = {
            "input_info": {"work_dir": self.job_root, **self.spec},
            "experiment_cloud_meta": experiment_cloud_meta,
        }
        return experiment_meta

    def generate_job_meta(self, *args):
        """Generate job metadata dict."""
        handler_root = get_handler_root(self.job_context.org_name, "experiments", self.job_context.handler_id, None)
        auto3dseg_url = self._get_bundle_url(handler_root)
        if not auto3dseg_url:
            raise RuntimeError("Cannot find the pretrained Auto3DSeg model.")
        ptm_meta = {"bundle_url": auto3dseg_url, "is_auto3dseg_inference": True}
        return {"ptm_meta": ptm_meta}


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
                        "gen_trt_engine": TrainVal,
                        "trtexec": TrainVal,
                        "purpose_built_models_ds_convert": TrainVal,
                        "odconvert": TrainVal,
                        "pyt_odconvert": TrainVal,
                        "data_services": TrainVal,
                        "medical_annotation": ContinualLearning,
                        "medical_automl_auto3dseg": Auto3DSegTrain,
                        "medical_train": BundleTrain,
                        "medical_automl_batchinfer": Auto3DSegInfer,
                        "medical_batchinfer": BatchInfer,
                        "medical_generate": BatchInfer,
                        }
