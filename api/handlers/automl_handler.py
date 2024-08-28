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

"""AutoML handler modules"""
import os
import sys
import json
import time
from copy import deepcopy

from handlers.stateless_handlers import get_handler_metadata, get_handler_type, get_root, get_jobs_root
from handlers.utilities import Code, decrypt_handler_metadata
from handlers.docker_images import DOCKER_IMAGE_MAPPER
from utils import safe_load_file, safe_dump_file
from automl.utils import merge_normal_and_automl_job_meta
from job_utils.automl_job_utils import on_delete_automl_job
from job_utils import executor as jobDriver

# TODO Make sure the image name is current docker tag of the API
image = DOCKER_IMAGE_MAPPER["api"]


class AutoMLHandler:
    """
    Handler class for AutoML jobs
    Start: Start controller as a K8s job
    Stop: Stop controller and stop the autoML recommendation that is running
    Resume: Same as Start (Since Controller's constructor allows for restore)
    Delete: Same as AppHandler
    Download: Same as AppHandler
    Retrieve: Construct the JobSchema based on Controller's status.json

    """

    @staticmethod
    def start(user_id, org_name, experiment_id, job_id, handler_metadata, name=""):
        """Starts an automl job by running automl_start.py file"""
        root = os.path.join(get_jobs_root(user_id, org_name), job_id)
        if not os.path.exists(root):
            os.makedirs(root)

        if not name:
            name = "automl train job"
        network = get_handler_type(handler_metadata)
        metric = handler_metadata.get("metric", "map")
        automl_settings = handler_metadata.get("automl_settings", {})
        automl_algorithm = automl_settings.get("automl_algorithm", "Bayesian")
        automl_max_recommendations = automl_settings.get("automl_max_recommendations", 20)
        automl_delete_intermediate_ckpt = automl_settings.get("automl_delete_intermediate_ckpt", True)
        automl_R = automl_settings.get("automl_R", 27)
        automl_nu = automl_settings.get("automl_nu", 3)
        epoch_multiplier = automl_settings.get("epoch_multiplier", 1)
        automl_add_hyperparameters = automl_settings.get("automl_add_hyperparameters", "[]")
        automl_remove_hyperparameters = automl_settings.get("automl_remove_hyperparameters", "[]")
        override_automl_disabled_params = automl_settings.get("override_automl_disabled_params", False)

        workspace_id = handler_metadata.get("workspace")
        workspace_metadata = get_handler_metadata(org_name, workspace_id, "workspaces")
        decrypted_workspace_metadata = deepcopy(workspace_metadata)
        decrypt_handler_metadata(decrypted_workspace_metadata)

        # Call the script
        print("Starting automl", job_id, file=sys.stderr)

        run_command = 'umask 0 && unzip -q /opt/ngccli/ngccli_linux.zip -d /opt/ngccli/ && /opt/ngccli/ngc-cli/ngc --version && '
        run_command += f"""/venv/bin/python3 automl_start.py --user_id={user_id} --org_name={org_name} --name='{name}' --root={root} --automl_job_id={job_id} --network={network} --experiment_id={experiment_id} --resume=False --automl_algorithm={automl_algorithm} --automl_max_recommendations={automl_max_recommendations} --automl_delete_intermediate_ckpt={automl_delete_intermediate_ckpt} --automl_R={automl_R} --automl_nu={automl_nu} --metric={metric} --epoch_multiplier={epoch_multiplier} --automl_add_hyperparameters="{automl_add_hyperparameters}" --automl_remove_hyperparameters="{automl_remove_hyperparameters}" --override_automl_disabled_params={override_automl_disabled_params} --decrypted_workspace_metadata='{json.dumps(decrypted_workspace_metadata)}'"""
        jobDriver.create(user_id, org_name, job_id, image, run_command, num_gpu=0, automl_brain=True)  # TODO: Commented for testing only

    @staticmethod
    def stop(user_id, org_name, experiment_id, job_id):
        """Stops a running automl job"""
        print("Stopping automl", file=sys.stderr)

        try:
            jobDriver.delete(job_id, use_ngc=False)
            k8s_status = jobDriver.status(org_name, experiment_id, job_id, "experiments", use_ngc=False)
            while k8s_status in ("Done", "Error", "Running", "Pending"):
                if k8s_status in ("Done", "Error"):
                    break
                k8s_status = jobDriver.status(org_name, experiment_id, job_id, "experiments", use_ngc=False)
                time.sleep(5)
            controller_path = os.path.join(get_jobs_root(user_id, org_name), job_id, "controller.json")
            recommendations = safe_load_file(controller_path)
            for recommendation in recommendations:
                recommendation_job_id = recommendation.get("job_id")
                if recommendation_job_id:
                    if recommendation.get("status") in ("pending", "running", "started"):
                        recommendation["status"] = "canceling"
                        safe_dump_file(controller_path, recommendations)
                    jobDriver.delete(recommendation_job_id)
                    rec_k8s_status = jobDriver.status(org_name, experiment_id, recommendation_job_id, "experiments")
                    while rec_k8s_status in ("Done", "Error", "Running", "Pending"):
                        if rec_k8s_status in ("Done", "Error"):
                            break
                        rec_k8s_status = jobDriver.status(org_name, experiment_id, recommendation_job_id, "experiments")
                        time.sleep(5)
                    if recommendation.get("status") in ("pending", "running", "started", "canceling"):
                        recommendation["status"] = "canceled"
                        safe_dump_file(controller_path, recommendations)
        except:
            return Code(404, [], "job cannot be stopped in platform")

        # Remove any pending jobs from Workflow queue
        try:
            on_delete_automl_job(org_name, experiment_id, job_id)
        except:
            return Code(200, {"message": f"job {job_id} cancelled, and no pending recommendations"})

        return Code(200, {"message": f"job {job_id} cancelled"})

    @staticmethod
    def retrieve(user_id, org_name, experiment_id, job_id):
        """Retrieves a running automl job and writes the stats to job_id.json in job_metadata folder"""
        print("Retrieving automl", file=sys.stderr)
        root = os.path.join(get_jobs_root(user_id, org_name), job_id)
        json_file = os.path.join(root, "automl_metadata.json")

        if not os.path.exists(json_file):
            return Code(400, {}, "No AutoML run found")

        # Check if job is running / error / done / pending
        # Skipping this as automl pods are deleted upon completion and we can't use kubernetes api to fetch the status
        # The status is updated in controller.py
        # k8s_status = jobDriver.status(job_id)
        # stateless_handlers.update_job_status(org_name,experiment_id,job_id,status=k8s_status)

        # Create a JobResult schema and update the jobs_metadata/<automl_job_id>.json
        path = os.path.join(get_root(), org_name, "experiments", experiment_id, "jobs_metadata", job_id + ".json")
        job_meta = safe_load_file(path)
        merge_normal_and_automl_job_meta(user_id, org_name, job_id, job_meta)

        return Code(200, job_meta, "Job retrieved")

    @staticmethod
    def resume(user_id, org_name, experiment_id, job_id, handler_metadata, name=""):
        """Resumes a stopped automl job"""
        print("Resuming automl", job_id, file=sys.stderr)

        root = os.path.join(get_jobs_root(user_id, org_name), job_id)
        if not os.path.exists(root):
            os.makedirs(root)

        if not name:
            name = "automl train job"
        network = get_handler_type(handler_metadata)
        metric = handler_metadata.get("metric", "map")
        automl_settings = handler_metadata.get("automl_settings", {})
        automl_algorithm = automl_settings.get("automl_algorithm", "Bayesian")
        automl_max_recommendations = automl_settings.get("automl_max_recommendations", 20)
        automl_delete_intermediate_ckpt = automl_settings.get("automl_delete_intermediate_ckpt", True)
        automl_R = automl_settings.get("automl_R", 27)
        automl_nu = automl_settings.get("automl_nu", 3)
        epoch_multiplier = automl_settings.get("epoch_multiplier", 1)
        automl_add_hyperparameters = automl_settings.get("automl_add_hyperparameters", "[]")
        automl_remove_hyperparameters = automl_settings.get("automl_remove_hyperparameters", "[]")
        override_automl_disabled_params = automl_settings.get("override_automl_disabled_params", False)

        workspace_id = handler_metadata.get("workspace")
        workspace_metadata = get_handler_metadata(org_name, workspace_id, "workspaces")
        decrypted_workspace_metadata = deepcopy(workspace_metadata)
        decrypt_handler_metadata(decrypted_workspace_metadata)

        # Call the script
        run_command = f"""/venv/bin/python3 automl_start.py --user_id={user_id} --org_name={org_name} --name='{name}' --root={root} --automl_job_id={job_id} --network={network} --experiment_id={experiment_id} --resume=True --automl_algorithm={automl_algorithm} --automl_max_recommendations={automl_max_recommendations} --automl_delete_intermediate_ckpt={automl_delete_intermediate_ckpt} --automl_R={automl_R} --automl_nu={automl_nu} --metric={metric} --epoch_multiplier={epoch_multiplier} --automl_add_hyperparameters="{automl_add_hyperparameters}" --automl_remove_hyperparameters="{automl_remove_hyperparameters}" --override_automl_disabled_params={override_automl_disabled_params} --decrypted_workspace_metadata='{json.dumps(decrypted_workspace_metadata)}'"""
        jobDriver.create(user_id, org_name, job_id, image, run_command, num_gpu=0, automl_brain=True)
