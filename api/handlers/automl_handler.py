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
from handlers import stateless_handlers
from handlers.stateless_handlers import get_handler_type, get_root, safe_load_file
from handlers.utilities import Code
from handlers.docker_images import DOCKER_IMAGE_MAPPER
from job_utils.automl_job_utils import on_delete_automl_job
from job_utils import executor as jobDriver
import sys
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
    def start(user_id, experiment_id, job_id, handler_metadata):
        """Starts an automl job by running automl_start.py file"""
        root = get_root() + f"{user_id}/experiments/{experiment_id}/{job_id}/"
        if not os.path.exists(root):
            os.makedirs(root)

        network = get_handler_type(user_id, experiment_id)
        automl_algorithm = handler_metadata.get("automl_algorithm", "bayesian")
        automl_max_recommendations = handler_metadata.get("automl_max_recommendations", 20)
        automl_delete_intermediate_ckpt = handler_metadata.get("automl_delete_intermediate_ckpt", True)
        automl_R = handler_metadata.get("automl_R", 27)
        automl_nu = handler_metadata.get("automl_nu", 3)
        metric = handler_metadata.get("metric", "map")
        epoch_multiplier = handler_metadata.get("epoch_multiplier", 1)
        automl_add_hyperparameters = handler_metadata.get("automl_add_hyperparameters", "[]")
        automl_remove_hyperparameters = handler_metadata.get("automl_remove_hyperparameters", "[]")
        override_automl_disabled_params = handler_metadata.get("override_automl_disabled_params", False)

        # Call the script
        print("Starting automl", job_id, file=sys.stderr)

        run_command = 'umask 0 && unzip -q /opt/ngccli/ngccli_linux.zip -d /opt/ngccli/ && /opt/ngccli/ngc-cli/ngc --version && '
        run_command += f'/venv/bin/python3 automl_start.py --user_id={user_id} --root={root} --automl_job_id={job_id} --network={network} --experiment_id={experiment_id} --resume=False --automl_algorithm={automl_algorithm} --automl_max_recommendations={automl_max_recommendations} --automl_delete_intermediate_ckpt={automl_delete_intermediate_ckpt} --automl_R={automl_R} --automl_nu={automl_nu} --metric={metric} --epoch_multiplier={epoch_multiplier} --automl_add_hyperparameters="{automl_add_hyperparameters}" --automl_remove_hyperparameters="{automl_remove_hyperparameters}" --override_automl_disabled_params={override_automl_disabled_params}'
        jobDriver.create(user_id, job_id, image, run_command, num_gpu=0)  # TODO: Commented for testing only

    @staticmethod
    def stop(user_id, experiment_id, job_id):
        """Stops a running automl job"""
        print("Stopping automl", file=sys.stderr)

        try:
            jobDriver.delete(job_id)
            controller_path = os.path.join(stateless_handlers.get_handler_root(user_id, "experiments", experiment_id, job_id), "controller.json")
            recommendations = stateless_handlers.safe_load_file(controller_path)
            for recommendation in recommendations:
                recommendation_job_id = recommendation.get("job_id")
                if recommendation_job_id:
                    jobDriver.delete(recommendation_job_id)
        except:
            return Code(404, [], "job cannot be stopped in platform")

        # Remove any pending jobs from Workflow queue
        try:
            on_delete_automl_job(user_id, experiment_id, job_id)
        except:
            return Code(200, [job_id], "job cancelled, and no pending recommendations")

        # TODO: Move the best model to weights/model.tlt

        return Code(200, job_id, "job cancelled")

    @staticmethod
    def retrieve(user_id, experiment_id, job_id):
        """Retrieves a running automl job and writes the stats to job_id.json in job_metadata folder"""
        print("Retrieving automl", file=sys.stderr)
        root = get_root() + f"{user_id}/experiments/{experiment_id}/{job_id}/"
        stats = {}

        json_file = os.path.join(root, "automl_metadata.json")

        if not os.path.exists(json_file):
            return Code(400, {}, "No AutoML run found")

        stats = safe_load_file(json_file)
        if not stats:
            stats["message"] = "Stats will be updated in a few seconds"

        # Check if job is running / error / done / pending
        # Skipping this as automl pods are deleted upon completion and we can't use kubernetes api to fetch the status
        # The status is updated in controller.py
        # k8s_status = jobDriver.status(job_id)
        # stateless_handlers.update_job_status(user_id,experiment_id,job_id,status=k8s_status)

        # Create a JobResult schema and update the jobs_metadata/<automl_job_id>.json
        path = os.path.join(stateless_handlers.get_root(), user_id, "experiments", experiment_id, "jobs_metadata", job_id + ".json")
        job_meta = stateless_handlers.safe_load_file(path)
        job_meta["result"] = {}
        job_meta["result"]["stats"] = []
        job_meta["result"]["automl_result"] = []
        for key, value in stats.items():
            if "best_" in key:
                job_meta["result"]["automl_result"].append({"metric": key, "value": value})
            else:
                job_meta["result"]["stats"].append({"metric": key, "value": str(value)})

        return Code(200, job_meta, "Job retrieved")

    @staticmethod
    def resume(user_id, experiment_id, job_id, handler_metadata):
        """Resumes a stopped automl job"""
        print("Resuming automl", job_id, file=sys.stderr)

        root = get_root() + f"{user_id}/experiments/{experiment_id}/{job_id}/"
        if not os.path.exists(root):
            os.makedirs(root)

        network = get_handler_type(user_id, experiment_id)
        automl_algorithm = handler_metadata.get("automl_algorithm", "bayesian")
        automl_max_recommendations = handler_metadata.get("automl_max_recommendations", 20)
        automl_delete_intermediate_ckpt = handler_metadata.get("automl_delete_intermediate_ckpt", True)
        automl_R = handler_metadata.get("automl_R", 27)
        automl_nu = handler_metadata.get("automl_nu", 3)
        metric = handler_metadata.get("metric", "map")
        epoch_multiplier = handler_metadata.get("epoch_multiplier", 1)
        automl_add_hyperparameters = handler_metadata.get("automl_add_hyperparameters", "[]")
        automl_remove_hyperparameters = handler_metadata.get("automl_remove_hyperparameters", "[]")
        override_automl_disabled_params = handler_metadata.get("override_automl_disabled_params", False)
        # Call the script
        run_command = f'/venv/bin/python3 automl_start.py --user_id={user_id} --root={root} --automl_job_id={job_id} --network={network} --experiment_id={experiment_id} --resume=True --automl_algorithm={automl_algorithm} --automl_max_recommendations={automl_max_recommendations} --automl_delete_intermediate_ckpt={automl_delete_intermediate_ckpt} --automl_R={automl_R} --automl_nu={automl_nu} --metric={metric} --epoch_multiplier={epoch_multiplier} --automl_add_hyperparameters="{automl_add_hyperparameters}" --automl_remove_hyperparameters="{automl_remove_hyperparameters}" --override_automl_disabled_params={override_automl_disabled_params}'
        jobDriver.create(user_id, job_id, image, run_command, num_gpu=0)
