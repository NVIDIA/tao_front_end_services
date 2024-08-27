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

"""AutoML controller modules"""
import os
import sys
import glob
import time
import uuid
import shutil
import traceback
from copy import deepcopy
from datetime import timedelta

from automl.utils import Recommendation, ResumeRecommendation, JobStates, report_healthy
from constants import _ITER_MODELS, NO_VAL_METRICS_DURING_TRAINING_NETWORKS, NETWORK_METRIC_MAPPING, MISSING_EPOCH_FORMAT_NETWORKS
from dgx_controller import overwrite_job_logs_from_bcp
from handlers.cloud_storage import CloudStorage
from handlers.utilities import StatusParser, get_total_epochs, get_file_list_from_cloud_storage, filter_files, format_epoch
from handlers.stateless_handlers import update_job_status, get_handler_metadata, write_handler_metadata, get_handler_job_metadata, update_job_metadata
from job_utils.automl_job_utils import on_new_automl_job, on_delete_automl_job, on_cancel_automl_job
from utils import read_network_config, safe_load_file, safe_dump_file

time_per_epoch = 0
time_per_epoch_counter = 0


class Controller:
    """
    Abstractly, just a collection of threads and a switch to start and stop them

    - start(): Start all threads needed to run AutoML
    - stop(): Stop all threads started by start()
    - generate_recommendations(): Runs the automl algorithm to generate and analyze recommendations
    - read_results(): Listens to experiments
    - write_results(): Routinely updates a controller_data.json to help Handlers
    """

    def __init__(self, root, network, brain, automl_context, max_recommendations, delete_intermediate_ckpt, metric, automl_algorithm, decrypted_workspace_metadata):
        """Initialize the Automl Controller class

        Args:
            root: handler root
            network: model name
            brain: Bayesian/Hyperband class object
            automl_context: job context with regards to automl
            max_recommendations: max_recommendation parameter value (for Bayesian)
            delete_intermediate_ckpt: boolean value to delete/not-delete checkpoints which don't correspond to the best model
            metric: metric name which will be used to choose best models
            automl_algorithm: automl algorithm name
        """
        self.brain = brain

        self.recommendations = []
        self.automl_context = automl_context

        self.root = root
        self.network = network
        self.checkpoint_delimiter = ""
        if self.network in MISSING_EPOCH_FORMAT_NETWORKS:
            self.checkpoint_delimiter = "_"
        self.completed_recommendations = 0
        self.max_recommendations = int(max_recommendations)
        self.delete_intermediate_ckpt = bool(delete_intermediate_ckpt)
        self.automl_algorithm = automl_algorithm
        self.handler_metadata = get_handler_metadata(self.automl_context.handler_id, "experiments")
        self.decrypted_workspace_metadata = decrypted_workspace_metadata
        self.metric = metric
        if self.automl_algorithm in ("hyperband", "h") and self.network in NO_VAL_METRICS_DURING_TRAINING_NETWORKS:
            self.metric_key = "loss"
            self.metric = "loss"
        elif self.metric == "kpi":
            self.metric_key = NETWORK_METRIC_MAPPING[self.network]
        else:
            self.metric_key = self.metric

        self.brain.reverse_sort = True
        self.min_max = max
        if self.metric == "loss" or self.metric_key in ("loss", "evaluation_cost "):
            self.brain.reverse_sort = False
            self.min_max = min

        self.total_epochs = 0
        self.first_epoch_number = -1
        self.best_epoch_number = {}
        self.brain_epoch_number = -2
        self.best_model_copied = False
        self.ckpt_path = {}

        self.old_bracket = 0
        self.hyperband_cancel_condition_seen = False

        self.eta = "Will be updated after completing one epoch"
        self.remaining_epochs_in_experiment = float("inf")
        self.average_time_per_epoch = float("inf")

        self.on_new_automl_job = lambda jc: on_new_automl_job(self.automl_context, jc)

        cloud_type = self.decrypted_workspace_metadata.get("cloud_type")
        cloud_specific_details = self.decrypted_workspace_metadata.get("cloud_specific_details")
        cloud_region = cloud_specific_details.get("cloud_region")
        cloud_bucket_name = cloud_specific_details.get("cloud_bucket_name")
        if cloud_type == "aws":
            self.cs_instance = CloudStorage("aws", cloud_bucket_name, region=cloud_region, access_key=cloud_specific_details.get("access_key"), secret_key=cloud_specific_details.get("secret_key"))
        elif cloud_type == "azure":
            self.cs_instance = CloudStorage("azure", cloud_bucket_name, access_key=cloud_specific_details.get("account_name"), secret_key=cloud_specific_details.get("access_key"))

        with open(f"{self.root}/automl_metadata.json", 'a', encoding='utf-8'):  # Creating Controller json if it doesn't exist
            pass

    def update_status_message(self):
        """Update job detailed status to indicate the best model was not copied"""
        if not self.best_model_copied:
            metadata = get_handler_job_metadata(self.automl_context.id)
            results = metadata.get("result", {})
            if results:
                detailed_status = results.get("detailed_status", {})
                if not detailed_status:
                    results["detailed_status"] = {}
            else:
                metadata["results"] = {}
                results["detailed_status"] = {}
            results["detailed_status"]["message"] = f"Checkpoint file doesn't exist in best model folder /results/{self.automl_context.id}"
            update_job_metadata(self.automl_context.handler_id, self.automl_context.id, metadata_key="result", data=results, kind="experiments")

    def cancel_recommendation_jobs(self):
        """Cleanup recommendation jobs"""
        for rec in self.recommendations:
            job_name = rec.job_id
            print("\nDeleting", job_name, file=sys.stderr)
            if not job_name:
                continue
            if not os.getenv("CI_PROJECT_DIR", None):
                on_cancel_automl_job(rec.job_id)
        on_delete_automl_job(self.automl_context.org_name, self.automl_context.handler_id, self.automl_context.id, "experiments")

    def start(self):
        """Starts the automl controller"""
        try:
            report_healthy(self.root + "/controller.log", "Starting", clear=False)
            self._execute_loop()
            status = "Error"
            if self.best_model_copied:
                status = "Done"
            self.update_status_message()
            update_job_status(self.automl_context.handler_id, self.automl_context.id, status=status, kind="experiments")
            self.cancel_recommendation_jobs()

        except Exception:
            self.update_status_message()
            self.cancel_recommendation_jobs()
            print(f"AutoMLpipeline loop for network {self.network} failed due to exception {traceback.format_exc()}", file=sys.stderr)
            update_job_status(self.automl_context.handler_id, self.automl_context.id, status="Error", kind="experiments")

    def save_state(self):
        """Save the self.recommendations into a controller.json"""
        recs_dict = [ele.__dict__ for ele in self.recommendations]
        file_path = self.root + "/controller.json"
        metadata = get_handler_job_metadata(self.automl_context.id)
        current_status = metadata.get("status", "")
        if current_status not in ("canceled", "canceling"):
            safe_dump_file(file_path, recs_dict)

    @staticmethod
    def load_state(root, network, brain, automl_context, max_recommendations, delete_intermediate_ckpt, metric, automl_algorithm, decrypted_workspace_metadata):
        """Loads a Controller object from pre-existing root"""
        ctrl = Controller(root, network, brain, automl_context, max_recommendations, delete_intermediate_ckpt, metric, automl_algorithm, decrypted_workspace_metadata)
        ctrl.recommendations = []
        # Restore the recommendations
        file_path = root + "/controller.json"
        recs_dict = safe_load_file(file_path)

        for rec_dict in recs_dict:
            rec = Recommendation(rec_dict["id"], rec_dict["specs"], ctrl.metric_key)
            rec.update_result(rec_dict["result"])
            rec.update_status(rec_dict["status"])
            rec.assign_job_id(rec_dict["job_id"])
            ctrl.recommendations.append(rec)
            ctrl.best_epoch_number[rec_dict["id"]] = rec_dict.get("best_epoch_number") if rec_dict.get("best_epoch_number") else 0

        # Handle temp_rec
        # temp_rec is a recommendation that started, but never ended
        # Usually, if the controller is stopped before a recommendation is done, it might have to be started / resumed again
        file_path = root + "/current_rec.json"
        temp_rec = safe_load_file(file_path)
        # if ctrl.recommendations[temp_rec].status != JobStates.canceled:
        #     ctrl.recommendations[temp_rec].update_status(JobStates.success)
        ctrl.save_state()
        if ctrl.recommendations[temp_rec].status == JobStates.canceled:
            print("Resuming stopped automl sub-experiment", temp_rec, file=sys.stderr)
            if ctrl.automl_algorithm == "hyperband":
                ctrl.brain.track_id = temp_rec
            ctrl.on_new_automl_job(ctrl.recommendations[temp_rec])

        return ctrl

    def _execute_loop(self):
        """A loop that does the 3 things in order
        1.See if any new recommendation is up to execute
        2.Reads results of newly done experiments
        3.Writes AutoML status into a file which can be shown to the end user
        """
        update_job_status(self.automl_context.handler_id, self.automl_context.id, status="Running", kind="experiments")
        while True:
            automl_status_file = self.root + "/controller.json"
            metadata = get_handler_job_metadata(self.automl_context.id)
            current_status = metadata.get("status", "")
            if current_status in ("canceled", "canceling"):
                return
            if os.path.exists(automl_status_file):
                automl_status = safe_load_file(automl_status_file)
                self.completed_recommendations = len(automl_status)
                if (self.completed_recommendations == self.max_recommendations and automl_status[self.max_recommendations - 1]['status'] in ('success', 'failure') and self.automl_algorithm in ("bayesian", "b")) or (self.automl_algorithm in ("hyperband", "h") and self.brain.done()):
                    report_healthy(self.root + "/controller.log", "Stopping", clear=False)
                    # Find best model based on mAP
                    print("Finding best model", file=sys.stderr)
                    self.find_best_model()
                    print("best_model_copied result", self.best_model_copied, file=sys.stderr)

                    if self.best_model_copied:
                        # Delete final extra checkpoints after finish training
                        for rec in self.recommendations:
                            expt_root = os.path.join("/results", rec.job_id)
                            self.get_best_checkpoint_path(expt_root, rec)
                            self.delete_not_best_model_checkpoints(expt_root, rec, True)
                            self.handler_metadata["checkpoint_epoch_number"][f"best_model_{self.automl_context.id}"] = self.best_epoch_number[rec.id]
                            self.handler_metadata["checkpoint_epoch_number"][f"latest_model_{self.automl_context.id}"] = self.best_epoch_number[rec.id]
                            write_handler_metadata(self.automl_context.handler_id, self.handler_metadata, "experiments")

                    self.eta = 0.0
                    self.remaining_epochs_in_experiment = 0.0
                    self.write_results()
                    return

            self.run_experiments()
            self.read_results()
            self.write_results()
            if not os.getenv("CI_PROJECT_DIR", None):
                time.sleep(10)

    def run_experiments(self):
        """Generate recommendation from brain
        if a new job is requested, add it to self.recommendations and execute it (add it to workflow)
        if a resume is requested, add the relevant recommendation to the workflow
        """
        if self.automl_algorithm in ("bayesian", "b") and len(self.recommendations) == self.max_recommendations:
            return
        history = deepcopy(self.recommendations)
        recommended_specs = self.brain.generate_recommendations(history)
        assert len(recommended_specs) in [0, 1], "At most one recommendation"
        for spec in recommended_specs:
            print(f"Recommendation recieved for {self.network}", file=sys.stderr)
            if type(spec) is dict:
                # Save brain state and update current recommendation
                self.hyperband_cancel_condition_seen = False
                self.brain.save_state()
                # update temp_rec
                new_id = len(self.recommendations)
                self.best_epoch_number[new_id] = 0
                file_path = self.root + "/current_rec.json"
                safe_dump_file(file_path, new_id)

                # Run new recommendation
                rec = Recommendation(new_id, spec, self.metric_key)
                job_id = str(uuid.uuid4())  # Assign job_id for this recommendation
                rec.assign_job_id(job_id)
                self.recommendations.append(rec)
                self.save_state()
                self.on_new_automl_job(rec)
                report_healthy(self.root + "/controller.log", "Job started", clear=False)

            elif type(spec) is ResumeRecommendation:
                self.hyperband_cancel_condition_seen = False
                rec_id = spec.id
                self.best_epoch_number[rec_id] = 0
                # Save brain state and update current recommendation
                self.brain.save_state()
                # update temp_rec
                file_path = self.root + "/current_rec.json"
                safe_dump_file(file_path, rec_id)
                assert self.recommendations[rec_id].id == rec_id  # Make sure the self.recommendations[rec_id] indeed has 'id' field = rec_id
                self.recommendations[rec_id].specs = spec.specs.copy()
                self.recommendations[rec_id].update_status(JobStates.pending)

                # Remove previous files (except checkpoints) from experiment folder.
                def remove_files(local_expt_path, cloud_expt_path):
                    expt_file_name = get_file_list_from_cloud_storage(self.decrypted_workspace_metadata, cloud_expt_path)
                    regex_pattern = r'.*(?:lightning_logs|events).*$|.*\.(json|txt)$'
                    expt_file_name = filter_files(expt_file_name, regex_pattern)
                    for file_name in expt_file_name:
                        self.cs_instance.delete_file(file_name)
                    if os.path.exists(local_expt_path):
                        expt_file_name = glob.glob(local_expt_path + "/**/*.txt", recursive=True) + glob.glob(local_expt_path + "/**/*.json", recursive=True)
                        print("Removing log and status json files", expt_file_name, file=sys.stderr)
                        for file_name in expt_file_name:
                            if os.path.isfile(file_name):
                                os.remove(file_name)
                expt_name = "experiment_" + str(rec_id)
                remove_files(os.path.join(self.root, expt_name), os.path.join("/results", self.recommendations[rec_id].job_id))

                self.save_state()
                self.on_new_automl_job(self.recommendations[rec_id])
                report_healthy(self.root + "/controller.log", "Job started", clear=False)

    def read_results(self):
        """Update results for each recommendation"""
        flag = False
        for rec in self.recommendations:
            old_status = rec.status

            job_name = rec.job_id
            if not job_name:
                continue

            expt_name = "experiment_" + str(rec.id)
            local_expt_root = os.path.join(self.root, expt_name)
            cloud_expt_root = os.path.join("/results", rec.job_id)

            # If rec already changed to Success, no need to check
            if rec.status in [JobStates.success, JobStates.failure]:
                if self.delete_intermediate_ckpt:
                    self.delete_checkpoint_files(cloud_expt_root, rec)
                    # Remove the checkpoints from not best model
                    brain_file_path = self.root + "/brain.json"
                    if os.path.exists(brain_file_path):
                        brain_dict = safe_load_file(brain_file_path)
                        if self.automl_algorithm in ("bayesian", "b") or self.old_bracket != brain_dict.get("bracket", 0):
                            flag = self.delete_not_best_model_checkpoints(cloud_expt_root, rec, flag)
                continue

            status_file = os.path.join(self.root, expt_name, "status.json")
            status_parser = StatusParser(status_file, self.network, local_expt_root, self.first_epoch_number)

            new_results = status_parser.update_results(automl=True)
            self.calculate_eta(new_results)
            new_results = status_parser.update_results(self.total_epochs, self.eta, self.total_epochs - self.remaining_epochs_in_experiment, True)
            if status_parser.first_epoch_number != -1:
                self.first_epoch_number = status_parser.first_epoch_number
            update_job_metadata(self.automl_context.handler_id, self.automl_context.id, metadata_key="result", data=new_results, kind="experiments")

            validation_map_processed = False
            # Force termination of the case for hyperband training
            if self.automl_algorithm in ("hyperband", "h"):
                brain_file_path = self.root + "/brain.json"
                if os.path.exists(brain_file_path):
                    brain_dict = safe_load_file(brain_file_path)
                    # if the experiment is in the last set of bracket, do not cancel job.
                    for result_key in new_results.keys():
                        if self.hyperband_cancel_condition_seen or result_key in ("automl_experiment_epoch", "cur_iter"):
                            if not isinstance(new_results.get(result_key, None), type(None)):
                                self.brain_epoch_number = float(brain_dict.get("epoch_number", float('inf')))
                                if len(brain_dict.get("ni", [float('-inf')])[str(brain_dict.get("bracket", 0))]) != (brain_dict.get("sh_iter", float('inf')) + 1):
                                    if self.hyperband_cancel_condition_seen or new_results.get(result_key) > self.brain_epoch_number or (self.network == "pointpillars" and (new_results.get(result_key) + 1 >= self.brain_epoch_number)):
                                        self.hyperband_cancel_condition_seen = True
                                        # Cancel the current running job and change the job state to success
                                        validation_map, self.best_epoch_number[rec.id], _ = status_parser.read_metric(results=new_results, metric=self.metric, automl_algorithm=self.automl_algorithm, automl_root=self.root, brain_epoch_number=self.brain_epoch_number)
                                        if validation_map != 0.0:
                                            format_epoch_number = format_epoch(self.network, self.best_epoch_number[rec.id])
                                            trained_files = get_file_list_from_cloud_storage(self.decrypted_workspace_metadata, cloud_expt_root)
                                            regex_pattern = fr'^(?!.*lightning_logs).*{self.checkpoint_delimiter}{format_epoch_number}\.(pth|tlt|hdf5)$'
                                            trained_files = filter_files(trained_files, regex_pattern)
                                            if trained_files:
                                                rec.update_status(JobStates.success)
                                                validation_map_processed = True
                                                self.hyperband_cancel_condition_seen = False
                                                on_cancel_automl_job(rec.job_id)
                                                self.delete_checkpoint_files(cloud_expt_root, rec)

            # Status is read from the status.json and not from K8s
            # status.json needs to be reliable
            status = ""
            if rec.status == JobStates.success:
                status = JobStates.success
            elif new_results.get("detailed_status"):
                status = new_results["detailed_status"].get("status", JobStates.pending).lower()
            if not status:
                status = JobStates.pending
            if status in [JobStates.success, JobStates.failure]:
                if not validation_map_processed:
                    brain_epoch_number = self.brain_epoch_number
                    if self.automl_algorithm in ("bayesian", "b"):
                        self.brain.num_epochs_per_experiment = get_total_epochs(self.automl_context, os.path.dirname(self.root), automl=True, automl_experiment_id=rec.id)
                        brain_epoch_number = self.brain.num_epochs_per_experiment
                    validation_map, self.best_epoch_number[rec.id], _ = status_parser.read_metric(results=new_results, metric=self.metric, automl_algorithm=self.automl_algorithm, automl_root=self.root, brain_epoch_number=brain_epoch_number)
                if status == JobStates.failure:
                    if self.brain.reverse_sort:
                        validation_map = 1e-7
                    else:
                        validation_map = float('inf')
                if validation_map != 0.0:
                    rec.update_result(validation_map)
                self.save_state()
                on_cancel_automl_job(rec.job_id)
            if old_status != status:
                rec.update_status(status)
                self.save_state()
                if status == JobStates.success:
                    container_log_file = f"{self.root}/experiment_{rec.id}/log.txt"
                    if os.getenv("BACKEND") in ("BCP", "NVCF"):
                        overwrite_job_logs_from_bcp(container_log_file, rec.job_id)
                    if os.path.exists(container_log_file):
                        with open(container_log_file, "a", encoding='utf-8') as f:
                            f.write("\nEOF\n")

            if rec.status in [JobStates.success, JobStates.failure] and self.delete_intermediate_ckpt:
                # Retain the latest checkpoint and remove others in experiment folder
                self.delete_checkpoint_files(cloud_expt_root, rec)

        if self.automl_algorithm in ("hyperband", "h"):
            if os.path.exists(brain_file_path):
                self.old_bracket = brain_dict.get("bracket", 0)

    def calculate_eta(self, new_results):
        """Calculate estimated time remaining for automl job"""
        global time_per_epoch
        global time_per_epoch_counter
        self.total_epochs = 0
        if self.automl_algorithm in ("bayesian", "b"):
            self.total_epochs = self.max_recommendations * self.brain.num_epochs_per_experiment
        elif self.automl_algorithm in ("hyperband", "h"):
            for key in self.brain.ni:
                experiments = self.brain.ni[key]
                epochs = self.brain.ri[key]
                for i, num_epochs in enumerate(epochs):
                    if i == 0:
                        self.total_epochs += experiments[i] * num_epochs
                    else:
                        self.total_epochs += experiments[i] * (epochs[i] - epochs[i - 1])
            self.total_epochs *= self.brain.epoch_multiplier

        for result_key in new_results.keys():
            if result_key in ("automl_experiment_epoch", "cur_iter") and new_results.get(result_key):
                current_epoch = new_results.get(result_key)
                if result_key == "cur_iter":
                    time_per_key = "time_per_iter"
                else:
                    time_per_key = "time_per_epoch"
                time_per_epoch_string = new_results.get(time_per_key, "0:0:0.0")
                if time_per_epoch_string:
                    format_time_per_epoch = time.strptime(time_per_epoch_string.split(".")[0], '%H:%M:%S')
                    time_per_epoch += (format_time_per_epoch.tm_hour * 60 * 60 + format_time_per_epoch.tm_min * 60 + format_time_per_epoch.tm_sec)
                else:
                    time_per_epoch = 0
                time_per_epoch_counter += 1
                self.average_time_per_epoch = time_per_epoch / time_per_epoch_counter

                if self.automl_algorithm in ("bayesian", "b"):
                    remaining_epochs = self.brain.num_epochs_per_experiment - current_epoch
                    self.remaining_epochs_in_experiment = remaining_epochs + (self.max_recommendations - self.completed_recommendations) * (self.brain.num_epochs_per_experiment)
                    self.eta = self.remaining_epochs_in_experiment * self.average_time_per_epoch

                elif self.automl_algorithm in ("hyperband", "h"):
                    current_sh_allowed_epochs = self.brain.ri[self.brain.bracket][self.brain.sh_iter] * self.brain.epoch_multiplier
                    if self.brain.bracket > 0:
                        current_sh_allowed_epochs = (self.brain.ri[self.brain.bracket][self.brain.sh_iter] - self.brain.ri[self.brain.bracket][self.brain.sh_iter - 1]) * self.brain.epoch_multiplier
                    if self.brain.bracket == 0 and self.brain.bracket == 0:
                        completed_epochs = self.brain.expt_iter * current_sh_allowed_epochs
                    else:
                        completed_epochs = (self.brain.bracket - 1) * current_sh_allowed_epochs + current_epoch

                    for bracket in range(0, self.brain.bracket + 1):
                        local_sh_iter = len(self.brain.ni[bracket])
                        if bracket == self.brain.bracket:
                            local_sh_iter = self.brain.sh_iter
                        for sh in range(0, local_sh_iter):
                            completed_epochs += self.brain.ni[bracket][sh] * self.brain.ri[bracket][sh]

                    self.remaining_epochs_in_experiment = self.total_epochs - completed_epochs
                    self.eta = self.remaining_epochs_in_experiment * self.average_time_per_epoch

        if self.remaining_epochs_in_experiment == float("inf") or self.remaining_epochs_in_experiment == float("-inf"):
            self.remaining_epochs_in_experiment = self.total_epochs

    def write_results(self):
        """Update stats value and write to automl_metadata.json"""
        automl_brain_metadata_json = os.path.join(self.root, "automl_metadata.json")
        # Best mAP seen till now
        result_dict = {}
        try:
            if self.recommendations[-1].result == 0.0:
                result_dict[f"best_{self.metric_key}"] = self.min_max(self.recommendations[:-1], key=lambda rec: rec.result).result
            else:
                result_dict[f"best_{self.metric_key}"] = self.min_max(self.recommendations, key=lambda rec: rec.result).result
        except:
            result_dict[f"best_{self.metric_key}"] = 0.0

        if type(self.eta) is float:
            self.eta = str(timedelta(seconds=self.eta))
        result_dict["Estimated time for automl completion"] = str(self.eta)
        result_dict["Current experiment number"] = len(self.recommendations)

        if self.network in _ITER_MODELS:
            result_dict["Number of iters yet to start"] = self.remaining_epochs_in_experiment
            result_dict["Time per iter in seconds"] = round(self.average_time_per_epoch, 2)
        else:
            result_dict["Number of epochs yet to start"] = self.remaining_epochs_in_experiment
            result_dict["Time per epoch in seconds"] = round(self.average_time_per_epoch, 2)

        safe_dump_file(automl_brain_metadata_json, result_dict)

    def find_best_model(self):
        """Find best model based on metric value chosen and move those artifacts to best_model folder"""
        print("Finding best recommendation config", file=sys.stderr)
        try:
            best_mAP = self.min_max(self.recommendations, key=lambda rec: rec.result).result
        except:
            best_mAP = 0.0
            return

        print("Best metric value", best_mAP, file=sys.stderr)
        for rec in self.recommendations:
            print("\nRecommendation in function find_best_model", rec, file=sys.stderr)
            job_name = rec.job_id
            if not job_name:
                continue
            expt_folder = os.path.join("/results", rec.job_id)
            checkpoint_files = get_file_list_from_cloud_storage(self.decrypted_workspace_metadata, expt_folder)
            regex_pattern = r'^(?!.*lightning_logs).*\.(pth|tlt|hdf5)$'
            checkpoint_files = filter_files(checkpoint_files, regex_pattern)
            print("Experiment folder", expt_folder, file=sys.stderr)
            print("Checkpoints in find best_model", checkpoint_files, file=sys.stderr)

            if checkpoint_files and (rec.status == JobStates.success and rec.result == best_mAP):
                api_params = read_network_config(self.network)["api_params"]
                spec_path = f"recommendation_{rec.id}.{api_params['spec_backend']}"
                log_path = os.path.join(f"experiment_{rec.id}", "log.txt")
                print("Best recommendation Spec path", spec_path, file=sys.stderr)
                print("Best recommendation Log path", log_path, file=sys.stderr)
                local_best_model_folder = os.path.join(self.root, "best_model")
                cloud_best_model_folder = f"/results/{self.automl_context.id}"
                print("cloud_best_model_folder", cloud_best_model_folder, file=sys.stderr)
                best_spec_path = os.path.join(self.root, spec_path)
                best_log_path = os.path.join(self.root, log_path)
                os.makedirs(local_best_model_folder)

                self.cs_instance.move_folder(expt_folder[1:], cloud_best_model_folder)
                self.cs_instance.upload_file(best_spec_path, os.path.join(cloud_best_model_folder, spec_path))
                shutil.copy(best_spec_path, local_best_model_folder)
                shutil.copy(best_log_path, local_best_model_folder)
                find_trained_tlt, find_trained_hdf5, find_trained_pth, _ = self.get_checkpoint_paths_matching_epoch_number(cloud_best_model_folder, rec.id)
                if find_trained_tlt or find_trained_hdf5 or find_trained_pth:
                    self.best_model_copied = True
                else:
                    print("Best model checkpoints couldn't be moved", file=sys.stderr)
                break

    def get_checkpoint_paths_matching_epoch_number(self, path, rec_id):
        """Get checkpoints from cloud_path and filter based on epoch number"""
        checkpoint_files = get_file_list_from_cloud_storage(self.decrypted_workspace_metadata, path)
        format_epoch_number = format_epoch(self.network, self.best_epoch_number[rec_id])
        find_trained_tlt = filter_files(checkpoint_files, regex_pattern=fr'.*{self.checkpoint_delimiter}{format_epoch_number}\.tlt$')
        find_trained_hdf5 = filter_files(checkpoint_files, regex_pattern=fr'.*{self.checkpoint_delimiter}{format_epoch_number}\.hdf5$')
        find_trained_pth = filter_files(checkpoint_files, regex_pattern=fr'.*{self.checkpoint_delimiter}{format_epoch_number}\.pth$')
        find_trained_ckzip = filter_files(checkpoint_files, regex_pattern=fr'.*{self.checkpoint_delimiter}{format_epoch_number}\.ckzip$')
        return find_trained_tlt, find_trained_hdf5, find_trained_pth, find_trained_ckzip

    def get_best_checkpoint_path(self, path, recommendation):
        """Assign the checkpoint with the best metric value for supported models; for others call the 'find latest checkpoint method'"""
        self.ckpt_path[path] = {}
        format_epoch_number = format_epoch(self.network, self.best_epoch_number[recommendation.id])
        recommendation.best_epoch_number = format_epoch_number
        print("Best epoch number", recommendation.best_epoch_number, path, file=sys.stderr)
        find_trained_tlt, find_trained_hdf5, find_trained_pth, find_trained_ckzip = self.get_checkpoint_paths_matching_epoch_number(path, recommendation.id)
        if find_trained_tlt:
            self.ckpt_path[path]["tlt"] = find_trained_tlt[0]
        if find_trained_hdf5:
            self.ckpt_path[path]["hdf5"] = find_trained_hdf5[0]
        if find_trained_pth:
            self.ckpt_path[path]["pth"] = find_trained_pth[0]
        if find_trained_ckzip:
            self.ckpt_path[path]["ckzip"] = find_trained_ckzip[0]

    def delete_checkpoint_files(self, path, rec):
        """Remove the extra checkpoints generated after the on_cancel_automl_job"""
        if not os.getenv("CI_PROJECT_DIR", None):
            time.sleep(30)  # Mounted paths can take time to reflect files generated on remote locally
        trained_files = get_file_list_from_cloud_storage(self.decrypted_workspace_metadata, path)
        regex_pattern = r'.*\.(tlt|hdf5|pth|ckzip|resume|lightning_logs)$'
        trained_files = filter_files(trained_files, regex_pattern)
        print("Available checkpoints in delete_checkpoint_files function", trained_files, file=sys.stderr)
        self.get_best_checkpoint_path(path, rec)
        print("self.ckpt_path in delete_checkpoint_files function", self.ckpt_path, file=sys.stderr)
        for files in trained_files:
            if files not in self.ckpt_path[path].values():
                if self.cs_instance.is_file(files):
                    print("Removing files in delete_checkpoint_files function", files, file=sys.stderr)
                    self.cs_instance.delete_file(files)
                elif ".tlt" in files and self.network == "unet":
                    print("Removing folder in delete_checkpoint_files function", files, file=sys.stderr)
                    self.cs_instance.delete_folder(files[1:])

    def delete_not_best_model_checkpoints(self, path, rec, flag):
        """Remove the checkpoints which don't correspond to the best result"""
        try:
            if self.recommendations[-1].result == 0.0:
                best_mAP = self.min_max(self.recommendations[:-1], key=lambda rec: rec.result).result
            else:
                best_mAP = self.min_max(self.recommendations, key=lambda rec: rec.result).result
        except:
            best_mAP = 0.0

        print("delete_not_best_model_checkpoints function arguments", path, rec, flag, file=sys.stderr)
        if rec.result != best_mAP or bool(flag):
            trained_files = get_file_list_from_cloud_storage(self.decrypted_workspace_metadata, path)
            regex_pattern = r'.*(?:lightning_logs|events).*$|.*\.(tlt|hdf5|pth|ckzip|resume)$'
            trained_files = filter_files(trained_files, regex_pattern)
            print("Available checkpoints in delete_not_best_model_checkpoints function", trained_files, file=sys.stderr)
            for files in trained_files:
                if self.cs_instance.is_file(files):
                    print("Removing files in delete_not_best_model_checkpoints function", files, file=sys.stderr)
                    self.cs_instance.delete_file(files)
                elif ".tlt" in files and self.network == "unet":
                    print("Removing folder in delete_not_best_model_checkpoints function", files, file=sys.stderr)
                    self.cs_instance.delete_folder(files[1:])
        else:
            flag = True
        return flag
