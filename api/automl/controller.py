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
import time
import sys
from copy import deepcopy
from automl.utils import Recommendation, ResumeRecommendation, JobStates, report_healthy
from handlers.utilities import StatusParser, _ITER_MODELS, NO_VAL_METRICS_DURING_TRAINING_NETWORKS, MISSING_EPOCH_FORMAT_NETWORKS, network_metric_mapping, read_network_config, get_total_epochs, archive_job_results, generate_job_tar_stats
from handlers.stateless_handlers import update_job_status, get_handler_metadata, write_handler_metadata, safe_load_file, safe_dump_file, get_root
from job_utils.automl_job_utils import on_new_automl_job, on_delete_automl_job, on_cancel_automl_job
import uuid
import shutil
import glob
import traceback

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

    def __init__(self, root, network, brain, automl_context, max_recommendations, delete_intermediate_ckpt, metric, automl_algorithm):
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

        self.root = root
        self.automl_experiments_root = root.replace(get_root(), get_root(ngc_runner_fetch=True))
        self.handler_root = "/".join(self.automl_experiments_root.split("/")[0:-2])
        self.network = network
        self.completed_recommendations = 0
        self.max_recommendations = int(max_recommendations)
        self.delete_intermediate_ckpt = bool(delete_intermediate_ckpt)
        self.automl_algorithm = automl_algorithm

        self.metric = metric
        if self.automl_algorithm in ("hyperband", "h") and self.network in NO_VAL_METRICS_DURING_TRAINING_NETWORKS:
            self.metric_key = "loss"
            self.metric = "loss"
        elif self.metric == "kpi":
            self.metric_key = network_metric_mapping[self.network]
        else:
            self.metric_key = self.metric

        self.brain.reverse_sort = True
        self.min_max = max
        if self.metric == "loss" or self.metric_key in ("loss", "evaluation_cost "):
            self.brain.reverse_sort = False
            self.min_max = min

        self.best_epoch_number = {}
        self.brain_epoch_number = -2
        self.best_model_copied = False
        self.ckpt_path = {}

        self.old_bracket = 0
        self.hyperband_cancel_condition_seen = False

        self.eta = "Will be updated after completing one epoch"
        self.remaining_epochs_in_experiment = float("inf")
        self.average_time_per_epoch = float("inf")

        self.automl_context = automl_context
        self.on_new_automl_job = lambda jc: on_new_automl_job(self.automl_context, jc)

        with open(f"{self.root}/automl_metadata.json", 'a', encoding='utf-8'):  # Creating Controller json if it doesn't exist
            pass

    def start(self):
        """Starts the automl controller"""
        try:
            report_healthy(self.root + "/controller.log", "Starting", clear=False)
            self._execute_loop()
            status = "Error"
            if self.best_model_copied:
                status = "Done"
            archive_job_results(self.handler_root, self.automl_context.id)
            generate_job_tar_stats(self.automl_context.user_id, self.automl_context.handler_id, self.automl_context.id, self.handler_root)
            update_job_status(self.automl_context.user_id, self.automl_context.handler_id, self.automl_context.id, status=status)

            on_delete_automl_job(self.automl_context.user_id, self.automl_context.handler_id, self.automl_context.id)
        except Exception:
            print(f"AutoMLpipeline loop for network {self.network} failed due to exception {traceback.format_exc()}", file=sys.stderr)
            update_job_status(self.automl_context.user_id, self.automl_context.handler_id, self.automl_context.id, status="Error")

    def save_state(self):
        """Save the self.recommendations into a controller.json"""
        recs_dict = [ele.__dict__ for ele in self.recommendations]
        file_path = self.root + "/controller.json"
        safe_dump_file(file_path, recs_dict)

    @staticmethod
    def load_state(root, network, brain, automl_context, max_recommendations, delete_intermediate_ckpt, metric, automl_algorithm):
        """Loads a Controller object from pre-existing root"""
        ctrl = Controller(root, network, brain, automl_context, max_recommendations, delete_intermediate_ckpt, metric, automl_algorithm)
        ctrl.recommendations = []
        # Restore the recommendations
        file_path = root + "/controller.json"
        recs_dict = safe_load_file(file_path)

        for rec_dict in recs_dict:
            rec = Recommendation(rec_dict["id"], rec_dict["specs"])
            rec.update_result(rec_dict["result"])
            rec.update_status(rec_dict["status"])
            rec.assign_job_id(rec_dict["job_id"])
            ctrl.recommendations.append(rec)

        # Handle temp_rec
        # temp_rec is a recommendation that started, but never ended
        # Usually, if the controller is stopped before a recommendation is done, it might have to be started / resumed again
        file_path = root + "/current_rec.json"
        temp_rec = safe_load_file(file_path)
        ctrl.recommendations[temp_rec].update_status(JobStates.success)
        ctrl.save_state()
        # ctrl.on_new_automl_job(ctrl.recommendations[temp_rec])

        return ctrl

    def _execute_loop(self):
        """A loop that does the 3 things in order
        1.See if any new recommendation is up to execute
        2.Reads results of newly done experiments
        3.Writes AutoML status into a file which can be shown to the end user
        """
        update_job_status(self.automl_context.user_id, self.automl_context.handler_id, self.automl_context.id, status="Running")
        while True:
            automl_status_file = self.root + "/controller.json"
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
                            expt_root = os.path.join(self.automl_experiments_root, "experiment_" + str(rec.id))
                            self.get_best_checkpoint_path(expt_root, rec)
                            self.delete_not_best_model_checkpoints(expt_root, rec, True)
                            handler_metadata = get_handler_metadata(self.automl_context.user_id, self.automl_context.handler_id)
                            handler_metadata["checkpoint_epoch_number"][f"best_model_{self.automl_context.id}"] = self.best_epoch_number[rec.id]
                            handler_metadata["checkpoint_epoch_number"][f"latest_model_{self.automl_context.id}"] = self.best_epoch_number[rec.id]
                            write_handler_metadata(self.automl_context.user_id, self.automl_context.handler_id, handler_metadata)

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
            print("Recommendation gotten", file=sys.stderr)
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
                rec = Recommendation(new_id, spec)
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
                def remove_files(expt_path):
                    if os.path.exists(expt_path):
                        expt_file_name = glob.glob(expt_path + "/**/*.txt", recursive=True) + glob.glob(expt_path + "/**/*.json", recursive=True) + glob.glob(expt_path + "/**/*event*", recursive=True) + glob.glob(expt_path + "/**/*lightning_logs*", recursive=True)
                        for file_name in expt_file_name:
                            if os.path.isfile(file_name):
                                os.remove(file_name)
                expt_name = "experiment_" + str(rec_id)
                remove_files(os.path.join(self.automl_experiments_root, expt_name))
                remove_files(os.path.join(self.root, expt_name))

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
            expt_root = os.path.join(self.automl_experiments_root, expt_name)

            # If rec already changed to Success, no need to check
            if rec.status in [JobStates.success, JobStates.failure]:
                if self.delete_intermediate_ckpt:
                    self.delete_checkpoint_files(expt_root, rec)
                    # Remove the checkpoints from not best model
                    brain_file_path = self.root + "/brain.json"
                    if os.path.exists(brain_file_path):
                        brain_dict = safe_load_file(brain_file_path)
                        if self.automl_algorithm in ("bayesian", "b") or self.old_bracket != brain_dict.get("bracket", 0):
                            flag = self.delete_not_best_model_checkpoints(expt_root, rec, flag)
                continue

            status_file = os.path.join(self.root, expt_name, "status.json")
            status_parser = StatusParser(status_file, self.network, expt_root)
            new_results = status_parser.update_results()
            self.calculate_eta(new_results)

            validation_map_processed = False
            # Force termination of the case for hyperband training
            if self.automl_algorithm in ("hyperband", "h"):
                brain_file_path = self.root + "/brain.json"
                if os.path.exists(brain_file_path):
                    brain_dict = safe_load_file(brain_file_path)
                    # if the experiment is in the last set of bracket, do not cancel job.
                    for result_key in new_results.keys():
                        if self.hyperband_cancel_condition_seen or result_key in ("epoch", "cur_iter"):
                            if not isinstance(new_results.get(result_key, None), type(None)):
                                self.brain_epoch_number = float(brain_dict.get("epoch_number", float('inf')))
                                if len(brain_dict.get("ni", [float('-inf')])[str(brain_dict.get("bracket", 0))]) != (brain_dict.get("sh_iter", float('inf')) + 1):
                                    if self.hyperband_cancel_condition_seen or new_results.get(result_key) > self.brain_epoch_number:
                                        self.hyperband_cancel_condition_seen = True
                                        # Cancel the current running job and change the job state to success
                                        validation_map, self.best_epoch_number[rec.id], _ = status_parser.read_metric(results=new_results, metric=self.metric, automl_algorithm=self.automl_algorithm, automl_root=self.root, brain_epoch_number=self.brain_epoch_number)
                                        if validation_map != 0.0:
                                            rec.update_status(JobStates.success)
                                            validation_map_processed = True
                                            self.hyperband_cancel_condition_seen = False
                                            on_cancel_automl_job(rec.job_id)
                                            self.delete_checkpoint_files(expt_root, rec)

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
                        self.brain.num_epochs_per_experiment = get_total_epochs(self.automl_context, self.handler_root, automl=True, automl_experiment_id=rec.id)
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
            if old_status != status:
                rec.update_status(status)
                self.save_state()
                if status == JobStates.success:
                    if os.getenv("NGC_RUNNER") == "True":
                        on_cancel_automl_job(rec.job_id)
                    container_log_file = f"{self.automl_experiments_root}/experiment_{rec.id}/log.txt"
                    if os.path.exists(container_log_file):
                        with open(container_log_file, "a", encoding='utf-8') as f:
                            f.write("\nEOF\n")

            if rec.status in [JobStates.success, JobStates.failure] and self.delete_intermediate_ckpt:
                # Retain the latest checkpoint and remove others in experiment folder
                self.delete_checkpoint_files(expt_root, rec)

        if self.automl_algorithm in ("hyperband", "h"):
            if os.path.exists(brain_file_path):
                self.old_bracket = brain_dict.get("bracket", 0)

    def calculate_eta(self, new_results):
        """Calculate estimated time remaining for automl job"""
        global time_per_epoch
        global time_per_epoch_counter
        for result_key in new_results.keys():
            if result_key in ("epoch", "cur_iter") and new_results.get(result_key):
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
                    self.eta /= 60
                    self.eta = round(self.eta, 2)

                elif self.automl_algorithm in ("hyperband", "h"):
                    current_sh_allowed_epochs = self.brain.ri[self.brain.bracket][self.brain.sh_iter] * self.brain.epoch_multiplier
                    current_sh_remaining_epochs = (self.brain.ni[self.brain.bracket][self.brain.sh_iter] - self.brain.expt_iter) * current_sh_allowed_epochs
                    if current_epoch < current_sh_allowed_epochs:
                        current_sh_remaining_epochs += ((current_sh_allowed_epochs - current_epoch))
                    future_sh_epochs = 0.0
                    for bracket in range(self.brain.bracket, len(self.brain.ni)):
                        for remaining_sh in range(self.brain.sh_iter + 1, len(self.brain.ri[bracket])):
                            current_sh_epochs = self.brain.ri[bracket][remaining_sh]
                            if remaining_sh != 0:
                                current_sh_epochs -= self.brain.ri[bracket][remaining_sh - 1]
                            future_sh_epochs += self.brain.ni[bracket][remaining_sh] * current_sh_epochs * self.brain.epoch_multiplier
                    self.remaining_epochs_in_experiment = current_sh_remaining_epochs + future_sh_epochs
                    self.eta = self.remaining_epochs_in_experiment * self.average_time_per_epoch
                    self.eta /= 60
                    self.eta = round(self.eta, 2)

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

        eta_msg_suffix = ""
        if type(self.eta) is float:
            eta_msg_suffix = " minutes remaining approximately"
        result_dict["Estimated time for automl completion"] = str(self.eta) + eta_msg_suffix
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

        for rec in self.recommendations:
            print("Recommendation in function find_best_model", rec, file=sys.stderr)
            job_name = rec.job_id
            if not job_name:
                continue
            expt_folder = os.path.join(self.automl_experiments_root, "experiment_" + str(rec.id))
            print("Experiment folder", expt_folder, file=sys.stderr)
            print("Checkpoints in find best_model", glob.glob(expt_folder + "/**/*.tlt", recursive=True) + glob.glob(expt_folder + "/**/*.hdf5", recursive=True) + glob.glob(expt_folder + "/**/*.pth", recursive=True), file=sys.stderr)
            if os.path.exists(expt_folder) and (rec.status == JobStates.success and rec.result == best_mAP) and (glob.glob(expt_folder + "/**/*.tlt", recursive=True) + glob.glob(expt_folder + "/**/*.hdf5", recursive=True) + glob.glob(expt_folder + "/**/*.pth", recursive=True)):
                self.best_model_copied = True
                api_params = read_network_config(self.network)["api_params"]
                spec_path = os.path.join(self.automl_experiments_root, f"recommendation_{rec.id}.{api_params['spec_backend']}")
                print("Best recommendation Spec path", spec_path, os.path.exists(spec_path), file=sys.stderr)
                best_model_folder = os.path.join(self.automl_experiments_root, "best_model")

                shutil.move(expt_folder, best_model_folder)
                shutil.copy(spec_path, best_model_folder)
                shutil.copy(os.path.join(self.root, "controller.json"), best_model_folder)
                break

    def get_best_checkpoint_path(self, path, recommendation):
        """Assign the checkpoint with the best metric value for supported models; for others call the 'find latest checkpoint method'"""
        self.ckpt_path[path] = {}
        if self.network in MISSING_EPOCH_FORMAT_NETWORKS:
            format_epoch_number = str(self.best_epoch_number[recommendation.id])
        else:
            format_epoch_number = f"{self.best_epoch_number[recommendation.id]:03}"
        recommendation.best_epoch_number = format_epoch_number
        print("Best epoch number", recommendation.best_epoch_number, path, file=sys.stderr)
        find_trained_tlt = glob.glob(f"{path}/*{format_epoch_number}.tlt") + glob.glob(f"{path}/train/*{format_epoch_number}.tlt") + glob.glob(f"{path}/weights/*{format_epoch_number}.tlt")
        find_trained_hdf5 = glob.glob(f"{path}/*{format_epoch_number}.hdf5") + glob.glob(f"{path}/train/*{format_epoch_number}.hdf5") + glob.glob(f"{path}/weights/*{format_epoch_number}.hdf5")
        find_trained_pth = glob.glob(f"{path}/*{format_epoch_number}.pth") + glob.glob(f"{path}/train/*{format_epoch_number}.pth") + glob.glob(f"{path}/weights/*{format_epoch_number}.pth")
        find_trained_ckzip = glob.glob(f"{path}/*{format_epoch_number}.ckzip") + glob.glob(f"{path}/train/*{format_epoch_number}.ckzip") + glob.glob(f"{path}/weights/*{format_epoch_number}.ckzip")
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
        trained_files = glob.glob(path + "/**/*.tlt", recursive=True) + glob.glob(path + "/**/*.hdf5", recursive=True) + glob.glob(path + "/**/*.pth", recursive=True) + glob.glob(path + "/**/*.ckzip", recursive=True) + glob.glob(path + "/**/*.resume", recursive=True) + glob.glob(path + "/**/*lightning_logs*", recursive=True)
        print("Available checkpoints in delete_checkpoint_files function", trained_files, file=sys.stderr)
        self.get_best_checkpoint_path(path, rec)
        print("self.ckpt_path in delete_checkpoint_files function", self.ckpt_path, file=sys.stderr)
        for files in trained_files:
            if files not in self.ckpt_path[path].values():
                if os.path.isfile(files):
                    print("Removing files in delete_checkpoint_files function", files, file=sys.stderr)
                    os.remove(files)
                elif ".tlt" in files and self.network == "unet":
                    print("Removing folder in delete_checkpoint_files function", files, file=sys.stderr)
                    shutil.rmtree(files)

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
            trained_files = glob.glob(path + "/**/*.tlt", recursive=True) + glob.glob(path + "/**/*.hdf5", recursive=True) + glob.glob(path + "/**/*.pth", recursive=True) + glob.glob(path + "/**/*.ckzip", recursive=True) + glob.glob(path + "/**/*.resume", recursive=True) + glob.glob(path + "/**/*event*", recursive=True) + glob.glob(path + "/**/*lightning_logs*", recursive=True)
            print("Available checkpoints in delete_not_best_model_checkpoints function", trained_files, file=sys.stderr)
            for files in trained_files:
                if os.path.isfile(files):
                    print("Removing files in delete_not_best_model_checkpoints function", files, file=sys.stderr)
                    os.remove(files)
                elif ".tlt" in files and self.network == "unet":
                    print("Removing folder in delete_not_best_model_checkpoints function", files, file=sys.stderr)
                    shutil.rmtree(files)
        else:
            flag = True
        return flag
