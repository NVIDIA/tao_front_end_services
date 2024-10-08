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

"""Utils for AutoML"""
import os
import math
import glob
import datetime
import time
from kubernetes import client, config

from utils import safe_load_file
from handlers.stateless_handlers import get_jobs_root


def fix_input_dimension(dimension_value, factor=32):
    """Return dimension as a multiple of factor"""
    if int(dimension_value) % factor == 0:
        return dimension_value
    return (int(dimension_value / factor) + 1) * factor


def clamp_value(value, v_min, v_max):
    """Clamps value within the given range"""
    if value >= v_max:
        epsilon = v_max / 10
        if epsilon == 0.0:
            epsilon = 0.0000001
        value = v_max - epsilon
    if value <= v_min:
        epsilon = v_min / 10
        if epsilon == 0.0:
            epsilon = 0.0000001
        value = v_min + epsilon
    return value


def get_valid_range(parameter_config, parent_params):
    """Compute the clamp range for the given parameter"""
    v_min = float(parameter_config.get("valid_min"))
    v_max = float(parameter_config.get("valid_max"))
    default_value = float(parameter_config.get("default_value"))
    if math.isinf(v_min):
        v_min = default_value
    if math.isinf(v_max):
        v_max = default_value

    dependent_on_param = parameter_config.get("depends_on", None)
    if type(dependent_on_param) is str:
        dependent_on_param_op = dependent_on_param.split(" ")[0]
        dependent_on_param_name = dependent_on_param.split(" ")[1]
        if dependent_on_param_name in parent_params.keys():
            limit_value = parent_params[dependent_on_param_name]
        else:
            limit_value = default_value

        epsilon = 0.000001
        if limit_value == epsilon:
            epsilon /= 10

        if dependent_on_param_op == ">":
            v_min = limit_value + epsilon
        elif dependent_on_param_op == ">=":
            v_min = limit_value
        elif dependent_on_param_op == "<":
            v_max = limit_value - epsilon
        elif dependent_on_param_op == "<=":
            v_max = limit_value

    return v_min, v_max


def report_healthy(path, message, clear=False):
    """Write health message to the provided file"""
    mode = "w" if clear else "a"
    with open(path, mode, encoding='utf-8') as f:
        f.write(f"Healthy at {datetime.datetime.now().isoformat()}\n")
        if message:
            f.write(str(message) + "\n")


def wait_for_job_completion(job_id):
    """Check if the provided job_id is actively running and wait until completion"""
    config.load_incluster_config()
    while True:
        dgx_active_jobs = []
        if os.getenv("BACKEND", "") in ("BCP", "NVCF"):
            custom_api = client.CustomObjectsApi()
            crd_group = 'dgx-job-manager.nvidia.io'
            crd_version = 'v1alpha1'
            crd_plural = 'dgxjobs'
            # List all instances of the Custom Resource across all namespaces
            custom_resources = custom_api.list_cluster_custom_object(crd_group, crd_version, crd_plural)
            dgx_active_jobs = [dgx_cr["spec"].get("name") for dgx_cr in custom_resources['items']]

        ret = client.BatchV1Api().list_job_for_all_namespaces()
        active_jobs = dgx_active_jobs + [job.metadata.name for job in ret.items]
        active_jobs = list(set(active_jobs))
        if job_id not in active_jobs:
            break
        time.sleep(5)


def delete_lingering_checkpoints(epoch_number, path):
    """Delete checkpoints which are present even after job deletion"""
    trained_files = glob.glob(path + "/**/*.tlt", recursive=True) + glob.glob(path + "/**/*.hdf5", recursive=True) + glob.glob(path + "/**/*.pth", recursive=True) + glob.glob(path + "/**/*.ckzip", recursive=True) + glob.glob(path + "/**/*lightning_logs*", recursive=True)
    for file_name in trained_files:
        if os.path.isfile(file_name):
            if not (f"{epoch_number}.tlt" in file_name or f"{epoch_number}.hdf5" in file_name or f"{epoch_number}.pth" in file_name or f"{epoch_number}.ckzip" in file_name):
                os.remove(file_name)


def merge_normal_and_automl_job_meta(user_id, org_name, job_id, job_meta):
    """Merge jobs_metadata.json and automl_metadata.json"""
    root = os.path.join(get_jobs_root(user_id, org_name), job_id)
    json_file = os.path.join(root, "automl_metadata.json")

    stats = safe_load_file(json_file)
    if not stats:
        stats["message"] = "Stats will be updated in a few seconds"

    if "result" not in job_meta:
        job_meta["result"] = {}
    if "stats" not in job_meta["result"]:
        job_meta["result"]["stats"] = []
    if "automl_result" not in job_meta["result"]:
        job_meta["result"]["automl_result"] = []
    for key, value in stats.items():
        if "best_" in key:
            job_meta["result"]["automl_result"].append({"metric": key, "value": value})
        else:
            job_meta["result"]["stats"].append({"metric": key, "value": str(value)})


class Recommendation:
    """Recommendation class for AutoML recommendations"""

    def __init__(self, identifier, specs, metric):
        """Initialize the Recommendation class
        Args:
            identity: the id of the recommendation
            specs: the specs/config of the recommendation
        """
        assert type(identifier) is int
        self.id = identifier

        assert type(specs) is dict
        self.specs = specs

        self.job_id = None
        self.status = JobStates.pending
        self.result = 0.0
        self.best_epoch_number = ""
        self.metric = metric

    def items(self):
        """Returns specs.items"""
        return self.specs.items()

    def get(self, key):
        """Returns value of requested key in the spec"""
        return self.specs.get(key, None)

    def assign_job_id(self, job_id):
        """Associates provided job id to the class objects job id"""
        assert type(job_id) is str
        self.job_id = job_id

    def update_result(self, result):
        """Update the result value"""
        result = float(result)
        assert type(result) is float
        self.result = result

    def update_status(self, status):
        """Update the status value"""
        assert type(status) is str
        self.status = status

    def __repr__(self):
        """Constructs a dictionary with the class members and returns them"""
        return f"id: {self.id}\njob_id: {self.job_id}\nresult: {self.result}\nstatus: {self.status}"


class ResumeRecommendation:
    """Recommendation class for Hyperband resume experiments"""

    def __init__(self, identity, specs):
        """Initialize the ResumeRecommendation class

        Args:
            identity: the id of the recommendation
            specs: the specs/config of the recommendation
        """
        self.id = identity
        self.specs = specs


class JobStates():
    """Various states of an automl job"""

    pending = "pending"
    started = "started"
    running = "running"
    success = "success"
    failure = "failure"
    error = "error"  # alias for failure
    done = "done"  # alias for success
    canceled = "canceled"
    canceling = "canceling"
