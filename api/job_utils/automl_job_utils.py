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

"""Util functions for AutoML jobs"""
import sys
from handlers.stateless_handlers import get_public_experiments
from job_utils.workflow import Workflow, Job, Dependency
from job_utils import executor as jobDriver


def get_base_experiment_id_from_recommendation(specs, network_arch):
    """Dynamicaly obtain base_experiment id based on the backbone and num_layers chosen"""
    #  TODO : generalize this for all models
    backbone_arch = specs.get("backbone", "resnet")
    num_layers = specs.get("num_layers", 34)
    match_string = f":{backbone_arch}{num_layers}"
    base_experiment_id = None
    for metadata in get_public_experiments():
        ngc_path_exists = metadata.get("ngc_path", None) is not None
        correct_arch = metadata.get("network_arch", "") == network_arch
        base_experiment_string_match = match_string in metadata.get("ngc_path", "")
        if ngc_path_exists and correct_arch and base_experiment_string_match:
            base_experiment_id = metadata.get("id", None)

    return base_experiment_id


def on_new_automl_job(automl_context, recommendation):
    """Assigns dependencies for the automl recommendation job;
    Creates job_context dictionary and enqueues the job to workflow
    """
    # Controller interacts with this
    # TODO: uncomment below block after fixing get_base_experiment_id_from_recommendation
    # # Download NGC pretrained model as a background process
    # base_experiment_id = get_base_experiment_id_from_recommendation(recommendation.specs, automl_context.network)
    # # Background process to download this base_experiment
    # if base_experiment_id:
    #     ptm_download_thread = threading.Thread(target=download_base_experiment, args=(base_experiment_id,))
    #     ptm_download_thread.start()

    # automl_context is same as JobContext that was created for AutoML job
    recommendation_id = recommendation.id
    deps = []
    deps.append(Dependency(type="automl", name=str(recommendation_id)))
    deps.append(Dependency(type="specs"))
    deps.append(Dependency(type="model"))
    deps.append(Dependency(type="dataset"))
    deps.append(Dependency(type="gpu"))

    job = {
        'user_id': automl_context.user_id,
        'org_name': automl_context.org_name,
        'num_gpu': automl_context.num_gpu,
        'platform': automl_context.platform,
        'kind': "experiment",
        'id': automl_context.id,
        'parent_id': None,
        'priority': 2,
        'action': "train",
        'network': automl_context.network,
        'handler_id': automl_context.handler_id,
        'created_on': automl_context.created_on,
        'last_modified': automl_context.last_modified,
        'dependencies': deps
    }
    j = Job(**job)
    Workflow.enqueue(j)
    print("Recommendation submitted to workflow", file=sys.stderr)


def on_delete_automl_job(org_name, handler_id, job_id, kind=""):
    """Dequeue the automl job"""
    # AutoML handler stop would handle this
    # automl_context is same as JobContext that was created for AutoML job
    Workflow.dequeue(org_name, handler_id, job_id, kind)


def on_cancel_automl_job(job_id):
    """Delete the job from k8's jobs"""
    jobDriver.delete(job_id)
