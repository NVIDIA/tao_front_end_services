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

"""Util functions for AutoML jobs"""
import sys
import threading
from handlers.stateless_handlers import get_public_models, get_handler_metadata
from handlers.utilities import download_ptm
from job_utils.workflow import Workflow, Job, Dependency
from job_utils import executor as jobDriver


def get_ptm_id_from_recommendation(specs, network_arch):
    """Dynamicaly obtain ptm id based on the backbone and num_layers chosen"""
    backbone_arch = specs.get("backbone", "resnet")
    num_layers = specs.get("num_layers", 34)
    match_string = f":{backbone_arch}{num_layers}"
    ptm_id = None
    for model_id in get_public_models():
        metadata = get_handler_metadata(model_id)
        ngc_path_exists = metadata.get("ngc_path", None) is not None
        correct_arch = metadata.get("network_arch", "") == network_arch
        ptm_string_match = match_string in metadata.get("ngc_path", "")
        if ngc_path_exists and correct_arch and ptm_string_match:
            ptm_id = metadata.get("id", None)

    return ptm_id


def on_new_automl_job(automl_context, recommendation):
    """Assigns dependencies for the automl recommendation job;
    Creates job_context dictionary and enqueues the job to workflow
    """
    # Controller interacts with this
    # Download NGC pretrained model as a background process
    ptm_id = get_ptm_id_from_recommendation(recommendation.specs, automl_context.network)
    # Background process to download this PTM
    if ptm_id:
        job_run_thread = threading.Thread(target=download_ptm, args=(ptm_id,))
        job_run_thread.start()

    # automl_context is same as JobContext that was created for AutoML job
    recommendation_id = recommendation.id
    deps = []
    deps.append(Dependency(type="automl", name=str(recommendation_id)))
    if ptm_id:
        deps.append(Dependency(type="automl_ptm", name=str(ptm_id)))
    deps.append(Dependency(type="dataset"))
    deps.append(Dependency(type="gpu"))
    deps.append(Dependency(type="automl_specs"))

    job = {
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


def on_delete_automl_job(handler_id, job_id):
    """Dequeue the automl job"""
    # AutoML handler stop would handle this
    # automl_context is same as JobContext that was created for AutoML job
    Workflow.dequeue(handler_id, job_id)


def on_cancel_automl_job(job_id):
    """Delete the job from k8's jobs"""
    jobDriver.delete(job_id)
