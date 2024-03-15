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

"""Workflow manager for normal model actions"""
import os

from job_utils.workflow import Workflow, Job, Dependency
from handlers.stateless_handlers import get_handler_spec_root
from handlers.utilities import load_json_spec, JobContext
from handlers.stateless_handlers import get_handler_type


def create_job_context(parent_job_id, action, job_id, handler_id, user_id, kind, specs=None, name=None, description=None):
    """Calls the create job contexts function"""
    network = get_handler_type(user_id, handler_id)
    if not network:
        return []

    job_id = str(job_id)
    assert specs
    # Create a jobcontext
    job_context = JobContext(job_id, parent_job_id, network, action, handler_id, user_id, kind, specs=specs, name=name, description=description)
    return job_context


def on_new_job(job_context):
    """Assigns dependencies for a new job;
    Creates job_context dictionary and enqueues the job to workflow
    """
    deps = []
    deps.append(Dependency(type="parent"))
    deps.append(Dependency(type="specs"))
    deps.append(Dependency(type="model"))
    deps.append(Dependency(type="dataset"))
    if job_context.action not in ["convert", "dataset_convert", "kmeans", "annotation"]:
        deps.append(Dependency(type="gpu"))
    elif job_context.action in ("convert", "gen_trt_engine"):
        spec_json_path = os.path.join(get_handler_spec_root(job_context.user_id, job_context.handler_id), f"{job_context.id}-{job_context.action}-spec.json")
        config = load_json_spec(spec_json_path)
        if config.get("platform"):
            deps.append(Dependency(type="gpu", name=config["platform"]))

    job = {
        'user_id': job_context.user_id,
        'kind': job_context.kind,
        'id': job_context.id,
        'parent_id': job_context.parent_id,
        'priority': 1,
        'action': job_context.action,
        'network': job_context.network,
        'handler_id': job_context.handler_id,
        'created_on': job_context.created_on,
        'last_modified': job_context.last_modified,
        'dependencies': deps,
        'specs': job_context.specs
    }
    j = Job(**job)
    Workflow.enqueue(j)


def on_delete_job(user_id, handler_id, job_id):
    """Dequeue a job"""
    Workflow.dequeue(user_id, handler_id, job_id)
