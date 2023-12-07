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

"""Workflow manager for normal model actions"""
import os
import json

from job_utils.workflow import Workflow, Job, Dependency
from handlers.stateless_handlers import get_handler_spec_root
from handlers.utilities import load_json_spec


def on_new_job(job_contexts):
    """Assigns dependencies for a new job;
    Creates job_context dictionary and enqueues the job to workflow
    """
    for job_context in job_contexts:
        deps = []
        deps.append(Dependency(type="parent"))
        if job_context.specs:
            spec_json_path = os.path.join(get_handler_spec_root(job_context.handler_id), f"{job_context.action}.json")
            with open(spec_json_path, "w", encoding='utf-8') as f:
                request_json_string = json.dumps(job_context.specs, indent=4)
                f.write(request_json_string)
        deps.append(Dependency(type="specs"))
        deps.append(Dependency(type="model"))
        deps.append(Dependency(type="dataset"))
        if job_context.action not in ["convert", "dataset_convert", "kmeans"]:
            deps.append(Dependency(type="gpu"))
        elif job_context.action in ("convert", "gen_trt_engine"):
            spec_json_path = os.path.join(get_handler_spec_root(job_context.handler_id), f"{job_context.action}.json")
            config = load_json_spec(spec_json_path)
            if config.get("platform"):
                deps.append(Dependency(type="gpu", name=config["platform"]))

        job = {
            'id': job_context.id,
            'parent_id': job_context.parent_id,
            'priority': 1,
            'action': job_context.action,
            'network': job_context.network,
            'handler_id': job_context.handler_id,
            'created_on': job_context.created_on,
            'last_modified': job_context.last_modified,
            'dependencies': deps
        }
        j = Job(**job)
        Workflow.enqueue(j)


def on_delete_job(handler_id, job_id):
    """Dequeue a job"""
    Workflow.dequeue(handler_id, job_id)
