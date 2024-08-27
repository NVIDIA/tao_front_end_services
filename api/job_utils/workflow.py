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

"""Job Workflow modules"""
import os
import sys
import threading
import functools
import time
import uuid
import glob
from pathlib import Path

from queue import PriorityQueue

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from constants import MEDICAL_AUTOML_ARCHITECT, MEDICAL_NETWORK_ARCHITECT
from handlers.utilities import JobContext
from handlers.actions import ACTIONS_TO_FUNCTIONS, AutoMLPipeline
from handlers.stateless_handlers import get_all_pending_jobs, get_handler_root, get_handler_job_metadata, get_jobs_root, update_job_metadata, safe_dump_file, safe_load_file, get_handler_type, get_handler_metadata
from handlers.automl_handler import AutoMLHandler
from utils import read_network_config
from job_utils.dependencies import dependency_type_map, dependency_check_default


def synchronized(wrapped):
    """Decorator function for synchronizing threaded functions"""
    lock = threading.Lock()

    @functools.wraps(wrapped)
    def _wrap(*args, **kwargs):
        with lock:
            return wrapped(*args, **kwargs)
    return _wrap


@dataclass
class IdedItem:
    """Base class for representing id's in uuid"""

    id: uuid.UUID = field(default=uuid.uuid4())


@dataclass(order=True)
class PrioritizedItem:
    """Base class for prioritizing items"""

    priority: int = field(default=1)
    created_on: str = field(default=datetime.now(tz=timezone.utc))


@dataclass
class Dependency:
    """Base class for representing dependecies"""

    type: str = field(default=None)
    name: str = field(default=None)
    num: int = field(default=1)


@dataclass
class Job(PrioritizedItem, IdedItem):
    """Class for representing jobs"""

    last_modified: str = field(compare=False, default=datetime.now(tz=timezone.utc))
    action: str = field(compare=False, default=None)
    dependencies: list = field(compare=False, default=None)
    # More parameters for Job from JobContext
    parent_id: uuid.UUID = field(compare=False, default=uuid.uuid4())
    network: str = field(compare=False, default=None)
    handler_id: uuid.UUID = field(compare=False, default=uuid.uuid4())
    user_id: uuid.UUID = field(compare=False, default=uuid.uuid4())
    org_name: str = field(compare=False, default=None)
    kind: str = field(compare=False, default=None)
    specs: dict = field(compare=False, default=None)
    num_gpu: int = field(compare=False, default=0)
    platform: str = field(compare=False, default=None)


def dependency_check(job_context, dependency):
    """Checks if depencies for the job are met"""
    dependency_check_fn = dependency_type_map.get(dependency.type, dependency_check_default)
    dependency_met = dependency_check_fn(job_context, dependency)
    return dependency_met


def execute_job(job_context):
    """Starts a thread on pipelines present in actions.py"""
    isautoml = False
    for dep in job_context.dependencies:
        if dep.type == "automl":
            isautoml = True
            break

    if not isautoml:
        # Get action, network
        action = job_context.action
        network = job_context.network
        # Get the correct ActionPipeline - build specs, build run command, launch K8s job, monitor status, run post-job steps
        network_config = read_network_config(network)
        action_pipeline_name = network_config["api_params"]["actions_pipe"].get(action, "")
        if network in MEDICAL_NETWORK_ARCHITECT:
            action_pipeline_name = "monai_" + action_pipeline_name
        elif network in MEDICAL_AUTOML_ARCHITECT:
            action_pipeline_name = "medical_automl_" + action_pipeline_name
        action_pipeline = ACTIONS_TO_FUNCTIONS[action_pipeline_name]
        _Actionpipeline = action_pipeline(job_context)
        # Thread this!
        job_run_thread = threading.Thread(target=_Actionpipeline.run, args=(), name=f'tao-job-thread-{job_context.id}')
        job_run_thread.start()
    else:
        # AUTOML Job
        # TODO: At test time, sequentially run it and not as a thread to catch errors
        _AutoMLPipeline = AutoMLPipeline(job_context)
        job_run_thread = threading.Thread(target=_AutoMLPipeline.run, args=(), name=f'tao-job-thread-{job_context.id}')
        job_run_thread.start()
        # AutoMLPipeline(job_context)
    return True


@synchronized
def still_exists(job_to_check):
    """Checks if the the job is yet to be executed/queued or not"""
    filename = os.path.join(get_handler_root(job_to_check.org_name, None, job_to_check.handler_id, None), "jobs.yaml")
    jobs = read_jobs(filename)
    for _, job in enumerate(jobs):
        if job.id == job_to_check.id:
            return True
    return False


@synchronized
def report_healthy(message, clear=False):
    """Writes healthy message with timestamp"""
    path = "/shared/health.txt"
    Path(path).touch()
    mode = "w" if clear else "a"
    with open(path, mode, encoding='utf-8') as f:
        f.write(f"Healthy at {datetime.now().isoformat()}\n")
        if message:
            f.write(str(message) + "\n")


def read_jobs(yaml_file):
    """Reads a job yaml file and convert to job contexts"""
    jobs = []
    if not os.path.exists(yaml_file):
        return []
    jobs_raw = safe_load_file(yaml_file, file_type="yaml")
    if not jobs_raw:
        return []
    # convert yaml to Jobs and put it in the priority queue
    for job in jobs_raw:
        j = Job(**job)
        j.dependencies = []
        for d in job.get('dependencies'):
            j.dependencies.append(Dependency(**d))
        jobs.append(j)

    return jobs


def write_jobs(yaml_file, jobs):
    """Writes list of Job objects into the yaml_file"""
    os.makedirs(os.path.dirname(yaml_file), exist_ok=True)
    data = [asdict(i) for i in jobs]
    safe_dump_file(yaml_file, data, file_type="yaml")


@synchronized
def scan_for_jobs():
    """Scans for new jobs and queues them if dependencies are met"""
    while True:
        report_healthy("Workflow has waken up", clear=True)
        # Create global queue
        queue = PriorityQueue()
        # Read all jobs.yaml files into one queue
        pattern = os.environ.get("TAO_ROOT", "/shared/orgs/") + "**/**/**/jobs.yaml"  # jobs.yaml still part of PVC
        job_files = glob.glob(pattern)
        for job_file in job_files:
            for j in read_jobs(job_file):
                queue.put(j)
        len_q = len(queue.queue)
        report_healthy(f"Found {len_q} pending jobs")
        # Parse to dequeue
        jobs_to_dequeue = []
        list.sort(queue.queue)
        for i in range(len(queue.queue)):
            # check dependencies
            job = queue.queue[i]
            report_healthy(f"{job.id} with action {job.action}: Checking dependencies")
            report_healthy(f"Total dependencies: {len(job.dependencies)}")
            all_met = True
            pending_reason_message = ""
            for dep in job.dependencies:
                dependency_met, message = dependency_check(job, dep)
                if not dependency_met:
                    pending_reason_message += f"{message} and, "
                    report_healthy(f"Unmet dependency: {dep.type} {pending_reason_message}")
                    all_met = False
                if "Parent job " in message and "errored out" in message:
                    jobs_to_dequeue.append(job)
                    break

            # Update detailed status message in response when appropriate message is available
            pending_reason_message = ''.join(pending_reason_message.rsplit(" and, ", 1))
            metadata = get_handler_job_metadata(job.id)
            results = metadata.get("result", {})
            if results:
                detailed_status = results.get("detailed_status", {})
                if not detailed_status:
                    results["detailed_status"] = {}
            else:
                metadata["results"] = {}
                results["detailed_status"] = {}
            results["detailed_status"]["message"] = pending_reason_message
            update_job_metadata(job.handler_id, job.id, metadata_key="result", data=results, kind=job.kind + "s")

            # if all dependencies are met
            if all_met and still_exists(job):
                # execute job
                # check is job is there in the jobs.yaml still
                report_healthy(f"{job.id} with action {job.action}: All dependencies met")
                if execute_job(job):
                    # dequeue job
                    jobs_to_dequeue.append(job)
        for job in jobs_to_dequeue:
            Workflow.dequeue(job.org_name, job.handler_id, job.id, job.kind + "s")
        report_healthy("Workflow going to sleep")
        time.sleep(15)


class Workflow:
    """
    Workflow is an abstraction that can run on multiple threads. Its use is to be
    able to perform dependency checks and spawn off K8s jobs

    Currently, jobs are packaged inside the ActionPipeline that runs as a thread.

    On application restart, it will check if there were any pending job monitoring threads that were interrupted and restart them.

    """

    @staticmethod
    def restart_threads():
        """Method used to restart unfinished job monitoring threads"""
        jobs = get_all_pending_jobs()
        automl_brain_restarted = False
        for job_dict in jobs:
            parent_job_id = job_dict.get("parent_id")
            action = job_dict.get("action")
            job_id = job_dict.get("id")
            name = job_dict.get("name")
            org_name = job_dict.get("org_name")
            specs = job_dict.get("specs")
            if 'experiment_id' in job_dict:
                kind = 'experiment'
                handler_id = job_dict['experiment_id']
            elif 'dataset_id' in job_dict:
                kind = 'dataset'
                handler_id = job_dict['dataset_id']
            elif 'workspace_id' in job_dict:
                kind = 'workspace'
                handler_id = job_dict['workspace_id']
            else:
                print(f"Warning: Job {job_id} monitoring unable to be restarted, cannot determine handler kind", file=sys.stderr)
                continue

            handler_metadata = get_handler_metadata(handler_id, kind)
            if not handler_metadata:
                print(f"Warning: Job {job_id} monitoring unable to be restarted, cannot find {kind} {handler_id}", file=sys.stderr)
                continue
            network = get_handler_type(handler_metadata)
            user_id = handler_metadata.get("user_id")
            if not org_name:
                if "org_name" not in handler_metadata:
                    print(f"Warning: Job {job_id} monitoring unable to be restarted, cannot determine org name", file=sys.stderr)
                    continue
                org_name = handler_metadata.get("org_name")
            num_gpu = handler_metadata.get("num_gpu", -1)
            isautoml = handler_metadata.get("automl_settings", {}).get("automl_enabled", False)

            job_context = JobContext(job_id, parent_job_id, network, action, handler_id, user_id, org_name, kind, name=name, num_gpu=num_gpu, specs=specs)
            # If job has yet to be executed, skip monitoring
            if still_exists(job_context):
                continue
            print(f"Found unfinished monitoring thread for job {job_id}, restarting job thread now", file=sys.stderr)

            if not isautoml:
                # Get the correct ActionPipeline and monitor status
                network_config = read_network_config(network)
                action_pipeline_name = network_config["api_params"]["actions_pipe"].get(action, "")
                if network in MEDICAL_NETWORK_ARCHITECT:
                    action_pipeline_name = "monai_" + action_pipeline_name
                elif network in MEDICAL_AUTOML_ARCHITECT:
                    action_pipeline_name = "medical_automl_" + action_pipeline_name
                action_pipeline = ACTIONS_TO_FUNCTIONS[action_pipeline_name]

                _Actionpipeline = action_pipeline(job_context)
                # Thread this!
                job_run_thread = threading.Thread(target=_Actionpipeline.monitor_job, args=(), name=f'tao-monitor-job-thread-{job_context.id}')
                job_run_thread.start()
                print(f"Monitoring thread for job {job_id} restarted", file=sys.stderr)
            else:
                # Restart AutoML job monitoring threads
                controller_path = os.path.join(get_jobs_root(user_id, org_name), job_id, "controller.json")
                recommendations = safe_load_file(controller_path)
                handler_metadata = get_handler_metadata(handler_id, kind + "s")
                if handler_metadata:
                    if not automl_brain_restarted:
                        AutoMLHandler.resume(user_id, org_name, handler_id, job_id, handler_metadata, name=name)
                        automl_brain_restarted = True
                    for recommendation in recommendations:
                        if recommendation.get("status", None) in ("pending", "running", "started") and recommendation.get("id", None):
                            rec_id = recommendation["id"]
                            deps = [Dependency(type="automl", name=str(rec_id))]
                            automl_context = JobContext(job_id, parent_job_id, network, action, handler_id, user_id, org_name, kind, name=name, num_gpu=num_gpu)
                            automl_context.dependencies = deps
                            _AutoMLPipeline = AutoMLPipeline(automl_context)
                            job_run_thread = threading.Thread(target=_AutoMLPipeline.monitor_job, args=(), name=f'tao-monitor-job-thread-{automl_context.id}')
                            job_run_thread.start()
                            print(f"Restarted AutoML monitoring thread for job {job_id} and recommendation {rec_id}", file=sys.stderr)

    @staticmethod
    def start():
        """Method used to initialize the workflow. Starts a thread if thread is not there from before"""
        # Make sure there is no other Workflow thread
        for thread in threading.enumerate():
            if thread.name == "WorkflowThreadTAO":
                return False
        # Restart unfinished monitoring threads, if any
        Workflow.restart_threads()
        t = threading.Thread(target=scan_for_jobs)
        t.name = 'WorkflowThreadTAO'
        t.daemon = True
        t.start()
        return True

    @staticmethod
    def enqueue(job):
        """Method used from outside to put a job into the workflow"""
        # Simply prints the job inside the filename
        # Called only by on_new_job()
        filename = os.path.join(get_handler_root(job.org_name, job.kind + "s", job.handler_id, None), "jobs.yaml")
        jobs = read_jobs(filename)
        jobs.append(job)
        write_jobs(filename, jobs)

    @staticmethod
    def dequeue(org_name, handler_id, job_id, kind=""):
        """Method used from outside to remove a job from the workflow"""
        # Simply remove the job from the filename
        # Read all jobs

        filename = os.path.join(get_handler_root(org_name, kind, handler_id, None), "jobs.yaml")
        jobs = read_jobs(filename)
        # Delete job_id's job from the list
        for idx, job in enumerate(jobs):
            if job.id == job_id:
                del jobs[idx]
        # Write it back as is
        write_jobs(filename, jobs)

    @staticmethod
    def healthy():
        """Method used to see if the workflow thread is running"""
        try:
            path = "/shared/health.txt"
            # current time and last health modified time must be less than 100 seconds
            last_updated_time = time.time() - os.path.getmtime(path)
            if last_updated_time > 3600:
                print(f"Health file was updated {last_updated_time} ago which is > 3600", file=sys.stderr)
            return last_updated_time <= 3600
        except:
            return False
