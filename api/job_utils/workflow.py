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

"""Job Workflow modules"""
import os
import yaml
import threading
import functools
import datetime
import time
import uuid
import glob
from pathlib import Path

from queue import PriorityQueue

from dataclasses import dataclass, field, asdict

from handlers.utilities import read_network_config
from handlers.actions import ACTIONS_TO_FUNCTIONS, AutoMLPipeline
from handlers.stateless_handlers import get_root, get_handler_root, get_handler_job_metadata, update_job_results
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
    created_on: str = field(default=datetime.datetime.now().isoformat())


@dataclass
class Dependency:
    """Base class for representing dependecies"""

    type: str = field(default=None)
    name: str = field(default=None)


@dataclass
class Job(PrioritizedItem, IdedItem):
    """Class for representing jobs"""

    last_modified: str = field(compare=False, default=datetime.datetime.now().isoformat())
    action: str = field(compare=False, default=None)
    dependencies: list = field(compare=False, default=None)
    # More parameters for Job from JobContext
    parent_id: uuid.UUID = field(compare=False, default=uuid.uuid4())
    network: str = field(compare=False, default=None)
    handler_id: uuid.UUID = field(compare=False, default=uuid.uuid4())


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
        action_pipeline = ACTIONS_TO_FUNCTIONS[action_pipeline_name]
        _Actionpipeline = action_pipeline(job_context)
        # Thread this!
        job_run_thread = threading.Thread(target=_Actionpipeline.run, args=())
        job_run_thread.start()
    else:
        # AUTOML Job
        # TODO: At test time, sequentially run it and not as a thread to catch errors
        _AutoMLPipeline = AutoMLPipeline(job_context)
        job_run_thread = threading.Thread(target=_AutoMLPipeline.run, args=())
        job_run_thread.start()
        # AutoMLPipeline(job_context)
    return True


@synchronized
def still_exists(job_to_check):
    """Checks if the the job is yet to be executed/queued or not"""
    filename = os.path.join(get_handler_root(job_to_check.handler_id), "jobs.yaml")
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
        f.write(f"Healthy at {datetime.datetime.now().isoformat()}\n")
        if message:
            f.write(str(message) + "\n")


@synchronized
def read_jobs(yaml_file):
    """Reads a job yaml file and convert to job contexts"""
    jobs = []
    if not os.path.exists(yaml_file):
        return []
    with open(yaml_file, "r", encoding='utf-8') as file:
        jobs_raw = yaml.safe_load(file)
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


@synchronized
def write_jobs(yaml_file, jobs):
    """Writes list of Job objects into the yaml_file"""
    with open(yaml_file, 'w', encoding='utf-8') as file:
        yaml.dump([asdict(i) for i in jobs], file, sort_keys=False)


@synchronized
def scan_for_jobs():
    """Scans for new jobs and queues them if dependencies are met"""
    while True:
        report_healthy("Workflow has waken up", clear=True)
        # Create global queue
        queue = PriorityQueue()
        # Read all jobs.yaml files into one queue
        pattern = get_root() + "**/**/**/jobs.yaml"
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
                    report_healthy(f"Unmet dependency: {dep.type}")
                    all_met = False

            # Update detailed status message in response when appropriate message is available
            pending_reason_message = ''.join(pending_reason_message.rsplit(" and, ", 1))
            metadata = get_handler_job_metadata(job.handler_id, job.id)
            results = metadata.get("result", {})
            if results:
                detailed_status = results.get("detailed_status", {})
                if not detailed_status:
                    results["detailed_status"] = {}
            else:
                metadata["results"] = {}
                results["detailed_status"] = {}
            results["detailed_status"]["message"] = pending_reason_message
            update_job_results(job.handler_id, job.id, results)

            # if all dependencies are met
            if all_met and still_exists(job):
                # execute job
                # check is job is there in the jobs.yaml still
                report_healthy(f"{job.id} with action {job.action}: All dependencies met")
                if execute_job(job):
                    # dequeue job
                    jobs_to_dequeue.append(job)
        for job in jobs_to_dequeue:
            Workflow.dequeue(job.handler_id, job.id)
        report_healthy("Workflow going to sleep")
        time.sleep(15)


class Workflow:
    """
    Workflow is an abstraction that can run on multiple threads. Its use is to be
    able to perform dependency checks and spawn off K8s jobs

    Currently, jobs are packaged inside the ActionPipeline that runs as a thread

    """

    @staticmethod
    def start():
        """Method used to initialize the workflow. Starts a thread if thread is not there from before"""
        # Make sure there is no other Workflow thread
        for thread in threading.enumerate():
            if thread.name == "WorkflowThreadTAO":
                return False
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
        filename = os.path.join(get_handler_root(job.handler_id), "jobs.yaml")
        jobs = read_jobs(filename)
        jobs.append(job)
        write_jobs(filename, jobs)

    @staticmethod
    def dequeue(handler_id, job_id):
        """Method used from outside to remove a job from the workflow"""
        # Simply remove the job from the filename
        # Read all jobs

        filename = os.path.join(get_handler_root(handler_id), "jobs.yaml")
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
            return (time.time() - os.path.getmtime(path)) <= 100
        except:
            return False
