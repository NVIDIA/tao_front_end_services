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

"""Job chaining modules"""
import copy

from handlers.utilities import JobContext
from handlers.utilities import read_network_config
from handlers.stateless_handlers import get_handler_type, get_handler_metadata_with_jobs


class ChainingRules:
    """Class for defining rules of chaining jobs together"""

    def __init__(self, chainable, chained_only):
        """Intialize ChainingRules class
        Args:
            chainable: defines chaining rules
            chained_only: actions that fail without a parent job ID
        """
        self._chainable = chainable
        self._chained_only = chained_only

    def chainable(self, child, parent):
        """Defines chaining runes"""
        assert child in list(self._chainable.keys()), "Action not part of Chainer pipeline"
        return parent in self._chainable[child]

    def chained_only(self, action):
        """Returns actions that can run only with a parent job(chaining)"""
        return action in self._chained_only


# Actions chaining rules are defined here
# Read as evaluate can be chained after ["train",..."export"]
_cvaction_rules = {"train": [],
                   "evaluate": ["train", "prune", "retrain", "export", "gen_trt_engine", "trtexec"],
                   "prune": ["train", "retrain"],
                   "inference": ["train", "prune", "retrain", "export", "gen_trt_engine", "trtexec"],
                   "inference_seq": ["train", "prune", "retrain", "export"],
                   "inference_trt": ["train", "prune", "retrain", "export"],
                   "retrain": ["train", "prune"],
                   "export": ["train", "prune", "retrain"],
                   "calibration_tensorfile": ["train", "prune", "retrain"],
                   "gen_trt_engine": ["export"],
                   "trtexec": ["export"],
                   "confmat": ["train", "prune", "retrain"]}
_cvaction_chainedonly = ["prune", "retrain", "export", "gen_trt_engine", "trtexec", "calibration_tensorfile"]
CVAction = ChainingRules(_cvaction_rules, _cvaction_chainedonly)

# OD Dataset chaining rules => Basically says that chaining does not matter
# NOTE: convert writes into tfrecords directory
ODAction = ChainingRules({"convert": ["augment"],
                          "convert_and_index": ["augment"],
                          "convert_efficientdet_tf1": ["augment"],
                          "convert_efficientdet_tf2": ["augment"],
                          "kmeans": [],
                          "augment": []}, [])

_dsaction_rules = {"generate": [],
                   "convert": [],
                   "validate": [],
                   "analyze": []}
DSAction = ChainingRules(_dsaction_rules, [])


CHAINING_RULES_TO_FUNCTIONS = {"cvaction": CVAction,
                               "dsaction": DSAction,
                               "odaction": ODAction}


def infer_action_from_job(handler_id, job_id):
    """Takes handler, job_id (UUID / str) and returns action corresponding to that jobID"""
    job_id = str(job_id)
    action = ""
    all_jobs = get_handler_metadata_with_jobs(handler_id)["jobs"]
    for job in all_jobs:
        if job["id"] == job_id:
            action = job["action"]
            break
    return action


def _create_job_contexts(parent_job_id, parent_action, actions, job_ids, network, chaining_rules, handler_id):
    """Create job contexts for the job_id's provided"""
    job_contexts = []
    for idx, jid in enumerate(job_ids):

        job_id = str(jid)
        action = actions[idx]

        # Create a jobconext
        job_context = JobContext(job_id, None, network, action, handler_id)
        job_contexts.append(job_context)

    completed_tasks_master = []
    # See if parent_job is given
    if parent_job_id:
        completed_tasks_master = [(parent_job_id, parent_action)]

    # Run actions one-by-one
    for idx, action in enumerate(actions):

        # Create a jobconext
        job_context = job_contexts[idx]
        job_id = job_context.id

        completed_tasks_itr = copy.deepcopy(completed_tasks_master)

        # Check for a proper parent job
        for par_job in reversed(completed_tasks_itr):
            par_job_id, par_action = par_job
            if chaining_rules.chainable(action, par_action):
                # Simply call the action Pipeline
                job_context.parent_id = par_job_id
                completed_tasks_master.append((job_id, action))
                break
        # If no proper parent job found
        else:
            # If action is only chained
            if chaining_rules.chained_only(action):
                job_context.status = "Error"
                continue
            # If action can be standalone with no parent.
            # Simply call the action Pipeline
            job_context.parent_id = None  # Update parent JobID
            completed_tasks_master.append((job_id, action))  # List of completed actions

    # Update the job contexts after chainer parsing is done
    for jc in job_contexts:
        jc.write()
    return job_contexts


def create_job_contexts(parent_job_id, actions, job_ids, handler_id):
    """Calls the create job contexts function after Obtains the necessary additional info fo"""
    parent_action = infer_action_from_job(handler_id, parent_job_id)

    network = get_handler_type(handler_id)
    if not network:
        return []

    network_config = read_network_config(network)
    chaining_rules = CHAINING_RULES_TO_FUNCTIONS[network_config["api_params"]["chaining_rules"]]

    return _create_job_contexts(parent_job_id, parent_action, actions, job_ids, network, chaining_rules, handler_id)
