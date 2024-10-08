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

"""Hyperband AutoML algorithm modules"""
import numpy as np
import os
import math

from automl.utils import ResumeRecommendation, JobStates, get_valid_range, clamp_value, report_healthy
from automl.automl_algorithm_base import AutoMLAlgorithmBase
from handlers.utilities import get_train_spec, get_flatten_specs
from utils import safe_load_file, safe_dump_file

np.random.seed(95051)


class HyperBand(AutoMLAlgorithmBase):
    """Hyperband AutoML algorithm class"""

    def __init__(self, job_context, root, network, parameters, R, nu, epoch_multiplier):
        """Initialize the Hyperband algorithm class
        Args:
            root: handler root
            network: model we are running AutoML on
            parameters: automl sweepable parameters
            R: the maximum amount of resource that can be allocated to a single configuration
            nu: an input that controls the proportion of configurations discarded in each round of SuccessiveHalving
            epoch_multiplier: multiplying factor for epochs
        """
        super().__init__(job_context, root, network, parameters)
        report_healthy(self.root + "/controller.log", "Hyperband init", clear=True)
        self.epoch_multiplier = int(epoch_multiplier)
        self.ni = {}
        self.ri = {}
        self.brackets_and_sh_sequence(R, nu)
        self.epoch_number = 0
        self.resume_epoch_number = 0
        # State variables
        self.bracket = 0  # Bracket
        self.override_num_epochs(self.ri[self.bracket][-1] * self.epoch_multiplier)
        self.sh_iter = 0  # SH iteration
        self.experiments_considered = []
        self.expt_iter = 0  # Recommendations within the SH
        self.complete = False
        self.reverse_sort = True

    def brackets_and_sh_sequence(self, R, nu):
        """Generate ni,ri arrays based on R and nu values"""
        smax = int(np.log(R) / np.log(nu))
        for itr, s in enumerate(range(smax, 0, -1)):  # This would be range(smax,-1,-1) to follow the paper, but the last bracket has n recommendations and no pruning and they are just random recs. Unless using a Bayesian HP recommendation, this can be 0
            self.ni[itr] = []
            self.ri[itr] = []
            n = int(math.ceil(int((smax + 1) / (s + 1)) * (nu**s)))
            r = int(R / (nu**s))
            for s_idx in range(s + 1):
                ni = int(n * (nu**(-s_idx)))
                ri = int(r * (nu**s_idx))
                self.ni[itr].append(ni)
                self.ri[itr].append(ri)

    def override_num_epochs(self, num_epochs):
        """Override num epochs parameter in train spec file"""
        spec = get_train_spec(self.job_context, os.path.join(self.handler_root, "specs"))
        for key1 in spec:
            if key1 in ("training_config", "train_config", "train"):
                for key2 in spec[key1]:
                    if key2 in ("num_epochs", "epochs", "n_epochs", "max_iters"):
                        spec[key1][key2] = num_epochs
                    elif key2 in ("train_config"):
                        for key3 in spec[key1][key2]:
                            if key3 == "runner":
                                for key4 in spec[key1][key2][key3]:
                                    if key4 == "max_epochs":
                                        spec[key1][key2][key3][key4] = num_epochs
            elif key1 in ("num_epochs"):
                spec[key1] = num_epochs
        safe_dump_file(self.handler_root + f"/specs/{self.job_context.id}-train-spec.json", spec)

    def generate_automl_param_rec_value(self, parameter_config):
        """Generate a random value for the parameter passed"""
        tp = parameter_config.get("value_type")
        default_value = parameter_config.get("default_value", None)
        parent_param = parameter_config.get("parent_param", None)

        if tp == "float":
            v_min = parameter_config.get("valid_min", "")
            v_max = parameter_config.get("valid_max", "")
            if v_min == "" or v_max == "":
                return float(default_value)
            if (type(v_min) is not str and math.isnan(v_min)) or (type(v_max) is not str and math.isnan(v_max)):
                return float(default_value)
            v_min, v_max = get_valid_range(parameter_config, self.parent_params)
            random_float = np.random.uniform(low=v_min, high=v_max)
            random_float = clamp_value(random_float, v_min, v_max)

            if not (type(parent_param) is float and math.isnan(parent_param)):
                if (type(parent_param) is str and parent_param != "nan" and parent_param == "TRUE") or (type(parent_param) is bool and parent_param):
                    self.parent_params[parameter_config.get("parameter")] = random_float
            return random_float

        return super().generate_automl_param_rec_value(parameter_config)

    def save_state(self):
        """Save the Hyperband algorithm related variables to brain.json"""
        state_dict = {}
        state_dict["bracket"] = self.bracket
        state_dict["sh_iter"] = self.sh_iter
        state_dict["expt_iter"] = self.expt_iter
        state_dict["complete"] = self.complete
        state_dict["epoch_number"] = self.epoch_number
        state_dict["resume_epoch_number"] = self.resume_epoch_number
        state_dict["epoch_multiplier"] = self.epoch_multiplier
        state_dict["ni"] = self.ni
        state_dict["ri"] = self.ri

        file_path = self.root + "/brain.json"
        safe_dump_file(file_path, state_dict)

    @staticmethod
    def load_state(job_context, root, network, parameters, R, nu, epoch_multiplier):
        """Load the Hyperband algorithm related variables to brain.json"""
        file_path = root + "/brain.json"
        if not os.path.exists(file_path):
            return HyperBand(job_context, root, network, parameters, R, nu, epoch_multiplier)

        json_loaded = safe_load_file(file_path)

        brain = HyperBand(job_context, root, network, parameters, R, nu, epoch_multiplier)
        # Load state (Remember everything)
        brain.bracket = json_loaded["bracket"]  # Bracket
        brain.sh_iter = json_loaded["sh_iter"]  # SH iteration
        brain.expt_iter = json_loaded["expt_iter"]  # Recommendations within the SH
        brain.complete = json_loaded["complete"]
        brain.epoch_number = json_loaded["epoch_number"]
        brain.resume_epoch_number = json_loaded["resume_epoch_number"]

        return brain

    def _generate_one_recommendation(self, history):
        """Updates the counter variables and performs successive halving"""
        if self.complete:
            return None

        num = self.ni[self.bracket][self.sh_iter]
        if self.expt_iter == num:
            self.expt_iter = 0
            self.sh_iter += 1
        if self.sh_iter == len(self.ni[self.bracket]):
            self.sh_iter = 0
            self.bracket += 1
            if self.bracket in self.ri.keys():
                self.override_num_epochs(self.ri[self.bracket][-1] * self.epoch_multiplier)
        if self.bracket > max(list(self.ni.keys())):
            self.complete = True
            return None

        if self.sh_iter == 0:
            specs = self._generate_random_parameters()
            self.epoch_number = self.ri[self.bracket][self.sh_iter] * self.epoch_multiplier
            to_return = specs
        else:
            # Do successive halving on the last bracket
            # Here, we are sloppy in defining the window, but we assume runs that run for more epochs will be better
            # We take history[-bracket_size:] and prune this at every SH step
            lower = -1 * self.ni.get(self.bracket, [0])[0]

            self.resume_epoch_number = int(self.ri[self.bracket][self.sh_iter - 1] * self.epoch_multiplier)
            if self.expt_iter == 0:
                if self.sh_iter == 1:
                    self.experiments_considered = sorted(history[lower:], key=lambda rec: rec.result, reverse=self.reverse_sort)[0:self.ni[self.bracket][self.sh_iter]]
                else:
                    for experiment in self.experiments_considered:
                        experiment.result = history[experiment.id].result
                    self.experiments_considered = sorted(self.experiments_considered, key=lambda rec: rec.result, reverse=self.reverse_sort)[0:self.ni[self.bracket][self.sh_iter]]

            self.epoch_number = self.ri[self.bracket][self.sh_iter] * self.epoch_multiplier
            resumerec = ResumeRecommendation(self.experiments_considered[self.expt_iter].id, self.experiments_considered[self.expt_iter].specs)
            to_return = resumerec
        self.expt_iter += 1

        return to_return

    def done(self):
        """Return if Hyperband algorithm is complete or not"""
        return self.complete

    def _generate_random_parameters(self):
        """Generates random parameter values for a recommendation"""
        hyperparam_dict = {}
        for param in self.parameters:
            name = param["parameter"]
            rec = self.generate_automl_param_rec_value(param)
            hyperparam_dict[name] = rec
        return hyperparam_dict

    def generate_recommendations(self, history):
        """Generates recommendations for the controller to run"""
        get_flatten_specs(self.default_train_spec, self.default_train_spec_flattened)
        if history == []:
            rec1 = self._generate_one_recommendation(history)
            assert type(rec1) is dict
            self.track_id = 0
            return [rec1]

        if history[self.track_id].status not in [JobStates.success, JobStates.failure]:
            return []
        rec = self._generate_one_recommendation(history)
        if type(rec) is dict:
            self.track_id = len(history)
            return [rec]
        if type(rec) is ResumeRecommendation:
            self.track_id = rec.id
            return [rec]
        return []
