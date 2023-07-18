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

"""Bayesian AutoML algorithm modules"""
import numpy as np
import os
import json
import math

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.stats import norm
from scipy.optimize import minimize
from automl.utils import JobStates, fix_input_dimension, get_valid_range, clamp_value, report_healthy
from handlers.utilities import load_json_spec

np.random.seed(95051)


class Bayesian:
    """Bayesian AutoML algorithm class"""

    def __init__(self, root, parameters):
        """Initialize the Bayesian algorithm class
        Args:
            root: handler root
            parameters: automl sweepable parameters
        """
        self.root = root
        self.parameters = parameters
        self.parent_params = {}
        length_scale = [1.0] * len(self.parameters)
        m52 = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=2.5)
        # m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) # is another option
        self.gp = GaussianProcessRegressor(
            kernel=m52,
            alpha=1e-10,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=10,
            random_state=95051
        )
        # The following 2 need to be stored
        self.Xs = []
        self.ys = []

        self.xi = 0.01
        self.num_restarts = 5

        self.num_epochs_per_experiment = self.get_total_epochs()

        report_healthy(self.root + "/controller.log", "Bayesian init", clear=True)

    def convert_parameter(self, parameter_config, suggestion):
        """Convert 0 to 1 GP prediction into a possible value"""
        tp = parameter_config.get("value_type")
        default_value = parameter_config.get("default_value", None)
        math_cond = parameter_config.get("math_cond", None)
        parent_param = parameter_config.get("parent_param", None)

        if tp in ("int", "integer"):
            if parameter_config["parameter"] == "augmentation_config.preprocessing.output_image_height":
                if "model_config.input_image_config.size_height_width.height" in self.parent_params.keys():
                    return self.parent_params["model_config.input_image_config.size_height_width.height"]
            if parameter_config["parameter"] == "augmentation_config.preprocessing.output_image_width":
                if "model_config.input_image_config.size_height_width.width" in self.parent_params.keys():
                    return self.parent_params["model_config.input_image_config.size_height_width.width"]

            v_min = parameter_config.get("valid_min", "")
            v_max = parameter_config.get("valid_max", "")
            if v_min == "" or v_max == "":
                return int(default_value)
            if (type(v_min) != str and math.isnan(v_min)) or (type(v_max) != str and math.isnan(v_max)):
                return int(default_value)

            v_min = int(v_min)
            if (type(v_max) != str and math.isinf(v_max)) or v_max == "inf":
                v_max = int(default_value)
            else:
                v_max = int(v_max)
            random_int = np.random.randint(v_min, v_max + 1)

            if type(math_cond) == str:
                factor = int(math_cond.split(" ")[1])
                random_int = fix_input_dimension(random_int, factor)

            if not (type(parent_param) == float and math.isnan(parent_param)):
                if (type(parent_param) == str and parent_param != "nan" and parent_param == "TRUE") or (type(parent_param) == bool and parent_param):
                    self.parent_params[parameter_config.get("parameter")] = random_int

            return random_int
        if tp == "float":
            v_min = parameter_config.get("valid_min", "")
            v_max = parameter_config.get("valid_max", "")
            if v_min == "" or v_max == "":
                return float(default_value)
            if (type(v_min) != str and math.isnan(v_min)) or (type(v_max) != str and math.isnan(v_max)):
                return float(default_value)

            v_min, v_max = get_valid_range(parameter_config, self.parent_params)
            normalized = suggestion * (v_max - v_min) + v_min
            quantized = clamp_value(normalized, v_min, v_max)

            if not (type(parent_param) == float and math.isnan(parent_param)):
                if (type(parent_param) == str and parent_param != "nan" and parent_param == "TRUE") or (type(parent_param) == bool and parent_param):
                    self.parent_params[parameter_config.get("parameter")] = quantized

            return quantized
        if tp == "bool":
            return np.random.randint(0, 2) == 1
        if tp == "ordered_int":
            if parameter_config.get("valid_options", "") == "":
                return default_value
            valid_values = parameter_config.get("valid_options")
            sample = int(np.random.choice(valid_values.split(",")))
            return sample
        if tp in ("categorical", "ordered"):
            if parameter_config.get("valid_options", "") == "":
                return default_value
            valid_values = parameter_config.get("valid_options")
            sample = np.random.choice(valid_values.split(","))
            return sample
        return default_value

    def save_state(self):
        """Save the Bayesian algorithm related variables to brain.json"""
        state_dict = {}
        state_dict["Xs"] = np.array(self.Xs).tolist()  # List of np arrays
        state_dict["ys"] = np.array(self.ys).tolist()  # List

        file_path = self.root + "/brain.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f,
                      separators=(',', ':'),
                      sort_keys=True,
                      indent=4)

    @staticmethod
    def load_state(root, parameters):
        """Load the Bayesian algorithm related variables to brain.json"""
        file_path = root + "/brain.json"
        if not os.path.exists(file_path):
            return Bayesian(root)

        with open(file_path, 'r', encoding='utf-8') as f:
            json_loaded = json.loads(f.read())
            Xs = []
            for x in json_loaded["Xs"]:
                Xs.append(np.array(x))
            ys = json_loaded["ys"]
        bayesian = Bayesian(root, parameters)
        # Load state (Remember everything)
        bayesian.Xs = Xs
        bayesian.ys = ys

        len_y = len(ys)
        bayesian.gp.fit(np.array(Xs[:len_y]), np.array(ys))

        return bayesian

    def generate_recommendations(self, history):
        """Generates parameter values and appends to recommendations"""
        if history == []:
            # default recommendation => random points
            # TODO: In production, this must be default values for a baseline
            suggestions = np.random.rand(len(self.parameters))
            self.Xs.append(suggestions)
            recommendations = []
            for param_dict, suggestion in zip(self.parameters, suggestions):
                recommendations.append(self.convert_parameter(param_dict, suggestion))
            return [dict(zip([param["parameter"] for param in self.parameters], recommendations))]
        # This function will be called every 5 seconds or so.
        # If no change in history, dont give a recommendation
        # ie - wait for previous recommendation to finish
        if history[-1].status not in [JobStates.success, JobStates.failure]:
            return []

        # Update the GP based on results
        self.ys.append(history[-1].result)
        self.update_gp()

        # Generate one recommendation
        # Generate "suggestions" which are in [0.0, 1.0] by optimizing EI
        suggestions = self.optimize_ei()  # length = len(self.parameters), np.array type
        self.Xs.append(suggestions)
        # Convert the suggestions to recommendations based on parameter type
        # Assume one:one mapping between self.parameters and suggestions
        recommendations = []
        assert len(self.parameters) == len(suggestions)
        for param_dict, suggestion in zip(self.parameters, suggestions):
            recommendations.append(self.convert_parameter(param_dict, suggestion))

        return [dict(zip([param["parameter"] for param in self.parameters], recommendations))]

    def update_gp(self):
        """Update gausian regressor parameters"""
        Xs_npy = np.array(self.Xs)
        ys_npy = np.array(self.ys)
        self.gp.fit(Xs_npy, ys_npy)

    def optimize_ei(self):
        """Optmize expected improvement functions"""
        best_ei = 1.0
        best_x = None

        dim = len(self.Xs[0])
        bounds = [(0, 1)] * len(self.parameters)

        for _ in range(self.num_restarts):
            x0 = np.random.rand(dim)
            res = minimize(self._expected_improvement, x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.fun < best_ei:
                best_ei = res.fun
                best_x = res.x
        return best_x.reshape(-1)

    """
    Used from:
    http://krasserm.github.io/2018/03/21/bayesian-optimization/
    """
    def _expected_improvement(self, X):
        """
        Computes the EI at points X based on existing samples X_sample
        and Y_sample using a Gaussian process surrogate model.
        Args:
            X: Points at which EI shall be computed (m x d).
            X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.
            xi: Exploitation-exploration trade-off parameter.

        Returns:
            Expected improvements at points X.
        """
        X = X.reshape(1, -1)

        mu, sigma = self.gp.predict(X, return_std=True)
        mu_sample = self.gp.predict(np.array(self.Xs))

        sigma = sigma.reshape(-1, 1)
        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        # See also section 2.4 in [1]
        mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return -1 * ei[0, 0]

    def get_total_epochs(self):
        """Get the epoch/iter number from train.json"""
        spec = load_json_spec(self.root + "/../specs/train.json")
        max_epoch = 100.0
        for key1 in spec:
            if key1 in ("training_config", "train_config", "train"):
                for key2 in spec[key1]:
                    if key2 in ("num_epochs", "epochs", "n_epochs", "max_iters"):
                        max_epoch = float(spec[key1][key2])
                    elif key2 in ("train_config"):
                        for key3 in spec[key1][key2]:
                            if key3 == "runner":
                                for key4 in spec[key1][key2][key3]:
                                    if key4 == "max_epochs":
                                        max_epoch = float(spec[key1][key2][key3][key4])
            elif key1 in ("num_epochs"):
                max_epoch = float(spec[key1])

        return max_epoch
