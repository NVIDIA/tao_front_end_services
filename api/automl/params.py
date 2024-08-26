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

"""AutoML read parameters modules"""
from constants import AUTOML_DISABLED_NETWORKS
from handlers.utilities import Code, get_flatten_specs
from handlers.stateless_handlers import safe_load_file
from specs_utils import csv_to_json_schema

import os
import pandas as pd

_VALID_TYPES = ["int", "integer",
                "float",
                "ordered_int", "bool",
                "ordered", "categorical",
                "list_1_backbone", "list_1_normal", "list_2", "list_3"]


def generate_hyperparams_to_search(job_context, automl_add_hyperparameters, automl_remove_hyperparameters, handler_root, override_automl_disabled_params=False):
    """Use train.csv spec of the network to choose the parameters of AutoML
    Returns: a list of dict for AutoML supported networks
    """
    network_arch = job_context.network
    if network_arch not in AUTOML_DISABLED_NETWORKS:
        DIR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        CSV_PATH = os.path.join(DIR_PATH, "specs_utils", "specs", network_arch, f"{network_arch} - train.csv")
        if not os.path.exists(CSV_PATH):
            return Code(404, {}, "Default specs do not exist for action")

        original_train_spec = csv_to_json_schema.convert(CSV_PATH)["default"]
        original_spec_with_keys_flattened = {}
        get_flatten_specs(original_train_spec, original_spec_with_keys_flattened)

        updated_train_spec = safe_load_file(f"{handler_root}/specs/{job_context.id}-{job_context.action}-spec.json")
        updated_spec_with_keys_flattened = {}
        get_flatten_specs(updated_train_spec, updated_spec_with_keys_flattened)

        deleted_params = original_spec_with_keys_flattened.keys() - updated_spec_with_keys_flattened

        data_frame = pd.read_csv(CSV_PATH)

        data_frame = data_frame[data_frame['value_type'].isin(_VALID_TYPES)]

        if not override_automl_disabled_params:
            data_frame = data_frame.loc[data_frame['automl_enabled'] != False]  # pylint: disable=C0121  # noqa: E712

        # Push params that are dependent on other params to the bottom
        data_frame = data_frame.sort_values(by=['depends_on'])
        data_frame = data_frame[::-1]

        data_frame.loc[data_frame.parameter.isin(automl_remove_hyperparameters), 'automl_enabled'] = False
        data_frame.loc[data_frame.parameter.isin(automl_add_hyperparameters), 'automl_enabled'] = True

        automl_params = data_frame.loc[data_frame['automl_enabled'] == True]  # pylint: disable=C0121  # noqa: E712
        automl_params = automl_params.loc[~automl_params['parameter'].isin(deleted_params)]
        automl_params = automl_params[["parameter", "value_type", "default_value", "valid_min", "valid_max", "valid_options", "math_cond", "parent_param", "depends_on"]]
        return automl_params.to_dict('records'), automl_params["parameter"].values
    return [{}], []
