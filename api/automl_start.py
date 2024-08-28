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

"""AutoML main handler"""
from automl.controller import Controller
from automl.bayesian import Bayesian
from automl.hyperband import HyperBand
from automl.params import generate_hyperparams_to_search
from handlers.utilities import JobContext
from handlers.stateless_handlers import update_job_status, update_job_metadata
import ast
import sys
import argparse
import traceback
import json


def automl_start(root, network, jc, resume, automl_algorithm, automl_max_recommendations, automl_delete_intermediate_ckpt, automl_R, automl_nu, metric, epoch_multiplier, automl_add_hyperparameters, automl_remove_hyperparameters, override_automl_disabled_params, decrypted_workspace_metadata):
    """Starts the automl controller"""
    parameters, parameter_names = generate_hyperparams_to_search(jc, automl_add_hyperparameters, automl_remove_hyperparameters, "/".join(root.split("/")[0:-2]), override_automl_disabled_params)

    # Check if automl algorithm is valid for specific use-cases
    if network == "classification_pyt":
        if "model.head.type" in parameter_names and automl_algorithm == "hyperband":
            error_message = "Hyperband not supported when non-epoch based models are chosen. Change algorithm to bayesian"
            result = {"message": error_message}
            update_job_metadata(jc.org_name, jc.handler_id, jc.id, metadata_key="result", data=result, kind="experiments")
            raise ValueError(error_message)

    if resume:
        if automl_algorithm.lower() in ("hyperband", "h"):
            brain = HyperBand.load_state(job_context=jc, root=root, network=network, parameters=parameters, R=int(automl_R), nu=int(automl_nu), epoch_multiplier=int(epoch_multiplier))
        elif automl_algorithm.lower() in ("bayesian", "b"):
            brain = Bayesian.load_state(jc, root, network, parameters)

        controller = Controller.load_state(root, network, brain, jc, automl_max_recommendations, automl_delete_intermediate_ckpt, metric, automl_algorithm.lower(), decrypted_workspace_metadata)
        controller.start()

    else:
        if automl_algorithm.lower() in ("hyperband", "h"):
            brain = HyperBand(job_context=jc, root=root, network=network, parameters=parameters, R=int(automl_R), nu=int(automl_nu), epoch_multiplier=int(epoch_multiplier))
        elif automl_algorithm.lower() in ("bayesian", "b"):
            brain = Bayesian(jc, root, network, parameters)
        controller = Controller(root, network, brain, jc, automl_max_recommendations, automl_delete_intermediate_ckpt, metric, automl_algorithm.lower(), decrypted_workspace_metadata)
        controller.start()
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='AutoML controller', description='Run AutoML.')
    parser.add_argument(
        '--user_id',
        type=str,
    )
    parser.add_argument(
        '--org_name',
        type=str,
    )
    parser.add_argument(
        '--name',
        type=str,
    )
    parser.add_argument(
        '--root',
        type=str,
    )
    parser.add_argument(
        '--automl_job_id',
        type=str,
    )
    parser.add_argument(
        '--network',
        type=str,
    )
    parser.add_argument(
        '--experiment_id',
        type=str,
    )
    parser.add_argument(
        '--resume',
        type=str,
    )
    parser.add_argument(
        '--automl_algorithm',
        type=str,
    )
    parser.add_argument(
        '--automl_max_recommendations',
        type=str,
    )
    parser.add_argument(
        '--automl_delete_intermediate_ckpt',
        type=str,
    )
    parser.add_argument(
        '--automl_R',
        type=str,
    )
    parser.add_argument(
        '--automl_nu',
        type=str,
    )
    parser.add_argument(
        '--metric',
        type=str,
    )
    parser.add_argument(
        '--epoch_multiplier',
        type=str,
    )
    parser.add_argument(
        '--automl_add_hyperparameters',
        type=str,
    )
    parser.add_argument(
        '--automl_remove_hyperparameters',
        type=str,
    )
    parser.add_argument(
        '--override_automl_disabled_params',
        type=str,
    )
    parser.add_argument(
        '--decrypted_workspace_metadata',
        type=json.loads,
    )

    args = parser.parse_args()
    automl_job_id = args.automl_job_id
    handler_id = args.experiment_id
    network = args.network
    user_id = args.user_id
    org_name = args.org_name
    try:
        root = args.root
        name = args.name
        jc = JobContext(automl_job_id, None, network, "train", handler_id, user_id, org_name, "experiment", name=name)
        resume = args.resume == "True"
        automl_algorithm = args.automl_algorithm
        automl_max_recommendations = args.automl_max_recommendations
        automl_delete_intermediate_ckpt = args.automl_delete_intermediate_ckpt
        automl_R = args.automl_R
        automl_nu = args.automl_nu
        metric = args.metric
        epoch_multiplier = args.epoch_multiplier
        automl_add_hyperparameters = ast.literal_eval(args.automl_add_hyperparameters)
        automl_remove_hyperparameters = ast.literal_eval(args.automl_remove_hyperparameters)
        override_automl_disabled_params = args.override_automl_disabled_params == "True"
        decrypted_workspace_metadata = args.decrypted_workspace_metadata
        automl_start(
            root=root,
            network=network,
            jc=jc,
            resume=resume,
            automl_algorithm=automl_algorithm,
            automl_max_recommendations=automl_max_recommendations,
            automl_delete_intermediate_ckpt=automl_delete_intermediate_ckpt,
            automl_R=automl_R,
            automl_nu=automl_nu,
            metric=metric,
            epoch_multiplier=epoch_multiplier,
            automl_add_hyperparameters=automl_add_hyperparameters,
            automl_remove_hyperparameters=automl_remove_hyperparameters,
            override_automl_disabled_params=override_automl_disabled_params,
            decrypted_workspace_metadata=decrypted_workspace_metadata)

    except Exception:
        print(f"AutoML start for network {network} failed due to exception {traceback.format_exc()}", file=sys.stderr)
        update_job_status(org_name, handler_id, automl_job_id, status="Error", kind="experiments")
