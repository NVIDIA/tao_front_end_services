"""MEDICAL Auto3DSeg training script.Exception"""

import datetime
import json
import os
import shutil
import sys
from inspect import signature

from medical.apps.auto3dseg import AutoRunner, AlgoEnsembleBuilder, AlgoEnsembleBestN
from medical.apps.auto3dseg.utils import import_bundle_algo_history
from medical.utils.enums import AlgoKeys


def train(spec, status_file):
    """Run MEDICAL AutoML training."""
    # -------------------------------------------------------------------------
    # Configure Auto3DSeg Input Parameters
    # -------------------------------------------------------------------------
    # Create input config
    input_cfg = {
        "task": "segmentation",
        "modality": spec.pop("modality"),
        "datalist": spec.pop("datalist"),
        "dataroot": "",  # not needed as the datalist has full path
        "multigpu": spec.pop("num_gpu") > 1,
        "class_names": spec.pop("class_names", None),
    }
    # Initialize Autorunner
    parameters = {}
    for key in signature(AutoRunner).parameters:
        if key in spec:
            parameters[key] = spec.pop(key)
    # Set ensemble to False (currently not supported)
    parameters["ensemble"] = False
    runner = AutoRunner(input=input_cfg, **parameters)

    # Provide additional parameters
    if "num_fold" in spec:
        runner.set_num_fold(num_fold=spec.pop("num_fold"))
    if "train_params" in spec:
        runner.set_training_params(params=spec.pop("train_params"))
    if "ensemble_params" in spec:
        runner.set_ensemble_method(**spec.pop("ensemble_params"))
    if "prediction_params" in spec:
        runner.set_prediction_params(params=spec.pop("prediction_params"))
    if "hpo_params" in spec:
        runner.set_hpo_params(params=spec.pop("hpo_params"))
    if "gpu_customization" in spec:
        runner.set_gpu_customization(**spec.pop("gpu_customization"))
    if "device_info" in spec:
        runner.set_device_info(**spec.pop("device_info"))
    if "image_save_transform" in spec:
        runner.set_image_save_transform(**spec.pop("image_save_transform"))

    status_dict = {
        "date": datetime.date.today().isoformat(),
        "time": datetime.datetime.now().isoformat(),
        "status": "Unknown",
        "message": "MEDICAL Auto3DSeg are configured and will be run. Please use MLFLow to check the metrics.",
    }
    with open(status_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(status_dict) + "\n")

    # -------------------------------------------------------------------------
    # Run Auto3DSeg
    # -------------------------------------------------------------------------
    # Check unused parameters and report it.
    spec.pop("cluster", None)
    spec.pop("output_experiment_name", None)
    spec.pop("output_experiment_description", None)
    if spec:
        print("Unused parameters:", spec, file=sys.stderr)
    runner.run()

    status_dict = {
        "date": datetime.date.today().isoformat(),
        "time": datetime.datetime.now().isoformat(),
        "status": "SUCCESS",
        "message": "MEDICAL Auto3DSeg job is completed. Please export model or use MLFlow to check results.",
    }
    with open(status_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(status_dict) + "\n")

    # -------------------------------------------------------------------------
    # Choose the best model
    # -------------------------------------------------------------------------
    # Import the training history
    history = import_bundle_algo_history(runner.work_dir, only_trained=False)
    history_untrained = [h for h in history if not h[AlgoKeys.IS_TRAINED]]
    if history_untrained:
        print(
            f"Model selection will skip {[h[AlgoKeys.ID] for h in history_untrained]} untrained algos."
            "Generally it means these algos did not complete training.",
            flush=True,
        )
        history = [h for h in history if h[AlgoKeys.IS_TRAINED]]
    if len(history) == 0:
        raise ValueError(
            f"The training results was not found {runner.work_dir}. Possibly the training step was not completed."
        )
    # Build the ensembler (best N algo) to choose the best model
    builder = AlgoEnsembleBuilder(history, runner.data_src_cfg_name)
    builder.set_ensemble_method(AlgoEnsembleBestN(n_best=1))
    ensembler = builder.get_ensemble()
    algos = [alg for alg in ensembler.algo_ensemble if alg is not None]
    if len(algos) == 0:
        raise ValueError("No trained model is available. Please check the training step.")
    if len(algos) > 1:
        print("Choosing the first best model. Ensembling is not available yet!", flush=True)
    algo = algos[0]

    status_dict = {
        "date": datetime.date.today().isoformat(),
        "time": datetime.datetime.now().isoformat(),
        "status": "SUCCESS",
        "message": "MEDICAL AutoML best model selection is done.",
        "best_model": algo["identifier"],
        "key_metric": algo["best_metric"],
    }
    with open(status_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(status_dict) + "\n")

    # -------------------------------------------------------------------------
    # Consolidate the best model
    # -------------------------------------------------------------------------
    # Copy the chosen model into best model directory
    best_model_path = os.path.join(runner.work_dir, "best_model")
    shutil.copytree(algo["algo_instance"].output_path, best_model_path)
    print(f"Best model is exported to {best_model_path}.", flush=True)

    # Clean up the best model
    shutil.rmtree(os.path.join(best_model_path, "__pycache__"), ignore_errors=True)
    shutil.rmtree(os.path.join(best_model_path, "pretrained_model"), ignore_errors=True)
    shutil.rmtree(os.path.join(best_model_path, "scripts", "__pycache__"), ignore_errors=True)
    try:
        os.remove(os.path.join(best_model_path, "algo_object.pkl"))
    except FileNotFoundError:
        pass
    print(f"Exported model is cleaned up: {best_model_path}.", flush=True)

    status_dict = {
        "date": datetime.date.today().isoformat(),
        "time": datetime.datetime.now().isoformat(),
        "status": "SUCCESS",
        "message": (
            "MEDICAL AutoML training is successfully completed. A new experiment will be generated upon this. "
            "Please check the post running job to get the information on the new experiment/model."
        )
    }
    with open(status_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(status_dict) + "\n")


if __name__ == "__main__":
    spec = json.loads(sys.argv[1])
    status_file = sys.argv[2]
    print(f"Running MEDICAL AutoML with spec: {spec}", flush=True)
    print(f"Writing status to {status_file}", flush=True)
    train(spec, status_file)
