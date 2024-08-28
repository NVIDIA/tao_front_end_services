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

"""Functions to infer data sources"""
import os
import tempfile
import sys

from handlers.medical_dataset_handler import MonaiDatasetHandler
from handlers.stateless_handlers import get_job_id_of_action, get_handler_metadata, get_handler_job_metadata, get_handler_id, get_workspace_string_identifier
from handlers.infer_params import CLI_CONFIG_TO_FUNCTIONS


def detectnet_v2(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for Detecnet v2"""
    pass


def unet(config, job_context, handler_metadata):
    pass


def segformer(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for Segformer"""
    workspace_cache = {}
    # Init
    if "dataset" not in list(config.keys()):
        config["dataset"] = {}

    # Training datasets
    if handler_metadata.get("train_datasets", []) != []:
        if "train_dataset" not in config["dataset"].keys():
            config["dataset"]["train_dataset"] = {}
        config["dataset"]["train_dataset"]["ann_dir"] = []
        config["dataset"]["train_dataset"]["img_dir"] = []
        for train_ds in handler_metadata.get("train_datasets", []):
            train_ds_metadata = get_handler_metadata(job_context.org_name, train_ds, kind="datasets")
            workspace_identifier = get_workspace_string_identifier(job_context.org_name, train_ds_metadata.get('workspace'), workspace_cache)
            if config["dataset"]["train_dataset"].get("ann_dir", None):
                config["dataset"]["train_dataset"]["ann_dir"].append(f"{workspace_identifier}{train_ds_metadata.get('cloud_file_path')}/masks/train.tar.gz")
                config["dataset"]["train_dataset"]["img_dir"].append(f"{workspace_identifier}{train_ds_metadata.get('cloud_file_path')}/images/train.tar.gz")
            else:
                config["dataset"]["train_dataset"]["ann_dir"] = [f"{workspace_identifier}{train_ds_metadata.get('cloud_file_path')}/masks/train.tar.gz"]
                config["dataset"]["train_dataset"]["img_dir"] = [f"{workspace_identifier}{train_ds_metadata.get('cloud_file_path')}/images/train.tar.gz"]
    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        if job_context.action == "train":
            eval_key = "val_dataset"
        else:
            eval_key = "test_dataset"
        config["dataset"][eval_key] = {}
        eval_ds_metadata = get_handler_metadata(job_context.org_name, eval_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, eval_ds_metadata.get('workspace'), workspace_cache)
        config["dataset"][eval_key]["ann_dir"] = f"{workspace_identifier}{eval_ds_metadata.get('cloud_file_path')}/masks/val.tar.gz"
        config["dataset"][eval_key]["img_dir"] = f"{workspace_identifier}{eval_ds_metadata.get('cloud_file_path')}/images/val.tar.gz"
    return config


def object_detection(config, job_context, handler_metadata):
    """Returns config directly as no changes are required"""
    return config


def efficientdet_tf2(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for EfficientDet tf2"""
    workspace_cache = {}
    # Dataset convert
    if handler_metadata.get("train_datasets", []) == []:
        if config.get("dataset_convert") is None:
            config["dataset_convert"] = {}
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, handler_metadata.get('workspace'), workspace_cache)
        config["dataset_convert"]["image_dir"] = f"{workspace_identifier}{handler_metadata.get('cloud_file_path')}" + "/images.tar.gz"
        config["dataset_convert"]["annotations_file"] = f"{workspace_identifier}{handler_metadata.get('cloud_file_path')}" + "/annotations.json"
        return config

    # Init
    if "data" not in list(config.keys()):
        config["dataset"] = {}

    # Training datasets
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        print("Warning: EfficientDet supports only one train dataset", file=sys.stderr)
        train_ds_metadata = get_handler_metadata(job_context.org_name, train_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, train_ds_metadata.get('workspace'), workspace_cache)
        handler_root = f"{workspace_identifier}{train_ds_metadata.get('cloud_file_path')}"
        train_ds_convert_job_id = get_job_id_of_action(job_context.org_name, train_ds, kind="datasets", action="convert_efficientdet_tf2")
        dataset_convert_root = f"{workspace_identifier}results/{train_ds_convert_job_id}"
        config["dataset"]["train_tfrecords"] = [dataset_convert_root + "/dataset_convert"]

    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        eval_ds_metadata = get_handler_metadata(job_context.org_name, eval_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, eval_ds_metadata.get('workspace'), workspace_cache)
        handler_root = f"{workspace_identifier}{eval_ds_metadata.get('cloud_file_path')}"
        config["dataset"]["val_json_file"] = handler_root + "/annotations.json"
        eval_ds_convert_job_id = get_job_id_of_action(job_context.org_name, eval_ds, kind="datasets", action="convert_efficientdet_tf2")
        dataset_convert_root = f"{workspace_identifier}results/{eval_ds_convert_job_id}"
        config["dataset"]["val_tfrecords"] = [dataset_convert_root + "/dataset_convert"]

    # calibration dataset
    calib_ds = handler_metadata.get("calibration_dataset", None)
    if calib_ds and job_context.action == "gen_trt_engine":
        calib_ds_metadata = get_handler_metadata(job_context.org_name, calib_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, calib_ds_metadata.get('workspace'), workspace_cache)
        config["gen_trt_engine"]["tensorrt"]["calibration"]["cal_image_dir"] = f"{workspace_identifier}{calib_ds_metadata.get('cloud_file_path')}" + "/images.tar.gz"

    infer_ds = handler_metadata.get("inference_dataset", None)
    if infer_ds:
        if "inference" not in config.keys():
            config["inference"] = {}
        infer_ds_metadata = get_handler_metadata(job_context.org_name, infer_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, infer_ds_metadata.get('workspace'), workspace_cache)
        infer_ds_root = f"{workspace_identifier}{infer_ds_metadata.get('cloud_file_path')}"
        if job_context.action in ("evaluate", "inference"):
            config["inference"]["label_map"] = os.path.join(infer_ds_root, "label_map.yaml")
        if job_context.action == "inference":
            config["inference"]["image_dir"] = os.path.join(infer_ds_root, "images.tar.gz")
    return config


def classification_tf2(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for Classification-tf2"""
    workspace_cache = {}
    if "dataset" not in list(config.keys()):
        config["dataset"] = {}

    print("Warning: Classification-tf2 supports only one train dataset", file=sys.stderr)
    print("Warning: Train, eval datasets are both required to run Classification actions - train, evaluate, retrain, inference", file=sys.stderr)
    train_datasets = handler_metadata.get("train_datasets", [])
    if train_datasets != []:
        train_ds_metadata = get_handler_metadata(job_context.org_name, train_datasets[0], kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, train_ds_metadata.get('workspace'), workspace_cache)
        config["dataset"]["train_dataset_path"] = f"{workspace_identifier}{train_ds_metadata.get('cloud_file_path')}" + "/images_train.tar.gz"

    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        eval_ds_metadata = get_handler_metadata(job_context.org_name, eval_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, eval_ds_metadata.get('workspace'), workspace_cache)
        config["dataset"]["val_dataset_path"] = f"{workspace_identifier}{eval_ds_metadata.get('cloud_file_path')}" + "/images_val.tar.gz"

    infer_ds = handler_metadata.get("inference_dataset", None)
    if infer_ds is not None:
        if "evaluate" not in config:
            config["evaluate"] = {}
        if "inference" not in config:
            config["inference"] = {}
        infer_ds_metadata = get_handler_metadata(job_context.org_name, infer_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, infer_ds_metadata.get('workspace'), workspace_cache)
        config["evaluate"]["dataset_path"] = f"{workspace_identifier}{infer_ds_metadata.get('cloud_file_path')}" + "/images_test.tar.gz"
        if job_context.action == "inference":
            config["inference"]["image_dir"] = f"{workspace_identifier}{infer_ds_metadata.get('cloud_file_path')}" + "/images_test.tar.gz"
        config["inference"]["classmap"] = os.path.join(f"{workspace_identifier}{infer_ds_metadata.get('cloud_file_path')}", "classmap.json")

    # calibration dataset
    calib_ds = handler_metadata.get("calibration_dataset", None)
    if calib_ds and job_context.action == "gen_trt_engine":
        calib_ds_metadata = get_handler_metadata(job_context.org_name, calib_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, calib_ds_metadata.get('workspace'), workspace_cache)
        config["gen_trt_engine"]["tensorrt"]["calibration"]["cal_image_dir"] = f"{workspace_identifier}{calib_ds_metadata.get('cloud_file_path')}" + "/images_train.tar.gz"
        if calib_ds == eval_ds:
            config["gen_trt_engine"]["tensorrt"]["calibration"]["cal_image_dir"] = f"{workspace_identifier}{calib_ds_metadata.get('cloud_file_path')}" + "/images_val.tar.gz"

    return config


def classification_pyt(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for Classification-pyt"""
    workspace_cache = {}

    print("Warning: Classification-pyt supports only one train dataset", file=sys.stderr)
    print("Warning: Train, eval datasets are both required to run Classification actions - train, evaluate, inference", file=sys.stderr)
    train_datasets = handler_metadata.get("train_datasets", [])
    if train_datasets != []:
        train_ds_metadata = get_handler_metadata(job_context.org_name, train_datasets[0], kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, train_ds_metadata.get('workspace'), workspace_cache)
        config["dataset"]["data"]["train"]["data_prefix"] = f"{workspace_identifier}{train_ds_metadata.get('cloud_file_path')}" + "/images_train.tar.gz"
        config["dataset"]["data"]["train"]["classes"] = f"{workspace_identifier}{train_ds_metadata.get('cloud_file_path')}" + "/classes.txt"

    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        if "val" not in list(config["dataset"]["data"].keys()):
            config["dataset"]["data"]["val"] = {}
        eval_ds_metadata = get_handler_metadata(job_context.org_name, eval_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, eval_ds_metadata.get('workspace'), workspace_cache)
        config["dataset"]["data"]["val"]["data_prefix"] = f"{workspace_identifier}{eval_ds_metadata.get('cloud_file_path')}" + "/images_val.tar.gz"
        config["dataset"]["data"]["val"]["classes"] = f"{workspace_identifier}{eval_ds_metadata.get('cloud_file_path')}" + "/classes.txt"

    infer_ds = handler_metadata.get("inference_dataset", None)
    if infer_ds is not None:
        if "test" not in list(config["dataset"]["data"].keys()):
            config["dataset"]["data"]["test"] = {}
        infer_ds_metadata = get_handler_metadata(job_context.org_name, infer_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, infer_ds_metadata.get('workspace'), workspace_cache)
        config["dataset"]["data"]["test"]["data_prefix"] = f"{workspace_identifier}{infer_ds_metadata.get('cloud_file_path')}" + "/images_test.tar.gz"
        config["dataset"]["data"]["test"]["classes"] = f"{workspace_identifier}{infer_ds_metadata.get('cloud_file_path')}" + "/classes.txt"

    return config


def action_recogntion_dynamic_config(config, action):
    """Dynamically drop out spec parameters based on certain other parameters"""
    model_type = config["model"]["model_type"]  # rgb/of/joint

    if model_type == "rgb":
        config["model"].pop("of_seq_length", None)
        if action == "train":
            config["model"].pop("of_pretrained_num_classes", None)
        config["dataset"]["augmentation_config"].pop("of_input_mean", None)
        config["dataset"]["augmentation_config"].pop("of_input_std", None)
        config["model"].pop("of_pretrained_model_path", None)
    elif model_type == "of":
        config["model"].pop("rgb_seq_length", None)
        if action == "train":
            config["model"].pop("rgb_pretrained_num_classes", None)
        config["dataset"]["augmentation_config"].pop("rgb_input_mean", None)
        config["dataset"]["augmentation_config"].pop("rgb_input_std", None)
        config["model"].pop("rgb_pretrained_model_path", None)
    elif model_type == "joint":
        if "rgb_pretrained_model_path" in config["model"].keys():
            ptm_paths = config["model"]["rgb_pretrained_model_path"].split(",")
            rgb_pretrained_model_path = ptm_paths[0] if ptm_paths[0].find("_rgb_") else ptm_paths[1]
            of_pretrained_model_path = ptm_paths[0] if ptm_paths[0].find("_of_") else ptm_paths[1]
            config["model"]["rgb_pretrained_model_path"] = rgb_pretrained_model_path
            config["model"]["of_pretrained_model_path"] = of_pretrained_model_path

    if "label_map" not in config["dataset"]:
        config["dataset"]["label_map"] = {}
    if not config["dataset"]["label_map"]:
        config["dataset"]["label_map"] = {"catch": 0, "smile": 1}
    return config


def action_recognition(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params and,
    makes changes to config baed on model and input_type for Action recognition
    """
    workspace_cache = {}
    config = action_recogntion_dynamic_config(config, job_context.action)
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_ds_metadata = get_handler_metadata(job_context.org_name, train_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, train_ds_metadata.get('workspace'), workspace_cache)
        root = f"{workspace_identifier}{train_ds_metadata.get('cloud_file_path')}"
        if job_context.action == "train":
            config["dataset"]["train_dataset_dir"] = os.path.join(root, "train.tar.gz")
            config["dataset"]["val_dataset_dir"] = os.path.join(root, "test.tar.gz")
        elif job_context.action == "evaluate":
            config["evaluate"]["test_dataset_dir"] = os.path.join(root, "test.tar.gz")
        elif job_context.action == "inference":
            config["inference"]["inference_dataset_dir"] = os.path.join(root, "test/smile.tar.gz")
    return config


def pointpillars(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for Pointpillars"""
    workspace_cache = {}
    train_ds = handler_metadata.get("train_datasets", [])
    if train_ds != []:
        train_ds = train_ds[0]
    else:
        train_ds = handler_metadata.get("id")

    train_ds_metadata = get_handler_metadata(job_context.org_name, train_ds, kind="datasets")
    workspace_identifier = get_workspace_string_identifier(job_context.org_name, train_ds_metadata.get('workspace'), workspace_cache)
    data_root = f"{workspace_identifier}{train_ds_metadata.get('cloud_file_path')}"
    config["dataset"]["data_path"] = data_root

    if handler_metadata.get("train_datasets", []) != []:
        ds_convert_job_id = get_job_id_of_action(job_context.org_name, train_ds, kind="datasets", action="dataset_convert")
        data_info_root = f"{workspace_identifier}results/{ds_convert_job_id}/data_info/"
        config["dataset"]["data_info_path"] = data_info_root
    return config


def pose_classification_dynamic_config(config, action):
    """Dynamically drop out spec parameters based on certain other parameters"""
    model_type = config["model"]["graph_layout"]  # openpose/nvidia
    if model_type == "nvidia":
        if action == "train":
            config["dataset"].pop("random_choose", None)
            config["dataset"].pop("random_move", None)
            config["dataset"].pop("window_size", None)
    elif model_type == "openpose":
        if action == "train":
            config["model"].pop("pretrained_model_path", None)
    return config


def pose_classification(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params and,
    makes changes to config baed on model type for Pose classification
    """
    workspace_cache = {}
    model_type = config["model"]["graph_layout"]  # openpose/nvidia
    if model_type == "openpose":
        model_type = "kinetics"
    pose_classification_dynamic_config(config, job_context.action)

    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_ds_metadata = get_handler_metadata(job_context.org_name, train_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, train_ds_metadata.get('workspace'), workspace_cache)
        root = f"{workspace_identifier}{train_ds_metadata.get('cloud_file_path')}"
        if job_context.action == "train":
            config["dataset"]["train_dataset"] = {}
            config["dataset"]["val_dataset"] = {}
            config["dataset"]["train_dataset"]["data_path"] = os.path.join(root, model_type, "train_data.npy")
            config["dataset"]["train_dataset"]["label_path"] = os.path.join(root, model_type, "train_label.pkl")
            config["dataset"]["val_dataset"]["data_path"] = os.path.join(root, model_type, "val_data.npy")
            config["dataset"]["val_dataset"]["label_path"] = os.path.join(root, model_type, "val_label.pkl")
        elif job_context.action in ("evaluate", "inference"):
            config[job_context.action]["test_dataset"] = {}
            config[job_context.action]["test_dataset"]["data_path"] = os.path.join(root, model_type, "val_data.npy")
            if job_context.action == "evalute":
                config[job_context.action]["test_dataset"]["label_path"] = os.path.join(root, model_type, "val_label.pkl")
    return config


def re_identification(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for Re-identification"""
    workspace_cache = {}
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_ds_metadata = get_handler_metadata(job_context.org_name, train_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, train_ds_metadata.get('workspace'), workspace_cache)
        root = f"{workspace_identifier}{train_ds_metadata.get('cloud_file_path')}"
        if job_context.action == "train":
            config["dataset"]["train_dataset_dir"] = os.path.join(root, "sample_train.tar.gz")
            config["dataset"]["test_dataset_dir"] = os.path.join(root, "sample_test.tar.gz")
            config["dataset"]["query_dataset_dir"] = os.path.join(root, "sample_query.tar.gz")
        elif job_context.action in ("evaluate", "inference"):
            config[job_context.action]["test_dataset"] = os.path.join(root, "sample_test.tar.gz")
            config[job_context.action]["query_dataset"] = os.path.join(root, "sample_query.tar.gz")
    return config


def deformable_detr(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for Deformable-Detr"""
    workspace_cache = {}
    # Train dataset
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_ds_metadata = get_handler_metadata(job_context.org_name, train_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, train_ds_metadata.get('workspace'), workspace_cache)
        train_root = f"{workspace_identifier}{train_ds_metadata.get('cloud_file_path')}"
        if job_context.action == "train":
            config["dataset"]["train_data_sources"] = [{}]
            config["dataset"]["train_data_sources"][0]["image_dir"] = os.path.join(train_root, "images.tar.gz")
            config["dataset"]["train_data_sources"][0]["json_file"] = os.path.join(train_root, "annotations.json")

    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        eval_ds_metadata = get_handler_metadata(job_context.org_name, eval_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, eval_ds_metadata.get('workspace'), workspace_cache)
        eval_root = f"{workspace_identifier}{eval_ds_metadata.get('cloud_file_path')}"
        if job_context.action == "train":
            config["dataset"]["val_data_sources"] = [{}]
            config["dataset"]["val_data_sources"][0]["image_dir"] = os.path.join(eval_root, "images.tar.gz")
            config["dataset"]["val_data_sources"][0]["json_file"] = os.path.join(eval_root, "annotations.json")
        if job_context.action == "evaluate":
            config["dataset"]["test_data_sources"] = {}
            config["dataset"]["test_data_sources"]["image_dir"] = os.path.join(eval_root, "images.tar.gz")
            config["dataset"]["test_data_sources"]["json_file"] = os.path.join(eval_root, "annotations.json")

    # Inference dataset
    infer_ds = handler_metadata.get("inference_dataset", None)
    if infer_ds is not None:
        infer_ds_metadata = get_handler_metadata(job_context.org_name, infer_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, infer_ds_metadata.get('workspace'), workspace_cache)
        infer_root = f"{workspace_identifier}{infer_ds_metadata.get('cloud_file_path')}"
        if job_context.action == "inference":
            config["dataset"]["infer_data_sources"] = {}
            config["dataset"]["infer_data_sources"]["image_dir"] = [os.path.join(infer_root, "images.tar.gz")]
            config["dataset"]["infer_data_sources"]["classmap"] = os.path.join(infer_root, "label_map.txt")

    # calibration dataset
    calib_ds = handler_metadata.get("calibration_dataset", None)
    if calib_ds and job_context.action == "gen_trt_engine":
        calib_ds_metadata = get_handler_metadata(job_context.org_name, calib_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, calib_ds_metadata.get('workspace'), workspace_cache)
        config["gen_trt_engine"]["tensorrt"]["calibration"]["cal_image_dir"] = [f"{workspace_identifier}{calib_ds_metadata.get('cloud_file_path')}" + "/images.tar.gz"]

    if job_context.action != "train":
        if "pretrained_backbone_path" in config["model"].keys():
            del config["model"]["pretrained_backbone_path"]
        if "resume_training_checkpoint_path" in config["train"].keys():
            del config["train"]["resume_training_checkpoint_path"]

    return config


def mal(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for MAL"""
    workspace_cache = {}
    # Train dataset
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_ds_metadata = get_handler_metadata(job_context.org_name, train_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, train_ds_metadata.get('workspace'), workspace_cache)
        train_root = f"{workspace_identifier}{train_ds_metadata.get('cloud_file_path')}"
        if job_context.action in ("evaluate", "inference", "train"):
            if "dataset" not in config.keys():
                config["dataset"] = {}
            config["dataset"]["train_img_dir"] = os.path.join(train_root, "images.tar.gz")
            config["dataset"]["train_ann_path"] = os.path.join(train_root, "annotations.json")

    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        eval_ds_metadata = get_handler_metadata(job_context.org_name, eval_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, eval_ds_metadata.get('workspace'), workspace_cache)
        eval_root = f"{workspace_identifier}{eval_ds_metadata.get('cloud_file_path')}"
        if job_context.action in ("evaluate", "inference", "train"):
            if "dataset" not in config.keys():
                config["dataset"] = {}
            config["dataset"]["val_img_dir"] = os.path.join(eval_root, "images.tar.gz")
            config["dataset"]["val_ann_path"] = os.path.join(eval_root, "annotations.json")

    # Inference dataset
    infer_ds = handler_metadata.get("inference_dataset", None)
    if infer_ds is not None:
        infer_ds_metadata = get_handler_metadata(job_context.org_name, infer_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, infer_ds_metadata.get('workspace'), workspace_cache)
        infer_root = f"{workspace_identifier}{infer_ds_metadata.get('cloud_file_path')}"
        if job_context.action == "inference":
            if "inference" not in config.keys():
                config["inference"] = {}
            config["inference"]["img_dir"] = os.path.join(infer_root, "images.tar.gz")
            config["inference"]["ann_path"] = os.path.join(infer_root, "annotations.json")

    return config


dino = deformable_detr
instance_segmentation = object_detection


def ml_recog(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for Metric Learning Recognition"""
    workspace_cache = {}
    # Train dataset
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_ds_metadata = get_handler_metadata(job_context.org_name, train_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, train_ds_metadata.get('workspace'), workspace_cache)
        train_root = f"{workspace_identifier}{train_ds_metadata.get('cloud_file_path')}"
        if "dataset" not in config.keys():
            config["dataset"] = {}
        if job_context.action == "train":
            config["dataset"]["train_dataset"] = f"{train_root}/metric_learning_recognition/retail-product-checkout-dataset_classification_demo/known_classes/train.tar.gz"
            config["dataset"]["val_dataset"] = {
                "reference": f"{train_root}/metric_learning_recognition/retail-product-checkout-dataset_classification_demo/known_classes/reference.tar.gz",
                "query": f"{train_root}/metric_learning_recognition/retail-product-checkout-dataset_classification_demo/known_classes/val.tar.gz"}
        if job_context.action == "evaluate":
            config["dataset"]["val_dataset"] = {
                "reference": f"{train_root}/metric_learning_recognition/retail-product-checkout-dataset_classification_demo/unknown_classes/reference.tar.gz",
                "query": f"{train_root}/metric_learning_recognition/retail-product-checkout-dataset_classification_demo/unknown_classes/test.tar.gz"}
        if job_context.action == "inference":
            if "inference" not in config.keys():
                config["inference"] = {}
            config["dataset"]["val_dataset"] = {
                "reference": f"{train_root}/metric_learning_recognition/retail-product-checkout-dataset_classification_demo/unknown_classes/reference.tar.gz",
                "query": ""}
            config["inference"]["input_path"] = f"{train_root}/metric_learning_recognition/retail-product-checkout-dataset_classification_demo/unknown_classes/test.tar.gz"

    # calibration dataset
    calib_ds = handler_metadata.get("calibration_dataset", None)
    if calib_ds and job_context.action == "gen_trt_engine":
        calib_ds_metadata = get_handler_metadata(job_context.org_name, calib_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, calib_ds_metadata.get('workspace'), workspace_cache)
        config["gen_trt_engine"]["tensorrt"]["calibration"]["cal_image_dir"] = [f"{workspace_identifier}{calib_ds_metadata.get('cloud_file_path')}" + "/metric_learning_recognition/retail-product-checkout-dataset_classification_demo/known_classes/test.tar.gz"]
    return config


def ocdnet(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for OCDNET"""
    workspace_cache = {}
    parent_action = get_handler_job_metadata(job_context.org_name, job_context.handler_id, job_context.parent_id).get("action")
    if parent_action == "retrain" or job_context.action == "retrain":
        config["model"]["load_pruned_graph"] = True
    # Train dataset
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_ds_metadata = get_handler_metadata(job_context.org_name, train_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, train_ds_metadata.get('workspace'), workspace_cache)
        train_root = f"{workspace_identifier}{train_ds_metadata.get('cloud_file_path')}"
        if "dataset" not in config.keys():
            config["dataset"] = {}
            config["dataset"]["train_dataset"] = {}
            config["dataset"]["validate_dataset"] = {}
        if job_context.action in ("train", "retrain"):
            config["dataset"]["train_dataset"]["data_path"] = [os.path.join(train_root, "train.tar.gz")]

    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        eval_ds_metadata = get_handler_metadata(job_context.org_name, eval_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, eval_ds_metadata.get('workspace'), workspace_cache)
        eval_root = f"{workspace_identifier}{eval_ds_metadata.get('cloud_file_path')}"
        if "dataset" not in config.keys():
            config["dataset"] = {}
            config["dataset"]["train_dataset"] = {}
            config["dataset"]["validate_dataset"] = {}
        config["dataset"]["validate_dataset"]["data_path"] = [os.path.join(eval_root, "test.tar.gz")]
        if job_context.action == "inference":
            if "inference" not in config.keys():
                config["inference"] = {}
            config["inference"]["input_folder"] = os.path.join(eval_root, "test/img.tar.gz")

    # calibration dataset
    calib_ds = handler_metadata.get("calibration_dataset", None)
    if calib_ds and job_context.action == "gen_trt_engine":
        calib_ds_metadata = get_handler_metadata(job_context.org_name, calib_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, calib_ds_metadata.get('workspace'), workspace_cache)
        config["gen_trt_engine"]["tensorrt"]["calibration"]["cal_image_dir"] = f"{workspace_identifier}{calib_ds_metadata.get('cloud_file_path')}" + "/train/img.tar.gz"
    return config


def ocrnet(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for OCRNET"""
    workspace_cache = {}
    if job_context.action == "dataset_convert":
        # ds = handler_metadata.get("id")
        intent = handler_metadata.get("use_for")
        format = ""
        if intent == ["training"]:
            format = "train"
        elif intent == ["evaluation"]:
            format = "test"
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, handler_metadata.get('workspace'), workspace_cache)
        root = f"{workspace_identifier}{handler_metadata.get('cloud_file_path')}"
        if "dataset_convert" not in config.keys():
            config["dataset_convert"] = {}
        config["dataset_convert"]["input_img_dir"] = f"{root}/{format}.tar.gz"
        config["dataset_convert"]["gt_file"] = f"{root}/{format}/gt_new.txt"

    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        eval_ds_metadata = get_handler_metadata(job_context.org_name, eval_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, eval_ds_metadata.get('workspace'), workspace_cache)
        eval_root = f"{workspace_identifier}{eval_ds_metadata.get('cloud_file_path')}"
        eval_ds_convert_job_id = get_job_id_of_action(job_context.org_name, eval_ds, kind="datasets", action="dataset_convert")
        eval_ds_convert_root = f"{workspace_identifier}results/{eval_ds_convert_job_id}"

    # Train dataset
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_ds_metadata = get_handler_metadata(job_context.org_name, train_ds, kind="datasets")
        if "dataset" not in config.keys():
            config["dataset"] = {}
        if job_context.action in ("train", "retrain"):
            train_ds_convert_job_id = get_job_id_of_action(job_context.org_name, train_ds, kind="datasets", action="dataset_convert")
            workspace_identifier = get_workspace_string_identifier(job_context.org_name, train_ds_metadata.get('workspace'), workspace_cache)
            train_ds_convert_root = f"{workspace_identifier}results/{train_ds_convert_job_id}"
            config["dataset"]["train_dataset_dir"] = [os.path.join(train_ds_convert_root, "dataset_convert/lmdb")]
            config["dataset"]["val_dataset_dir"] = os.path.join(eval_ds_convert_root, "dataset_convert/lmdb")
        config["dataset"]["character_list_file"] = f"{eval_root}/character_list"

    if eval_ds is not None:
        if job_context.action == "evaluate":
            if "evaluate" not in config.keys():
                config["evaluate"] = {}
            config["evaluate"]["test_dataset_dir"] = os.path.join(eval_ds_convert_root, "dataset_convert/lmdb")
        if job_context.action == "inference":
            if "inference" not in config.keys():
                config["inference"] = {}
            config["inference"]["inference_dataset_dir"] = os.path.join(eval_root, "test.tar.gz")

    # calibration dataset
    calib_ds = handler_metadata.get("calibration_dataset", None)
    if calib_ds and job_context.action == "gen_trt_engine":
        calib_ds_metadata = get_handler_metadata(job_context.org_name, calib_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, calib_ds_metadata.get('workspace'), workspace_cache)
        config["gen_trt_engine"]["tensorrt"]["calibration"]["cal_image_dir"] = [f"{workspace_identifier}{calib_ds_metadata.get('cloud_file_path')}" + "/train.tar.gz"]
        if calib_ds == eval_ds:
            config["gen_trt_engine"]["tensorrt"]["calibration"]["cal_image_dir"] = [f"{workspace_identifier}{calib_ds_metadata.get('cloud_file_path')}" + "/test.tar.gz"]
    return config


def optical_inspection(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for optical inspection"""
    workspace_cache = {}
    # Train dataset
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_ds_metadata = get_handler_metadata(job_context.org_name, train_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, train_ds_metadata.get('workspace'), workspace_cache)
        train_root = f"{workspace_identifier}{train_ds_metadata.get('cloud_file_path')}"
        if "dataset" not in config.keys():
            config["dataset"] = {}
        if "train_dataset" not in config["dataset"].keys():
            config["dataset"]["train_dataset"] = {}
        config["dataset"]["train_dataset"]["images_dir"] = os.path.join(train_root, "images.tar.gz")
        config["dataset"]["train_dataset"]["csv_path"] = os.path.join(train_root, "dataset.csv")

    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        eval_ds_metadata = get_handler_metadata(job_context.org_name, eval_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, eval_ds_metadata.get('workspace'), workspace_cache)
        eval_root = f"{workspace_identifier}{eval_ds_metadata.get('cloud_file_path')}"
        if "dataset" not in config.keys():
            config["dataset"] = {}
        if "validation_dataset" not in config["dataset"].keys():
            config["dataset"]["validation_dataset"] = {}
        if "test_dataset" not in config["dataset"].keys():
            config["dataset"]["test_dataset"] = {}
        config["dataset"]["validation_dataset"]["images_dir"] = os.path.join(eval_root, "images.tar.gz")
        config["dataset"]["validation_dataset"]["csv_path"] = os.path.join(eval_root, "dataset.csv")
        config["dataset"]["test_dataset"]["images_dir"] = os.path.join(eval_root, "images.tar.gz")
        config["dataset"]["test_dataset"]["csv_path"] = os.path.join(eval_root, "dataset.csv")

    # Inference dataset
    infer_ds = handler_metadata.get("inference_dataset", None)
    if infer_ds is not None:
        infer_ds_metadata = get_handler_metadata(job_context.org_name, infer_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, infer_ds_metadata.get('workspace'), workspace_cache)
        infer_root = f"{workspace_identifier}{infer_ds_metadata.get('cloud_file_path')}"
        if "dataset" not in config.keys():
            config["dataset"] = {}
        if "infer_dataset" not in config["dataset"].keys():
            config["dataset"]["infer_dataset"] = {}
        config["dataset"]["infer_dataset"]["images_dir"] = os.path.join(infer_root, "images.tar.gz")
        config["dataset"]["infer_dataset"]["csv_path"] = os.path.join(infer_root, "dataset.csv")

    # calibration dataset
    calib_ds = handler_metadata.get("calibration_dataset", None)
    if calib_ds and job_context.action == "gen_trt_engine":
        calib_ds_metadata = get_handler_metadata(job_context.org_name, calib_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, calib_ds_metadata.get('workspace'), workspace_cache)
        config["gen_trt_engine"]["tensorrt"]["calibration"]["cal_image_dir"] = f"{workspace_identifier}{calib_ds_metadata.get('cloud_file_path')}" + "/images.tar.gz"
    return config


def centerpose(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for CenterPose"""
    workspace_cache = {}
    # Train dataset
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_ds_metadata = get_handler_metadata(job_context.org_name, train_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, train_ds_metadata.get('workspace'), workspace_cache)
        train_root = f"{workspace_identifier}{train_ds_metadata.get('cloud_file_path')}"
        if job_context.action == "train":
            config["dataset"]["train_data"] = {}
            config["dataset"]["train_data"] = os.path.join(train_root, 'train.tar.gz')

    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        eval_ds_metadata = get_handler_metadata(job_context.org_name, eval_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, eval_ds_metadata.get('workspace'), workspace_cache)
        eval_root = f"{workspace_identifier}{eval_ds_metadata.get('cloud_file_path')}"
        if job_context.action == "train":
            config["dataset"]["val_data"] = {}
            config["dataset"]["val_data"] = os.path.join(eval_root, "val.tar.gz")
        if job_context.action == "evaluate":
            config["dataset"]["test_data"] = {}
            config["dataset"]["test_data"] = os.path.join(eval_root, "test.tar.gz")

    # Inference dataset
    infer_ds = handler_metadata.get("inference_dataset", None)
    if infer_ds is not None:
        infer_ds_metadata = get_handler_metadata(job_context.org_name, infer_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, infer_ds_metadata.get('workspace'), workspace_cache)
        infer_root = f"{workspace_identifier}{infer_ds_metadata.get('cloud_file_path')}"
        if job_context.action == "inference":
            config["dataset"]["inference_data"] = {}
            config["dataset"]["inference_data"] = os.path.join(infer_root, "val.tar.gz")

    # calibration dataset
    calib_ds = handler_metadata.get("calibration_dataset", None)
    if calib_ds and job_context.action == "gen_trt_engine":
        calib_ds_metadata = get_handler_metadata(job_context.org_name, calib_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, calib_ds_metadata.get('workspace'), workspace_cache)
        config["gen_trt_engine"]["tensorrt"]["calibration"]["cal_image_dir"] = f"{workspace_identifier}{calib_ds_metadata.get('cloud_file_path')}" + "/train.tar.gz"

    if job_context.action != "train":
        if "backbone" in config["model"].keys():
            if "pretrained_backbone_path" in config["model"]["backbone"].keys():
                del config["model"]["backbone"]["pretrained_backbone_path"]
        if "train" in config.keys() and "resume_training_checkpoint_path" in config["train"].keys():
            del config["train"]["resume_training_checkpoint_path"]

    return config


def visual_changenet_classify(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for visual_changenet_classify"""
    workspace_cache = {}
    # Train dataset
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_ds_metadata = get_handler_metadata(job_context.org_name, train_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, train_ds_metadata.get('workspace'), workspace_cache)
        train_root = f"{workspace_identifier}{train_ds_metadata.get('cloud_file_path')}"
        if "dataset" not in config.keys():
            config["dataset"] = {}
        if "classify" not in config["dataset"].keys():
            config["dataset"]["classify"] = {}
        if "train_dataset" not in config["dataset"]["classify"].keys():
            config["dataset"]["classify"]["train_dataset"] = {}
        config["dataset"]["classify"]["train_dataset"]["images_dir"] = os.path.join(train_root, "images.tar.gz")
        config["dataset"]["classify"]["train_dataset"]["csv_path"] = os.path.join(train_root, "dataset.csv")

    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        eval_ds_metadata = get_handler_metadata(job_context.org_name, eval_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, eval_ds_metadata.get('workspace'), workspace_cache)
        eval_root = f"{workspace_identifier}{eval_ds_metadata.get('cloud_file_path')}"
        if "dataset" not in config.keys():
            config["dataset"] = {}
        if "classify" not in config["dataset"].keys():
            config["dataset"]["classify"] = {}
        if "validation_dataset" not in config["dataset"]["classify"].keys():
            config["dataset"]["classify"]["validation_dataset"] = {}
        if "test_dataset" not in config["dataset"]["classify"].keys():
            config["dataset"]["classify"]["test_dataset"] = {}
        config["dataset"]["classify"]["validation_dataset"]["images_dir"] = os.path.join(eval_root, "images.tar.gz")
        config["dataset"]["classify"]["validation_dataset"]["csv_path"] = os.path.join(eval_root, "dataset.csv")
        config["dataset"]["classify"]["test_dataset"]["images_dir"] = os.path.join(eval_root, "images.tar.gz")
        config["dataset"]["classify"]["test_dataset"]["csv_path"] = os.path.join(eval_root, "dataset.csv")

    # Inference dataset
    infer_ds = handler_metadata.get("inference_dataset", None)
    if infer_ds is not None:
        infer_ds_metadata = get_handler_metadata(job_context.org_name, infer_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, infer_ds_metadata.get('workspace'), workspace_cache)
        infer_root = f"{workspace_identifier}{infer_ds_metadata.get('cloud_file_path')}"
        if "dataset" not in config.keys():
            config["dataset"] = {}
        if "classify" not in config["dataset"].keys():
            config["dataset"]["classify"] = {}
        if "infer_dataset" not in config["dataset"]["classify"].keys():
            config["dataset"]["classify"]["infer_dataset"] = {}
        config["dataset"]["classify"]["infer_dataset"]["images_dir"] = os.path.join(infer_root, "images.tar.gz")
        config["dataset"]["classify"]["infer_dataset"]["csv_path"] = os.path.join(infer_root, "dataset.csv")

    # calibration dataset
    calib_ds = handler_metadata.get("calibration_dataset", None)
    if calib_ds and job_context.action == "gen_trt_engine":
        calib_ds_metadata = get_handler_metadata(job_context.org_name, calib_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, calib_ds_metadata.get('workspace'), workspace_cache)
        config["gen_trt_engine"]["tensorrt"]["calibration"]["cal_image_dir"] = f"{workspace_identifier}{calib_ds_metadata.get('cloud_file_path')}" + "/images.tar.gz"
    return config


def visual_changenet(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for visual changenet"""
    workspace_cache = {}
    # Train dataset
    if handler_metadata.get("train_datasets", []) != []:
        print("Warning: checking handler visual changenet", handler_metadata.get("train_datasets"), file=sys.stderr)
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_dataset_metadata = get_handler_metadata(job_context.org_name, train_ds)
        if train_dataset_metadata.get('format') == 'visual_changenet_classify':
            return visual_changenet_classify(config, job_context, handler_metadata)
        train_ds_metadata = get_handler_metadata(job_context.org_name, train_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, train_ds_metadata.get('workspace'), workspace_cache)
        train_root = f"{workspace_identifier}{train_ds_metadata.get('cloud_file_path')}"
        if "dataset" not in config.keys():
            config["dataset"] = {}
        if "segment" not in config["dataset"].keys():
            config["dataset"]["segment"] = {}
        config["dataset"]["segment"]["root_dir"] = train_root

    return config


def analytics(config, job_context, handler_metadata):
    """Function to create data sources for analytics module"""
    workspace_cache = {}
    workspace_identifier = get_workspace_string_identifier(job_context.org_name, handler_metadata.get('workspace'), workspace_cache)
    config["data"]["image_dir"] = f"{workspace_identifier}{handler_metadata.get('cloud_file_path')}" + "/images.tar.gz"
    if config["data"]["input_format"] == "COCO":
        config["data"]["ann_path"] = f"{workspace_identifier}{handler_metadata.get('cloud_file_path')}" + "/annotations.json"  # Annotation format conversion action's parent id
        if job_context.parent_id:
            parent_job_metadata = get_handler_job_metadata(job_context.org_name, handler_metadata.get("id"), job_context.parent_id, kind="datasets")
            parent_action = parent_job_metadata.get("action")
            if parent_action == "annotation_format_convert":
                output_format = parent_job_metadata.get("specs", {}).get("data", {}).get("output_format")
                if output_format == "COCO":
                    config["data"]["ann_path"] = f"{workspace_identifier}results/{job_context.parent_id}" + "/annotations.json"
    elif config["data"]["input_format"] == "KITTI":
        config["data"]["ann_path"] = f"{workspace_identifier}{handler_metadata.get('cloud_file_path')}" + "/labels.tar.gz"
    return config


def data_services_image(config, job_context, handler_metadata):
    """Function to create data sources for data_service's image module"""
    workspace_cache = {}
    infer_ds = handler_metadata.get("inference_dataset", None)
    infer_ds_metadata = get_handler_metadata(job_context.org_name, infer_ds, kind="datasets")
    workspace_identifier = get_workspace_string_identifier(job_context.org_name, infer_ds_metadata.get('workspace'), workspace_cache)
    infer_root = f"{workspace_identifier}{infer_ds_metadata.get('cloud_file_path')}"
    if "data" not in config.keys():
        config["data"] = {}
    config["data"]["image_dir"] = infer_root
    return config


def annotations(config, job_context, handler_metadata):
    """Function to create data sources for annotations module"""
    workspace_cache = {}
    workspace_identifier = get_workspace_string_identifier(job_context.org_name, handler_metadata.get('workspace'), workspace_cache)
    if config["data"]["input_format"] == "COCO":
        if "coco" not in config.keys():
            config["coco"] = {}
        config["coco"]["ann_file"] = f"{workspace_identifier}{handler_metadata.get('cloud_file_path')}" + "/annotations.json"
    elif config["data"]["input_format"] == "KITTI":
        if "kitti" not in config.keys():
            config["kitti"] = {}
        config["kitti"]["image_dir"] = f"{workspace_identifier}{handler_metadata.get('cloud_file_path')}" + "/images.tar.gz"
        config["kitti"]["label_dir"] = f"{workspace_identifier}{handler_metadata.get('cloud_file_path')}" + "/labels.tar.gz"
    return config


def augmentation(config, job_context, handler_metadata):
    """Function to create data sources for augmentation module"""
    workspace_cache = {}
    infer_ds = handler_metadata["id"]
    if infer_ds:
        infer_ds_metadata = get_handler_metadata(job_context.org_name, infer_ds, kind="datasets")
        workspace_identifier = get_workspace_string_identifier(job_context.org_name, infer_ds_metadata.get('workspace'), workspace_cache)
        config["data"]["image_dir"] = f"{workspace_identifier}{infer_ds_metadata.get('cloud_file_path')}" + "/images.tar.gz"
        if config["data"]["dataset_type"] == "kitti":
            config["data"]["ann_path"] = f"{workspace_identifier}{infer_ds_metadata.get('cloud_file_path')}" + "/labels.tar.gz"
            config["spatial_aug"]["rotation"]["refine_box"]["gt_cache"] = os.path.join(f"{workspace_identifier}{infer_ds_metadata.get('cloud_file_path')}", "label.json")
        elif config["data"]["dataset_type"] == "coco":
            config["data"]["ann_path"] = f"{workspace_identifier}{infer_ds_metadata.get('cloud_file_path')}" + "/annotations.json"
            if job_context.parent_id:
                auto_label_exp_id = get_handler_id(job_context.org_name, job_context.parent_id)
                parent_job_metadata = get_handler_job_metadata(job_context.org_name, auto_label_exp_id, job_context.parent_id, kind="experiments")  # Auto label action's parent id
                parent_action = parent_job_metadata.get("action")
                if parent_action == "generate":
                    config["data"]["ann_path"] = f"{workspace_identifier}results/{job_context.parent_id}" + "/label.json"
            config["spatial_aug"]["rotation"]["refine_box"]["gt_cache"] = config["data"]["ann_path"]
    return config


def auto_label(config, job_context, handler_metadata):
    """Function to create data sources for auto_label module"""
    workspace_cache = {}
    infer_ds = handler_metadata["inference_dataset"]
    infer_ds_metadata = get_handler_metadata(job_context.org_name, infer_ds, kind="datasets")
    workspace_identifier = get_workspace_string_identifier(job_context.org_name, infer_ds_metadata.get('workspace'), workspace_cache)
    config["mal"]["inference"]["img_dir"] = f"{workspace_identifier}{infer_ds_metadata.get('cloud_file_path')}" + "/images.tar.gz"
    config["mal"]["inference"]["ann_path"] = f"{workspace_identifier}{infer_ds_metadata.get('cloud_file_path')}" + "/annotations.json"
    if job_context.parent_id:
        parent_job_metadata = get_handler_job_metadata(job_context.org_name, infer_ds, job_context.parent_id, kind="datasets")  # Annotation format conversion action's parent id
        parent_action = parent_job_metadata.get("action")
        if parent_action == "annotation_format_convert":
            output_format = parent_job_metadata.get("specs", {}).get("data", {}).get("output_format")
            if output_format == "COCO":
                config["mal"]["inference"]["ann_path"] = f"{workspace_identifier}results/{job_context.parent_id}" + "/annotations.json"
    config["mal"]["inference"]["label_dump_path"] = f"/results/{job_context.id}/label.json"
    return config


def prepare_job_datalist(config, job_context, handler_metadata):
    """Function to create data sources for medical bundles."""
    job_action = job_context.action
    train_datasets = handler_metadata["train_datasets"]
    if len(train_datasets) > 1:
        raise ValueError(f"Only one train dataset is supported, but {len(train_datasets)} are given.")
    datasets = {}
    datasets['train'] = train_datasets[0] if len(train_datasets) == 1 else None
    datasets['validate'] = handler_metadata["eval_dataset"]
    datasets['batchinfer'] = handler_metadata["inference_dataset"]
    _, _, overridden_output_dir = CLI_CONFIG_TO_FUNCTIONS["medical_output_dir"](job_context, handler_metadata)
    local_job = (job_context.specs and "cluster" in job_context.specs and job_context.specs["cluster"] == "local")
    if job_action == "train":
        datasets.pop("batchinfer")

    if job_action in ["batchinfer", "generate"]:
        datasets.pop("train")
        datasets.pop("validate")

    # Workaround for the ci test.
    from handlers.medical.dataset.dicom import DicomEndpoint
    config["datasets_info"] = {}
    for dataset_usage, dataset_id in datasets.items():
        if dataset_id is None:
            continue

        datalist_json = os.path.join(overridden_output_dir, f"{dataset_usage}_datalist.json")
        dataset_metadata = get_handler_metadata(job_context.org_name, dataset_id, "datasets")
        endpoint = MonaiDatasetHandler.endpoint(dataset_metadata)
        config[f"{dataset_usage}#dataset#data"] = f"%{datalist_json}"
        config["datasets_info"][dataset_usage] = {"url": endpoint.url,
                                                  "client_id": endpoint.client_id,
                                                  "client_secret": endpoint.client_secret,
                                                  "secret_env": f"{dataset_usage.upper()}_SECRET",
                                                  "is_dicom": False,
                                                  "filepath": tempfile.TemporaryDirectory().name,  # pylint: disable=R1732
                                                  "datalist_path": datalist_json,
                                                  "skip_label": False}
        if job_action == "batchinfer":
            config["dataset#data"] = f"%{datalist_json}"
            # Skip the lable check.
            config["datasets_info"][dataset_usage]["skip_label"] = True

        if isinstance(endpoint, DicomEndpoint):
            if not local_job:
                raise RuntimeError("Batch actions only support object storage datasets.")
            config["datasets_info"][dataset_usage]["is_dicom"] = True

    return config


DS_CONFIG_TO_FUNCTIONS = {"detectnet_v2": detectnet_v2,
                          "unet": unet,
                          "segformer": segformer,
                          "efficientdet_tf2": efficientdet_tf2,
                          "classification_pyt": classification_pyt,
                          "classification_tf2": classification_tf2,
                          "action_recognition": action_recognition,
                          "mal": mal,
                          "ml_recog": ml_recog,
                          "medical_annotation": prepare_job_datalist,
                          "medical_vista3d": prepare_job_datalist,
                          "medical_vista2d": prepare_job_datalist,
                          "medical_automl": prepare_job_datalist,
                          "medical_automl_generated": prepare_job_datalist,
                          "medical_custom": prepare_job_datalist,
                          "medical_classification": prepare_job_datalist,
                          "medical_detection": prepare_job_datalist,
                          "medical_segmentation": prepare_job_datalist,
                          "medical_genai": prepare_job_datalist,
                          "medical_maisi": prepare_job_datalist,
                          "ocdnet": ocdnet,
                          "ocrnet": ocrnet,
                          "optical_inspection": optical_inspection,
                          "pointpillars": pointpillars,
                          "pose_classification": pose_classification,
                          "re_identification": re_identification,
                          "deformable_detr": deformable_detr,
                          "dino": dino,
                          "object_detection": object_detection,
                          "centerpose": centerpose,
                          "instance_segmentation": instance_segmentation,
                          "analytics": analytics,
                          "annotations": annotations,
                          "augmentation": augmentation,
                          "image": data_services_image,
                          "auto_label": auto_label,
                          "visual_changenet": visual_changenet}
