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
import glob
import json
import sys
import shutil

from handlers.medical_dataset_handler import MedicalDatasetHandler
from handlers.stateless_handlers import get_handler_root, get_handler_metadata, get_handler_job_metadata, get_handler_user, safe_load_file, safe_dump_file
from handlers.infer_params import CLI_CONFIG_TO_FUNCTIONS


def detectnet_v2(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for Detecnet v2"""
    # Creates data sources based on what is available
    # if eval is not given, train could fail
    # this is because by defn, we want to use all "train" data for learning
    # a model

    # Init
    if "dataset_config" not in list(config.keys()):
        config["dataset_config"] = {}

    # Training datasets
    if handler_metadata.get("train_datasets", []) != []:
        config["dataset_config"]["data_sources"] = []
        for train_ds in handler_metadata.get("train_datasets", []):
            ds_source_dict = {}
            ds_source_dict["tfrecords_path"] = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True) + "/tfrecords/*"
            ds_source_dict["image_directory_path"] = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True) + "/"
            config["dataset_config"]["data_sources"].append(ds_source_dict)

    if config["dataset_config"].get("validation_fold") is not None:
        del config["dataset_config"]["validation_fold"]
    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        config["dataset_config"]["validation_data_source"] = {}
        config["dataset_config"]["validation_data_source"]["tfrecords_path"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/tfrecords/*"
        config["dataset_config"]["validation_data_source"]["image_directory_path"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/"

    return config


def unet(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for Unet"""
    # Init
    if "dataset_config" not in list(config.keys()):
        config["dataset_config"] = {}

    # Training datasets
    if handler_metadata.get("train_datasets", []) != []:
        for train_ds in handler_metadata.get("train_datasets", []):
            config["dataset_config"]["train_masks_path"] = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True) + "/masks/train"
            config["dataset_config"]["train_images_path"] = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True) + "/images/train"
    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        config["dataset_config"]["val_masks_path"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/masks/val"
        config["dataset_config"]["val_images_path"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/images/val"

    # Infer dataset
    infer_ds = handler_metadata.get("inference_dataset", None)
    if infer_ds is not None:
        config["dataset_config"]["test_images_path"] = get_handler_root(job_context.user_id, "datasets", infer_ds, None, ngc_runner_fetch=True) + "/images/test"

    return config


def segformer(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for Segformer"""
    # Init
    if "dataset" not in list(config.keys()):
        config["dataset"] = {}

    # Training datasets
    if handler_metadata.get("train_datasets", []) != []:
        for train_ds in handler_metadata.get("train_datasets", []):
            if "train_dataset" not in config["dataset"].keys():
                config["dataset"]["train_dataset"] = {}
            if config["dataset"]["train_dataset"].get("ann_dir", None):
                config["dataset"]["train_dataset"]["ann_dir"].append(get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True) + "/masks/train")
                config["dataset"]["train_dataset"]["img_dir"].append(get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True) + "/images/train")
            else:
                config["dataset"]["train_dataset"]["ann_dir"] = [get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True) + "/masks/train"]
                config["dataset"]["train_dataset"]["img_dir"] = [get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True) + "/images/train"]
    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        if job_context.action == "train":
            eval_key = "val_dataset"
        else:
            eval_key = "test_dataset"
        config["dataset"][eval_key] = {}
        config["dataset"][eval_key]["ann_dir"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/masks/val"
        config["dataset"][eval_key]["img_dir"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/images/val"
    return config


faster_rcnn = detectnet_v2


def yolo_v4(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for Yolo_v4"""
    # Identical to detectnet_v2: validation_data_sources instead of validation_data_source

    # Init
    if "dataset_config" not in list(config.keys()):
        config["dataset_config"] = {}

    # Training datasets
    if handler_metadata.get("train_datasets", []) != []:
        config["dataset_config"]["data_sources"] = []
        for train_ds in handler_metadata.get("train_datasets", []):
            ds_source_dict = {}
            ds_source_dict["tfrecords_path"] = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True) + "/tfrecords/*"
            ds_source_dict["image_directory_path"] = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True) + "/"
            config["dataset_config"]["data_sources"].append(ds_source_dict)

    if config["dataset_config"].get("validation_fold") is not None:
        del config["dataset_config"]["validation_fold"]
    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        config["dataset_config"]["validation_data_sources"] = {}
        config["dataset_config"]["validation_data_sources"]["tfrecords_path"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/tfrecords/*"
        config["dataset_config"]["validation_data_sources"]["image_directory_path"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/"

    return config


yolo_v3 = yolo_v4
yolo_v4_tiny = yolo_v4


def ssd(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for SSD"""
    # Identical to yolo_v4: tfrecords_path ends with -* as opposed to *

    # Init
    if "dataset_config" not in list(config.keys()):
        config["dataset_config"] = {}

    # Training datasets
    if handler_metadata.get("train_datasets", []) != []:
        config["dataset_config"]["data_sources"] = []
        for train_ds in handler_metadata.get("train_datasets", []):
            ds_source_dict = {}
            ds_source_dict["tfrecords_path"] = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True) + "/tfrecords/tfrecords-*"
            # ds_source_dict["image_directory_path"] =  get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True)+"/"
            config["dataset_config"]["data_sources"].append(ds_source_dict)

    if config["dataset_config"].get("validation_fold") is not None:
        del config["dataset_config"]["validation_fold"]
    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        config["dataset_config"]["validation_data_sources"] = {}
        config["dataset_config"]["validation_data_sources"]["label_directory_path"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/labels"
        config["dataset_config"]["validation_data_sources"]["image_directory_path"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/images"

    return config


retinanet = ssd
dssd = ssd


def object_detection(config, job_context, handler_metadata):
    """Returns config directly as no changes are required"""
    return config


def lprnet(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for LPRNet"""
    # Assumes every train dataset and the eval dataset - all have same characters.txt

    # Init
    if "dataset_config" not in list(config.keys()):
        config["dataset_config"] = {}

    # Training datasets
    if handler_metadata.get("train_datasets", []) != []:
        config["dataset_config"]["data_sources"] = []
        for train_ds in handler_metadata.get("train_datasets", []):
            ds_source_dict = {}
            ds_source_dict["label_directory_path"] = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True) + "/label"
            ds_source_dict["image_directory_path"] = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True) + "/image"
            config["dataset_config"]["data_sources"].append(ds_source_dict)
            config["dataset_config"]["characters_list_file"] = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True) + "/characters.txt"

    if config["dataset_config"].get("validation_fold") is not None:
        del config["dataset_config"]["validation_fold"]
    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        config["dataset_config"]["validation_data_sources"] = {}
        config["dataset_config"]["validation_data_sources"]["label_directory_path"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/label"
        config["dataset_config"]["validation_data_sources"]["image_directory_path"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/image"
        config["dataset_config"]["characters_list_file"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/characters.txt"

    return config


def efficientdet_tf1(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for EfficientDetTf1"""
    # Init
    if "dataset_config" not in list(config.keys()):
        config["dataset_config"] = {}

    # Training datasets
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        print("Warning: EfficientDet supports only one train dataset", file=sys.stderr)
        config["dataset_config"]["training_file_pattern"] = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True) + "/tfrecords/*.tfrecord"

    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        config["dataset_config"]["validation_file_pattern"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/tfrecords/*.tfrecord"
        config["dataset_config"]["validation_json_file"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/annotations.json"

    return config


def efficientdet_tf2(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for EfficientDet tf2"""
    # Init
    if "data" not in list(config.keys()):
        config["dataset"] = {}

    # Training datasets
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        print("Warning: EfficientDet supports only one train dataset", file=sys.stderr)
        handler_root = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True)
        parent_dir = os.path.dirname(glob.glob(handler_root + "/**/*.tfrecord", recursive=True)[0])
        config["dataset"]["train_tfrecords"] = [parent_dir + "/*.tfrecord"]

    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        handler_root = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True)
        parent_dir = os.path.dirname(glob.glob(handler_root + "/**/*.tfrecord", recursive=True)[0])
        config["dataset"]["val_tfrecords"] = [parent_dir + "/*.tfrecord"]
        config["dataset"]["val_json_file"] = handler_root + "/annotations.json"

    return config


def mask_rcnn(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for Mask RCNN"""
    # Init
    if "data_config" not in list(config.keys()):
        config["data_config"] = {}

    # Training datasets
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        print("Warning: MaskRCNN supports only one train dataset", file=sys.stderr)
        config["data_config"]["training_file_pattern"] = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True) + "/tfrecords/*.tfrecord"

    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        config["data_config"]["validation_file_pattern"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/tfrecords/*.tfrecord"
        config["data_config"]["val_json_file"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/annotations.json"

    return config


def multitask_classification(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for Multi-task classification"""
    # Init
    if "dataset_config" not in list(config.keys()):
        config["dataset_config"] = {}

    parent_action = get_handler_job_metadata(job_context.user_id, job_context.handler_id, job_context.parent_id).get("action")
    if job_context.action in ("train", "retrain"):
        if handler_metadata.get("train_datasets", []) != []:
            train_ds = handler_metadata.get("train_datasets", [])[0]
            root = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True)
            print("Warning: Multitask Classification supports only one train dataset", file=sys.stderr)
            config["dataset_config"]["train_csv_path"] = root + "/train.csv"
            config["dataset_config"]["val_csv_path"] = root + "/val.csv"
            config["dataset_config"]["image_directory_path"] = root + "/images_train"

    elif job_context.action == "evaluate" or (job_context.action == "inference" and parent_action in ("gen_trt_engine", "trtexec")):
        if handler_metadata.get("eval_dataset", None) is not None:
            eval_ds = handler_metadata.get("eval_dataset", None)
            root = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True)
            config["dataset_config"]["val_csv_path"] = root + "/val.csv"
            config["dataset_config"]["image_directory_path"] = root + "/images_val"

    return config


def classification_tf1(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for Classification-tf1"""
    if "train_config" not in list(config.keys()):
        config["train_config"] = {}
    if "eval_config" not in list(config.keys()):
        config["eval_config"] = {}
    print("Warning: Classification supports only one train dataset", file=sys.stderr)
    print("Warning: Train, eval datasets are both required to run Classification actions - train, evaluate, retrain, inference", file=sys.stderr)
    train_datasets = handler_metadata.get("train_datasets", [])
    if train_datasets != []:
        config["train_config"]["train_dataset_path"] = get_handler_root(job_context.user_id, "datasets", train_datasets[0], None, ngc_runner_fetch=True) + "/images_train"
    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        if os.path.exists(get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/images_val"):
            config["train_config"]["val_dataset_path"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/images_val"
            config["eval_config"]["eval_dataset_path"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/images_val"
        else:
            print("Warning: eval_ds+/images_val does not exist", file=sys.stderr)

    infer_ds = handler_metadata.get("inference_dataset", None)
    if infer_ds is not None:
        if os.path.exists(get_handler_root(job_context.user_id, "datasets", infer_ds, None, ngc_runner_fetch=True) + "/images_test"):
            config["eval_config"]["eval_dataset_path"] = get_handler_root(job_context.user_id, "datasets", infer_ds, None, ngc_runner_fetch=True) + "/images_test"
        else:
            print("Warning: infer_ds+/images_test does not exist", file=sys.stderr)

    return config


def classification_tf2(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for Classification-tf2"""
    if "dataset" not in list(config.keys()):
        config["dataset"] = {}
    if "evaluate" not in list(config.keys()):
        config["evaluate"] = {}

    print("Warning: Classification-tf2 supports only one train dataset", file=sys.stderr)
    print("Warning: Train, eval datasets are both required to run Classification actions - train, evaluate, retrain, inference", file=sys.stderr)
    train_datasets = handler_metadata.get("train_datasets", [])
    if train_datasets != []:
        config["dataset"]["train_dataset_path"] = get_handler_root(job_context.user_id, "datasets", train_datasets[0], None, ngc_runner_fetch=True) + "/images_train"

    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        if os.path.exists(get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/images_val"):
            config["dataset"]["val_dataset_path"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/images_val"
        else:
            print("Warning: eval_ds+/images_val does not exist", file=sys.stderr)

    infer_ds = handler_metadata.get("inference_dataset", None)
    if infer_ds is not None:
        if os.path.exists(get_handler_root(job_context.user_id, "datasets", infer_ds, None, ngc_runner_fetch=True) + "/images_test"):
            config["evaluate"]["dataset_path"] = get_handler_root(job_context.user_id, "datasets", infer_ds, None, ngc_runner_fetch=True) + "/images_test"
        else:
            print("Warning: infer_ds+/images_test does not exist", file=sys.stderr)

    return config


def classification_pyt(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for Classification-pyt"""
    if "val" not in list(config["dataset"]["data"].keys()):
        config["dataset"]["data"]["val"] = {}
    if "test" not in list(config["dataset"]["data"].keys()):
        config["dataset"]["data"]["test"] = {}

    print("Warning: Classification-pyt supports only one train dataset", file=sys.stderr)
    print("Warning: Train, eval datasets are both required to run Classification actions - train, evaluate, inference", file=sys.stderr)
    train_datasets = handler_metadata.get("train_datasets", [])
    if train_datasets != []:
        config["dataset"]["data"]["train"]["data_prefix"] = get_handler_root(job_context.user_id, "datasets", train_datasets[0], None, ngc_runner_fetch=True) + "/images_train"
        config["dataset"]["data"]["train"]["classes"] = get_handler_root(job_context.user_id, "datasets", train_datasets[0], None, ngc_runner_fetch=True) + "/classes.txt"

    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        if os.path.exists(get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/images_val"):
            config["dataset"]["data"]["val"]["data_prefix"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/images_val"
            config["dataset"]["data"]["val"]["classes"] = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True) + "/classes.txt"
        else:
            print("Warning: eval_ds+/images_val does not exist", file=sys.stderr)

    infer_ds = handler_metadata.get("inference_dataset", None)
    if infer_ds is not None:
        if os.path.exists(get_handler_root(job_context.user_id, "datasets", infer_ds, None, ngc_runner_fetch=True) + "/images_test"):
            config["dataset"]["data"]["test"]["data_prefix"] = get_handler_root(job_context.user_id, "datasets", infer_ds, None, ngc_runner_fetch=True) + "/images_test"
            config["dataset"]["data"]["test"]["classes"] = get_handler_root(job_context.user_id, "datasets", infer_ds, None, ngc_runner_fetch=True) + "/classes.txt"
        else:
            print("Warning: infer_ds+/images_test does not exist", file=sys.stderr)

    return config


def bpnet(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for BPNET"""
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        root = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True)

        coco_spec_json = safe_load_file(root + "/coco_spec.json")
        coco_spec_json["root_directory_path"] = root + "/"
        safe_dump_file(root + "/coco_spec.json", coco_spec_json)

        if job_context.action in ("train", "retrain"):
            config["dataloader"]["pose_config"]["pose_config_path"] = root + "/bpnet_18joints.json"
            config["dataloader"]["dataset_config"]["root_data_path"] = root + "/"
            config["dataloader"]["dataset_config"]["train_records_folder_path"] = root + "/"
            config["dataloader"]["dataset_config"]["val_records_folder_path"] = root + "/"
            config["dataloader"]["dataset_config"]["dataset_specs"]["coco"] = root + "/coco_spec.json"
            config["inference_spec"] = root + "/infer_spec.yaml"
    else:
        train_ds = handler_metadata.get("id")
        root = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True)
        config["root_directory_path"] = root + "/"
    return config


def fpenet(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for FPENET"""
    afw_suffix = ""
    if "num_keypoints" in config.keys():
        if config["num_keypoints"] == 10:
            afw_suffix = "_10"
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        root = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True)
        config["dataloader"]["dataset_info"]["tfrecords_directory_path"] = root + "/data/tfrecords/"
        if job_context.action != "export":
            config["dataloader"]["dataset_info"]["tfrecords_set_id_train"] = f"afw{afw_suffix}"
            config["dataloader"]["dataset_info"]["tfrecords_set_id_val"] = f"afw{afw_suffix}"
            config["dataloader"]["kpiset_info"]["tfrecords_set_id_kpi"] = f"afw{afw_suffix}"
            if config["num_keypoints"] == 10:
                config["dataloader"]["augmentation_info"]["modulus_spatial_augmentation"]["hflip_probability"] = 0.0
        if job_context.action == "inference":
            inference_json = safe_load_file(root + "/data.json")

            modified_inference_json = []
            for img_info in inference_json:
                img_info["filename"] = os.path.join(root, "data", "afw", os.path.basename(img_info["filename"]))
                modified_inference_json.append(img_info)

            safe_dump_file(root + "/data.json", modified_inference_json)

    else:
        train_ds = handler_metadata.get("id")
        root = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True)
        config["sets"] = [f"afw{afw_suffix}"]
        config["gt_root_path"] = root + "/"
        config["save_root_path"] = root + "/"
        config["image_root_path"] = root + "/"

    return config


def action_recogntion_dynamic_config(config, action):
    """Dynamically drop out spec parameters based on certain other parameters"""
    model_type = config["model"]["model_type"]  # rgb/of/joint
    input_type = config["model"]["input_type"]  # 3d/2d

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

    config["dataset"]["label_map"] = config["dataset"]["label_map_" + input_type]
    config["dataset"].pop("label_map_2d", None)
    config["dataset"].pop("label_map_3d", None)
    return config


def action_recognition(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params and,
    makes changes to config baed on model and input_type for Action recognition
    """
    config = action_recogntion_dynamic_config(config, job_context.action)
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        root = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True)
        if job_context.action == "train":
            config["dataset"]["train_dataset_dir"] = os.path.join(root, "train")
            config["dataset"]["val_dataset_dir"] = os.path.join(root, "test")
        elif job_context.action == "evaluate":
            config["evaluate"]["test_dataset_dir"] = os.path.join(root, "test")
        elif job_context.action == "inference":
            config["inference"]["inference_dataset_dir"] = os.path.join(root, "test/smile")
    return config


def pointpillars(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for Pointpillars"""
    train_ds = handler_metadata.get("train_datasets", [])
    if train_ds != []:
        train_ds = train_ds[0]
    else:
        train_ds = handler_metadata.get("id")
    root = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True)
    config["dataset"]["data_path"] = root
    return config


def pose_classification_dynamic_config(config, action):
    """Dynamically drop out spec parameters based on certain other parameters"""
    model_type = config["model"]["graph_layout"]  # openpose/nvidia
    if model_type == "nvidia":
        if action == "train":
            config["dataset"].pop("random_choose", None)
            config["dataset"].pop("random_move", None)
            config["dataset"].pop("window_size", None)
        config["dataset"]["label_map"] = config["dataset"]["label_map_nvidia"]
    elif model_type == "openpose":
        if action == "train":
            config["model"].pop("pretrained_model_path", None)
        config["dataset"]["label_map"] = config["dataset"]["label_map_kinetics"]

    config["dataset"].pop("label_map_kinetics", None)
    config["dataset"].pop("label_map_nvidia", None)
    return config


def pose_classification(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params and,
    makes changes to config baed on model type for Pose classification
    """
    model_type = config["model"]["graph_layout"]  # openpose/nvidia
    if model_type == "openpose":
        model_type = "kinetics"
    pose_classification_dynamic_config(config, job_context.action)

    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        root = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True)
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
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        root = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True)
        if job_context.action == "train":
            config["dataset"]["train_dataset_dir"] = os.path.join(root, "sample_train")
            config["dataset"]["test_dataset_dir"] = os.path.join(root, "sample_test")
            config["dataset"]["query_dataset_dir"] = os.path.join(root, "sample_query")
        elif job_context.action in ("evaluate", "inference"):
            config[job_context.action]["test_dataset"] = os.path.join(root, "sample_test")
            config[job_context.action]["query_dataset"] = os.path.join(root, "sample_query")
    return config


def deformable_detr(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for Deformable-Detr"""
    # Train dataset
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_root = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True)
        if job_context.action == "train":
            config["dataset"]["train_data_sources"] = [{}]
            config["dataset"]["train_data_sources"][0]["image_dir"] = os.path.join(train_root, "images")
            config["dataset"]["train_data_sources"][0]["json_file"] = os.path.join(train_root, "annotations.json")

    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    eval_root = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True)
    if eval_ds is not None:
        if job_context.action == "train":
            config["dataset"]["val_data_sources"] = [{}]
            config["dataset"]["val_data_sources"][0]["image_dir"] = os.path.join(eval_root, "images")
            config["dataset"]["val_data_sources"][0]["json_file"] = os.path.join(eval_root, "annotations.json")
        if job_context.action == "evaluate":
            config["dataset"]["test_data_sources"] = {}
            config["dataset"]["test_data_sources"]["image_dir"] = os.path.join(eval_root, "images")
            config["dataset"]["test_data_sources"]["json_file"] = os.path.join(eval_root, "annotations.json")

    # Inference dataset
    infer_ds = handler_metadata.get("inference_dataset", None)
    infer_root = get_handler_root(job_context.user_id, "datasets", infer_ds, None, ngc_runner_fetch=True)
    if infer_ds is not None:
        if job_context.action == "inference":
            config["dataset"]["infer_data_sources"] = {}
            config["dataset"]["infer_data_sources"]["image_dir"] = [os.path.join(infer_root, "images")]
            config["dataset"]["infer_data_sources"]["classmap"] = os.path.join(infer_root, "label_map.txt")

    return config


def mal(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for MAL"""
    # Train dataset
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_root = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True)
        if job_context.action in ("evaluate", "inference", "train"):
            if "dataset" not in config.keys():
                config["dataset"] = {}
            config["dataset"]["train_img_dir"] = os.path.join(train_root, "images")
            config["dataset"]["train_ann_path"] = os.path.join(train_root, "annotations.json")

    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    eval_root = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True)
    if eval_ds is not None:
        if job_context.action in ("evaluate", "inference", "train"):
            if "dataset" not in config.keys():
                config["dataset"] = {}
            config["dataset"]["val_img_dir"] = os.path.join(eval_root, "images")
            config["dataset"]["val_ann_path"] = os.path.join(eval_root, "annotations.json")

    # Inference dataset
    infer_ds = handler_metadata.get("inference_dataset", None)
    infer_root = get_handler_root(job_context.user_id, "datasets", infer_ds, None, ngc_runner_fetch=True)
    if infer_ds is not None:
        if job_context.action == "inference":
            if "inference" not in config.keys():
                config["inference"] = {}
            config["inference"]["img_dir"] = os.path.join(infer_root, "images")
            config["inference"]["ann_path"] = os.path.join(infer_root, "annotations.json")

    return config


dino = deformable_detr
instance_segmentation = object_detection


def ml_recog(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for Metric Learning Recognition"""
    # Train dataset
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_root = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True)
        if "dataset" not in config.keys():
            config["dataset"] = {}
        if job_context.action == "train":
            config["dataset"]["train_dataset"] = os.path.join(train_root, "metric_learning_recognition", "retail-product-checkout-dataset_classification_demo", "known_classes", "train")
            config["dataset"]["val_dataset"] = {
                "reference": os.path.join(train_root, "metric_learning_recognition", "retail-product-checkout-dataset_classification_demo", "known_classes", "reference"),
                "query": os.path.join(train_root, "metric_learning_recognition", "retail-product-checkout-dataset_classification_demo", "known_classes", "val")}
        if job_context.action == "evaluate":
            config["dataset"]["val_dataset"] = {
                "reference": os.path.join(train_root, "metric_learning_recognition", "retail-product-checkout-dataset_classification_demo", "unknown_classes", "reference"),
                "query": os.path.join(train_root, "metric_learning_recognition", "retail-product-checkout-dataset_classification_demo", "unknown_classes", "test")}
        if job_context.action == "inference":
            if "inference" not in config.keys():
                config["inference"] = {}
            config["dataset"]["val_dataset"] = {
                "reference": os.path.join(train_root, "metric_learning_recognition", "retail-product-checkout-dataset_classification_demo", "unknown_classes", "reference"),
                "query": ""}
            config["inference"]["input_path"] = os.path.join(train_root, "metric_learning_recognition", "retail-product-checkout-dataset_classification_demo", "unknown_classes", "test")

    return config


def ocdnet(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for OCDNET"""
    parent_action = get_handler_job_metadata(job_context.user_id, job_context.handler_id, job_context.parent_id).get("action")
    if parent_action == "retrain":
        config["model"]["load_pruned_graph"] = True
    # Train dataset
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_root = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True)
        if "dataset" not in config.keys():
            config["dataset"] = {}
            config["dataset"]["train_dataset"] = {}
            config["dataset"]["validate_dataset"] = {}
        if job_context.action in ("train", "retrain"):
            config["dataset"]["train_dataset"]["data_path"] = [os.path.join(train_root, "train")]

    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    eval_root = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True)
    if eval_ds is not None:
        if "dataset" not in config.keys():
            config["dataset"] = {}
            config["dataset"]["train_dataset"] = {}
            config["dataset"]["validate_dataset"] = {}
        config["dataset"]["validate_dataset"]["data_path"] = [os.path.join(eval_root, "test")]
        if job_context.action == "inference":
            if "inference" not in config.keys():
                config["inference"] = {}
            config["inference"]["input_folder"] = os.path.join(eval_root, "test/img")

    return config


def ocrnet(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for OCRNET"""
    if job_context.action == "dataset_convert":
        ds = handler_metadata.get("id")
        root = get_handler_root(job_context.user_id, "datasets", ds, None, ngc_runner_fetch=True)
        if "dataset_convert" not in config.keys():
            config["dataset_convert"] = {}
        sub_folder = "train"
        if "test" in os.listdir(root):
            sub_folder = "test"
        config["dataset_convert"]["input_img_dir"] = f"{root}/{sub_folder}"
        config["dataset_convert"]["gt_file"] = f"{root}/{sub_folder}/gt_new.txt"
        config["dataset_convert"]["results_dir"] = f"{root}/{sub_folder}/lmdb"

    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    if eval_ds is not None:
        eval_root = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True)

    # Train dataset
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_root = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True)
        if "dataset" not in config.keys():
            config["dataset"] = {}
        if job_context.action in ("train", "retrain"):
            config["dataset"]["train_dataset_dir"] = [os.path.join(train_root, "train/lmdb")]
            config["dataset"]["val_dataset_dir"] = os.path.join(eval_root, "test/lmdb")
        config["dataset"]["character_list_file"] = os.path.join(eval_root, "character_list")

    if eval_ds is not None:
        if job_context.action == "evaluate":
            if "evaluate" not in config.keys():
                config["evaluate"] = {}
            config["evaluate"]["test_dataset_dir"] = os.path.join(eval_root, "test/lmdb")
        if job_context.action == "inference":
            if "inference" not in config.keys():
                config["inference"] = {}
            config["inference"]["inference_dataset_dir"] = os.path.join(eval_root, "test")

    return config


def optical_inspection(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for optical inspection"""
    # Train dataset
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_root = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True)
        if "dataset" not in config.keys():
            config["dataset"] = {}
        if "train_dataset" not in config["dataset"].keys():
            config["dataset"]["train_dataset"] = {}
        config["dataset"]["train_dataset"]["images_dir"] = os.path.join(train_root, "images")
        config["dataset"]["train_dataset"]["csv_path"] = os.path.join(train_root, "dataset.csv")

    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    eval_root = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True)
    if eval_ds is not None:
        if "dataset" not in config.keys():
            config["dataset"] = {}
        if "validation_dataset" not in config["dataset"].keys():
            config["dataset"]["validation_dataset"] = {}
        if "test_dataset" not in config["dataset"].keys():
            config["dataset"]["test_dataset"] = {}
        config["dataset"]["validation_dataset"]["images_dir"] = os.path.join(eval_root, "images")
        config["dataset"]["validation_dataset"]["csv_path"] = os.path.join(eval_root, "dataset.csv")
        config["dataset"]["test_dataset"]["images_dir"] = os.path.join(eval_root, "images")
        config["dataset"]["test_dataset"]["csv_path"] = os.path.join(eval_root, "dataset.csv")

    # Inference dataset
    infer_ds = handler_metadata.get("inference_dataset", None)
    infer_root = get_handler_root(job_context.user_id, "datasets", infer_ds, None, ngc_runner_fetch=True)
    if infer_ds is not None:
        if "dataset" not in config.keys():
            config["dataset"] = {}
        if "infer_dataset" not in config["dataset"].keys():
            config["dataset"]["infer_dataset"] = {}
        config["dataset"]["infer_dataset"]["images_dir"] = os.path.join(infer_root, "images")
        config["dataset"]["infer_dataset"]["csv_path"] = os.path.join(infer_root, "dataset.csv")

    return config


def centerpose(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for CenterPose"""
    # Train dataset
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_root = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True)
        if job_context.action == "train":
            config["dataset"]["train_data"] = {}
            config["dataset"]["train_data"] = os.path.join(train_root, 'train')

    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    eval_root = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True)
    if eval_ds is not None:
        if job_context.action == "train":
            config["dataset"]["val_data"] = {}
            config["dataset"]["val_data"] = os.path.join(eval_root, "val")
        if job_context.action == "evaluate":
            config["dataset"]["test_data"] = {}
            config["dataset"]["test_data"] = os.path.join(eval_root, "test")

    # Inference dataset
    infer_ds = handler_metadata.get("inference_dataset", None)
    infer_root = get_handler_root(job_context.user_id, "datasets", infer_ds, None, ngc_runner_fetch=True)
    if infer_ds is not None:
        if job_context.action == "inference":
            config["dataset"]["inference_data"] = {}
            config["dataset"]["inference_data"] = os.path.join(infer_root, "val")

    return config


def visual_changenet_classify(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for visual_changenet_classify"""
    # Train dataset
    if handler_metadata.get("train_datasets", []) != []:
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_root = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True)
        if "dataset" not in config.keys():
            config["dataset"] = {}
        if "classify" not in config["dataset"].keys():
            config["dataset"]["classify"] = {}
        if "train_dataset" not in config["dataset"]["classify"].keys():
            config["dataset"]["classify"]["train_dataset"] = {}
        config["dataset"]["classify"]["train_dataset"]["images_dir"] = os.path.join(train_root, "images")
        config["dataset"]["classify"]["train_dataset"]["csv_path"] = os.path.join(train_root, "dataset.csv")

    # Eval dataset
    eval_ds = handler_metadata.get("eval_dataset", None)
    eval_root = get_handler_root(job_context.user_id, "datasets", eval_ds, None, ngc_runner_fetch=True)
    if eval_ds is not None:
        if "dataset" not in config.keys():
            config["dataset"] = {}
        if "classify" not in config["dataset"].keys():
            config["dataset"]["classify"] = {}
        if "validation_dataset" not in config["dataset"]["classify"].keys():
            config["dataset"]["classify"]["validation_dataset"] = {}
        if "test_dataset" not in config["dataset"]["classify"].keys():
            config["dataset"]["classify"]["test_dataset"] = {}
        config["dataset"]["classify"]["validation_dataset"]["images_dir"] = os.path.join(eval_root, "images")
        config["dataset"]["classify"]["validation_dataset"]["csv_path"] = os.path.join(eval_root, "dataset.csv")
        config["dataset"]["classify"]["test_dataset"]["images_dir"] = os.path.join(eval_root, "images")
        config["dataset"]["classify"]["test_dataset"]["csv_path"] = os.path.join(eval_root, "dataset.csv")

    # Inference dataset
    infer_ds = handler_metadata.get("inference_dataset", None)
    infer_root = get_handler_root(job_context.user_id, "datasets", infer_ds, None, ngc_runner_fetch=True)
    if infer_ds is not None:
        if "dataset" not in config.keys():
            config["dataset"] = {}
        if "classify" not in config["dataset"].keys():
            config["dataset"]["classify"] = {}
        if "infer_dataset" not in config["dataset"]["classify"].keys():
            config["dataset"]["classify"]["infer_dataset"] = {}
        config["dataset"]["classify"]["infer_dataset"]["images_dir"] = os.path.join(infer_root, "images")
        config["dataset"]["classify"]["infer_dataset"]["csv_path"] = os.path.join(infer_root, "dataset.csv")

    return config


def visual_changenet(config, job_context, handler_metadata):
    """Assigns paths of data sources to the respective config params for visual changenet"""
    # Train dataset
    if handler_metadata.get("train_datasets", []) != []:
        print("Warning: checking handler visual changenet", handler_metadata.get("train_datasets"), file=sys.stderr)
        train_ds = handler_metadata.get("train_datasets", [])[0]
        train_dataset_metadata = get_handler_metadata(job_context.user_id, train_ds)
        if train_dataset_metadata.get('format') == 'visual_changenet_classify':
            return visual_changenet_classify(config, job_context, handler_metadata)
        train_root = get_handler_root(job_context.user_id, "datasets", train_ds, None, ngc_runner_fetch=True)
        if "dataset" not in config.keys():
            config["dataset"] = {}
        if "segment" not in config["dataset"].keys():
            config["dataset"]["segment"] = {}
        config["dataset"]["segment"]["root_dir"] = train_root

    return config


def analytics(config, job_context, handler_metadata):
    """Function to create data sources for analytics module"""
    config["data"]["image_dir"] = os.path.join(get_handler_root(job_context.user_id, "datasets", handler_metadata["inference_dataset"], None, ngc_runner_fetch=True), "images")
    if config["data"]["input_format"] == "COCO":
        config["data"]["ann_path"] = os.path.join(get_handler_root(job_context.user_id, "datasets", handler_metadata["inference_dataset"], None, ngc_runner_fetch=True), "annotations.json")
    elif config["data"]["input_format"] == "KITTI":
        config["data"]["ann_path"] = os.path.join(get_handler_root(job_context.user_id, "datasets", handler_metadata["inference_dataset"], None, ngc_runner_fetch=True), "labels")
    return config


def annotations(config, job_context, handler_metadata):
    """Function to create data sources for annotations module"""
    if config["data"]["input_format"] == "COCO":
        config["coco"]["ann_file"] = os.path.join(get_handler_root(job_context.user_id, "datasets", handler_metadata["inference_dataset"], None, ngc_runner_fetch=True), "annotations.json")
    elif config["data"]["input_format"] == "KITTI":
        config["kitti"]["image_dir"] = os.path.join(get_handler_root(job_context.user_id, "datasets", handler_metadata["inference_dataset"], None, ngc_runner_fetch=True), "images")
        config["kitti"]["label_dir"] = os.path.join(get_handler_root(job_context.user_id, "datasets", handler_metadata["inference_dataset"], None, ngc_runner_fetch=True), "labels")
    return config


def augmentation(config, job_context, handler_metadata):
    """Function to create data sources for augmentation module"""
    config["data"]["image_dir"] = os.path.join(get_handler_root(job_context.user_id, "datasets", handler_metadata["inference_dataset"], None, ngc_runner_fetch=True), "images")
    if config["data"]["dataset_type"] == "kitti":
        config["data"]["ann_path"] = os.path.join(get_handler_root(job_context.user_id, "datasets", handler_metadata["inference_dataset"], None, ngc_runner_fetch=True), "labels")
    elif config["data"]["dataset_type"] == "coco":
        config["data"]["ann_path"] = os.path.join(get_handler_root(job_context.user_id, "datasets", handler_metadata["inference_dataset"], None, ngc_runner_fetch=True), "annotations.json")
    return config


def auto_label(config, job_context, handler_metadata):
    """Function to create data sources for auto_label module"""
    config["inference"]["img_dir"] = os.path.join(get_handler_root(job_context.user_id, "datasets", handler_metadata["inference_dataset"], None, ngc_runner_fetch=True), "images")
    config["inference"]["ann_path"] = os.path.join(get_handler_root(job_context.user_id, "datasets", handler_metadata["inference_dataset"], None, ngc_runner_fetch=True), "annotations.json")
    return config


def prepare_job_datalist(config, job_context, handler_metadata):
    """Function to create data sources for medical bundles."""
    job_action = job_context.action
    train_datasets = handler_metadata["train_datasets"]
    if len(train_datasets) != 1:
        raise ValueError(f"Only one train dataset is supported, but {len(train_datasets)} are given.")

    eval_dataset = handler_metadata["eval_dataset"]  # Can be None.
    train_dataset_id = train_datasets[0]
    eval_dataset_id = eval_dataset if eval_dataset else None

    user_id = get_handler_user(job_context.handler_id)
    ptm_model_path, bundle_name, overriden_output_dir = CLI_CONFIG_TO_FUNCTIONS["medical_output_dir"](job_context, handler_metadata)

    dataset_usages = ["train", "validate"]
    dataset_ids = [train_dataset_id, eval_dataset_id]
    ngc_runner_fetch = not (job_context.specs and "cluster" in job_context.specs and job_context.specs["cluster"] == "local")

    src_dir = os.path.join(ptm_model_path, bundle_name)
    dst_dir = os.path.join(overriden_output_dir, bundle_name)
    if not ngc_runner_fetch:
        # this is a local run, so we can copy the bundle from base dir to experiment dir
        shutil.copytree(src_dir, dst_dir)
    else:
        config["copy_in_job"] = f"cp -r {src_dir} {dst_dir}"
    # Workaround for the ci test.
    from handlers.medical.dataset.dicom import DicomEndpoint
    for dataset_usage, dataset_id in zip(dataset_usages, dataset_ids):
        if dataset_id is None:
            continue
        if job_action == "train":
            datalist_json = os.path.join(overriden_output_dir, f"{dataset_usage}_datalist.json")
            ds = MedicalDatasetHandler.prepare_datalist(user_id=user_id, dataset_id=dataset_id, ngc_runner_fetch=ngc_runner_fetch)
            with open(datalist_json, "w", encoding="utf-8") as fp:
                json.dump(ds, fp, indent=4)

            config[f"{dataset_usage}#dataset#data"] = f"%{datalist_json}"

            dataset_metadata = get_handler_metadata(job_context.user_id, dataset_id)
            endpoint = MedicalDatasetHandler.endpoint(dataset_metadata)
            if isinstance(endpoint, DicomEndpoint):
                config["dicom_convert"] = True

    return config


def prepare_medical_datalist(config, job_context, handler_metadata, workspace_dir):
    """Function to create medical datalist"""
    datasets = {}
    # get training dataset id
    dataset_id = handler_metadata["train_datasets"]
    if dataset_id:
        if len(dataset_id) > 1:
            raise ValueError(f"One and only one training dataset is supported, but {len(dataset_id)} is given.")
        if len(dataset_id) == 1:
            datasets["training"] = dataset_id[0]
    # get validation dataset id
    dataset_id = handler_metadata["eval_dataset"]
    if dataset_id:
        datasets["validation"] = dataset_id
    # get inference dataset id (first from action spec, if not available, then from model spec)
    dataset_id = job_context.specs.get("inference_dataset")
    if dataset_id:
        datasets["testing"] = dataset_id
    else:
        dataset_id = handler_metadata["inference_dataset"]
        if dataset_id:
            datasets["testing"] = dataset_id

    # get user id
    user_id = get_handler_user(job_context.handler_id)

    # create datalist sections (training and testing) and fill in with "data" field of manifests
    datalist_dict = {}
    for usage, dataset in datasets.items():
        datalist_dict[usage] = MedicalDatasetHandler.get_data_from_manifest(
            user_id=user_id, dataset_id=dataset, workspace_dir=workspace_dir
        )

    # save datalist to json file and update config
    datalist_json_path = os.path.join(workspace_dir, "datalist.json")
    with open(datalist_json_path, "w", encoding="utf-8") as fp:
        json.dump(datalist_dict, fp, indent=2)
    config["datalist"] = datalist_json_path
    config["work_dir"] = workspace_dir
    if config.get("ensemble") is None:
        config["ensemble"] = False

    return config


DS_CONFIG_TO_FUNCTIONS = {"detectnet_v2": detectnet_v2,
                          "faster_rcnn": faster_rcnn,
                          "yolo_v4": yolo_v4,
                          "yolo_v4_tiny": yolo_v4_tiny,
                          "yolo_v3": yolo_v3,
                          "ssd": ssd,
                          "dssd": dssd,
                          "retinanet": retinanet,
                          "unet": unet,
                          "segformer": segformer,
                          "lprnet": lprnet,
                          "efficientdet_tf1": efficientdet_tf1,
                          "efficientdet_tf2": efficientdet_tf2,
                          "mask_rcnn": mask_rcnn,
                          "multitask_classification": multitask_classification,
                          "classification_pyt": classification_pyt,
                          "classification_tf1": classification_tf1,
                          "classification_tf2": classification_tf2,
                          "bpnet": bpnet,
                          "fpenet": fpenet,
                          "action_recognition": action_recognition,
                          "mal": mal,
                          "ml_recog": ml_recog,
                          "medical_annotation": prepare_job_datalist,
                          "medical_vista3d": prepare_job_datalist,
                          "medical_automl": prepare_medical_datalist,
                          "medical_segmentation": prepare_medical_datalist,
                          "medical_custom": prepare_job_datalist,
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
                          "auto_label": auto_label,
                          "visual_changenet": visual_changenet}
