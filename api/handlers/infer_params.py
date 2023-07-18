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

"""
Functions to infer values
Each function takes as input:
- handler (dataset / model)
- app_handler
- job_context
"""
import os
import glob
import sys
import shutil
import uuid

from handlers.utilities import search_for_ptm, get_model_results_path, read_network_config
from handlers.stateless_handlers import get_handler_root, get_handler_spec_root, get_handler_job_metadata, load_json_data


def infer_verbose(job_context, handler_metadata):
    """Return True to enable verbose commands"""
    return True


def infer_key(job_context, handler_metadata):
    """Returns the encryption key associated with the model"""
    try:
        return handler_metadata.get("encryption_key", "tlt_encode")
    except:
        return None


def infer_output_dir(job_context, handler_metadata):
    """Creates output directory within handler root"""
    job_id = str(job_context.id)
    outroot = get_handler_root(handler_metadata.get("id"))
    results_dir = os.path.join(outroot, job_id)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    return results_dir + "/"


def infer_spec_file(job_context, handler_metadata):
    """Returns path of the spec file of a job"""
    network = job_context.network
    network_config = read_network_config(network)
    api_params = network_config.get("api_params", {})

    spec_root = get_handler_spec_root(handler_metadata.get("id"))
    job_id = str(job_context.id)

    if job_context.action == "convert_efficientdet_tf2":
        spec_path = os.path.join(spec_root, job_id + ".yaml")
    else:
        spec_path = os.path.join(spec_root, job_id + "." + api_params["spec_backend"])
    return spec_path


# NOTE: Only supports those with ngc_path to be PTMs
def infer_ptm(job_context, handler_metadata):
    """Returns a list of path of the ptm files of a network"""
    network = job_context.network
    handler_ptms = handler_metadata.get("ptm", None)
    if handler_ptms is None:
        return None
    ptm_file = []
    for handler_ptm in handler_ptms:
        if handler_ptm:
            ptm_root = get_handler_root(handler_ptm)
            ptm_file.append(search_for_ptm(ptm_root, network=network))
    return ",".join(ptm_file)


def infer_pruned_model(job_context, handler_metadata):
    """Returns path of the pruned model"""
    handler_root = get_handler_root(handler_metadata.get("id"))
    if not handler_root:
        return None
    if handler_metadata["network_arch"] in ("efficientdet_tf2", "classification_tf2"):
        return os.path.join(handler_root, job_context.id, "pruned_model.tlt")
    pruned_model = os.path.join(handler_root, job_context.parent_id, "pruned_model.tlt")
    if os.path.exists(pruned_model):
        return pruned_model
    if os.path.exists(pruned_model.replace(".tlt", ".pth")):
        return pruned_model.replace(".tlt", ".pth")
    if os.path.exists(pruned_model.replace(".tlt", ".hdf5")):
        return pruned_model.replace(".tlt", ".hdf5")
    return None


def infer_parent_model(job_context, handler_metadata):
    """Returns path of the weight file of the parent job"""
    parent_model = get_model_results_path(handler_metadata, job_context.parent_id)
    if os.path.exists(str(parent_model)):
        return parent_model
    return None


def infer_resume_model(job_context, handler_metadata):
    """Returns path of the weight file of the current job"""
    parent_model = get_model_results_path(handler_metadata, job_context.id)
    if os.path.exists(str(parent_model)):
        return parent_model
    return None


def infer_resume_model_or_ptm(job_context, handler_metadata):
    """Returns path of the weight file of the current job if exists else returns path of the ptm files"""
    resume_model = infer_resume_model(job_context, handler_metadata)
    if resume_model:
        return resume_model
    return infer_ptm(job_context, handler_metadata)


def infer_ptm_if_no_resume_model(job_context, handler_metadata):
    """Returns path of path of the ptm files if there is no model to resume"""
    resume_model = infer_resume_model(job_context, handler_metadata)
    if resume_model:
        return None
    return infer_ptm(job_context, handler_metadata)


def infer_automl_assign_ptm(job_context, handler_metadata, job_root, rec_number):
    """Returns path of path of the ptm files if there is no model to resume for AutoML"""
    expt_root = infer_automl_output_dir(job_context, handler_metadata, job_root, rec_number)
    resume_model = glob.glob(expt_root + "/**/*.tlt", recursive=True) + glob.glob(expt_root + "/**/*.hdf5", recursive=True) + glob.glob(expt_root + "/**/*.pth", recursive=True)
    if not resume_model:
        return infer_ptm(job_context, handler_metadata)
    return None


def infer_automl_resume_model(job_context, handler_metadata, job_root, rec_number):
    """Returns path of the checkpoint file for the automl recommendation to resume on"""
    expt_root = infer_automl_output_dir(job_context, handler_metadata, job_root, rec_number)
    resume_model = glob.glob(expt_root + "/**/*.tlt", recursive=True) + glob.glob(expt_root + "/**/*.hdf5", recursive=True) + glob.glob(expt_root + "/**/*.pth", recursive=True)
    resume_model.sort(reverse=False)
    if resume_model:
        resume_model = resume_model[0]
    return resume_model


def infer_automl_ptm_if_no_resume_model(job_context, handler_metadata, job_root, rec_number):
    """Returns path of the checkpoint file for the automl recommendation to resume on"""
    expt_root = infer_automl_output_dir(job_context, handler_metadata, job_root, rec_number)
    resume_model = glob.glob(expt_root + "/**/*.tlt", recursive=True) + glob.glob(expt_root + "/**/*.hdf5", recursive=True) + glob.glob(expt_root + "/**/*.pth", recursive=True)
    resume_model.sort(reverse=False)
    if resume_model:
        return resume_model[0]
    return infer_ptm(job_context, handler_metadata)


def infer_automl_experiment_spec(job_context, handler_metadata, job_root, rec_number):
    """Returns path automl spec file"""
    network = job_context.network
    network_config = read_network_config(network)
    api_params = network_config.get("api_params", {})
    experiment_spec = f"{job_root}/recommendation_{rec_number}.{api_params['spec_backend']}"
    return experiment_spec


def infer_automl_assign_resume_epoch(job_context, handler_metadata, job_root, rec_number):
    """Returns path automl spec file"""
    additional_epoch = 0
    if job_context.network != "efficientdet_tf2":
        additional_epoch = 1  # epoch numbers indexed by 1
    resume_epoch_number = 0 + additional_epoch
    if infer_automl_resume_model(job_context, handler_metadata, job_root, rec_number):
        brain_dict = load_json_data(json_file=f"{job_root}/brain.json")
        resume_epoch_number = int(brain_dict.get("resume_epoch_number", -1)) + additional_epoch
    return resume_epoch_number


def infer_automl_output_dir(job_context, handler_metadata, job_root, rec_number):
    """Returns path of the automl experiment folder"""
    expt_root = os.path.join(job_root, f"experiment_{rec_number}/")
    return expt_root


def infer_parent_model_evaluate(job_context, handler_metadata):
    """Returns path of the weight file of the parent job if exists else returns path of the ptm files"""
    # Assumes: <results_dir/weights> is stored
    # If extension is None: output is based on RESULTS_RELPATH
    # If extension exists, then search for that extension
    parent_job_id = job_context.parent_id
    handler_id = handler_metadata.get("id")
    parent_action = get_handler_job_metadata(handler_id, parent_job_id).get("action")

    if parent_action == "export":
        parent_model = os.path.join(get_handler_root(handler_metadata.get("id")), str(job_context.parent_id), "model.engine")
    else:
        parent_model = get_model_results_path(handler_metadata, job_context.parent_id)

    if os.path.exists(str(parent_model)):
        return parent_model
    # This means, running eval without a parent => eval a PTM!
    # It is the duty of user to have given a PTM. Else job will error out without launching.
    ptm = infer_ptm(job_context, handler_metadata)
    return ptm


def infer_framework_evaluate(job_context, handler_metadata):
    """Returns framework to evaluate model on based on the parent action"""
    parent_job_id = job_context.parent_id
    handler_id = handler_metadata.get("id")
    parent_action = get_handler_job_metadata(handler_id, parent_job_id).get("action")

    if parent_action == "export":
        return "tensorrt"
    return "tlt"


def infer_framework_evaluate_storetrue(job_context, handler_metadata):
    """Returns whether the evaluation framework is tensorrt or not"""
    framework = infer_framework_evaluate(job_context, handler_metadata)
    return framework == "tensorrt"


def infer_output_file(job_context, handler_metadata, extension):
    """Create output folder based on the filepath"""
    # Create all directories up until the file name
    outdir = infer_output_dir(job_context, handler_metadata)
    path = os.path.join(outdir, extension)
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return path


def infer_output_hdf5(job_context, handler_metadata):
    """Calls infer_output_file for model.hdf5"""
    return infer_output_file(job_context, handler_metadata, "model.hdf5")


def infer_output_pth(job_context, handler_metadata):
    """Calls infer_output_file for model.pth"""
    return infer_output_file(job_context, handler_metadata, "model.pth")


def infer_output_onnx(job_context, handler_metadata):
    """Calls infer_output_file for model.onnx"""
    return infer_output_file(job_context, handler_metadata, "model.onnx")


def infer_output_trt(job_context, handler_metadata):
    """Calls infer_output_file for model.engine"""
    return infer_output_file(job_context, handler_metadata, "model.engine")


def infer_output_weights_tlt(job_context, handler_metadata):
    """Calls infer_output_file for weights/model.tlt"""
    return infer_output_file(job_context, handler_metadata, "weights/model.tlt")


def infer_merged_json(job_context, handler_metadata):
    """Calls infer_output_file for merged.json"""
    return infer_output_file(job_context, handler_metadata, "merged.json")


def infer_cal_cache(job_context, handler_metadata):
    """Calls infer_output_file for cal.bin"""
    return infer_output_file(job_context, handler_metadata, "cal.bin")


def infer_cal_data_file(job_context, handler_metadata):
    """Calls infer_output_file for calibration.tensorfile"""
    return infer_output_file(job_context, handler_metadata, "calibration.tensorfile")


def infer_create_inference_result_file_pose(job_context, handler_metadata):
    """Calls infer_output_file for results.txt"""
    return infer_output_file(job_context, handler_metadata, "results.txt")


def infer_create_evaluate_matches_plot_reid(job_context, handler_metadata):
    """Calls infer_output_file for sampled_matches.png"""
    return infer_output_file(job_context, handler_metadata, "sampled_matches.png")


def infer_create_evaluate_cmc_plot_reid(job_context, handler_metadata):
    """Calls infer_output_file for cmc_curve.png"""
    return infer_output_file(job_context, handler_metadata, "cmc_curve.png")


def infer_create_inference_result_file_json(job_context, handler_metadata):
    """Calls infer_output_file for inference.json"""
    return infer_output_file(job_context, handler_metadata, "inference.json")


def infer_parent_spec(job_context, handler_metadata):
    """Returns path of the spec file of the parent job"""
    handler_id = handler_metadata.get("id")
    parent_job_id = job_context.parent_id

    network = job_context.network
    network_config = read_network_config(network)
    api_params = network_config.get("api_params", {})

    parent_action = get_handler_job_metadata(handler_metadata.get("id"), job_context.parent_id).get("action")
    if handler_metadata.get("automl_enabled") is True and parent_action == "train":
        root = get_handler_root(handler_id)
        automl_root = os.path.join(root, parent_job_id, "best_model")
        spec_file = (glob.glob(f"{automl_root}/*recommendation*.protobuf") + glob.glob(f"{automl_root}/*recommendation*.yaml"))[0]
        spec_file_copy = os.path.join(get_handler_spec_root(handler_id), job_context.id + "." + api_params["spec_backend"])
    else:
        spec_file = os.path.join(get_handler_spec_root(handler_id), parent_job_id + "." + api_params["spec_backend"])
        spec_file_copy = spec_file.replace(parent_job_id, job_context.id)

    os.makedirs(os.path.dirname(os.path.abspath(spec_file_copy)), exist_ok=True)
    shutil.copy(spec_file, spec_file_copy)
    return spec_file


def infer_parents_parent_spec(job_context, handler_metadata):
    """Returns path of the spec file of the parent's parent job"""
    handler_id = handler_metadata.get("id")
    parent_job_id = job_context.parent_id
    parents_parent_job_id = get_handler_job_metadata(handler_id, parent_job_id).get("parent_id", "")
    parents_parent_action = get_handler_job_metadata(handler_metadata.get("id"), parents_parent_job_id).get("action")

    if parents_parent_action == "dataset_convert":
        print("Dataset convert spec can't be used for this job, returning parent's spec now", file=sys.stderr)
        return infer_parent_spec(job_context, handler_metadata)
    try:
        uuid.UUID(parents_parent_job_id)
    except:
        print("Parent's parent job id can't be found, Searching for parent's spec now", file=sys.stderr)
        return infer_parent_spec(job_context, handler_metadata)

    network = job_context.network
    network_config = read_network_config(network)
    api_params = network_config.get("api_params", {})

    if handler_metadata.get("automl_enabled") is True and parents_parent_action == "train":
        root = get_handler_root(handler_id)
        automl_root = os.path.join(root, parents_parent_job_id, "best_model")
        spec_file = (glob.glob(f"{automl_root}/*recommendation*.protobuf") + glob.glob(f"{automl_root}/*recommendation*.yaml"))[0]
        spec_file_copy = os.path.join(get_handler_spec_root(handler_id), job_context.id + "." + api_params["spec_backend"])
    else:
        spec_file = os.path.join(get_handler_spec_root(handler_id), parents_parent_job_id + "." + api_params["spec_backend"])
        spec_file_copy = spec_file.replace(parents_parent_job_id, job_context.id)

    if not os.path.exists(spec_file):
        print("Parent's parent spec can't be found, Searching for parent's spec now", file=sys.stderr)
        return infer_parent_spec(job_context, handler_metadata)
    os.makedirs(os.path.dirname(os.path.abspath(spec_file_copy)), exist_ok=True)
    shutil.copy(spec_file, spec_file_copy)
    return spec_file


def infer_parent_spec_copied(job_context, handler_metadata):
    """Returns path of the spec file path copied from the parent job"""
    handler_id = handler_metadata.get("id")
    parent_job_id = job_context.parent_id

    network = job_context.network
    network_config = read_network_config(network)
    api_params = network_config.get("api_params", {})

    parent_action = get_handler_job_metadata(handler_metadata.get("id"), job_context.parent_id).get("action")
    if handler_metadata.get("automl_enabled") is True and parent_action == "train":
        root = get_handler_root(handler_id)
        automl_root = os.path.join(root, parent_job_id, "best_model")
        spec_file = (glob.glob(f"{automl_root}/*recommendation*.protobuf") + glob.glob(f"{automl_root}/*recommendation*.yaml"))[0]
        spec_file_copy = os.path.join(get_handler_spec_root(handler_id), job_context.id + "." + api_params["spec_backend"])
    else:
        spec_file = os.path.join(get_handler_spec_root(handler_id), parent_job_id + "." + api_params["spec_backend"])
        spec_file_copy = spec_file.replace(parent_job_id, job_context.id)

    os.makedirs(os.path.dirname(os.path.abspath(spec_file_copy)), exist_ok=True)
    shutil.copy(spec_file, spec_file_copy)
    return spec_file_copy


def infer_parent_cal_cache(job_context, handler_metadata):
    """Returns path of the cal.bin of the parent job"""
    parent_job_id = job_context.parent_id
    cal_file = os.path.join(get_handler_root(handler_metadata.get("id")), parent_job_id, "cal.bin")

    if os.path.exists(cal_file):
        return cal_file
    return None


def infer_lprnet_inference_input(job_context, handler_metadata):
    """Returns path of the inference images for lprnet"""
    # Returns root + "/image/" if it exists

    infer_ds = handler_metadata.get("inference_dataset", None)
    if infer_ds is None:
        return None

    images_path = get_handler_root(infer_ds) + "/image/"
    if os.path.exists(images_path):
        return images_path
    return None


def infer_classification_val_input(job_context, handler_metadata):
    """Returns path of the inference images for object_detection networks"""
    infer_ds = handler_metadata.get("eval_dataset", None)
    if infer_ds is None:
        return None

    images_root = get_handler_root(infer_ds)
    if os.path.exists(images_root + "/images/"):
        images_path = images_root + "/images/"
        return images_path
    if os.path.exists(images_root + "/images_val/"):
        images_path = images_root + "/images_val/"
        return images_path
    print(f"Warning: Image directory not found in {images_root}", file=sys.stderr)
    return None


# OD helper functions


def infer_od_inference_input(job_context, handler_metadata):
    """Returns path of the inference images for object_detection networks"""
    infer_ds = handler_metadata.get("inference_dataset", None)
    if infer_ds is None:
        return None

    images_root = get_handler_root(infer_ds)
    if os.path.exists(images_root + "/images/"):
        images_path = images_root + "/images/"
        return images_path
    if os.path.exists(images_root + "/images_test/"):
        images_path = images_root + "/images_test/"
        return images_path
    print(f"Warning: Image directory not found in {images_root}", file=sys.stderr)
    return None


def infer_od_inference_labels(job_context, handler_metadata):
    """Returns path of the inference labels for object_detection networks"""
    infer_ds = handler_metadata.get("inference_dataset", None)
    if infer_ds is None:
        return None

    images_root = get_handler_root(infer_ds)
    if os.path.exists(images_root + "/labels/"):
        images_path = images_root + "/labels/"
        return images_path
    print(f"Warning: Labels directory not found in {images_root}", file=sys.stderr)
    return None


def infer_od_inference_label_map(job_context, handler_metadata):
    """Returns path of label_map.txt for object_detection networks"""
    infer_ds = handler_metadata.get("inference_dataset", None)
    if infer_ds is None:
        return None

    label_map_path = get_handler_root(infer_ds) + "/label_map.txt"
    if os.path.exists(label_map_path):
        return label_map_path
    if os.path.exists(label_map_path.replace(".txt", ".yaml")):
        return label_map_path.replace(".txt", ".yaml")
    return None


def infer_od_inference_input_image(job_context, handler_metadata):
    """Returns path of a single inference image for object_detection networks"""
    print("Warning: Only single image can be inferred for multitask classification", file=sys.stderr)
    images_path = infer_od_inference_input(job_context, handler_metadata)
    if images_path:
        im_path = glob.glob(images_path + "/*")[0]
        return im_path

    return None


def infer_od_dir(job_context, handler_metadata, dirname):
    """Returns joined-path of handler_root and dirname"""
    handler_root = get_handler_root(handler_metadata.get("id"))
    path = f"{handler_root}/{dirname}"
    if os.path.exists(path):
        return path
    return None


def infer_od_images(job_context, handler_metadata):
    """Calls infer_od_dir on images directory"""
    return infer_od_dir(job_context, handler_metadata, "images/")


def infer_od_labels(job_context, handler_metadata):
    """Calls infer_od_dir on labels directory"""
    return infer_od_dir(job_context, handler_metadata, "labels/")


def infer_unet_val_images(job_context, handler_metadata):
    """Returns path of the images for unet dataset type networks"""
    infer_ds = handler_metadata.get("eval_dataset", None)
    if infer_ds is None:
        return None

    images_root = get_handler_root(infer_ds)
    if os.path.exists(images_root + "/images/val/"):
        images_path = images_root + "/images/val/"
        return images_path
    print(f"Warning: Labels directory not found in {images_root}", file=sys.stderr)
    return None


def infer_unet_val_labels(job_context, handler_metadata):
    """Returns path of the labels for unet dataset type networks"""
    infer_ds = handler_metadata.get("eval_dataset", None)
    if infer_ds is None:
        return None

    images_root = get_handler_root(infer_ds)
    if os.path.exists(images_root + "/masks/val/"):
        images_path = images_root + "/masks/val/"
        return images_path
    print(f"Warning: Labels directory not found in {images_root}", file=sys.stderr)
    return None


def infer_unet_test_images(job_context, handler_metadata):
    """Returns path of the images for unet dataset type networks"""
    infer_ds = handler_metadata.get("eval_dataset", None)
    if infer_ds is None:
        return None

    images_root = get_handler_root(infer_ds)
    if os.path.exists(images_root + "/images/test/"):
        images_path = images_root + "/images/test/"
        return images_path
    print(f"Warning: Labels directory not found in {images_root}", file=sys.stderr)
    return None


def infer_unet_test_labels(job_context, handler_metadata):
    """Returns path of the labels for unet dataset type networks"""
    infer_ds = handler_metadata.get("eval_dataset", None)
    if infer_ds is None:
        return None

    images_root = get_handler_root(infer_ds)
    if os.path.exists(images_root + "/masks/test/"):
        images_path = images_root + "/masks/test/"
        return images_path
    print(f"Warning: Labels directory not found in {images_root}", file=sys.stderr)
    return None


def infer_od_annotations(job_context, handler_metadata):
    """Calls infer_od_dir on annotations.json file"""
    return infer_od_dir(job_context, handler_metadata, "annotations.json")


def infer_parent_classmap(job_context, handler_metadata):
    """Returns path of classmap file of parent job"""
    parent_job_id = job_context.parent_id

    classmap_path = None

    # Check if inference dataset has classmap file
    infer_ds = handler_metadata.get("inference_dataset", None)
    if infer_ds is not None:
        classmap_path = os.path.join(get_handler_root(infer_ds), "classmap.json")
        if os.path.exists(str(classmap_path)):
            return classmap_path

    # Else check for classmap presence in the parent job's artifacts
    parent_action = get_handler_job_metadata(handler_metadata.get("id"), job_context.parent_id).get("action")
    automl_path = ""
    if handler_metadata.get("automl_enabled") is True and parent_action == "train":
        automl_path = "best_model"

    if parent_job_id:
        classmap_path = glob.glob(f'{os.path.join(get_handler_root(handler_metadata.get("id")), str(parent_job_id), automl_path)}/**/*classmap.json', recursive=True)
        if not classmap_path:
            classmap_path = glob.glob(f'{os.path.join(get_handler_root(handler_metadata.get("id")), str(parent_job_id), automl_path)}/**/*class_mapping.json', recursive=True)
    if classmap_path and os.path.exists(str(classmap_path[0])):
        # Copy parent classmap as current classmap - needed for consecutive jobs which uses parent classmap
        os.makedirs(os.path.join(get_handler_root(handler_metadata.get("id")), str(job_context.id)), exist_ok=True)
        shutil.copy(classmap_path[0], classmap_path[0].replace(parent_job_id, job_context.id).replace(automl_path, "").replace(parent_action, ""))
        return classmap_path[0]
    print("Warning: classmap.json needs to be uploaded with inference dataset", file=sys.stderr)
    return None


def infer_cal_image_dir(job_context, handler_metadata):
    """Returns path of calibration image directory"""
    # Infer calibration image dir
    # Idea is to use calibration_dataset's root/images/ directory
    # If not present, we simply error out
    calib_ds = handler_metadata.get("calibration_dataset", None)
    if calib_ds is None:
        return None

    if job_context.network == "unet":
        images_path = get_handler_root(calib_ds) + "/images/train/"
    elif job_context.network == "ocdnet":
        images_path = get_handler_root(calib_ds) + "/train/img/"
    else:
        images_path = get_handler_root(calib_ds) + "/images/"

    if os.path.exists(images_path):
        return images_path
    if os.path.exists(images_path.replace("/images/", "/images_train/")):
        return images_path.replace("/images/", "/images_train/")
    if os.path.exists(images_path.replace("/images/", "/train2017/")):
        return images_path.replace("/images/", "/train2017/")
    return None


def infer_cal_image_dir_list(job_context, handler_metadata):
    """Returns list of path of calibration images"""
    # Infer calibration image dir
    # Idea is to use calibration_dataset's root/images/ directory
    # If not present, we simply error out

    calib_ds = handler_metadata.get("calibration_dataset", None)
    if calib_ds is None:
        return None

    if job_context.network == "ml_recog":
        images_path = get_handler_root(calib_ds) + "/metric_learning_recognition/retail-product-checkout-dataset_classification_demo/known_classes/test/"
    elif job_context.network == "ocrnet":
        images_path = get_handler_root(calib_ds) + "/train/"
    else:
        images_path = get_handler_root(calib_ds) + "/images/"
    if os.path.exists(images_path):
        return [images_path]
    if os.path.exists(images_path.replace("/images/", "/images_train/")):
        return [images_path.replace("/images/", "/images_train/")]
    if os.path.exists(images_path.replace("/images/", "/train2017/")):
        return [images_path.replace("/images/", "/train2017/")]
    return None


def infer_bpnet_coco_spec(job_context, handler_metadata):
    """Returns path of coco_spec file for bpnet"""
    train_ds = handler_metadata.get("train_datasets", [])[0]
    handler_root = get_handler_root(train_ds)
    infer_json = handler_root + "/coco_spec.json"
    return infer_json


def infer_bpnet_inference(job_context, handler_metadata):
    """Returns path of inference dataset for bpnet"""
    train_ds = handler_metadata.get("train_datasets", [])[0]
    handler_root = get_handler_root(train_ds)
    infer_path = handler_root + "/val2017"
    return infer_path


def infer_data_json(job_context, handler_metadata):
    """Returns path of data json"""
    train_ds = handler_metadata.get("train_datasets", [])
    if train_ds != []:
        handler_root = get_handler_root(train_ds[0])
    else:
        handler_root = get_handler_root(handler_metadata.get("id"))
    return os.path.join(handler_root, "data.json")


def infer_inference_data(job_context, handler_metadata):
    """Returns path of dataset to run sample inference on"""
    train_ds = handler_metadata.get("train_datasets", [])[0]
    handler_root = get_handler_root(train_ds)
    return handler_root


def infer_gt_cache(job_context, handler_metadata):
    """Returns path of label.json for auto_labeling"""
    infer_ds = handler_metadata.get("inference_dataset", None)
    if infer_ds is None:
        return None

    gt_cache_path = os.path.join(get_handler_root(infer_ds), "label.json")
    if os.path.exists(gt_cache_path):
        return gt_cache_path
    return None


def infer_label_output(job_context, handler_metadata):
    """Returns path of label.json for auto_labeling"""
    results_dir = infer_output_dir(job_context, handler_metadata)
    label_output = os.path.join(results_dir, "label.json")
    return label_output


CLI_CONFIG_TO_FUNCTIONS = {"output_dir": infer_output_dir,
                           "key": infer_key,
                           "experiment_spec": infer_spec_file,
                           "pruned_model": infer_pruned_model,
                           "parent_model": infer_parent_model,
                           "parent_model_evaluate": infer_parent_model_evaluate,
                           "resume_model": infer_resume_model,
                           "resume_model_or_ptm": infer_resume_model_or_ptm,
                           "ptm_if_no_resume_model": infer_ptm_if_no_resume_model,
                           "automl_assign_ptm": infer_automl_assign_ptm,
                           "automl_resume_model": infer_automl_resume_model,
                           "automl_ptm_if_no_resume_model": infer_automl_ptm_if_no_resume_model,
                           "automl_experiment_spec": infer_automl_experiment_spec,
                           "automl_output_dir": infer_automl_output_dir,
                           "automl_assign_resume_epoch": infer_automl_assign_resume_epoch,
                           "framework": infer_framework_evaluate,
                           "framework_storetrue": infer_framework_evaluate_storetrue,
                           "verbose": infer_verbose,
                           "ptm": infer_ptm,
                           "create_hdf5_file": infer_output_hdf5,
                           "create_pth_file": infer_output_pth,
                           "create_onnx_file": infer_output_onnx,
                           "create_engine_file": infer_output_trt,
                           "create_weights_tlt_file": infer_output_weights_tlt,
                           "create_cal_cache": infer_cal_cache,
                           "create_cal_data_file": infer_cal_data_file,
                           "parent_spec": infer_parent_spec,
                           "parents_parent_spec": infer_parents_parent_spec,
                           "parent_spec_copied": infer_parent_spec_copied,
                           "parent_cal_cache": infer_parent_cal_cache,
                           "merged_json": infer_merged_json,
                           "create_inference_result_file_pose": infer_create_inference_result_file_pose,
                           "create_evaluate_matches_plot_reid": infer_create_evaluate_matches_plot_reid,
                           "create_evaluate_cmc_plot_reid": infer_create_evaluate_cmc_plot_reid,
                           "create_inference_result_file_reid": infer_create_inference_result_file_json,
                           "create_inference_result_file_mal": infer_create_inference_result_file_json,
                           "lprnet_inference_input": infer_lprnet_inference_input,
                           "classification_val_input": infer_classification_val_input,
                           "od_inference_input": infer_od_inference_input,
                           "od_inference_input_image": infer_od_inference_input_image,
                           "cal_image_dir": infer_cal_image_dir,
                           "cal_image_dir_list": infer_cal_image_dir_list,
                           "od_images": infer_od_images,
                           "od_labels": infer_od_labels,
                           "od_annotations": infer_od_annotations,
                           "od_inference_label_map": infer_od_inference_label_map,
                           "od_inference_labels": infer_od_inference_labels,
                           "unet_val_images": infer_unet_val_images,
                           "unet_val_labels": infer_unet_val_labels,
                           "unet_test_images": infer_unet_test_images,
                           "unet_test_labels": infer_unet_test_labels,
                           "create_od_tfrecords": lambda a, b: get_handler_root(b.get("id")) + "/tfrecords/",
                           "output_dir_images_annotated": lambda a, b: infer_output_dir(a, b) + "/images_annotated/",
                           "output_dir_labels": lambda a, b: infer_output_dir(a, b) + "/labels/",
                           "output_dir_inference_json": lambda a, b: infer_output_dir(a, b) + "/annotations_mal.json",
                           "root": lambda a, b: get_handler_root(b.get("id")),  # Just return the root of the handler object
                           "augment_out": lambda a, b: get_handler_root(b.get("id")) + "/augment",
                           "from_csv": lambda a, b: None,  # Used to infer the param from spec sheet
                           "parent_classmap": infer_parent_classmap,
                           "bpnet_coco_spec": infer_bpnet_coco_spec,
                           "bpnet_infer": infer_bpnet_inference,
                           "fpenet_data_json": infer_data_json,
                           "fpenet_inference_data": infer_inference_data,
                           "label_gt_cache": infer_gt_cache,
                           "auto_label_output": infer_label_output}
