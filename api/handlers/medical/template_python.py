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

"""Templates for Python files."""


TEMPLATE_TIS_MODEL = """
import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils
from medical.bundle import ConfigWorkflow
import torch
import base64
import ast
import sys

def unlink(path):
    if os.path.islink(path):
        os.unlink(path)

class TritonPythonModel:

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.model_version = args["model_version"]
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_file_dir)
        bundle_root = os.path.join(parent_dir, "{bundle_name}")
        sys.path = [bundle_root] + sys.path

        self.workflow = ConfigWorkflow(
            workflow_type="infer",
            bundle_root=bundle_root,
            config_file=os.path.join(bundle_root, "configs/inference.json"),
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
        )
        # update workflow
        override = {override}
        self.workflow.parser.update(override)
        self.workflow.initialize()
        print("model initialized!!!")

    def execute(self, requests):
        responses = []
        for request in requests:
            # input path is a folder of dicom series, and can contain multiple files
            input_path = pb_utils.get_input_tensor_by_name(request, "INPUT_PATH").as_numpy()[0].decode()
            # ensure no trailing `/`
            input_path = input_path.rstrip("/")
            output_dir = pb_utils.get_input_tensor_by_name(request, "OUTPUT_DIR").as_numpy()[0].decode()
            encode_prompts = pb_utils.get_input_tensor_by_name(request, "PROMPTS").as_numpy()[0].decode()
            prompts = ast.literal_eval(base64.b64decode(encode_prompts).decode("utf-8"))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            unlink("{output_dir}")
            os.symlink(output_dir, "{output_dir}")

            input_data = {{"image": input_path}}
            input_data.update(prompts)
            self.workflow.parser.ref_resolver.items["dataset"].config["data"][0] = input_data
            print("model version: ", self.model_version)

            # get inference file
            if os.path.isfile(input_path):
                filename_prefix, ext = os.path.splitext(os.path.basename(input_path))
                if ext == ".gz":
                    # assume only when ext is .gz, the file has two extensions (otherwise one exteinsion)
                    # reference:
                    # medical.data.utils.create_file_basename (used by the SaveImaged transform)
                    filename_prefix, ext = os.path.splitext(filename_prefix)
            else:
                filename_prefix = os.path.basename(input_path)
            pred_path = os.path.join(output_dir, f"{{filename_prefix}}_{output_postfix}{output_ext}")
            if os.path.exists(pred_path):
                os.remove(pred_path)
            output0_tensor = pb_utils.Tensor("OUTPUT_PATH", np.array([pred_path], dtype=np.object_))

            error_msg = ""
            try:
                # TODO: better to call self.workflow.run() (it requires the enhancement of medical.bundle code)
                self.workflow.evaluator.run()
                if not os.path.exists(pred_path):
                    error_msg = "Cannot find output file"
            except Exception as e:
                error_msg = str(e)
                torch.cuda.empty_cache()
                print(f"Got an unexcepted error: {{error_msg}}")
                # reinitialize if meet exceptions
                self.workflow.initialize()
            output1_tensor = pb_utils.Tensor("ERROR_MESSAGE", np.array([error_msg], dtype=np.object_))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output0_tensor, output1_tensor],
            )
            responses.append(inference_response)

            # unlink
            unlink("{output_dir}")

        return responses

    def finalize(self):
        self.workflow.finalize()

"""

TEMPLATE_TIS_CONFIG = """
name: "{tis_model}"
backend: "python"

input [
  {{
    name: "INPUT_PATH"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]

input [
  {{
    name: "OUTPUT_DIR"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]

input [
  {{
    name: "PROMPTS"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]

output [
  {{
    name: "OUTPUT_PATH"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]

output [
  {{
    name: "ERROR_MESSAGE"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]

instance_group [
  {{
    kind: KIND_GPU
    count: 1
    gpus: [ 0 ]
  }}
]

version_policy: {{
    latest: {{ num_versions: 1}}
}}
"""


TEMPLATE_MB_TRAIN = """
import datetime
import json
import sys

from medical.bundle import ConfigWorkflow
from medical.utils.enums import EngineStatsKeys as ESKeys

sys.path.append("{bundle_root}")

def train():
    status_dict = {{
        "date": datetime.date.today().isoformat(),
        "time": datetime.datetime.now().isoformat(),
        "status": "Unknown",
        "message": "MEDICAL jobs don't provide job status during training. Please download model artifacts to check."
    }}
    with open("{status_file}", "w") as f:
        f.write(json.dumps(status_dict) + "\\n")

    override = {override}
    tracking = override.pop("tracking", None)
    workflow = ConfigWorkflow(
        workflow_type="train",
        bundle_root="{bundle_root}",
        config_file={config_file_str},
        logging_file={logging_file_str},
        meta_file={meta_file_str},
        tracking = tracking,
    )
    workflow.parser.update(override)
    workflow.initialize()
    workflow.run()
    workflow.finalize()
    if workflow.evaluator is not None:
        stats = workflow.evaluator.get_stats()

        status_dict.update({{
            "date": datetime.date.today().isoformat(),
            "time": datetime.datetime.now().isoformat(),
            "status": "SUCCESS",
            "message": f"MEDICAL job completed. Please download model artifacts to check.",
            "epoch": stats[ESKeys.BEST_VALIDATION_EPOCH],
            "key_metric": stats[ESKeys.BEST_VALIDATION_METRIC],
        }})
    else:
        status_dict.update({{
            "date": datetime.date.today().isoformat(),
            "time": datetime.datetime.now().isoformat(),
            "status": "SUCCESS",
            "message": f"MEDICAL job completed. Please download model artifacts to check.",
            "epoch": -1,
            "key_metric": -1,
        }})
    with open("{status_file}", "a") as f:
        f.write(json.dumps(status_dict) + "\\n")

if __name__ == "__main__":
    train()
"""


TEMPLATE_DICOM_SEG_CONVERTER = """
import json
import os

import numpy as np
import pydicom

from uuid import uuid4
from medical.data.meta_tensor import MetaTensor
from medical.transforms import LoadImage, SaveImage

labels = {labels}

warn_set = set()
def get_label_index(label_name):
    # Find the index for the given label name
    for index, name in labels.items():
        _name = name.replace(" ", "_").lower()
        _label_name = label_name.replace(" ", "_").lower()
        if _name == _label_name:
            return int(index)
    if label_name not in warn_set:
        warn_set.add(label_name)
        print(f"labeled {{label_name}} not found model_params labels. Using 0 as default.")
    return 0

def extract_segment_info(dicom_data):
    segment_info = {{}}

    # Check if the SegmentSequence exists
    if "SegmentSequence" in dicom_data:
        for segment in dicom_data.SegmentSequence:
            # Extract segment number and label
            segment_number = segment.SegmentNumber
            segment_label = segment.SegmentLabel

            # Pair them in a dictionary
            segment_info[segment_number] = segment_label

    return segment_info

def change_label(item):
    dicom_dir = item["image"]
    seg_dir = item["label"]

    if not os.path.isdir(seg_dir):
        return
    filelist = os.listdir(seg_dir)
    seg_file_name = [x for x in filelist if ".dcm" in x][0]
    seg_file_path = os.path.join(seg_dir, seg_file_name)
    prefix = os.path.basename(seg_dir.rstrip("/"))
    output_postfix = str(uuid4())
    converted_file = os.path.join(seg_dir, prefix, prefix + "_" + output_postfix + ".nii.gz")
    dicom_img = LoadImage()(dicom_dir)
    if isinstance(dicom_img, tuple):
        dicom_img = dicom_img[0]
    seg_vol = np.zeros(dicom_img.shape)

    # Read the segmentation DICOM file
    seg_dcm = pydicom.dcmread(seg_file_path)
    # Extract the seg_record from the segmentation DICOM file
    try:
        seg_record = seg_dcm.pixel_array  # assuming that seg_dcm.pixel_array gives the segmentation record
        seg_info = extract_segment_info(seg_dcm)

        if len(seg_record.shape) == 2:
            seg_record = seg_record[np.newaxis, :, :]

        # Iterate over the PerFrameFunctionalGroupsSequence and update the seg_vol array
        for i, frame_item in enumerate(seg_dcm.PerFrameFunctionalGroupsSequence):
            frame_content_seq = frame_item.FrameContentSequence[0]
            class_id, slice_index = frame_content_seq.DimensionIndexValues
            class_name = seg_info[class_id]

            # Assuming seg_record[i, y, x] gives the i-th segmentation record for seg_vol[x, y, slice_index - 1]
            seg_slice = seg_record[i, :, :].T
            seg_vol[:, :, slice_index - 1][seg_slice > 0] = get_label_index(class_name)
    except TypeError as err:
        if "has no len()" in str(err):
            print("No segmentation record found in the DICOM file. Assume the segmentation is empty.")

    seg_vol = seg_vol[:, :, ::-1].copy()
    seg = MetaTensor(seg_vol, meta=dicom_img.meta)
    SaveImage(output_dir=seg_dir, output_postfix=output_postfix, output_dtype="uint16")(seg)
    item["label"] = converted_file


if __name__ == "__main__":
    for datalist_path in ["{train_datalist_path}", "{valid_datalist_path}"]:
        with open(datalist_path, "r") as fp:
            datalist = json.load(fp)

        for item in datalist:
            change_label(item)

        with open(datalist_path, "w") as fp:
            json.dump(datalist, fp, indent=2)
"""


TEMPLATE_CONTINUAL_LEARNING = """
import os
import shutil
import sys
import traceback
from time import sleep

sys.path.append("/opt/api")

from handlers.medical.helpers import DynamicSorter
from handlers.stateless_handlers import safe_load_file, update_job_status, printc
from job_utils.annotation_job_utils import (check_for_cancelation,
                                            handle_job_updates,
                                            initialize_cl_tracker,
                                            load_initial_state,
                                            process_notification_record)

notify_record = "{notify_record}"
job_context_dict = {job_context_dict}
logfile = "{logfile}"
handler_root = "{handler_root}"


if __name__ == "__main__":
    try:
        (user_id, experiment_id, cl_job_id, train_spec, round_size, stop_criteria,
         job_metadata_file, latest_mod_time, latest_record) = load_initial_state(job_context_dict, handler_root, notify_record)

        cl_tracker, cl_state = initialize_cl_tracker(stop_criteria)

        metric_sorter = DynamicSorter()
        jobs_trigger = []
        jobs_done = []

        while (not cl_tracker.should_stop(cl_state)):
            metadata = safe_load_file(job_metadata_file)
            check_for_cancelation(metadata, jobs_trigger, jobs_done, job_context_dict)
            update_job_status(user_id, experiment_id, cl_job_id, status="Running")
            # Process notification record updates and potentially trigger a new training job
            latest_mod_time, latest_record = process_notification_record(
                notify_record, latest_mod_time, latest_record, train_spec, round_size, cl_state, job_context_dict, jobs_trigger
            )

            # Check the status of all training jobs and handle triton model updates
            handle_job_updates(cl_state, jobs_trigger, jobs_done, metric_sorter, job_context_dict)
            sleep(15)  # Check every 15 seconds
    except Exception as e:
        # Something went wrong inside...
        print(traceback.format_exc(), file=sys.stderr)
        shutil.copy(logfile, os.path.join(handler_root, cl_job_id, "logs_from_toolkit.txt"))
        update_job_status(user_id, experiment_id, cl_job_id, status="Error")
        sys.exit(1)

    update_job_status(user_id, experiment_id, cl_job_id, status="Done")
    printc("Continual Learning job completed.", context=job_context_dict, keys="handler_id", file=sys.stderr)

"""
