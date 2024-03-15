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

"""MEDICAL Model Handler module."""
import base64
import os
import sys
from time import sleep
from uuid import uuid4

import numpy as np
import tritonclient.grpc as grpcclient
import validators
from tritonclient.utils import InferenceServerException, np_to_triton_dtype

from handlers.medical.dataset.cache import CacheInfo
from handlers.medical_dataset_handler import MedicalDatasetHandler
from handlers.utilities import Code
from job_utils import executor as jobDriver


class MedicalModelHandler:
    """MEDICAL Model Handler class."""

    @staticmethod
    def get_schema(action):
        """Provide schema for each action in MEDICAL models/bundles."""
        if action == "inference":
            return {
                "image": "",
                "dataset_id": "",
                "bundle_params": {}
            }
        return {"default": {}}

    @staticmethod
    def run_inference(user_id, handler_id, handler_metadata, spec):
        """
        Method for medical triton client.

        Args:
            user_id: User ID
            handler_id: Handler ID, usually experiment_id.
            handler_metadata: Handler MetaData
            spec: Spec for Infer action which contains
                  - `image_url` to fetch dicom images from dicom web server.
                  - `bundle_params` for prompts to be used for inference.

        """
        image = spec.get("image", None)
        if image is None:
            return Code(400, [], "image is required for inference action")

        max_attempts = 10
        tis_service_id = f"service-{handler_id}"

        # TODO: for model repository update, we can probably remove the next 6 lines.
        while jobDriver.status_tis_service(tis_service_id).get("status", "Unknown") != "Running":
            if max_attempts == 0:
                return Code(400, [], "Triton Inference Server is not running")
            # Triton Inference Server is not running yet. It might be that the model is being swapped by CL.
            max_attempts -= 1
            sleep(5)

        model_name = handler_metadata["realtime_infer_model_name"]
        pod_ip = handler_metadata["realtime_infer_endpoint"]

        dataset_id = spec.get("dataset_id")
        if not dataset_id and not validators.url(image):
            infer_ds = handler_metadata.get("inference_dataset")
            train_ds = handler_metadata.get("train_datasets")
            train_ds = train_ds[0] if train_ds else None
            dataset_id = infer_ds if infer_ds else train_ds

        # Get the input path from cacheimage
        response = MedicalDatasetHandler.from_cache(user_id, dataset_id, image)
        if response.code != 201:
            return response
        if response.data is None:
            return Code(400, [], "failed to fetch/determine image source")

        cache_info: CacheInfo = CacheInfo(c=response.data)
        input_path = cache_info.image

        # Make the output_path in user cache dir
        tmp_job_id = str(uuid4())
        output_dir = os.path.join(os.path.join(os.path.dirname(input_path), "labels"), tmp_job_id)
        os.makedirs(output_dir, exist_ok=False)
        os.chmod(output_dir, 0o777)

        # TODO: make port number configurable
        url = f"{pod_ip}:8001"
        client = grpcclient.InferenceServerClient(url=url, verbose=False)
        inputs = [
            grpcclient.InferInput("INPUT_PATH", [1], np_to_triton_dtype(np.object_)),
            grpcclient.InferInput("OUTPUT_DIR", [1], np_to_triton_dtype(np.object_)),
            grpcclient.InferInput("PROMPTS", [1], np_to_triton_dtype(np.object_)),
        ]
        outputs = [grpcclient.InferRequestedOutput("OUTPUT_PATH"), grpcclient.InferRequestedOutput("ERROR_MESSAGE")]

        inputs[0].set_data_from_numpy(np.array([input_path], dtype=np.object_))
        inputs[1].set_data_from_numpy(np.array([output_dir], dtype=np.object_))

        # FIXME: please make use of the bundle_params in OHIF
        bundle_params = spec.get("bundle_params", {})
        encode_prompts = base64.b64encode(str(bundle_params).encode("utf-8"))
        inputs[2].set_data_from_numpy(np.array([encode_prompts], dtype=np.object_))

        try:
            response = client.infer(model_name, inputs, request_id=str(uuid4().hex), outputs=outputs)
            output_path = response.as_numpy("OUTPUT_PATH")[0].decode()
            error_msg = response.as_numpy("ERROR_MESSAGE")[0].decode()
            if not os.path.exists(output_path):
                print(f"Run inference on input {input_path} with model {model_name} got error: {error_msg}", file=sys.stderr)
                return Code(400, [], f"Error: {error_msg}")
            res = Code(201, {"pred": output_path}, "Triton Inference Success" if error_msg == "" else error_msg)
            res.attachment_key = "pred"
            return res
        except InferenceServerException as e:
            return Code(400, [], f"Error: {e}")
