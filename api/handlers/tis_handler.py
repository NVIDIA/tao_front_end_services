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

"""Triton Inference Service handler modules"""
import os
import sys

from handlers.docker_images import DOCKER_IMAGE_MAPPER
from handlers.utilities import Code, get_model_bundle_root
from job_utils import executor as jobDriver
import tritonclient.grpc as grpcclient
from time import sleep

image = DOCKER_IMAGE_MAPPER.get("medical-tis")


class TISHandler:
    """
    Handler class for Triton Inference Service jobs
    Start: Start controller as a K8s job
    Stop: Stop controller and stop the triton inference server that is running

    """

    @staticmethod
    def start(org_name, model_id, handler_metadata, model_name, replicas=1):
        """Starts a Triton Inference Service job by running tis_start.py file"""
        print(f"Starting deploy triton inference server {model_id}", file=sys.stderr)
        model_repo = get_model_bundle_root(org_name, model_id)
        bundle_requirements_file = os.path.join(model_repo, model_name, "requirements.txt")
        # TODO: can leverage the shared pv to print logs inside the triton server.
        # https://gitlab-master.nvidia.com/dlmed/medical-service/-/merge_requests/71#note_17855255
        pre_command = f"umask 0 && pip install -r {bundle_requirements_file}" if os.path.exists(bundle_requirements_file) else "umask 0"
        run_command = f"{pre_command} && /opt/tritonserver/bin/tritonserver --model-repository={model_repo} --model-control-mode=explicit --load-model=*"
        # ports for http, grpc and metrics
        ports = (8000, 8001, 8002)
        jobDriver.create_triton_deployment(model_id, image, run_command, replicas=replicas, num_gpu=1, ports=ports)
        # TODO: the actual timeout is 2 * realtime_infer_request_timeout because in inner loop for tis service
        # we also use this number. We can enhance it in the future.
        timeout = handler_metadata.get("realtime_infer_request_timeout", 60)
        not_ready_log = False
        print("Check deployment status", file=sys.stderr)
        while (timeout > 0):
            stat_dict = jobDriver.status_triton_deployment(model_id, replicas=replicas)
            status = stat_dict.get("status", "Unknown")
            if status == "Running":
                print(f"Deployed triton inference server {model_id}", file=sys.stderr)
                # k8s service naming rule requres to start with an alphabetic character
                # https://kubernetes.io/docs/concepts/overview/working-with-objects/names/
                tis_service_id = f"service-{model_id}"
                return TISHandler.start_tis_service(tis_service_id, deploy_label=model_id, ports=ports, handler_metadata=handler_metadata)
            if status == "ReplicaNotReady" and not_ready_log is False:
                print("TIS is deployed but replica not ready.", file=sys.stderr)
                not_ready_log = True
            sleep(1)
            timeout -= 1
        print(f"Failed to deploy triton inference server {model_id}", file=sys.stderr)
        return Code(400, {}, f"Timeout Error: Triton Inference Server status: {status} after {timeout} seconds")

    @staticmethod
    def update(model_id, model_name):
        """Updates a Triton Inference Service deployment by reloading the model"""
        print(f"Updating {model_id}", file=sys.stderr)
        pods_ip = jobDriver.get_triton_deployment_pods(model_id)
        if len(pods_ip) == 0:
            return Code(400, [], f"Cannot find pods for {model_id}.")
        try:
            for pod_ip in pods_ip:
                # TODO: hardcode the port for now
                url = f"{pod_ip}:8001"
                client = grpcclient.InferenceServerClient(url=url, verbose=False)
                client.load_model(model_name)
                print(f"Updated deployment {model_id} replica {pod_ip}", file=sys.stderr)
            return Code(201, [], f"Updated deployment {model_id}")
        except Exception as e:
            print(f"Failed to update deployment {model_id}", file=sys.stderr)
            return Code(400, [], f"Failed to update deployment {model_id} with error {e}")

    @staticmethod
    def start_tis_service(tis_service_id, deploy_label, ports, handler_metadata):
        """
        Starts a Triton Inference Service job by running tis_start.py file
        Args:
            tis_service_id: the id of the tis service
            deploy_label: the label of the deployment
            ports: the ports for the tis service
            handler_metadata: the metadata of the model handler
        """
        jobDriver.create_tis_service(tis_service_id, deploy_label, ports=ports)
        tis_timeout = handler_metadata.get("realtime_infer_request_timeout", 60)
        not_ready_log = False
        print("Check TIS Service status", file=sys.stderr)
        while (tis_timeout > 0):
            service_stat_dict = jobDriver.status_tis_service(tis_service_id, ports=ports)
            service_status = service_stat_dict.get("status", "Unknown")
            if service_status == "Running":
                print(f"Created TIS service {tis_service_id}", file=sys.stderr)
                tis_service_ip = service_stat_dict.get("tis_service_ip", None)
                return Code(201, {"pod_ip": tis_service_ip}, "TIS Service Running")
            if service_status == "NotReady" and not_ready_log is False:
                print("TIS Service is started but not ready.", file=sys.stderr)
                not_ready_log = True
            sleep(1)
            tis_timeout -= 1
        print(f"Failed to create TIS service {tis_service_id}", file=sys.stderr)
        return Code(400, {}, f"Error: TIS service status: {service_status}")

    @staticmethod
    def stop(model_id, handler_metadata):
        """Stops a Triton Inference Service job"""
        print("Stopping triton inference server", file=sys.stderr)
        timeout = handler_metadata.get("realtime_infer_request_timeout", 60)
        jobDriver.delete_triton_deployment(model_id)
        while (timeout > 0):
            stat_dict = jobDriver.status_triton_deployment(model_id)
            status = stat_dict.get("status", "Unknown")
            if status == "NotFound":
                print(f"Stopped triton deployment {model_id}", file=sys.stderr)
                tis_service_id = f"service-{model_id}"
                return TISHandler.stop_tis_service(tis_service_id, handler_metadata)
            sleep(1)
            timeout -= 1

        print(f"Failed to delete triton inference server {model_id}", file=sys.stderr)
        return Code(400, [], f"TIS model {model_id} cannot be stopped in platform")

    @staticmethod
    def stop_tis_service(tis_service_id, handler_metadata):
        """
        Stops a Triton Inference Service job
        Args:
            tis_service_id: the id of the tis service
            handler_metadata: the metadata of the model handler
        Returns:
            Code: the code of the result, 201 if success, 400 if failed
        """
        jobDriver.delete_tis_service(tis_service_id)
        tis_timeout = handler_metadata.get("realtime_infer_request_timeout", 60)
        while (tis_timeout > 0):
            service_stat_dict = jobDriver.status_tis_service(tis_service_id)
            service_status = service_stat_dict.get("status", "Unknown")
            if service_status == "NotFound":
                print(f"Stopped TIS service {tis_service_id}", file=sys.stderr)
                return Code(201, {}, "Triton Inference Server Stopped")
            sleep(1)
            tis_timeout -= 1
        print(f"Failed to delete TIS Service {tis_service_id}", file=sys.stderr)
        return Code(400, [], f"TIS Service {tis_service_id} cannot be stopped in platform")
