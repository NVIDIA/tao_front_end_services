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

"""Kubernetes job manager modules"""
import os
import sys
import time
import uuid
import requests
import traceback

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from constants import TAO_NETWORKS, NETWORK_CONTAINER_MAPPING
from handlers.stateless_handlers import BACKEND, get_handler_job_metadata, get_toolkit_status
from handlers.utilities import send_microservice_request
from handlers.mongo_handler import mongo_secret
from nvcf_controller import get_nvcf_job_status


def _get_name_space():
    """Returns the namespace of the environment"""
    if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
        name_space = os.getenv('NAMESPACE', default="default")
        config.load_kube_config()
    else:
        with open('/var/run/secrets/kubernetes.io/serviceaccount/namespace', 'r', encoding='utf-8') as f:
            current_name_space = f.read()
        name_space = os.getenv('NAMESPACE', default=current_name_space)
        config.load_incluster_config()
    return name_space


def get_service_in_cluster_ip(service_name, namespace="default"):
    """Get the cluster IP of a service"""
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    service = v1.read_namespaced_service(namespace=namespace, name=service_name)
    return service.spec.cluster_ip


def create(user_id, org_name, job_name, image, command, num_gpu=-1, accelerator=None, docker_env_vars=None, port=False, nv_job_metadata=None, automl_brain=False, cl_medical=False, local_cluster=False):
    """Creates a kubernetes job"""
    name_space = _get_name_space()
    host_base_url = os.getenv("HOSTBASEURL", "no_url")
    if host_base_url == "no_url":
        raise ValueError("Base URL not set in values yaml. Please set it as http(s)://<ip_address>:<ingress-nginx-controller service's port number>")
    if BACKEND in ("BCP", "NVCF") and nv_job_metadata:
        config.load_incluster_config()

        # Initialize the custom objects API
        api_instance = client.CustomObjectsApi()

        if BACKEND == "BCP":
            # Define the CR body based on your CRD
            crd_group = 'dgx-job-manager.nvidia.io'
            crd_version = 'v1alpha1'
            crd_plural = 'dgxjobs'

            nv_job_body = {
                "apiVersion": f"{crd_group}/{crd_version}",
                "kind": "DgxJob",
                "metadata": {
                    "name": job_name + "-dgx",
                    "namespace": name_space,
                },
                "spec": {
                    "user_id": os.getenv("CALLBACK_UUID", user_id),
                    "name": job_name,
                    "job_id": job_name + "-dgx",
                    "dockerImageName": nv_job_metadata["dockerImageName"],
                    "command": nv_job_metadata["command"],
                    "orgName": nv_job_metadata["orgName"],
                    "teamName": nv_job_metadata["teamName"],
                    "aceName": nv_job_metadata["aceName"],
                    "aceInstance": nv_job_metadata["aceInstance"],
                    "runPolicy": nv_job_metadata["runPolicy"],
                    "resultContainerMountPoint": nv_job_metadata["resultContainerMountPoint"],
                    "envs": nv_job_metadata["envs"],
                },
                "status": {
                    "phase": "Pending"
                }
            }
        elif BACKEND == "NVCF":
            # Define the CR body based on your CRD
            crd_group = 'nvcf-job-manager.nvidia.io'
            crd_version = 'v1alpha1'
            crd_plural = 'nvcfjobs'

            nv_job_body = {
                "apiVersion": f"{crd_group}/{crd_version}",
                "kind": "NvcfJob",
                "metadata": {
                    "name": job_name + "-nvcf",
                    "namespace": name_space,
                },
                "spec": {
                    "org_name": nv_job_metadata["orgName"],
                    "action": nv_job_metadata["action"],
                    "workspace_ids": nv_job_metadata["workspace_ids"],
                    "deployment_string": nv_job_metadata["deployment_string"],
                    "container": nv_job_metadata["dockerImageName"],
                    "job_id": job_name,
                    "network": nv_job_metadata["network"],
                    "ngc_api_key": nv_job_metadata["TAO_USER_KEY"],
                    "spec_file_path": nv_job_metadata["spec_file_path"],
                    "tao_api_ui_cookie": nv_job_metadata["TAO_COOKIE_SET"],
                    "tao_api_admin_key": nv_job_metadata["TAO_ADMIN_KEY"],
                    "tao_api_base_url": nv_job_metadata["TAO_API_SERVER"],
                    "tao_api_status_callback_url": nv_job_metadata["TAO_LOGGING_SERVER_URL"],
                    "use_ngc_staging": nv_job_metadata["USE_NGC_STAGING"],
                    "automl_experiment_number": nv_job_metadata.get("AUTOML_EXPERIMENT_NUMBER", "0"),
                },
                "status": {
                    "phase": "Pending"
                }
            }
            print("nv_job_body", nv_job_body, file=sys.stderr)

        # Create the custom resource in the specified namespace
        try:
            api_instance.create_namespaced_custom_object(crd_group, crd_version, name_space, crd_plural, nv_job_body)
            print(f"NvJob CR '{job_name}' created successfully.", file=sys.stderr)
        except Exception as e:
            print(f"Failed to create NvJob CR '{job_name}': {e}", file=sys.stderr)
        return

    command = 'umask 0 && ' + command
    if num_gpu == -1:
        num_gpu = int(os.getenv('NUM_GPU_PER_NODE', default='1'))
    node_selector = {'accelerator': str(accelerator)}
    if not accelerator:
        node_selector = None
    claim_name = os.getenv('CLAIMNAME', 'tao-api-pvc')
    image_pull_secret = os.getenv('IMAGEPULLSECRET', default='imagepullsecret')
    name_space = _get_name_space()
    api_instance = client.BatchV1Api()

    volume_mounts = []
    if automl_brain or cl_medical:
        experiments_volume_mount = client.V1VolumeMount(
            name="shared-data",
            mount_path=f"/shared/orgs/{org_name}/experiments/",
            sub_path=f"orgs/{org_name}/experiments")
        jobs_volume_mount = client.V1VolumeMount(
            name="shared-data",
            mount_path=f"/shared/orgs/{org_name}/users/{user_id}/",
            sub_path=f"orgs/{org_name}/users/{user_id}")
        volume_mounts.append(experiments_volume_mount)
        volume_mounts.append(jobs_volume_mount)
    if cl_medical:
        # Continual learning needs to monitor the change of datasets from the notification
        data_volume_mount = client.V1VolumeMount(
            name="shared-data",
            mount_path=f"/shared/orgs/{org_name}/datasets/",
            sub_path=f"orgs/{org_name}/datasets")
        volume_mounts.append(data_volume_mount)
    if local_cluster:
        # Local job needs to mount the local job folder to avoid uploading/download the trained model
        local_cluster_volume_mount = client.V1VolumeMount(
            name="shared-data",
            mount_path=f"/shared/orgs/{org_name}/users/{user_id}/jobs/{job_name}",
            sub_path=f"orgs/{org_name}/users/{user_id}/jobs/{job_name}")
        volume_mounts.append(local_cluster_volume_mount)
        in_cluster_ip = get_service_in_cluster_ip("ingress-nginx-controller", namespace=name_space)
        # change the host_base_url to the in-cluster ip
        in_cluster_url = f"http://{in_cluster_ip}:80" if nv_job_metadata is None else None
        if "TAO_API_SERVER" in docker_env_vars:
            docker_env_vars["TAO_API_SERVER"] = docker_env_vars["TAO_API_SERVER"].replace(host_base_url, in_cluster_url)
        if "TAO_LOGGING_SERVER_URL" in docker_env_vars:
            docker_env_vars["TAO_LOGGING_SERVER_URL"] = docker_env_vars["TAO_LOGGING_SERVER_URL"].replace(host_base_url, in_cluster_url)
    dshm_volume_mount = client.V1VolumeMount(
        name="dshm",
        mount_path="/dev/shm")
    volume_mounts.append(dshm_volume_mount)

    resources = client.V1ResourceRequirements(
        limits={
            'nvidia.com/gpu': str(num_gpu)
        })
    capabilities = client.V1Capabilities(
        add=['SYS_PTRACE']
    )
    security_context = client.V1SecurityContext(
        capabilities=capabilities
    )
    backend_env = client.V1EnvVar(
        name="BACKEND",
        value=BACKEND)
    # CL job needs to set the environment variable to pass GPU checks (validate_num_gpu) for training jobs
    num_gpu_env = client.V1EnvVar(
        name="NUM_GPU_PER_NODE",
        value=str(num_gpu) if not cl_medical else os.getenv('NUM_GPU_PER_NODE', default='1'))
    mongo_secret_env = client.V1EnvVar(
        name="MONGOSECRET",
        value=mongo_secret
    )
    dynamic_docker_envs = []
    if docker_env_vars:
        for docker_env_var_key, docker_env_var_value in docker_env_vars.items():
            kubernetes_env = client.V1EnvVar(
                name=docker_env_var_key,
                value=docker_env_var_value)
            dynamic_docker_envs.append(kubernetes_env)

    tis_ports = [
        client.V1ContainerPort(container_port=8000, name="http-triton"),
        client.V1ContainerPort(container_port=8001, name="grpc-triton"),
        client.V1ContainerPort(container_port=8002, name="metrics-triton")
    ]
    container = client.V1Container(
        name="container",
        image=image,
        env=[backend_env, num_gpu_env, mongo_secret_env] + dynamic_docker_envs,
        command=["/bin/bash", "-c"],
        args=[command],
        resources=resources,
        volume_mounts=volume_mounts,
        ports=[] if port is False else tis_ports,
        security_context=security_context)
    shared_volume = client.V1Volume(
        name="shared-data",
        persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name=claim_name))
    dshm_volume = client.V1Volume(
        name="dshm",
        empty_dir=client.V1EmptyDirVolumeSource(medium='Memory'))
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(
            labels={"purpose": "tao-toolkit-job"}
        ),
        spec=client.V1PodSpec(
            image_pull_secrets=[client.V1LocalObjectReference(name=image_pull_secret)],
            containers=[container],
            volumes=[shared_volume, dshm_volume],
            node_selector=node_selector,
            restart_policy="Never"))
    spec = client.V1JobSpec(
        ttl_seconds_after_finished=100,
        template=template,
        backoff_limit=0)
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=job_name),
        spec=spec)

    try:
        api_instance.create_namespaced_job(
            body=job,
            namespace=name_space)
        return
    except:
        print(traceback.format_exc(), file=sys.stderr)
        return


def create_flask_service(job_id):
    """Create a service for a microservice pod"""
    try:
        name_space = _get_name_space()
        core_v1 = client.CoreV1Api()
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(name=f"flask-service-{job_id}"),
            spec=client.V1ServiceSpec(
                cluster_ip=None,  # Headless service
                selector={
                    "app": "flask",
                    "job-id": job_id
                },
                ports=[client.V1ServicePort(port=8000, target_port=8000)]
            )
        )
        core_v1.create_namespaced_service(namespace=name_space, body=service)
    except:
        print(traceback.format_exc(), file=sys.stderr)
        return


def delete_service(job_id):
    """Delete a microservice pod's service"""
    try:
        name_space = _get_name_space()
        service_name = f"flask-service-{job_id}"
        core_v1 = client.CoreV1Api()
        core_v1.delete_namespaced_service(name=service_name, namespace=name_space)
    except:
        print(traceback.format_exc(), file=sys.stderr)
        return


def create_microservice_pod(job_name, image, num_gpu=-1, accelerator=None):
    """Create pod to invoke microservices"""
    create_flask_service(job_name)
    name_space = _get_name_space()
    api_instance = client.BatchV1Api()

    image_pull_secret = os.getenv('IMAGEPULLSECRET', default='imagepullsecret')
    node_selector = {'accelerator': str(accelerator)}
    if not accelerator:
        node_selector = None

    dshm_volume = client.V1Volume(
        name="dshm",
        empty_dir=client.V1EmptyDirVolumeSource(medium='Memory'))

    dshm_volume_mount = client.V1VolumeMount(
        name="dshm",
        mount_path="/dev/shm")

    resources = client.V1ResourceRequirements(
        limits={
            'nvidia.com/gpu': str(num_gpu)
        })
    capabilities = client.V1Capabilities(
        add=['SYS_PTRACE']
    )
    security_context = client.V1SecurityContext(
        capabilities=capabilities
    )

    container = client.V1Container(
        name="container",
        image=image,
        command=["/bin/bash", "-c"],
        args=["flask run --host 0.0.0.0 --port 8000"],
        resources=resources,
        volume_mounts=[dshm_volume_mount],
        ports=[],
        security_context=security_context)

    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(
            labels={
                "purpose": "tao-toolkit-job",
                "app": "flask",
                "job-id": job_name
            }
        ),
        spec=client.V1PodSpec(
            image_pull_secrets=[client.V1LocalObjectReference(name=image_pull_secret)],
            containers=[container],
            volumes=[dshm_volume],
            node_selector=node_selector,
            restart_policy="Never"))

    spec = client.V1JobSpec(
        ttl_seconds_after_finished=100,
        template=template,
        backoff_limit=0)

    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=job_name),
        spec=spec)

    try:
        api_instance.create_namespaced_job(
            body=job,
            namespace=name_space)

        # Wait for the pod to be running
        core_v1 = client.CoreV1Api()
        pod_name = None
        while not pod_name:
            pods = core_v1.list_namespaced_pod(namespace='default', label_selector=f'job-name={job_name}')
            if pods.items:
                pod_name = pods.items[0].metadata.name
            time.sleep(10)

        print(f"Pod {pod_name} is running. Waiting for it to be ready", file=sys.stderr)

        # Ensure the pod is ready
        pod_ip = None
        pod_ready = False
        while not pod_ready or not pod_ip:
            pod = core_v1.read_namespaced_pod(name=pod_name, namespace='default')
            pod_ready = all(
                condition.status == 'True'
                for condition in pod.status.conditions
                if condition.type == 'Ready'
            )
            pod_ip = pod.status.pod_ip
            time.sleep(10)

        print(f"Pod {pod_name} is ready with IP {pod_ip}.", file=sys.stderr)
        time.sleep(10)
    except:
        print(traceback.format_exc(), file=sys.stderr)


def check_service_ready(service_name, namespace):
    """
    Check if the specified service is ready.

    Args:
        service_name (str): The name of the service to check.
        namespace (str): The namespace where the service is located.

    Returns:
        bool: True if the service is found, False otherwise.
    """
    try:
        _ = client.CoreV1Api().read_namespaced_service(name=service_name, namespace=namespace)
        return True
    except ApiException as e:
        if e.status == 404:
            return False
        raise e


def check_endpoints_ready(service_name, namespace):
    """
    Check if the specified service has ready endpoints.

    Args:
        service_name (str): The name of the service to check.
        namespace (str): The namespace where the service is located.

    Returns:
        bool: True if the service has ready endpoints, False otherwise.
    """
    try:
        endpoints = client.CoreV1Api().read_namespaced_endpoints(name=service_name, namespace=namespace)
        if not endpoints.subsets:
            return False
        for subset in endpoints.subsets:
            if subset.addresses:
                return True
        return False
    except ApiException as e:
        if e.status == 404:
            return False
        raise e


def wait_for_service(org_name, handler_id, job_id, handler_kind):
    """
    Wait until the specified service is ready or timeout is reached.

    Args:
        org_name (str): Org name under which job is submitted.
        handler_id (uuid): The handler id associated with the job.
        job_id (uuid): The job_id associated with the name of the service to wait for.
        handler_kind (str): If the job belongs to datasets or experiments.

    Returns:
        bool: True if the service is ready within the timeout period, False otherwise.
    """
    service_name = f"flask-service-{job_id}"
    namespace = _get_name_space()
    start_time = time.time()
    while time.time() - start_time < 300:
        metadata_status = get_handler_job_metadata(job_id).get("status")
        if metadata_status in ("Canceled", "Canceling", "Paused", "Pausing"):
            return metadata_status
        if check_service_ready(service_name, namespace) and check_endpoints_ready(service_name, namespace):
            print(f"Service '{service_name}' is ready.", file=sys.stderr)
            return "Running"
        print(f"Waiting for service '{service_name}' to be ready...", file=sys.stderr)
        time.sleep(10)
    print(f"Timed out waiting for service '{service_name}' to be ready.", file=sys.stderr)
    return "Error"


def create_microservice_and_send_request(api_endpoint, network, action, ngc_api_key="", cloud_metadata={}, specs={}, microservice_pod_id="", tao_api_admin_key="", tao_api_base_url="", tao_api_status_callback_url="", tao_api_ui_cookie="", use_ngc_staging="", automl_experiment_number="", num_gpu=-1, microservice_container="", org_name="", handler_id="", handler_kind=""):
    """Create a DNN container microservice pod and send request to the POD IP"""
    try:
        if not microservice_pod_id:
            microservice_pod_id = str(uuid.uuid4())
        if num_gpu == -1:
            num_gpu = int(os.getenv('NUM_GPU_PER_NODE', default='1'))
        if not microservice_container:
            microservice_container = os.getenv(f'IMAGE_{NETWORK_CONTAINER_MAPPING[network]}')
            if action == "gen_trt_engine":
                microservice_container = os.getenv('IMAGE_TAO_DEPLOY')
        create_microservice_pod(microservice_pod_id, microservice_container, num_gpu=num_gpu)
        if wait_for_service(org_name, handler_id, microservice_pod_id, handler_kind):
            response = send_microservice_request(api_endpoint, network, action,
                                                 ngc_api_key=ngc_api_key,
                                                 cloud_metadata=cloud_metadata,
                                                 specs=specs,
                                                 job_id=microservice_pod_id,
                                                 tao_api_admin_key=tao_api_admin_key,
                                                 tao_api_base_url=tao_api_base_url,
                                                 tao_api_status_callback_url=tao_api_status_callback_url,
                                                 tao_api_ui_cookie=tao_api_ui_cookie,
                                                 use_ngc_staging=use_ngc_staging,
                                                 automl_experiment_number=automl_experiment_number)
            if api_endpoint != "post_action":
                delete(microservice_pod_id, use_ngc=False)
            return response
        return None
    except:
        print("Exception in create ms pod and send request", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        delete(microservice_pod_id, use_ngc=False)
        return None


def create_triton_deployment(deployment_name, image, command, replicas, num_gpu=-1, ports=(8000, 8001, 8002)):
    """Creates a Triton deployment"""
    # You can use NFS for local development

    claim_name = os.getenv('CLAIMNAME', 'tao-api-pvc')
    image_pull_secret = os.getenv('IMAGEPULLSECRET', default='imagepullsecret')
    name_space = _get_name_space()
    api_instance = client.AppsV1Api()
    shared_volume_mount = client.V1VolumeMount(
        name="shared-data",
        mount_path="/shared")
    dshm_volume_mount = client.V1VolumeMount(
        name="dshm",
        mount_path="/dev/shm")
    resources = client.V1ResourceRequirements(
        limits={
            'nvidia.com/gpu': 1
            # can add other resources like cpu, memory
        })
    capabilities = client.V1Capabilities(
        add=['SYS_PTRACE']
    )
    security_context = client.V1SecurityContext(
        capabilities=capabilities
    )
    tis_ports = [
        client.V1ContainerPort(container_port=ports[0], name="http-triton"),
        client.V1ContainerPort(container_port=ports[1], name="grpc-triton"),
        client.V1ContainerPort(container_port=ports[2], name="metrics-triton")
    ]
    container = client.V1Container(
        name="container",
        image=image,
        command=["/bin/bash", "-c"],
        args=[command],
        resources=resources,
        volume_mounts=[shared_volume_mount, dshm_volume_mount],
        ports=tis_ports,
        security_context=security_context)

    shared_volume = client.V1Volume(
        name="shared-data",
        persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name=claim_name))

    dshm_volume = client.V1Volume(
        name="dshm",
        empty_dir=client.V1EmptyDirVolumeSource(medium='Memory'))
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(
            labels={
                "purpose": "tao-toolkit-job",
                "app": deployment_name,  # use deployment_name as the selector name
            }
        ),
        spec=client.V1PodSpec(
            image_pull_secrets=[client.V1LocalObjectReference(name=image_pull_secret)],
            containers=[container],
            volumes=[shared_volume, dshm_volume],
        ))
    spec = client.V1DeploymentSpec(
        replicas=replicas,
        template=template,
        selector={"matchLabels": {"app": deployment_name}})
    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=deployment_name),
        spec=spec)
    print("Prepared deployment configs", file=sys.stderr)
    try:
        api_instance.create_namespaced_deployment(
            body=deployment,
            namespace=name_space)
        print("Start create deployment", file=sys.stderr)
        return
    except Exception as e:
        print(f"Create deployment got error: {e}", file=sys.stderr)
        return


def create_tis_service(tis_service_name, deploy_label, ports=(8000, 8001, 8002)):
    """Create TIS service"""
    name_space = _get_name_space()
    tis_ports = [
        client.V1ServicePort(name="http", protocol="TCP", port=ports[0], target_port=ports[0]),
        client.V1ServicePort(name="grpc", protocol="TCP", port=ports[1], target_port=ports[1]),
        client.V1ServicePort(name="metrics", protocol="TCP", port=ports[2], target_port=ports[2]),
    ]
    spec = client.V1ServiceSpec(ports=tis_ports, selector={"app": deploy_label}, type="LoadBalancer")
    # add annotation, it will only works in Azure, but will not affect other cloud
    annotation = {
        "service.beta.kubernetes.io/azure-load-balancer-internal": "true"
    }
    service = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(name=tis_service_name, labels={"app": tis_service_name}, annotations=annotation),
        spec=spec,
    )
    api_instance = client.CoreV1Api()
    print("Prepared TIS Service configs", file=sys.stderr)
    try:
        api_instance.create_namespaced_service(
            body=service,
            namespace=name_space)
        print("Start create TIS Service", file=sys.stderr)
        return
    except Exception as e:
        print(f"Create TIS Service got error: {e}", file=sys.stderr)
        return


def get_triton_deployment_pods(deployment_name):
    """Returns pods of a Triton deployment"""
    name_space = _get_name_space()
    api_instance = client.CoreV1Api()
    label_selector = f"app={deployment_name}"
    try:
        pods = api_instance.list_namespaced_pod(
            namespace=name_space,
            label_selector=label_selector)
        pods_ip = []
        for pod in pods.items:
            pods_ip.append(pod.status.pod_ip)
        return pods_ip
    except Exception as e:
        print(f"Got {type(e)} error: {e}", file=sys.stderr)
        return []


def status_triton_deployment(deployment_name, replicas=1):
    """
    Returns status of Triton deployment
    Status definition:

    Running: The Triton deployment is ready and running
    ReplicaNotReady: at least one replica of the deployment is not ready.
    NotFound: cannot find the deployment.  This status is useful to check if the deployment is stopped.
    Error: meet exceptions except not found error when check the status.

    """
    name_space = _get_name_space()
    api_instance = client.AppsV1Api()
    try:
        api_response = api_instance.read_namespaced_deployment_status(
            name=deployment_name,
            namespace=name_space)
        available_replicas = api_response.status.available_replicas
        if not isinstance(available_replicas, int) or available_replicas < replicas:
            return {"status": "ReplicaNotReady"}
        return {"status": "Running"}
    except ApiException as e:
        if e.status == 404:
            print("Trion Deployment not found.", file=sys.stderr)
            # TODO: here defined a new status to find the situation that the deployment does not exists
            # This status is useful to check if the deployment is deleted or not created
            return {"status": "NotFound"}
        print(f"Got other ApiException error: {e}", file=sys.stderr)
        return {"status": "Error"}
    except Exception as e:
        print(f"Got {type(e)} error: {e}", file=sys.stderr)
        return {"status": "Error"}


def status_tis_service(tis_service_name, ports=(8000, 8001, 8002)):
    """
    Returns status of TIS Service
    Status definition:

    Running: The TIS Service is ready and running
    NotReady: the TIS Service is not ready.
    NotFound: cannot find the TIS Service. This status is useful to check if the service is stopped.
    Error: meet exceptions except not found error when check the status.
    """
    name_space = _get_name_space()
    try:
        tis_service_ip = get_service_in_cluster_ip(tis_service_name, namespace=name_space)
        # need to double confirm if the service is ready
        url = f"http://{tis_service_ip}:{ports[0]}/v2/health/ready"
        try:
            endpoint_response = requests.get(url)
            if endpoint_response.status_code == 200:
                return {"status": "Running", "tis_service_ip": tis_service_ip}
            # TODO: here defined a new status, in order to find the situation that the TIS Service is started but not ready.
            return {"status": "NotReady"}
        except:
            return {"status": "NotReady"}
    except ApiException as e:
        if e.status == 404:
            print("TIS Service not found.", file=sys.stderr)
            # TODO: here defined a new status, in order to find the situation that the TIS Service not exists
            # This status is useful to check if the TIS Service is deleted or not created
            return {"status": "NotFound"}
        print(f"Got other ApiException error: {e}", file=sys.stderr)
        return {"status": "Error"}
    except Exception as e:
        print(f"Got {type(e)} error: {e}", file=sys.stderr)
        return {"status": "Error"}


def override_k8_status(job_name, k8_status):
    """Override kubernetes job status with toolkit status"""
    toolkit_status = get_toolkit_status(job_name)
    override_status = ""
    if k8_status == "Pending":  # We don't want to reverse done/error status to running
        if toolkit_status in ("STARTED", "RUNNING"):
            override_status = "Running"
    if toolkit_status == "SUCCESS":
        override_status = "Done"
    if toolkit_status == "FAILURE":
        override_status = "Error"
    return override_status


def status(org_name, handler_id, job_name, handler_kind, use_ngc=True, network="", action=""):
    """Returns status of kubernetes job"""
    name_space = _get_name_space()
    if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
        config.load_kube_config()
    else:
        config.load_incluster_config()
    dgxjob_api_response = None
    if BACKEND in ("BCP", "NVCF") and use_ngc:
        if BACKEND == "BCP":
            crd_group = 'dgx-job-manager.nvidia.io'
            crd_version = 'v1alpha1'
            crd_plural = 'dgxjobs'
            name = job_name + "-dgx"
        elif BACKEND == "NVCF":
            crd_group = 'nvcf-job-manager.nvidia.io'
            crd_version = 'v1alpha1'
            crd_plural = 'nvcfjobs'
            name = job_name + "-nvcf"

        api_instance = client.CustomObjectsApi(None)
        try:
            dgxjob_api_response = api_instance.get_namespaced_custom_object(crd_group, crd_version, name_space, crd_plural, name)
            k8_status = dgxjob_api_response.get("status", {}).get("phase", "")
            if not k8_status:
                k8_status = "Pending"
            override_status = override_k8_status(job_name, k8_status)
            if override_status and override_status != k8_status:
                print(f"K8 status is {k8_status}, Toolkit Status is {override_status}, so overwriting", file=sys.stderr)
                k8_status = override_status
                deployment_string = dgxjob_api_response["spec"].get("deployment_string")
                function_id, version_id = "", ""
                if deployment_string.find(":") != -1:
                    function_id, version_id = deployment_string.split(":")
                get_nvcf_job_status(dgxjob_api_response, status=k8_status, function_id=function_id, version_id=version_id)
            return k8_status
        except Exception as e:
            print("Exception caught", e, file=sys.stderr)
            return "Error"

    # For local cluster jobs
    if network in TAO_NETWORKS:
        service_status = wait_for_service(org_name, handler_id, job_name, handler_kind)
        if service_status == "Running":
            response = send_microservice_request(api_endpoint="get_job_status", network=network, action=action, job_id=job_name)
            if response and response.ok:
                job_status_json = response.json()
                status = job_status_json.get("status")
                if status == "Processing":
                    status = "Running"
                return status
        elif service_status in ("Canceled", "Canceling", "Paused", "Pausing"):
            return service_status
        return "Error"

    api_instance = client.BatchV1Api()
    try:
        api_response = api_instance.read_namespaced_job_status(
            name=job_name,
            namespace=name_space)
        # print("Job status='%s'" % str(api_response.status), file=sys.stderr)
        # active_pods = 0 if api_response.status.active is None else api_response.status.active #TODO: NOTE: Currently assuming one pod
        if api_response.status.succeeded is not None:
            return "Done"
        if api_response.status.failed is not None:
            return "Error"
        return "Running"
    except ApiException as e:
        print(traceback.format_exc(), file=sys.stderr)
        if e.status == 404:
            print("Job not found.", file=sys.stderr)
            return "NotFound"
        return "Error"
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        return "Error"


def delete_triton_deployment(deployment_name):
    """Deletes a Triton deployment"""
    name_space = _get_name_space()
    api_instance = client.AppsV1Api()
    try:
        api_response = api_instance.delete_namespaced_deployment(
            name=deployment_name,
            namespace=name_space,
            body=client.V1DeleteOptions(
                propagation_policy='Foreground',
                grace_period_seconds=5))
        print(f"Triton Deployment deleted. status='{str(api_response.status)}'", file=sys.stderr)
        return
    except Exception as e:
        print(f"Triton Deployment failed to delete, got error: {e}", file=sys.stderr)
        return


def delete_tis_service(tis_service_name):
    """Deletes TIS service"""
    name_space = _get_name_space()
    api_instance = client.CoreV1Api()
    try:
        api_response = api_instance.delete_namespaced_service(
            name=tis_service_name,
            namespace=name_space,
        )
        print(f"TIS Service deleted. status='{str(api_response.status)}'", file=sys.stderr)
        return
    except Exception as e:
        print(f"TIS Service failed to delete, got error: {e}", file=sys.stderr)
        return


def delete(job_name, use_ngc=True):
    """Deletes a kubernetes job"""
    name_space = _get_name_space()
    if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
        config.load_kube_config()
    else:
        config.load_incluster_config()
    if BACKEND in ("BCP", "NVCF") and use_ngc:
        if BACKEND == "BCP":
            crd_group = 'dgx-job-manager.nvidia.io'
            crd_version = 'v1alpha1'
            crd_plural = 'dgxjobs'
            name = job_name + "-dgx"
        elif BACKEND == "NVCF":
            crd_group = 'nvcf-job-manager.nvidia.io'
            crd_version = 'v1alpha1'
            crd_plural = 'nvcfjobs'
            name = job_name + "-nvcf"
        api_instance = client.CustomObjectsApi(None)

        namespace = name_space
        grace_period_seconds = 10  # int | The duration in seconds before the object should be deleted. Value must be non-negative integer. The value zero indicates delete immediately. If this value is nil, the default grace period for the specified type will be used. Defaults to a per object value if not specified. zero means delete immediately. (optional)
        orphan_dependents = True  # bool | Deprecated: please use the PropagationPolicy, this field will be deprecated in 1.7. Should the dependent objects be orphaned. If true/false, the \"orphan\" finalizer will be added to/removed from the object's finalizers list. Either this field or PropagationPolicy may be set, but not both. (optional)
        body = client.V1DeleteOptions()  # V1DeleteOptions |  (optional)
        try:
            api_response = api_instance.delete_namespaced_custom_object(crd_group, crd_version, namespace, crd_plural, name, grace_period_seconds=grace_period_seconds, orphan_dependents=orphan_dependents, body=body, propagation_policy='Foreground')
            return
        except Exception as e:
            print("NvJob failed to delete.")
            print(e)
            return
    else:
        api_instance = client.BatchV1Api()
        try:
            delete_service(job_name)
            api_response = api_instance.delete_namespaced_job(
                name=job_name,
                namespace=name_space,
                body=client.V1DeleteOptions(
                    propagation_policy='Foreground',
                    grace_period_seconds=5))
            print(f"Job deleted. status='{str(api_response.status)}'", file=sys.stderr)
            return
        except:
            print("Job failed to delete.", file=sys.stderr)
            return


def list_namespace_jobs():
    """List kubernetes job in a namespace"""
    name_space = _get_name_space()
    api_instance = client.BatchV1Api()
    api_response = None
    try:
        api_response = api_instance.list_namespaced_job(namespace=name_space, label_selector="purpose=tao-toolkit-job", watch=False, limit=1000)
    except:
        pass
    return api_response


def dependency_check(num_gpu=-1, accelerator=None):
    """Checks for GPU dependency"""
    if os.getenv("BACKEND", "") not in ("local-k8s", "local-microservices"):
        return True
    if num_gpu == -1:
        num_gpu = int(os.getenv('NUM_GPU_PER_NODE', default='1'))
    label_selector = 'accelerator=' + str(accelerator)
    if not accelerator:
        label_selector = None
    if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
        config.load_kube_config()
    else:
        config.load_incluster_config()
    v1 = client.CoreV1Api()
    nodes = {}
    # how many GPUs allocatable per node
    ret = v1.list_node(label_selector=label_selector)
    if ret.items:
        for i in ret.items:
            if i.status and i.status.allocatable:
                for k, v in i.status.allocatable.items():
                    if k == 'nvidia.com/gpu':
                        nodes[i.metadata.name] = int(v)
                        break
    # how many GPUs requested for each node
    ret = v1.list_pod_for_all_namespaces()
    if ret.items:
        for i in ret.items:
            if i.spec.node_name is not None:
                if i.spec and i.spec.containers:
                    for c in i.spec.containers:
                        if c.resources and c.resources.requests:
                            for k, v in c.resources.requests.items():
                                if k == 'nvidia.com/gpu':
                                    current = nodes.get(i.spec.node_name, 0)
                                    nodes[i.spec.node_name] = max(0, current - int(v))
    # do I have enough GPUs on one of the nodes
    for k, v in nodes.items():
        if v >= num_gpu:
            return True
    return False


def create_tensorboard_deployment(deployment_name, image, command, logs_image, logs_command, replicas):
    """Creates Tensorboard Deployment"""
    name_space = _get_name_space()
    api_instance = client.AppsV1Api()
    logs_volume_mount = client.V1VolumeMount(
        name="tb-data",
        mount_path="/tfevents")
    capabilities = client.V1Capabilities(
        add=['SYS_PTRACE']
    )
    security_context = client.V1SecurityContext(
        capabilities=capabilities
    )

    tb_port = [
        client.V1ContainerPort(container_port=6006)
    ]
    resources = client.V1ResourceRequirements(
        limits={
            'memory': "600Mi",
            'cpu': "10m",
        },
        requests={
            'memory': '300Mi',
            'cpu': "5m"
        }
    )
    no_gpu = client.V1EnvVar(
        name="NVIDIA_VISIBLE_DEVICES",
        value="none")
    mongo_secret_env = client.V1EnvVar(
        name="MONGOSECRET",
        value=mongo_secret
    )
    backend_env = client.V1EnvVar(
        name="BACKEND",
        value=BACKEND
    )
    vault_path = client.V1EnvVar(
        name="VAULT_SECRET_PATH",
        value=os.getenv("VAULT_SECRET_PATH", None)
    )
    tb_container = client.V1Container(
        name="tb-container",
        image=image,
        env=[no_gpu],
        command=["/bin/sh", "-c"],
        args=[command],
        resources=resources,
        volume_mounts=[logs_volume_mount],
        ports=tb_port,
        security_context=security_context)

    tb_logs_container = client.V1Container(
        name="tb-logs-container",
        image=logs_image,
        env=[no_gpu, mongo_secret_env, backend_env, vault_path],
        command=["/bin/sh", "-c"],
        resources=resources,
        args=[logs_command],
        volume_mounts=[logs_volume_mount],
        security_context=security_context,
    )

    logs_volume = client.V1Volume(
        name="tb-data",
        empty_dir=client.V1EmptyDirVolumeSource())

    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(
            labels={
                "purpose": "tao-toolkit-tensorboard-job",
                "resource-type": "tensorboard",
                "app": deployment_name,  # use deployment_name as the selector name
            }
        ),
        spec=client.V1PodSpec(
            containers=[tb_container, tb_logs_container],
            volumes=[logs_volume],
        ))

    spec = client.V1DeploymentSpec(
        replicas=replicas,
        template=template,
        selector={"matchLabels": {"app": deployment_name}})

    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=deployment_name, labels={
            "resource-type": "tensorboard"
        }),
        spec=spec)

    print("Prepared deployment configs", file=sys.stderr)
    try:
        api_instance.create_namespaced_deployment(
            body=deployment,
            namespace=name_space)
        print("Start create deployment", file=sys.stderr)
        return
    except Exception as e:
        print(f"Create deployment got error: {e}", file=sys.stderr)
        return


def create_tensorboard_service(tb_service_name, deploy_label):
    """Creates Tensorboard Service"""
    name_space = _get_name_space()
    tb_port = [
        client.V1ServicePort(name='tb-default-port', port=6006, target_port=6006, protocol="TCP")
    ]
    spec = client.V1ServiceSpec(ports=tb_port, selector={"app": deploy_label})
    # add annotation, it will only works in Azure, but will not affect other cloud
    annotation = {
        "service.beta.kubernetes.io/azure-load-balancer-internal": "true"
    }
    service = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(name=tb_service_name, labels={"app": tb_service_name, "resource-type": "tensorboard"}, annotations=annotation),
        spec=spec,
    )
    api_instance = client.CoreV1Api()

    print("Prepared Tensorboard Service configs", file=sys.stderr)
    try:
        api_instance.create_namespaced_service(
            body=service,
            namespace=name_space)
        print("Start create Tensorboard Service", file=sys.stderr)
        return
    except Exception as e:
        print(f"Create Tensorboard Service got error: {e}", file=sys.stderr)
        return


def create_tensorboard_ingress(tb_service_name, tb_ingress_name, tb_ingress_path):
    """Creates Tensorboard Ingress"""
    name_space = _get_name_space()
    networking_v1_api = client.NetworkingV1Api()
    release_name = os.getenv("RELEASE_NAME", 'tao-api')
    auth_url = f'http://{release_name}-service.{name_space}.svc.cluster.local:8000/api/v1/auth'
    ingress = client.V1Ingress(
        api_version="networking.k8s.io/v1",
        kind="Ingress",
        metadata=client.V1ObjectMeta(name=tb_ingress_name, namespace=name_space, labels={
            "resource-type": "tensorboard"
        }, annotations={
            "kubernetes.io/ingress.class": "nginx",
            "nginx.ingress.kubernetes.io/auth-url": auth_url,
            "nginx.ingress.kubernetes.io/client-max-body-size": "0m",
            "nginx.ingress.kubernetes.io/proxy-body-size": "0m",
            "nginx.ingress.kubernetes.io/body-size": "0m",
            "nginx.ingress.kubernetes.io/client-body-buffer-size": "50m",
            "nginx.ingress.kubernetes.io/proxy-buffer-size": "128k",
            "nginx.ingress.kubernetes.io/proxy-buffers-number": "4",
            "nginx.ingress.kubernetes.io/proxy-connect-timeout": "3600",
            "nginx.ingress.kubernetes.io/proxy-read-timeout": "3600",
            "nginx.ingress.kubernetes.io/proxy-send-timeout": "3600",
        }),
        spec=client.V1IngressSpec(
            rules=[client.V1IngressRule(
                http=client.V1HTTPIngressRuleValue(
                    paths=[client.V1HTTPIngressPath(
                        path=tb_ingress_path,
                        path_type="Prefix",
                        backend=client.V1IngressBackend(
                            service=client.V1IngressServiceBackend(
                                port=client.V1ServiceBackendPort(
                                    name='tb-default-port'
                                ),
                                name=tb_service_name
                            )
                        )
                    )]
                )
            )]
        )
    )

    try:
        networking_v1_api.create_namespaced_ingress(
            body=ingress,
            namespace=name_space
        )
        print("Created Tensorboard Ingress", file=sys.stderr)
        return
    except Exception as e:
        print(f"Create Tensorboard Ingress got error: {e}", file=sys.stderr)
        return


def status_tensorboard_deployment(deployment_name, replicas=1):
    """
    Returns status of Tensorboard deployment
    Status definition:

    Running: The Tensorboard deployment is ready and running
    ReplicaNotReady: at least one replica of the deployment is not ready.
    NotFound: cannot find the deployment.  This status is useful to check if the deployment is stopped.
    Error: meet exceptions except not found error when check the status.

    """
    name_space = _get_name_space()
    api_instance = client.AppsV1Api()
    try:
        api_response = api_instance.read_namespaced_deployment_status(
            name=deployment_name,
            namespace=name_space)
        available_replicas = api_response.status.available_replicas
        if not isinstance(available_replicas, int) or available_replicas < replicas:
            return {"status": "ReplicaNotReady"}
        return {"status": "Running"}
    except ApiException as e:
        if e.status == 404:
            print("Tensorboard Deployment not found.", file=sys.stderr)
            # TODO: here defined a new status to find the situation that the deployment does not exists
            # This status is useful to check if the deployment is deleted or not created
            return {"status": "NotFound"}
        print(f"Got other ApiException error: {e}", file=sys.stderr)
        return {"status": "Error"}
    except Exception as e:
        print(f"Got {type(e)} error: {e}", file=sys.stderr)
        return {"status": "Error"}


def status_tb_service(tb_service_name, port=6006):
    """
    Returns status of TB Service
    Status definition:

    Running: The TB Service is ready and running
    NotReady: the TB Service is not ready.
    NotFound: cannot find the TB Service. This status is useful to check if the service is stopped.
    Error: meet exceptions except not found error when check the status.
    """
    name_space = _get_name_space()
    api_instance = client.CoreV1Api()

    try:
        api_response = api_instance.read_namespaced_service(
            name=tb_service_name,
            namespace=name_space,
        )
        print(f'TB Service API Response: {api_response}')
        tb_service_ip = api_response.spec.cluster_ip
        return {"status": "Running", "tb_service_ip": tb_service_ip}
    except ApiException as e:
        if e.status == 404:
            print("TIS Service not found.", file=sys.stderr)
            # TODO: here defined a new status, in order to find the situation that the TIS Service not exists
            # This status is useful to check if the TIS Service is deleted or not created
            return {"status": "NotFound"}
        print(f"Got other ApiException error: {e}", file=sys.stderr)
        return {"status": "Error"}
    except Exception as e:
        print(f"Got {type(e)} error: {e}", file=sys.stderr)
        return {"status": "Error"}


def delete_tensorboard_deployment(deployment_name):
    """Deletes Tensorboard Deployment"""
    name_space = _get_name_space()
    api_instance = client.AppsV1Api()
    try:
        api_response = api_instance.delete_namespaced_deployment(
            name=deployment_name,
            namespace=name_space,
            body=client.V1DeleteOptions(
                propagation_policy='Foreground',
                grace_period_seconds=5))
        print(f"Tensorboard Deployment deleted. status='{str(api_response.status)}'", file=sys.stderr)
        return
    except Exception as e:
        print(f"Tensorboard Deployment failed to delete, got error: {e}", file=sys.stderr)
        return


def delete_tensorboard_service(tb_service_name):
    """Deletes Tensorboard service"""
    name_space = _get_name_space()
    api_instance = client.CoreV1Api()
    try:
        api_response = api_instance.delete_namespaced_service(
            name=tb_service_name,
            namespace=name_space,
        )
        print(f"Tensorboard Service deleted. status='{str(api_response.status)}'", file=sys.stderr)
        return
    except Exception as e:
        print(f"Tensorboard Service failed to delete, got error: {e}", file=sys.stderr)
        return


def delete_tensorboard_ingress(tb_ingress_name):
    """Delete Tensorboard Ingress"""
    name_space = _get_name_space()
    networking_v1_api = client.NetworkingV1Api()
    try:
        api_response = networking_v1_api.delete_namespaced_ingress(
            name=tb_ingress_name,
            namespace=name_space
        )
        print(f"Tensorboard Ingress deleted. status='{str(api_response.status)}'", file=sys.stderr)
        return
    except Exception as e:
        print(f"Tensorboard Ingress failed to delete, got error: {e}", file=sys.stderr)
        return


def get_cluster_ip(namespace='default'):
    """Get cluster IP of service"""
    try:
        # Load kubeconfig file (optional if running in-cluster)
        if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
            config.load_kube_config()
        else:
            config.load_incluster_config()
        api_instance = client.CoreV1Api()
        service = api_instance.read_namespaced_service("tao-api-service", namespace)
        cluster_ip = service.spec.cluster_ip
        cluster_port = 8000
        for port in service.spec.ports:
            if port.name == "api":
                cluster_port = port.port
        return cluster_ip, cluster_port
    except Exception as e:
        print(f"Error fetching ClusterIP: {e}", file=sys.stderr)
        return None, None
