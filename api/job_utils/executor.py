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
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import os
import sys
import requests
import traceback
from handlers.stateless_handlers import admin_uuid, ngc_runner


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


def create(user_id, job_name, image, command, num_gpu=-1, accelerator=None, docker_env_vars=None, port=False, dgx_job_metadata=None):
    """Creates a kubernetes job"""
    name_space = _get_name_space()
    if ngc_runner == "True" and dgx_job_metadata:
        config.load_incluster_config()

        # Initialize the custom objects API
        api_instance = client.CustomObjectsApi()

        # Define the CR body based on your CRD
        crd_group = 'dgx-job-manager.nvidia.io'
        crd_version = 'v1alpha1'
        crd_plural = 'dgxjobs'

        dgxjob_body = {
            "apiVersion": f"{crd_group}/{crd_version}",
            "kind": "DgxJob",
            "metadata": {
                "name": job_name + "-dgx",
                "namespace": name_space,
            },
            "spec": {
                "user_id": dgx_job_metadata["user_id"],
                "name": job_name,
                "job_id": job_name + "-dgx",
                "dockerImageName": dgx_job_metadata["dockerImageName"],
                "command": dgx_job_metadata["command"],
                "orgName": dgx_job_metadata["orgName"],
                "teamName": dgx_job_metadata["teamName"],
                "aceName": dgx_job_metadata["aceName"],
                "aceInstance": dgx_job_metadata["aceInstance"],
                "workspaceMounts": dgx_job_metadata["workspaceMounts"],
                "runPolicy": dgx_job_metadata["runPolicy"],
                "resultContainerMountPoint": dgx_job_metadata["resultContainerMountPoint"],
                "envs": dgx_job_metadata["envs"],
            },
            "status": {
                "phase": "Pending"
            }
        }

        # Create the custom resource in the specified namespace
        try:
            api_instance.create_namespaced_custom_object(crd_group, crd_version, name_space, crd_plural, dgxjob_body)
            print(f"DgxJob CR '{job_name}' created successfully.", file=sys.stderr)
        except Exception as e:
            print(f"Failed to create DgxJob CR '{job_name}': {e}", file=sys.stderr)
        return

    command = 'umask 0 && ' + command
    if num_gpu == -1:
        num_gpu = int(os.getenv('NUM_GPU_PER_NODE', default='1'))
    telemetry_opt_out = os.getenv('TELEMETRY_OPT_OUT', default='no')
    node_selector = {'accelerator': str(accelerator)}
    if not accelerator:
        node_selector = None
    claim_name = os.getenv('CLAIMNAME', 'nvtl-api-pvc')
    image_pull_secret = os.getenv('IMAGEPULLSECRET', default='imagepullsecret')
    name_space = _get_name_space()
    api_instance = client.BatchV1Api()
    shared_volume_mount = client.V1VolumeMount(
        name="shared-data",
        mount_path=f"/shared/users/{admin_uuid}",
        sub_path=f"users/{admin_uuid}")
    user_volume_mount = client.V1VolumeMount(
        name="shared-data",
        mount_path="/shared/users/" + user_id,
        sub_path="users/" + user_id)
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
    if ngc_runner == "True":
        security_context.privileged = True  # To mount workspaces
    ngc_runner_env = client.V1EnvVar(
        name="NGC_RUNNER",
        value=ngc_runner)
    num_gpu_env = client.V1EnvVar(
        name="NUM_GPU_PER_NODE",
        value=str(num_gpu))
    telemetry_opt_out_env = client.V1EnvVar(
        name="TELEMETRY_OPT_OUT",
        value=telemetry_opt_out)
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
        env=[ngc_runner_env, num_gpu_env, telemetry_opt_out_env] + dynamic_docker_envs,
        command=["/bin/bash", "-c"],
        args=[command],
        resources=resources,
        volume_mounts=[shared_volume_mount, user_volume_mount, dshm_volume_mount],
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


def create_triton_deployment(deployment_name, image, command, replicas, num_gpu=-1, ports=(8000, 8001, 8002)):
    """Creates a Triton deployment"""
    # You can use NFS for local development

    claim_name = os.getenv('CLAIMNAME', 'nvtl-api-pvc')
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
    api_instance = client.CoreV1Api()

    try:
        api_response = api_instance.read_namespaced_service(
            name=tis_service_name,
            namespace=name_space,
        )
        tis_service_ip = api_response.spec.cluster_ip
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


def status(job_name, use_ngc=True):
    """Returns status of kubernetes job"""
    name_space = _get_name_space()
    if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
        config.load_kube_config()
    else:
        config.load_incluster_config()
    dgxjob_api_response = None
    if ngc_runner == "True" and use_ngc:
        api_instance = client.CustomObjectsApi(None)
        crd_group = 'dgx-job-manager.nvidia.io'
        crd_version = 'v1alpha1'
        crd_plural = 'dgxjobs'
        name = job_name + "-dgx"  # str | the custom object's name
        try:
            dgxjob_api_response = api_instance.get_namespaced_custom_object(crd_group, crd_version, name_space, crd_plural, name)
            status = dgxjob_api_response.get("status", {}).get("phase", "")
            if not status:
                status = "Pending"
            return status
        except Exception as e:
            print("Exception caught", e, file=sys.stderr)
            return "Error"

    # For local cluster jobs
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
        if e.status == 404:
            print("Job not found.", file=sys.stderr)
            return "NotFound"
        print(traceback.format_exc(), file=sys.stderr)
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
    if ngc_runner == "True" and use_ngc:
        api_instance = client.CustomObjectsApi(None)
        crd_group = 'dgx-job-manager.nvidia.io'
        crd_version = 'v1alpha1'
        crd_plural = 'dgxjobs'
        name = job_name + "-dgx"  # str | the custom object's name
        namespace = name_space
        grace_period_seconds = 10  # int | The duration in seconds before the object should be deleted. Value must be non-negative integer. The value zero indicates delete immediately. If this value is nil, the default grace period for the specified type will be used. Defaults to a per object value if not specified. zero means delete immediately. (optional)
        orphan_dependents = True  # bool | Deprecated: please use the PropagationPolicy, this field will be deprecated in 1.7. Should the dependent objects be orphaned. If true/false, the \"orphan\" finalizer will be added to/removed from the object's finalizers list. Either this field or PropagationPolicy may be set, but not both. (optional)
        body = client.V1DeleteOptions()  # V1DeleteOptions |  (optional)
        try:
            api_response = api_instance.delete_namespaced_custom_object(crd_group, crd_version, namespace, crd_plural, name, grace_period_seconds=grace_period_seconds, orphan_dependents=orphan_dependents, body=body, propagation_policy='Foreground')
            return
        except Exception as e:
            print("DGXJob failed to delete.")
            print(e)
            return
    else:
        api_instance = client.BatchV1Api()
        try:
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
    if os.getenv("NGC_RUNNER", "") == "True":
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
