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

"""Kubernetes job manager modules"""
from kubernetes import client, config
import os
import sys


def create(job_name, image, command, num_gpu=-1, accelerator=None, docker_env_vars=None):
    """Creates a kubernetes job"""
    command = 'umask 0 && ' + command
    if num_gpu == -1:
        num_gpu = int(os.getenv('NUM_GPU_PER_NODE', default='1'))
    telemetry_opt_out = os.getenv('TELEMETRY_OPT_OUT', default='no')
    node_selector = {'accelerator': str(accelerator)}
    if not accelerator:
        node_selector = None
    with open('/var/run/secrets/kubernetes.io/serviceaccount/namespace', 'r', encoding='utf-8') as f:
        current_name_space = f.read()
    name_space = os.getenv('NAMESPACE', default=current_name_space)
    claim_name = os.getenv('CLAIMNAME', 'tao-toolkit-api-pvc')
    image_pull_secret = os.getenv('IMAGEPULLSECRET', default='imagepullsecret')
    config.load_incluster_config()
    api_instance = client.BatchV1Api()
    shared_volume_mount = client.V1VolumeMount(
        name="shared-data",
        mount_path="/shared")
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
    container = client.V1Container(
        name="container",
        image=image,
        env=[num_gpu_env, telemetry_opt_out_env] + dynamic_docker_envs,
        command=["/bin/bash", "-c"],
        args=[command],
        resources=resources,
        volume_mounts=[shared_volume_mount, dshm_volume_mount],
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

    if os.getenv('BACKEND') == "moebius-cloud":
        # Create an instance of the API class
        # example https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/CustomObjectsApi.md#create_namespaced_custom_object
        api_instance = client.CustomObjectsApi(None)
        group = 'moebius-job-manager.nvidia.io'  # str | The custom resource's group name
        version = 'v1alpha1'  # str | The custom resource's version
        namespace = name_space
        plural = 'cloudjobs'  # str | The custom resource's plural name. For TPRs this would be lowercase plural kind.
        cloud_job_body = {
            "apiVersion": "moebius-job-manager.nvidia.io/v1alpha1",
            "kind": "CloudJob",
            "metadata": {
                "name":    job_name + "-moebius",
                "labels": {"job-name": job_name,
                           "purpose": "tao-toolkit-job",
                           "gputype": str(accelerator)
                           }
            },
            "spec": {"job": job,
                     "jobName": job_name,
                     "jobAction": "validate",
                     "jobGpu": str(accelerator),
                     "jobType": "train_model"
                     }
        }
        try:
            api_instance.create_namespaced_custom_object(group, version, namespace, plural, body=cloud_job_body)
            return
        except Exception as e:
            print("CloudJob creation failuers")
            print(e)
            return
    else:
        try:
            api_instance.create_namespaced_job(
                body=job,
                namespace=name_space)
            return
        except:
            return


def status(job_name):
    """Returns status of kubernetes job"""
    with open('/var/run/secrets/kubernetes.io/serviceaccount/namespace', 'r', encoding='utf-8') as f:
        current_name_space = f.read()
    name_space = os.getenv('NAMESPACE', default=current_name_space)
    config.load_incluster_config()
    cloudjob_api_response = None
    if os.getenv('BACKEND') == "moebius-cloud":
        api_instance = client.CustomObjectsApi(None)
        group = 'moebius-job-manager.nvidia.io'  # str | The custom resource's group name
        version = 'v1alpha1'  # str | The custom resource's version
        plural = 'cloudjobs'  # str | The custom resource's plural name. For TPRs this would be lowercase plural kind.
        name = job_name + "-moebius"  # str | the custom object's name
        try:
            cloudjob_api_response = api_instance.get_namespaced_custom_object(group, version, name_space, plural, name)
        except Exception as e:
            print(e)
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
    except:
        # moebius-cloud is in process of creating batchjob.
        if os.getenv('BACKEND') == "moebius-cloud":
            if cloudjob_api_response is not None:
                return "Creating"
        return "Error"


def delete(job_name):
    """Deletes a kubernetes job"""
    with open('/var/run/secrets/kubernetes.io/serviceaccount/namespace', 'r', encoding='utf-8') as f:
        current_name_space = f.read()
    name_space = os.getenv('NAMESPACE', default=current_name_space)
    config.load_incluster_config()
    if os.getenv('BACKEND') == "moebius-cloud":
        api_instance = client.CustomObjectsApi(None)
        group = 'moebius-job-manager.nvidia.io'  # str | The custom resource's group name
        version = 'v1alpha1'  # str | The custom resource's version
        namespace = name_space
        plural = 'cloudjobs'  # str | The custom resource's plural name. For TPRs this would be lowercase plural kind.
        namespace = name_space
        name = job_name + "-moebius"  # str | the custom object's name
        grace_period_seconds = 10  # int | The duration in seconds before the object should be deleted. Value must be non-negative integer. The value zero indicates delete immediately. If this value is nil, the default grace period for the specified type will be used. Defaults to a per object value if not specified. zero means delete immediately. (optional)
        orphan_dependents = True  # bool | Deprecated: please use the PropagationPolicy, this field will be deprecated in 1.7. Should the dependent objects be orphaned. If true/false, the \"orphan\" finalizer will be added to/removed from the object's finalizers list. Either this field or PropagationPolicy may be set, but not both. (optional)
        body = client.V1DeleteOptions()  # V1DeleteOptions |  (optional)
        try:
            api_response = api_instance.delete_namespaced_custom_object(group, version, namespace, plural, name, grace_period_seconds=grace_period_seconds, orphan_dependents=orphan_dependents, body=body, propagation_policy='Foreground')
            return
        except Exception as e:
            print("CloudJob failed to delete.")
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
    with open('/var/run/secrets/kubernetes.io/serviceaccount/namespace', 'r', encoding='utf-8') as f:
        current_name_space = f.read()
    name_space = os.getenv('NAMESPACE', default=current_name_space)
    config.load_incluster_config()
    api_instance = client.BatchV1Api()
    api_response = None
    try:
        api_response = api_instance.list_namespaced_job(namespace=name_space, label_selector="purpose=tao-toolkit-job", watch=False, limit=1000)
    except:
        pass
    return api_response


def dependency_check(num_gpu=-1, accelerator=None):
    """Checks for GPU dependency"""
    if num_gpu == -1:
        num_gpu = int(os.getenv('NUM_GPU_PER_NODE', default='1'))
    label_selector = 'accelerator=' + str(accelerator)
    if not accelerator:
        label_selector = None
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
