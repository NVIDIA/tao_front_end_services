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

"""NVCF job's kubernetes controller"""
import os
import sys
import time
import asyncio
import traceback
from kubernetes import client, config
from concurrent.futures import ThreadPoolExecutor

from handlers.nvcf_handler import invoke_function, get_status_of_invoked_function, create_function, deploy_function, get_function, delete_function_version
from handlers.utilities import get_cloud_metadata
from utils import safe_load_file

if not os.getenv("CI_PROJECT_DIR", None):
    config.load_incluster_config()
api_instance = client.CustomObjectsApi()
executor = ThreadPoolExecutor(max_workers=10)

job_tracker = {}
logs_tracker = set([])
active_tasks = set([])


def update_cr_status(namespace, custom_resource_name, status):
    """Update status of the NVCF Custom resource"""
    updated_cr = api_instance.patch_namespaced_custom_object(
        group="nvcf-job-manager.nvidia.io",
        version="v1alpha1",
        namespace=namespace,
        plural="nvcfjobs",
        name=custom_resource_name,
        body={"status": {"phase": status}}
    )
    return updated_cr


def create_and_deploy_function_sync(container):
    """Create and deploy a NVCF function (blocking)"""
    try:
        create_response = create_function(container)
        if create_response.ok:
            print("Function created successfully", file=sys.stderr)
            function_metadata = create_response.json()
            deploy_response = deploy_function(function_metadata)
            if deploy_response.ok:
                print("Function deployment initiated successfully", file=sys.stderr)
                while True:
                    function_id = function_metadata["function"]["id"]
                    version_id = function_metadata["function"]["versionId"]
                    current_function_response = get_function(function_id, version_id)
                    if current_function_response.ok:
                        current_function_metadata = current_function_response.json()
                        if current_function_metadata.get("function", {}).get("status") == "ACTIVE":
                            deployment_string = f"{function_metadata['function']['id']}:{function_metadata['function']['versionId']}"
                            print("Function deployed successfully", file=sys.stderr)
                            return deployment_string
                        if current_function_metadata.get("function", {}).get("status") == "ERROR":
                            return "False"
                        print(f"Function status: {current_function_metadata.get('function', {}).get('status')}", file=sys.stderr)
                    else:
                        return "False"
                    time.sleep(10)
            else:
                return "False"
        else:
            return "False"
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        print(f"Error in create_and_deploy_function: {str(e)}", file=sys.stderr)
        raise Exception(e) from e


async def create_and_deploy_function(container):
    """Create and deploy a NVCF function (non-blocking)"""
    loop = asyncio.get_event_loop()
    deployment_string = await loop.run_in_executor(executor, create_and_deploy_function_sync, container)
    return deployment_string


async def create_nvcf_job(nvcf_cr):
    """Construct requests call for triggering Job on NVCF cloud"""
    config.load_incluster_config()

    assert nvcf_cr.get("metadata", "")
    assert nvcf_cr.get("spec", "")

    namespace = nvcf_cr['metadata']['namespace']
    custom_resource_name = nvcf_cr['metadata']['name']
    org_name = nvcf_cr["spec"].get("org_name")

    action = nvcf_cr["spec"].get("action")
    network = nvcf_cr["spec"].get("network")
    deployment_string = nvcf_cr["spec"].get("deployment_string")
    container = nvcf_cr["spec"].get("container")
    tao_api_admin_key = nvcf_cr["spec"].get("tao_api_admin_key")
    tao_api_base_url = nvcf_cr["spec"].get("tao_api_base_url")
    tao_api_status_callback_url = nvcf_cr["spec"].get("tao_api_status_callback_url")
    tao_api_ui_cookie = nvcf_cr["spec"].get("tao_api_ui_cookie")
    use_ngc_staging = nvcf_cr["spec"].get("use_ngc_staging")
    automl_experiment_number = nvcf_cr["spec"].get("automl_experiment_number")

    # if not deployment_string:
    deployment_string = await create_and_deploy_function(container)
    if deployment_string == "False":
        print("Unable to deploy function, Retry TAO job", file=sys.stderr)
        return update_cr_status(namespace, custom_resource_name, "Error")

    updated_spec = {"deployment_string": deployment_string}
    # Patch the custom resource with the updated deployment_string
    updated_cr = api_instance.patch_namespaced_custom_object(
        group="nvcf-job-manager.nvidia.io",
        version="v1alpha1",
        namespace=namespace,
        plural="nvcfjobs",
        name=custom_resource_name,
        body={"spec": updated_spec},
    )

    cloud_metadata = {}
    get_cloud_metadata(org_name, nvcf_cr["spec"].get("workspace_ids"), cloud_metadata)

    ngc_api_key = nvcf_cr["spec"].get("ngc_api_key")

    spec_file_path = nvcf_cr["spec"].get("spec_file_path")
    if not spec_file_path:
        print("spec_file_path not set", file=sys.stderr)
        return update_cr_status(namespace, custom_resource_name, "Error")
    specs = safe_load_file(spec_file_path, file_type="yaml")

    job_create_response = invoke_function(deployment_string,
                                          network,
                                          action,
                                          microservice_action="post_action",
                                          cloud_metadata=cloud_metadata,
                                          specs=specs,
                                          ngc_api_key=ngc_api_key,
                                          job_id=custom_resource_name.replace("-nvcf", ""),
                                          tao_api_admin_key=tao_api_admin_key,
                                          tao_api_base_url=tao_api_base_url,
                                          tao_api_status_callback_url=tao_api_status_callback_url,
                                          tao_api_ui_cookie=tao_api_ui_cookie,
                                          use_ngc_staging=use_ngc_staging,
                                          automl_experiment_number=automl_experiment_number)

    if job_create_response.status_code not in [200, 202]:
        print("Invocation error response code", job_create_response.status_code, file=sys.stderr)
        print("Invocation error response json", job_create_response.json(), file=sys.stderr)
        return update_cr_status(namespace, custom_resource_name, "Error")

    job_create_response_json = job_create_response.json()
    print("Microservice job successfully created", job_create_response_json, file=sys.stderr)
    req_id = job_create_response_json.get("reqId", "")
    job_id = job_create_response_json.get("response", {}).get("job_id")

    if job_create_response.status_code == 202:
        while True:
            polling_response = get_status_of_invoked_function(req_id)
            if polling_response.status_code == 404:
                if polling_response.json().get("title") != "Not Found":
                    print("Polling(job_create) response failed", polling_response.status_code, file=sys.stderr)
                    return update_cr_status(namespace, custom_resource_name, "Error")
            if polling_response.status_code != 202:
                break
            time.sleep(10)

        if polling_response.status_code != 200:
            print("Polling(job_create) response status code is not 200", polling_response.status_code, file=sys.stderr)
            return update_cr_status(namespace, custom_resource_name, "Error")
        job_id = polling_response.json().get("response", {}).get("job_id")

    if not job_id:
        print("Job ID couldn't be fetched", file=sys.stderr)
        return update_cr_status(namespace, custom_resource_name, "Error")

    return updated_cr


def delete_nvcf_job(nvcf_cr):
    """Construct requests call for deleting Job on NVCF cloud"""
    deployment_string = nvcf_cr["spec"].get("deployment_string")
    if deployment_string.find(":") == -1:
        print(f"Deployment not active yet for custom resource {nvcf_cr['metadata']['name']}", file=sys.stderr)
        return
    function_id, version_id = deployment_string.split(":")
    delete_function_version(function_id, version_id)


def get_job_logs(user_id, job_id, orgName):
    """Get job logs from BCP"""
    return


def print_job_logs(user_id, job_id, orgName, custom_resource_name):
    """Print logs of NVCF job on controller pod"""
    return


def overwrite_job_logs_from_bcp(logfile, job_name):
    """Get job logs from BCP and overwrite it with existing logs"""
    return


def get_nvcf_job_status(nvcf_cr, status=""):
    """Get and update NVCF custom resource status"""
    if not status:
        custom_resource_name = nvcf_cr["metadata"].get('name')
        namespace = nvcf_cr['metadata']['namespace']

        action = nvcf_cr["spec"].get("action")
        network = nvcf_cr["spec"].get("network")
        job_id = nvcf_cr["spec"].get("job_id")
        deployment_string = nvcf_cr["spec"].get("deployment_string")
        if deployment_string.find(":") == -1:
            print(f"Deployment not active yet for job {job_id}", file=sys.stderr)
            status = "Pending"
            return status

        function_id, version_id = deployment_string.split(":")

        print("update status", deployment_string, job_tracker.keys(), file=sys.stderr)
        job_monitor_response = invoke_function(deployment_string, network, action, microservice_action="get_job_status", job_id=job_id)
        if job_monitor_response.status_code == 404:
            status = "Error"
            if job_monitor_response.json().get("title") == "Not Found":
                print("NVCF function was deleted, setting status as done", file=sys.stderr)
                status = "Done"

        if job_monitor_response.status_code == 202:
            req_id = job_monitor_response.get("reqId", "")
            while True:
                job_monitor_response = get_status_of_invoked_function(req_id)
                if job_monitor_response.status_code == 404:
                    if job_monitor_response.json().get("title") != "Not Found":
                        print("Polling(job_monitor) response failed", job_monitor_response.status_code, file=sys.stderr)
                        status = "Error"
                if job_monitor_response.status_code != 202:
                    break
                time.sleep(10)

            if job_monitor_response.status_code != 200:
                print("Polling(job_monitor) response status code is not 200", job_monitor_response.status_code, file=sys.stderr)
                status = "Error"

        if not status:
            job_monitor_response_json = job_monitor_response.json()
            status = job_monitor_response_json.get("response", {}).get("status")
            if status:
                if status == "Processing":
                    status = "Running"
                elif status not in ("Pending", "Done"):
                    status = "Error"
            else:
                status = "Pending"

    if status in ("Done", "Error"):
        print(f"Status is {status}. Hence, deleting the function", file=sys.stderr)
        delete_function_version(function_id, version_id)

    if not status:
        print("Status couldn't be inferred", file=sys.stderr)
        status = "Pending"
    try:
        namespace = nvcf_cr['metadata']['namespace']
        update_cr_status(namespace, custom_resource_name, status)
    except Exception:
        pass
    return status


def update_status(job_tracker, logs_tracker):
    """Update the status of the custom resources based on status of NGC job"""
    for _, nvcf_cr in job_tracker.items():
        job_id = nvcf_cr["spec"].get("job_id")
        deployment_string = nvcf_cr["spec"].get("deployment_string")
        if deployment_string.find(":") == -1:
            print(f"Deployment not active yet for job {job_id}", file=sys.stderr)
            continue
        custom_resource_name = nvcf_cr["metadata"].get('name')
        status = get_nvcf_job_status(nvcf_cr)

        if status in ("Done", "Error"):
            logs_tracker.add(custom_resource_name)
            print(f"Status is {status}. Hence, deleting the function", file=sys.stderr)


def remove_deleted_custom_resources(job_tracker, logs_tracker):
    """Remove deleted custom resources from tracker variables"""
    try:
        # Fetch the list of NVCF job custom resources
        nvcf_jobs = api_instance.list_cluster_custom_object(
            group="nvcf-job-manager.nvidia.io",
            version="v1alpha1",
            plural="nvcfjobs",
        )

        existing_jobs = set(job_tracker.keys())
        current_jobs = set(item['metadata']['name'] for item in nvcf_jobs['items'])
        deleted_jobs = existing_jobs - current_jobs

        for deleted_job in deleted_jobs:
            print(f"NVCF CR deleted: {deleted_job}")
            delete_nvcf_job(job_tracker[deleted_job])
            del job_tracker[deleted_job]

    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        print(f"Error in removing custom resource from tracker: {str(e)}", file=sys.stderr)


async def handle_new_nvcf_job(custom_resource_name, item):
    """Handle new NVCF job creation and deployment."""
    global job_tracker
    job_tracker = job_tracker if 'job_tracker' in globals() else {}
    global active_tasks
    active_tasks = active_tasks if 'active_tasks' in globals() else set()
    # Check if the job is already being processed
    if custom_resource_name not in job_tracker:
        print(f"Job {custom_resource_name} is already being processed", file=sys.stderr)
        return
    updated_item = await create_nvcf_job(item)
    if updated_item is not None:
        job_tracker[custom_resource_name] = updated_item
    active_tasks.remove(asyncio.current_task())


async def process_events():
    """Process NVCF JOB events"""
    global job_tracker
    job_tracker = job_tracker if 'job_tracker' in globals() else {}
    global logs_tracker
    logs_tracker = logs_tracker if 'logs_tracker' in globals() else {}
    global active_tasks
    active_tasks = active_tasks if 'active_tasks' in globals() else set()

    while True:
        try:
            # Fetch the list of NVCF job custom resources
            nvcf_jobs = api_instance.list_cluster_custom_object(
                group="nvcf-job-manager.nvidia.io",
                version="v1alpha1",
                plural="nvcfjobs",
            )

            for item in nvcf_jobs['items']:
                custom_resource_name = item['metadata']['name']

                if custom_resource_name not in job_tracker:
                    # Handle added event
                    print(f"NVCF CR added: {custom_resource_name}", file=sys.stderr)
                    task = asyncio.create_task(handle_new_nvcf_job(custom_resource_name, item))
                    job_tracker[custom_resource_name] = item
                    active_tasks.add(task)
                    task.add_done_callback(active_tasks.discard)

            remove_deleted_custom_resources(job_tracker, logs_tracker)
            await asyncio.sleep(10)

        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)
            print(f"Error in the event processing loop: {str(e)}", file=sys.stderr)


async def main():
    """Controller Main function"""
    global job_tracker
    job_tracker = job_tracker if 'job_tracker' in globals() else {}
    global logs_tracker
    logs_tracker = logs_tracker if 'logs_tracker' in globals() else {}
    global active_tasks
    active_tasks = active_tasks if 'active_tasks' in globals() else set()

    asyncio.create_task(process_events())
    while True:
        remove_deleted_custom_resources(job_tracker, logs_tracker)
        update_status(job_tracker, logs_tracker)
        await asyncio.sleep(10)

if __name__ == "__main__":
    # Run the main function asynchronously
    asyncio.run(main())
