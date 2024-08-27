#!/usr/bin/env python3

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

"""Start Tensorboard Events pulling from cloud storage"""
import argparse
from handlers.stateless_handlers import get_handler_metadata_with_jobs, get_handler_metadata
from handlers.utilities import filter_file_objects
from handlers.cloud_storage import create_cs_instance
from time import sleep
import os
from datetime import datetime, timezone

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Tensorboard Events Controller', description='Periodically pull tfevents files from cloud storage')
    parser.add_argument(
        '--experiment_id',
        type=str,
    )
    parser.add_argument(
        '--org_name',
        type=str,
    )
    args = parser.parse_args()
    experiment_id = args.experiment_id
    org_name = args.org_name
    print(f"Starting Tensorboard Events Pull for experiment {experiment_id}", flush=True)
    while True:
        sleep(30)
        handler_metadata = get_handler_metadata_with_jobs(experiment_id, "experiment")
        user_id = handler_metadata.get("user_id", None)
        workspace_id = handler_metadata.get("workspace", None)
        if not user_id:
            print("User ID not defined for Tensorboard Events Pull", flush=True)
            continue
        if not workspace_id:
            print("Workspace not defined for Tensorboard Events Pull", flush=True)
            continue
        workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
        results_root = "/results"
        cs_instance, _ = create_cs_instance(workspace_metadata)
        if not cs_instance:
            print(f"Unable to create cloud storage instance for Tensorboard Events Pull for experiment {experiment_id}", flush=True)
            continue

        jobs = handler_metadata.get("jobs", [])
        if len(jobs) == 0:
            print(f"No jobs found for experiment {experiment_id}", flush=True)
            continue
        for job in jobs:
            action = job.get('action', None)
            job_id = job.get('id', None)
            if action and job_id:
                results_dir = os.path.join(results_root, job_id)
                tf_events_path = results_dir + "/" + action
                tf_events_path = tf_events_path.lstrip('/')
                if cs_instance.is_folder(tf_events_path):
                    _, objects = cs_instance.list_files_in_folder(tf_events_path)
                    tf_events_objects = filter_file_objects(objects, regex_pattern=r'.*\.tfevents.+$')
                    if len(tf_events_objects) == 0:
                        print(f"No tfevents files present in {tf_events_path}", flush=True)
                    for obj in tf_events_objects:
                        file = obj.name
                        basename = os.path.basename(file)
                        destination = f'/tfevents/{action}/{basename}'
                        if not os.path.exists(destination):
                            cs_instance.download_file(file, destination)
                            print(f"Downloaded tfevents file to {destination}", flush=True)
                        else:
                            current_last_modified = os.path.getmtime(destination)
                            if hasattr(obj, 'last_modified'):
                                obj_last_modified = obj.last_modified
                            else:
                                obj_last_modified = obj.extra['last_modified']
                            date_obj = datetime.strptime(obj_last_modified, '%Y-%m-%dT%H:%M:%S.%fZ')
                            timestamp_float = date_obj.replace(tzinfo=timezone.utc).timestamp()
                            if timestamp_float > current_last_modified:
                                print("File has been modified, downloading file now", flush=True)
                                cs_instance.download_file(file, destination)
                                print(f"Downloaded tfevents file to {destination}", flush=True)
                else:
                    print(f"Path {tf_events_path} does not exist in cloud storage for experiment {experiment_id}", flush=True)
