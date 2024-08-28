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

"""Create metadata info for all base experiments supported"""
import argparse
import ast
import csv
import datetime
import json
import operator
import os
import uuid
import requests
import traceback
import io
import zipfile
from packaging import version

from utils import read_network_config, get_admin_api_key, get_ngc_artifact_base_url, send_get_request_with_retry, safe_load_file, safe_dump_file
from constants import TAO_NETWORKS

base_exp_uuid = "00000000-0000-0000-0000-000000000000"

DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "PROD")

ngc_api_base_url = "https://api.ngc.nvidia.com/v2" if DEPLOYMENT_MODE == "PROD" else "https://api.stg.ngc.nvidia.com/v2"


class BaseExperimentMetadata:
    """Base Experiment Metadata class"""

    def __init__(
        self,
        shared_folder_path: str,
        org_teams: str,
        ngc_api_key: str | None = None,
        override: bool = False,
        dry_run: bool = False,
    ):
        """Initialize Base Experiment Metadata class

        Args:
            shared_folder_path (str): Root path for base experiments
            org_teams (str): Organization and team names. Each pair of org/team separated by a comma.
            ngc_api_key (str, optional): NGC API Key. Defaults to None.
            override (bool, optional): Override existing base experiments. Defaults to False.
        """
        self.shared_folder_path = shared_folder_path
        self.ngc_api_key = ngc_api_key or get_admin_api_key()
        self.org_team_list = self.prepare_org_team(org_teams)
        self.override = override
        self.metadata: dict = {}
        self.dry_run = dry_run

        if self.override and self.dry_run:
            raise ValueError("Cannot use both `--override` and `--dry-run` flags together!")

        # set default uuids
        self.base_exp_uuid = uuid.UUID(base_exp_uuid)
        self.ptm_uuid = uuid.UUID(base_exp_uuid)

        # create rootdir and metadata file path
        self.rootdir = os.path.abspath(
            os.path.join(self.shared_folder_path, "orgs", str(self.base_exp_uuid), "experiments", str(self.ptm_uuid))
        )
        self.metadata_file = os.path.join(self.rootdir, "ptm_metadatas.json")

        # create rootdir if it doesn't exist
        os.makedirs(self.rootdir, exist_ok=True)

        # create a list of all supported network architectures
        self.supported_network_archs = self.get_supported_netowrk_archs()

        # set tao version and comparison operators
        self.tao_version: version.Version | None = None
        self.comparison_operators = {
            "<=": operator.le,
            "<": operator.lt,
            ">": operator.gt,
            ">=": operator.ge,
            "==": operator.eq,
            "!=": operator.ne,
        }

    def get_tao_version(self):
        """Return current version of Nvidia TAO API."""
        version_locals = {}
        with open("version.py", "r", encoding="utf-8") as version_file:
            exec(version_file.read(), {}, version_locals)  # pylint: disable=W0122
        return version_locals["__version__"]

    def check_version_compatibility(self, version_list: list):
        """Check if the current NVTL version is compatible with the provided version list"""
        if self.tao_version is None:
            try:
                self.tao_version = version.Version(self.get_tao_version())
            except FileNotFoundError as e:
                print("Skipping NVTL Version Check!!! Failed to get current NVTL version! >>>", e)
                return True
        version_ok = True
        for version_str in version_list:
            ver = version_str.strip(r"<>=! ")
            op = version_str[: -len(ver)].strip()
            if op not in self.comparison_operators:
                raise ValueError(f"Invalid version comparison operator: {op}")
            version_ok = self.comparison_operators[op](self.tao_version, version.Version(ver))
            if not version_ok:
                break
        return version_ok

    def get_ngc_token(self, org: str = "", team: str = ""):
        """Authenticate to NGC"""
        # Get the NGC login token
        url = "https://stg.authn.nvidia.com/token"
        if DEPLOYMENT_MODE == "PROD":
            url = "https://authn.nvidia.com/token"
        params = {"service": "ngc", "scope": "group/ngc"}
        if org:
            params["scope"] = f"group/ngc:{org}"
        if team:
            params["scope"] += f"&group/ngc:{org}/{team}"
        headers = {"Accept": "application/json"}
        auth = ("$oauthtoken", self.ngc_api_key)
        response = requests.get(url, headers=headers, auth=auth, params=params)
        return response.json()["token"]

    def prepare_org_team(self, org_teams: str):
        """Prepare org team list"""
        if org_teams:
            org_team_list = []
            for org_team in org_teams.split(","):
                org_team = org_team.replace("/no-team", "")
                org = org_team
                team = ""
                if "/" in org_team:
                    org, team = org_team.split("/")
                org_team_list.append((org, team))
        else:
            print("> No org/team is provided by `--org-team`.")
            org_team_list = self.get_org_teams()
        return org_team_list

    def get_org_teams(self):
        """Get all orgs and teams for the user"""
        print("--------------------------------------------------------")
        print("Getting accessible org/team for the provided NGC API key")
        print("--------------------------------------------------------")
        ngc_token = self.get_ngc_token()
        headers = {"Accept": "application/json", "Authorization": f"Bearer {ngc_token}"}
        url = f"{ngc_api_base_url}/orgs"
        response = requests.get(url, headers=headers, params={"page-size": 1000})
        if response.status_code != 200:
            raise ValueError(response.json())
        orgs = [org["name"] for org in response.json()["organizations"]]
        org_teams = []
        # get team for each org
        for org in orgs:
            url = f"{ngc_api_base_url}/org/{org}/teams"
            response = requests.get(url, headers=headers, params={"page-size": 1000})
            if response.status_code != 200:
                print(response.json())
                continue
            teams = [team["name"] for team in response.json()["teams"]]
            print(f"{org}:", teams)
            org_teams.extend([(org, team) for team in teams])
            org_teams.append((org, ""))
        print("nvidia: ['tao']")
        org_teams.append(("nvidia", "tao"))

        print(f"Created {len(org_teams)} org/team pairs for the provided NGC API key ")
        print("--------------------------------------------------------")
        return org_teams

    @staticmethod
    def get_supported_netowrk_archs():
        """Get the list of all supported network architectures by API"""
        # remove .config.json (12 charachter) from the end of the file name
        return [
            arch[:-12] for arch in os.listdir(f"{os.path.dirname(os.path.abspath(__file__))}/handlers/network_configs/")
        ]

    @staticmethod
    def split_ngc_path(ngc_path):
        """Split ngc path into org, team and model name, model version"""
        path_split = ngc_path.replace("/no-team", "").split("/")
        if len(path_split) == 3:
            org, team, model_name = path_split
        elif len(path_split) == 2:
            org, model_name = path_split
            team = ""
        else:
            raise ValueError(f"Invalid ngc_path: {ngc_path}")
        if ":" in model_name:
            model_name, model_version = model_name.split(":")
        else:
            model_version = ""
        return org, team, model_name, model_version

    @staticmethod
    def get_model_info_from_ngc(
        ngc_token: str, org: str, team: str, model_name: str, model_version: str, file: str = ""
    ):
        """Get model info from NGC"""
        url = ngc_api_base_url
        if team:
            url += f"/org/{org}/team/{team}/models/{model_name}/versions/{model_version}"
        else:
            url += f"/org/{org}/models/{model_name}/versions/{model_version}"
        if file:
            url += f"/files/{file}"
        headers = {"Accept": "application/json", "Authorization": f"Bearer {ngc_token}"}
        response = requests.get(url, headers=headers, params={"page-size": 1000})
        if response.status_code != 200:
            raise ValueError(
                f"Failed to get model info for {model_name}:{model_version} ({response.status_code} {response.reason})"
            )
        return response.json()

    def load_base_experiments_from_csv(self) -> dict:
        """Get base experiments from CSV file"""
        base_experiments: dict[str, dict] = {}
        with open(f"{os.path.dirname(os.path.abspath(__file__))}/pretrained_models.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                display_name, ngc_path, network_arch = row
                org, team, _, _ = self.split_ngc_path(ngc_path)
                ngc_token = self.get_ngc_token(org, team)
                self.add_experiment(base_experiments, display_name, ngc_path, network_arch, ngc_token)
        return base_experiments

    def add_experiment(self, base_experiments, display_name, ngc_path, network_arch, ngc_token):
        """Add experiment to the base experiments lis with unique id"""
        hash_str = f"{ngc_path}:{network_arch}"
        exp_id = str(uuid.uuid5(self.base_exp_uuid, hash_str))
        base_experiments[exp_id] = {
            "id": exp_id,
            "name": display_name,
            "ngc_path": ngc_path,
            "network_arch": network_arch,
            "base_experiment_metadata": {
                "spec_file_present": self.is_base_spec_file_present(ngc_path, exp_id, ngc_token)
            }
        }

    def load_base_experiments_from_ngc(self, page_size: int = 1000) -> dict:
        """Get base experiments from NGC"""
        base_experiments: dict[str, dict] = {}
        for org, team in self.org_team_list:
            print(f"Querying base experiments from '{org}{'/' + team if team else ''}'")
            ngc_token = self.get_ngc_token(org, team)
            headers = {"Accept": "application/json", "Authorization": f"Bearer {ngc_token}"}
            url = f"{ngc_api_base_url}/search/resources/MODEL"

            # Create the query to filter models and the required return fields
            query = f"resourceId:{org}/{team + '/' if team else ''}*"

            # Get the number of pages
            params = {"q": json.dumps({"pageSize": page_size, "query": query})}
            response = requests.get(url, headers=headers, params=params)
            n_pages = response.json()["resultPageTotal"]

            # Get the list of models
            for page_number in range(n_pages):
                params = {
                    "q": json.dumps(
                        {
                            "fields": ["resourceId", "name", "displayName", "orgName", "teamName"],
                            "page": page_number,
                            "pageSize": page_size,
                            "query": query,
                        }
                    )
                }
                response = requests.get(url, headers=headers, params=params)
                results = response.json()["results"]

                # Iterate through the list of models
                for model_list in results:
                    for model in model_list["resources"]:
                        try:
                            model_meta = self.get_model_info_from_ngc(
                                ngc_token, model["orgName"], model.get("teamName", ""), model["name"], ""
                            )
                        except ValueError as e:
                            print(e)
                            continue
                        if "modelVersions" in model_meta:
                            for model_version in model_meta["modelVersions"]:
                                if "customMetrics" in model_version:
                                    ngc_path = f'{model["resourceId"]}:{model_version["versionId"]}'
                                    for customMetrics in model_version["customMetrics"]:
                                        endpoints = []
                                        for key_value in customMetrics.get("attributes", []):
                                            if key_value["key"] == "endpoints":
                                                try:
                                                    endpoints = ast.literal_eval(key_value["value"])
                                                except (SyntaxError, ValueError):
                                                    print(f"{key_value} not loadable by `ast.literal_eval`.")
                                        for network_arch in endpoints:
                                            self.add_experiment(
                                                base_experiments, model.get("displayName"), ngc_path, network_arch, ngc_token
                                            )
        return base_experiments

    def is_base_spec_file_present(self, ngc_path, exp_id, ngc_token):
        """Checks if base experiment has experiment.yaml file"""
        headers = {"Accept": "application/json", "Authorization": f"Bearer {ngc_token}"}
        endpoint, model, version = get_ngc_artifact_base_url(ngc_path)
        download_endpoint = f"{endpoint}/files/experiment.yaml/zip/download"
        response = send_get_request_with_retry(download_endpoint, headers=headers)
        if response.ok:
            dest_path = f"{self.rootdir}/{exp_id}/{model}_v{version}"
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(dest_path)
            return True
        return False

    def extract_metadata(self, model_info, experiment_info, additional_metadata, model_name):
        """Create metadata for the model"""
        if model_info is None:
            raise ValueError(f"Failed to get model info for {experiment_info['ngc_path']}")
        if model_info["modelVersion"]["status"] != "UPLOAD_COMPLETE":
            raise ValueError(f"Model {experiment_info['ngc_path']} is not in UPLOAD_COMPLETE status!")
        network_arch = experiment_info["network_arch"]
        if network_arch not in self.supported_network_archs:
            raise ValueError(f"Network architecture of `{network_arch}` is not supported by API!")

        # Adhoc supported models
        attr = {}
        if model_info["modelVersion"].get("customMetrics"):
            for customMetrics in model_info["modelVersion"]["customMetrics"]:
                for key_value in customMetrics.get("attributes", []):
                    attr_key = key_value.get("key")
                    if attr_key:
                        try:
                            attr[attr_key] = ast.literal_eval(key_value.get("value"))
                        except (SyntaxError, ValueError):
                            attr[attr_key] = key_value.get("value")
            if attr.get("tao_version"):
                attr["tao_version_check"] = self.check_version_compatibility(attr["tao_version"])
                if not attr["tao_version_check"]:
                    raise ValueError(
                        f"Model {experiment_info['ngc_path']} requires API version of {attr['tao_version']} "
                        "but the current API version is {self.tao_version}!"
                    )
            if not attr.get("trainable"):
                raise ValueError(f"Model {experiment_info['ngc_path']} is not trainable!")
            for endpoint in attr.get("endpoints", []):
                if endpoint not in self.supported_network_archs:
                    print(
                        f"Skipping the 'endpoint' metadata for {experiment_info['ngc_path']}. "
                        f"'endpoint' metadata [{endpoint}] is not supported by API!"
                        "This may prevent base experiment creation in the future releases."
                    )
                    # raise ValueError(f"Network architecture of `{endpoint}` is not supported by API!")

        api_params = read_network_config(network_arch)["api_params"]
        accepted_ds_intents = api_params.get("accepted_ds_intents", [])
        if "visual_changenet" in experiment_info["ngc_path"] and "segment" in experiment_info["ngc_path"]:
            accepted_ds_intents = ["training"]
        base_experiment_pull_complete = "starting"
        if network_arch in TAO_NETWORKS:
            base_experiment_pull_complete = "pull_complete"
        metadata = {
            "id": experiment_info["id"],
            "public": True,
            "read_only": model_info["model"].get("isReadOnly", True),
            "base_experiment": [],
            "train_datasets": [],
            "eval_dataset": None,
            "calibration_dataset": None,
            "inference_dataset": None,
            "checkpoint_choose_method": "best_model",
            "checkpoint_epoch_number": {"id": 0},
            "logo": "https://www.nvidia.com",
            "network_arch": network_arch,
            "dataset_type": api_params["dataset_type"],
            "dataset_formats": api_params.get("formats", read_network_config(api_params["dataset_type"]).get("api_params", {}).get("formats", None)),
            "accepted_dataset_intents": accepted_ds_intents,
            "actions": api_params["actions"],
            "name": experiment_info["name"],
            "description": model_info["modelVersion"].get("description", "") or
                    model_info["model"].get("shortDescription", f"Base Experiment for {network_arch}"),
            "model_description": model_info["model"].get("shortDescription", f"Base Experiment for {network_arch}"),
            "version": model_info["modelVersion"].get("versionId", ""),
            "created_on": model_info["modelVersion"].get("createdDate", datetime.datetime.now().isoformat()),
            "last_modified": model_info["model"].get("updatedDate", datetime.datetime.now().isoformat()),
            "ngc_path": experiment_info["ngc_path"],
            "realtime_infer_support": api_params.get("realtime_infer_support", False),
            "sha256_digest": attr.get("sha256_digest", {}),
            "base_experiment_metadata": {
                "is_backbone":  attr.get("is_backbone", True),
                "is_trainable": attr.get("trainable", False),
                "spec_file_present": experiment_info["base_experiment_metadata"]["spec_file_present"],
                "task": attr.get("task", None),
                "domain": attr.get("domain", None),
                "backbone_type": attr.get("backbone_type", None),
                "backbone_class": attr.get("backbone_class", None),
                "license": attr.get("license", None),
                "model_card_link": f"https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/{model_name}",
            },
            "base_experiment_pull_complete": base_experiment_pull_complete,
            "type": "medical" if network_arch.startswith("medical_") else "vision",
        }
        # some additional specific metadata
        if additional_metadata:
            channel_def = (
                additional_metadata.get("network_data_format", {})
                .get("outputs", {})
                .get("pred", {})
                .get("channel_def", {})
            ).items()
            metadata["model_params"] = {"labels": {k: v.lower() for k, v in channel_def if v.lower() != "background"}}
            metadata["description"] = additional_metadata.get("description", metadata["description"])
        return metadata

    def get_ngc_hosted_base_experiments(self):
        """Get base experiments hosted on NGC"""
        model_info = {}
        valid_base_experiments = {}
        ngc_base_experiments = self.load_base_experiments_from_ngc()
        print("Loaded base experiments from NGC:", len(ngc_base_experiments))
        print("--------------------------------------------------------")
        if DEPLOYMENT_MODE == "PROD":
            print("--------------------------------------------------------")
            experiments_form_csv = self.load_base_experiments_from_csv()
            print("Loaded base experiments from CSV", len(experiments_form_csv))
            ngc_base_experiments = {**ngc_base_experiments, **experiments_form_csv}

        for exp_id, base_experiment in ngc_base_experiments.items():
            ngc_path = base_experiment["ngc_path"]
            org, team, model_name, model_version = self.split_ngc_path(ngc_path)
            ngc_token = self.get_ngc_token(org, team)
            if (org, team) in self.org_team_list:
                # Get ngc model metadata and cache it
                try:
                    if ngc_path not in model_info:
                        model_info[ngc_path] = self.get_model_info_from_ngc(
                            ngc_token, org, team, model_name, model_version
                        )
                        if base_experiment["network_arch"].startswith("medical_"):
                            medical_metadata = self.get_model_info_from_ngc(
                                ngc_token, org, team, model_name, model_version, "configs/metadata.json"
                            )
                        else:
                            medical_metadata = {}

                    # Update metadata for each experiment
                    valid_base_experiments[exp_id] = self.extract_metadata(
                        model_info[ngc_path], base_experiment, medical_metadata, model_name
                    )
                    print(f"Successfully created a base experiment for {ngc_path},{base_experiment['network_arch']}")
                except ValueError as e:
                    print(traceback.format_exc())
                    print(f"Failed to create a base experiment for for {ngc_path} >>> {e}")
                    continue
        return valid_base_experiments

    def get_existing_base_experiments(self):
        """Get existing base experiments"""
        if os.path.isfile(self.metadata_file):
            existing_models = safe_load_file(self.metadata_file)
            return existing_models
        return {}

    def sync(self):
        """Sync metadata info for ngc hosted base experiments"""
        existing_base_experiments = self.get_existing_base_experiments()
        ngc_hosted_base_experiments = self.get_ngc_hosted_base_experiments()
        self.metadata = {**existing_base_experiments, **ngc_hosted_base_experiments}
        if not self.dry_run:
            safe_dump_file(self.metadata_file, self.metadata)
            print(f"Base experiments metadata file created at {self.metadata_file}")
        else:
            print("Skipping NGC metadata edit in dry run mode!")
        print("--------------------------------------------------------")
        print("Existing base experiments:", len(existing_base_experiments))
        print("New base experiments:", len(ngc_hosted_base_experiments))
        print("Total base experiments:", len(self.metadata))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate base experiment metadata file")
    parser.add_argument("--shared-folder-path", help="Root path for base experiments", default="shared")
    parser.add_argument("--org-teams", help="Organization and team names. Each pair of org/team separated by a comma.")
    parser.add_argument("--ngc-api-key", help="NGC API Key")
    parser.add_argument("--dry-run", help="Dry run mode", default=False, action="store_true")
    parser.add_argument("--override", help="Override existing base experiments", action="store_true")
    args = parser.parse_args()
    bem = BaseExperimentMetadata(args.shared_folder_path, args.org_teams, args.ngc_api_key, args.override, args.dry_run)
    bem.sync()
