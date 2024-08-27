import sys
import os
from time import sleep
from constants import TENSORBOARD_EXPERIMENT_LIMIT
from handlers.mongo_handler import MongoHandler
from handlers.stateless_handlers import get_user
from handlers.utilities import Code
from handlers.docker_images import DOCKER_IMAGE_MAPPER
from job_utils import executor as jobDriver

release_name = os.getenv("RELEASE_NAME", 'tao-api')


class TensorboardHandler:

    """
    Handler class for Tensorboard Services
    Create: Start Tensorboard Service
    Delete: Delete Tensorboard Service
    """

    @staticmethod
    def start(org, experiment_id, user_id, replicas=1):
        print(f'Starting Tensorboard Service for experiment {experiment_id}', file=sys.stderr)
        tb_deployment_name = f'{release_name}-tb-deployment-{experiment_id}'
        tb_service_name = f"{release_name}-tb-service-{experiment_id}"
        tb_ingress_name = f'{release_name}-tb-ingress-{experiment_id}'
        tb_ingress_path = f'/tensorboard/v1/orgs/{org}/experiments/{experiment_id}'
        command = f"tensorboard --logdir /tfevents --bind_all --path_prefix={tb_ingress_path}"
        logs_command = f"umask 0 && /venv/bin/python3 tb_events_pull_start.py --experiment_id={experiment_id} --org_name={org}"
        tb_image = DOCKER_IMAGE_MAPPER["tensorboard"]
        logs_image = DOCKER_IMAGE_MAPPER["api"]
        jobDriver.create_tensorboard_deployment(tb_deployment_name, tb_image, command, logs_image, logs_command, replicas=replicas)
        timeout = 60
        not_ready_log = False
        print("Check deployment status", file=sys.stderr)
        while (timeout > 0):
            stat_dict = jobDriver.status_tensorboard_deployment(tb_deployment_name, replicas=replicas)
            status = stat_dict.get("status", "Unknown")
            if status == "Running":
                print(f"Deployed Tensorboard for {experiment_id}", file=sys.stderr)
                # k8s service naming rule requres to start with an alphabetic character
                # https://kubernetes.io/docs/concepts/overview/working-with-objects/names/
                TensorboardHandler.add_to_user_metadata(user_id)
                return TensorboardHandler.start_tb_service(tb_service_name, deploy_label=tb_deployment_name, tb_ingress_name=tb_ingress_name, tb_ingress_path=tb_ingress_path)
            if status == "ReplicaNotReady" and not_ready_log is False:
                print("TensorboardService is deployed but replica not ready.", file=sys.stderr)
                not_ready_log = True
            sleep(1)
            timeout -= 1
        print(f"Failed to deploy Tensorboard {experiment_id}", file=sys.stderr)
        return Code(500, {}, f"Timeout Error: Tensorboard status: {status} after {timeout} seconds")

    @staticmethod
    def stop(experiment_id, user_id):
        print(f"Stopping Tensorboard job for {experiment_id}", file=sys.stderr)
        deployment_name = f'{release_name}-tb-deployment-{experiment_id}'
        tb_service_name = f"{release_name}-tb-service-{experiment_id}"
        tb_ingress_name = f'{release_name}-tb-ingress-{experiment_id}'
        jobDriver.delete_tensorboard_deployment(deployment_name)
        jobDriver.delete_tensorboard_service(tb_service_name)
        jobDriver.delete_tensorboard_ingress(tb_ingress_name)
        TensorboardHandler.remove_from_user_metadata(user_id)
        return Code(200, {}, "Delete Tensorboard Started")

    @staticmethod
    def start_tb_service(tb_service_name, deploy_label, tb_ingress_name, tb_ingress_path):
        jobDriver.create_tensorboard_service(tb_service_name, deploy_label)
        timeout = 60
        print("Check TB Service status", file=sys.stderr)
        not_ready_log = False
        while (timeout > 0):
            service_stat_dict = jobDriver.status_tb_service(tb_service_name)
            service_status = service_stat_dict.get("status", "Unknown")
            if service_status == "Running":
                jobDriver.create_tensorboard_ingress(tb_service_name, tb_ingress_name, tb_ingress_path)
                tb_service_ip = service_stat_dict.get("tb_service_ip", None)
                print(f"Created Tensorboard service {tb_service_name} at {tb_service_ip}", file=sys.stderr)
                return Code(201, "Created Tensorboard Service")
            if service_status == "NotReady" and not_ready_log is False:
                print("TB Service is started but not ready.", file=sys.stderr)
                not_ready_log = True
            sleep(1)
            timeout -= 1
        print(f"Failed to create Tensorboard service {tb_service_name}", file=sys.stderr)
        return Code(500, {}, f"Error: Tensorboard service status: {service_status}")

    @staticmethod
    def add_to_user_metadata(user_id):
        mongo_users = MongoHandler("tao", "users")
        user_metadata = get_user(user_id, mongo_users)
        tensorboard_experiment_count = user_metadata.get("tensorboard_experiment_count", 0)
        tensorboard_experiment_count += 1
        user_metadata["tensorboard_experiment_count"] = tensorboard_experiment_count
        mongo_users.upsert({'id': user_id}, user_metadata)
        print(f"Number of Tensorboard Experiments for user {user_id} is {tensorboard_experiment_count}", file=sys.stderr)

    @staticmethod
    def remove_from_user_metadata(user_id):
        mongo_users = MongoHandler("tao", "users")
        user_metadata = get_user(user_id, mongo_users)
        tensorboard_experiment_count = user_metadata.get("tensorboard_experiment_count", 0)
        if tensorboard_experiment_count > 0:
            tensorboard_experiment_count -= 1
            user_metadata["tensorboard_experiment_count"] = tensorboard_experiment_count
            mongo_users.upsert({'id': user_id}, user_metadata)
            print(f"Number of Tensorboard Experiments for user {user_id} is {tensorboard_experiment_count}", file=sys.stderr)

    @staticmethod
    def check_user_metadata(user_id):
        user_metadata = get_user(user_id)
        tensorboard_experiment_count = user_metadata.get("tensorboard_experiment_count", 0)
        if tensorboard_experiment_count >= TENSORBOARD_EXPERIMENT_LIMIT:
            return False
        return True
