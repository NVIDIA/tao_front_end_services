import json
import subprocess
import time

def remove_corrupted_images_workflow(workspace_id, dataset_id):
    """Workflow to call Data-Services action to move out corrupted images"""

    model_name = "image"
    # Create experiment
    experiment_id = subprocess.getoutput(f"tao-client {model_name} experiment-create --network_arch {model_name} --encryption_key tlt_encode --workspace {workspace_id}")

    # Assign dataset to experiment
    dataset_information = {"inference_dataset":dataset_id}
    subprocess.getoutput(f"tao-client {model_name} patch-artifact-metadata --id {experiment_id} --job_type experiment --update_info '{json.dumps(dataset_information)}' ")

    # Get default spec schema
    specs = subprocess.getoutput(f"tao-client {model_name} get-spec --action validate --job_type experiment --id {experiment_id}")
    specs = json.loads(specs)

    # Run action
    job_id = subprocess.getoutput(f"tao-client {model_name} experiment-run-action --action validate --id {experiment_id} --specs '{json.dumps(specs)}'")

    # Monitor job
    while True:
        response = subprocess.getoutput(f"tao-client {model_name} get-action-status --job_type experiment --id {experiment_id} --job {job_id}")
        response = json.loads(response)
        if "error_desc" in response.keys() and response["error_desc"] in ("Job trying to retrieve not found", "No AutoML run found"):
            print("Job is being created")
            time.sleep(5)
            continue
        assert "status" in response.keys() and response.get("status") != "Error"
        if response.get("status") in ["Done","Error"]:
            break
        time.sleep(15)

    # Delete experiment
    subprocess.getoutput(f"tao-client {model_name} experiment-delete --id {experiment_id}")
    return job_id
