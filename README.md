# NVIDIA Transfer Learning API

* [Overview](#Overview)
    * [NVTL-Client](#NVTL-Client)
* [Pre-Requisites](#Pre-Requisites)
    * [Requirements](#Requirements)
        * [Hardware Requirements](#HardwareRequirements)
        * [Software Requirements](#SoftwareRequirements)
* [Getting Started](#GettingStarted)
	* [Updating docker](#Updatingdocker)
		* [Build docker](#Builddocker)
* [NV Vault Deployment](#NVVaultDeployment)
* [MONAI Service Deployment](#MONAIServiceDeployment)
* [DEV MODE](#DevMode)
* [Run MONAI premerge test locally](#MONAIPremerge)
* [Contribution Guidelines](#ContributionGuidelines)
* [License](#License)

## <a name='Overview'></a>Overview

NVIDIA Transfer Learning API is a cloud service that enables building end-to-end AI models using custom datasets. In addition to exposing NVIDIA Transfer Learning functionality through APIs, the service also enables a client to build end-to-end workflows - creating datasets, models, obtaining pretrained models from NGC, obtaining default specs, training, evaluating, optimizing, and exporting models for deployment on edge. NVTL jobs run on GPUs within a multi-node cloud cluster.

You can develop client applications on top of the provided API, or use the provided NVTL remote client CLI.

This repository includes the essential components for enabling clients to interact with NVTL DNN services through a well-defined set of RESTful endpoints. The NVTL API services can be deployed on any of the major cloud services like AWS, Azure or Google Cloud. NVTL DNN services includes a large pool of Deep Learnig models under various domains like Object Detection, Image Classification, Image Segmentation, Auto-Labeling, Data-Analytics, and many Deep Learning models curated to a particular use-case like Action recognition, Re-identification etc.

### <a name='NVTL-Client'></a>NVTL Client

NVTL-Client provides an command line interface to interact with the NVIDIA Transfer Learning API server by using click python package and requests to format and make RestFUL API calls, instead of relying on direct API calls

## <a name='Pre-Requisites'></a>Pre-Requisites
### <a name='Requirements'></a>Requirements

#### <a name='HardwareRequirements'></a>Hardware Requirements

* 32 GB system RAM
* 32 GB of GPU RAM
* 8 core CPU
* 1 NVIDIA GPU
* 100 GB of SSD space

#### <a name='SoftwareRequirements'></a>Software Requirements

| **Software**                     | **Version** |
| :--- | :--- |
| Ubuntu LTS                       | >=18.04     |
| python                           | >=3.10.x     |
| docker-ce                        | >19.03.5    |

The host machine needs to have containerization technologies like kubernetes and helm installed to deploy different services. NVTL provides quick start scripts for deploying NVTL API onto your machine [here](https://docs.nvidia.com/tao/tao-toolkit/text/tao_toolkit_api/api_setup.html).

## <a name='GettingStarted'></a>Getting Started

Once the repository is cloned and NVTL API service is deployed, you can start working on the models supported via jupyternotebook interface. In your browser go to the following address `http://<api_hosted_machine_ip>:31951/notebook/` and enter into the api or cli folders.

The notebooks in the api folder make REST API calls directly, whereas the notebooks in the cli folder abstract these restAPI calls into CLI commands to execute the workflow.
### <a name='Updatingdocker'></a>Updating docker

In the case where you would like to modify versions of the the third party dependencies supported by default in the API docker container, please follow the steps below:

#### <a name='Builddocker'></a>Build docker

The dev docker is defined in `$NV_NVTL_API_TOP/docker/Dockerfile`. The python packages required for the NVTL dev is defined in `$NV_NVTL_API_TOP/docker/requirements.txt`. Once you have made the required change, please update the docker using the build script in the same directory.

Take necessary backups of previous runs as we are going to remove the existing pvc folder before re-deploying the api service.
```
sudo rm -rf /mnt/nfs_share/* && make docker_build && make helm_install && make cli_install
```

The above step produces a digest file associated with the docker. This is a unique identifier for the docker. So please note this, and update all references of the old digest in the repository with the new digest. You may find the old digest in the `$NV_NVTL_API_TOP/docker/manifest.json`.

Push you final updated changes to the repository so that other developers can leverage and sync with the new dev environment.

Please note that if for some reason you would like to force build the docker without using a cache from the previous docker, you may do so by using the `--force` option.

```sh
bash $NV_NVTL_API_TOP/docker/build.sh --build --push --force
```

## <a name='NVVaultDeployment'></a>NV Vault Deployment
Install NV Vault to protect sensitive information of users in NVIDIA hosted service.
```
helm repo add hashicorp https://helm.releases.hashicorp.com
helm install vault hashicorp/vault \
         --namespace <YOUR-K8S-NAMESPACE> \
         --set='injector.externalVaultAddr=https://<stg|prod>.vault.nvidia.com/' \
         --set='injector.agentImage.repository=nvcr.io/nvidian/nvault-agent' \
         --set='injector.agentImage.tag=<latest-tag>'
```

Apply the config map for NV Vault sidecar.
```
kubectl apply -f vault-agent-configmap.yaml
```
For details about config NV Vault as a sidecar to the k8s pods please refer to [this link](https://gitlab-master.nvidia.com/kaizen/services/vault/docs/-/blob/main/guides/integrations/kubernetes/4-deploy-vault-agent-sidecar-injector.md?ref_type=heads).

## <a name='MONAIServiceDeployment'></a>MONAI Service Deployment
MONAI service deployment with some extra steps to above. Here are some links to MONAI service deployment, which show how to deploy MONAI service on different cloud service providers step-by-step.
1. [Azure deployment](https://confluence.nvidia.com/pages/viewpage.action?spaceKey=CLARA&title=MONAI+Service+Azure+Deployment)


## <a name='DevMode'></a>Develop with DEV_MODE

Please refer to this [Confluence Documentation](https://confluence.nvidia.com/display/CLARA/Enable+Developer+Mode+%28DEV_MODE%29+for+TAO-Toolkit-API) to enable `DEV_MODE` for setup

## <a name='MONAIPremerge'></a>Run MONAI premerge test locally

To run the MONAI premerge test locally, you'll have to `sudo make docker_build` first, and then
```
export NGC_KEY=<ngc key that has access to medical service ea NGC organization>
bash scripts/medical_premerge.sh
```

If you would like to use another image:
```
export IMAGE_API=<image to use>
```

If you would like to run the script outside of root folder
```
export NV_NVTL_API_TOP=<path to the root of this repo>
```

## <a name='ContributionGuidelines'></a>Contribution Guidelines
NVIDIA Transfer Learning API is not accepting contributions as part of the NVTL 5.2 release, but will be open in the future.

## <a name='License'></a>License
This project is licensed under the [Apache-2.0](./LICENSE) License.
