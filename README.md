# TAO Toolkit API

* [Overview](#Overview)
    * [Taoclient](#Taoclient)
* [Pre-Requisites](#Pre-Requisites)
    * [Requirements](#Requirements)
        * [Hardware Requirements](#HardwareRequirements)
        * [Software Requirements](#SoftwareRequirements)
* [Getting Started](#GettingStarted)
	* [Updating docker](#Updatingdocker)
		* [Build docker](#Builddocker)
* [Contribution Guidelines](#ContributionGuidelines)
* [License](#License)

## <a name='Overview'></a>Overview

TAO Toolkit is a Python package hosted on the NVIDIA Python Package Index. It interacts with lower-level TAO dockers available from the NVIDIA GPU Accelerated Container Registry (NGC). The TAO containers come pre-installed with all dependencies required for training. The output of the TAO workflow is a trained model that can be deployed for inference on NVIDIA devices using DeepStream, TensorRT and Triton.

This repository includes the essential components for enabling clients to interact with TAO DNN services through a well-defined set of RESTful endpoints. The TAO API services can be deployed on any of the major cloud services like AWS, Azure or Google Cloud. TAO DNN services includes a large pool of Deep Learnig models under various domains like Object Detection, Image Classification, Image Segmentation, Auto-Labeling, Data-Analytics, and many Deep Learning models curated to a particular use-case like Action recognition, Re-identification etc.

### <a name='Taoclient'></a>TAO Client

TAO-Client provides an command line interface to interact with the TAO Toolkit API server by using click python package and requests to format and make RestFUL API calls, instead of relying on direct API calls

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

The host machine needs to have containerization technologies like kubernetes and helm installed to deploy different services. TAO provides quick start scripts for deploying TAO API onto your machine [here](https://docs.nvidia.com/tao/tao-toolkit/text/tao_toolkit_api/api_setup.html).

## <a name='GettingStarted'></a>Getting Started

Once the repository is cloned and TAO API service is deployed, you can start working on the models supported via jupyternotebook interface. In your browser go to the following address `http://<api_hosted_machine_ip>:31951/notebook/` and enter into the api or cli folders.

The notebooks in the api folder make REST API calls directly, whereas the notebooks in the cli folder abstract these restAPI calls into CLI commands to execute the workflow.
### <a name='Updatingdocker'></a>Updating docker

In the case where you would like to modify versions of the the third party dependencies supported by default in the API docker container, please follow the steps below:

#### <a name='Builddocker'></a>Build docker

The dev docker is defined in `$NV_TAO_API_TOP/docker/Dockerfile`. The python packages required for the TAO dev is defined in `$NV_TAO_API_TOP/docker/requirements.txt`. Once you have made the required change, please update the docker using the build script in the same directory.

Take necessary backups of previous runs as we are going to remove the existing pvc folder before re-deploying the api service
```
sudo rm -rf /mnt/nfs_share/* && make docker_build && make helm_install && make cli_install
```

The above step produces a digest file associated with the docker. This is a unique identifier for the docker. So please note this, and update all references of the old digest in the repository with the new digest. You may find the old digest in the `$NV_TAO_API_TOP/docker/manifest.json`.

Push you final updated changes to the repository so that other developers can leverage and sync with the new dev environment.

Please note that if for some reason you would like to force build the docker without using a cache from the previous docker, you may do so by using the `--force` option.

```sh
bash $NV_TAO_API_TOP/docker/build.sh --build --push --force
```

## <a name='ContributionGuidelines'></a>Contribution Guidelines
TAO Toolkit API is not accepting contributions as part of the TAO 5.0 release, but will be open in the future.

## <a name='License'></a>License
This project is licensed under the [Apache-2.0](./LICENSE) License.
