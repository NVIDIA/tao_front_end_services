#!/usr/bin/env bash

set -eo pipefail
cd "$( dirname "${BASH_SOURCE[0]}" )"

TAO_VERSION="5.5.0"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
GIT_COMMIT_SHA="$(git rev-parse HEAD)"
GIT_COMMIT_TIME="$(TZ=UTC0 git show --quiet --date='format-local:%Y%m%dT%H%M%SZ' --format='%cd')"

TAO_API_REGISTRY="nvcr.io"
TAO_API_ORG="nvidia"
TAO_API_TEAM="tao"

NGC_KEY="$(grep apikey ~/.ngc/config 2>/dev/null | cut -d'=' -f2 | xargs; true)"
[ -z "$NGC_KEY" ] && NGC_KEY=no_key_val

UUID_TAO_HELM=$(uuidgen)
export UUID_TAO_HELM=$UUID_TAO_HELM
tag="$USER-$UUID_TAO_HELM"

DEPLOYMENT_MODE="PROD"

# Build parameters.
BUILD_DOCKER="0"
PUSH_DOCKER="0"
FORCE="0"


# Parse command line.
while [[ $# -gt 0 ]]
do
key="$1"
param_key=$(echo $key | cut -f1 -d= )
param_value=$(echo $key | sed 's/.*=//')
case $param_key in
    -b|--build)
    BUILD_DOCKER="1"
    shift # past argument
    ;;
    -n|--ngc_api_key)
    NGC_KEY=$param_value
    shift # past argument
    ;;
    --tao_version)
    TAO_VERSION=$param_value
    shift # past argument
    ;;
    --tao_api_registry)
    TAO_API_REGISTRY=$param_value
    shift # past argument
    ;;
    --tao_api_org)
    TAO_API_ORG=$param_value
    shift # past argument
    ;;
    --tao_api_team)
    TAO_API_TEAM=$param_value
    shift # past argument
    ;;
    --deployment_mode)
    DEPLOYMENT_MODE=$param_value
    shift # past argument
    ;;
    -p|--push)
    PUSH_DOCKER="1"
    shift # past argument
    ;;
    -f|--force)
    FORCE=1
    shift
    ;;
    --default)
    BUILD_DOCKER="1"
    PUSH_DOCKER="0"
    FORCE="0"
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done

echo "TAO_API_REGISTRY=${TAO_API_REGISTRY}"
echo "TAO_API_ORG=${TAO_API_ORG}"
echo "TAO_API_TEAM=${TAO_API_TEAM}"
echo "DEPLOYMENT_MODE=${DEPLOYMENT_MODE}"

TAO_API_REPOSITORY=$TAO_API_REGISTRY/$TAO_API_ORG/$TAO_API_TEAM/tao-api
export TAO_API_REPOSITORY=$TAO_API_REPOSITORY
echo $TAO_API_REPOSITORY > TAO_API_REPOSITORY.txt
echo $UUID_TAO_HELM > UUID_TAO_HELM.txt
chmod 777 TAO_API_REPOSITORY.txt UUID_TAO_HELM.txt

if [ $DEPLOYMENT_MODE = "PROD" ]; then
    NGC_CLI_URL="https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.31.0/files/ngccli_linux.zip"
    PTM_ORG_TEAMS="nvidia/tao"
else
    NGC_CLI_URL="https://api.stg.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.95.12/files/ngccli_linux.zip"
    PTM_ORG_TEAMS="nvidia/tao"
fi

# Build docker
if [ $BUILD_DOCKER = "1" ]; then
    echo "Building base docker ..."
    if [ $FORCE = "1" ]; then
        echo "Forcing docker build without cache ..."
        NO_CACHE="--no-cache"
    else
        NO_CACHE=""
    fi
    DOCKER_BUILDKIT=1 docker build --build-arg TAO_VERSION=$TAO_VERSION --build-arg GIT_BRANCH=$GIT_BRANCH --build-arg GIT_COMMIT_SHA=$GIT_COMMIT_SHA --build-arg GIT_COMMIT_TIME=$GIT_COMMIT_TIME --build-arg NGC_KEY=$NGC_KEY --build-arg DEPLOYMENT_MODE=$DEPLOYMENT_MODE --build-arg NGC_CLI_URL=$NGC_CLI_URL --build-arg PTM_ORG_TEAMS=$PTM_ORG_TEAMS --build-arg ORG_NAME=$TAO_API_ORG --build-arg TEAM_NAME=$TAO_API_TEAM --pull -f $NV_TAO_API_TOP/docker/Dockerfile -t $TAO_API_REPOSITORY:$tag $NO_CACHE \
        --network=host $NV_TAO_API_TOP/.
    if [ $PUSH_DOCKER = "1" ]; then
        echo "Pushing docker ..."
        docker push $TAO_API_REPOSITORY:$tag
        digest=$(docker inspect --format='{{index .RepoDigests 0}}' $TAO_API_REPOSITORY:$tag)
        echo -e "\033[1;33mUpdate the digest in the manifest.json file to:\033[0m"
        echo $digest
    else
        echo "Skip pushing docker ..."
    fi
# Exit by printing usage.
else
    echo "Usage: ./build.sh --build [--push] [--force]"
fi
