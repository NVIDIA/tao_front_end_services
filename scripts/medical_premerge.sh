#! /bin/bash

set +e

# input parameters
if [ -z "$IMAGE_API" ]; then
    echo "IMAGE_API is not set"
    IMAGE_API=$(docker images --format '{{.CreatedAt}}\t{{.Repository}}:{{.Tag}}' | sort -r | head -n 1 | cut -f2-)
    echo "Using the latest image: $IMAGE_API"
fi

if [ -z "$NGC_KEY" ]; then
    echo "NGC_KEY is not set"
    exit 1
fi

if [ -z "$NV_NVTL_API_TOP"]; then
    echo "NV_NVTL_API_TOP is not set"
    NV_NVTL_API_TOP=$(pwd)
    echo "Using the current directory: $NV_NVTL_API_TOP"
    # check if the current directory has "runtests.sh"
    if [ ! -f "$NV_NVTL_API_TOP/runtests.sh" ]; then
        echo "runtests.sh is not found in the current directory"
        exit 1
    fi
fi

# Start the container
docker run -itd --rm --name premerge --ipc host --network host -e NGC_CLI_TRACE_DISABLE=1 -e NGC_KEY=$NGC_KEY $IMAGE_API /bin/bash

# Copy the NV_NVTL_API_TOP folder into the container
docker cp $NV_NVTL_API_TOP premerge:/opt/tao-toolkit-api

# Check the version of the NGC CLI
docker exec -it premerge /bin/bash -c '/opt/ngccli/ngc-cli/ngc --version'

# Set the NGC API key
docker exec premerge /bin/sh -c "sed -i s/NGC_API_KEY_FIXME/\${NGC_KEY}/g /opt/tao-toolkit-api/docker/ngc_config_file && mkdir -p /root/.ngc/ && cp /opt/tao-toolkit-api/docker/ngc_config_file /root/.ngc/config && chmod -R 777 /root/.ngc"

# Run the tests
docker exec -it premerge /bin/bash -c 'export PATH=/opt/ngccli/ngc-cli:\${PATH}'

docker exec -it premerge /bin/bash -c 'cd /opt/tao-toolkit-api && ./runtests.sh'

docker stop premerge
