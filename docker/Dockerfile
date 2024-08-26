
################# BASE IMAGE #################

FROM nvcr.io/nvidia/base/ubuntu:20.04_x64_2022-09-23 AS base
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get upgrade --yes && \
    apt-get remove --yes python3.10 && \
    apt-get install --yes software-properties-common git curl && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install --no-install-suggests --no-install-recommends --yes python3.11 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    apt-get install --no-install-suggests --no-install-recommends --yes \
      libpython3.11-dev && \
    apt-get autoremove --yes

RUN apt-get update && apt-get install -y sshfs fuse jq rsync curl python-certifi
################ BUILD IMAGE #################

FROM base AS build
RUN apt-get install --no-install-suggests --no-install-recommends --yes \
      gcc \
      wget \
      unzip \
      python3.11-venv && \
    python3 -m venv /venv && \
    chmod -R a+w /venv && \
    /venv/bin/pip install --upgrade pip setuptools==65.5.1 wheel==0.38.1

################ STAGING IMAGE #################

FROM build AS staging

# Add python kubernetes client
# NOTE: Install with different package name because NGC scan gets mixed up with OpenShift kubernetes api server
RUN wget https://github.com/kubernetes-client/python/archive/refs/tags/v23.6.0.zip && \
    unzip v23.6.0.zip && \
    cd python-23.6.0 && \
    sed -i "s/PACKAGE_NAME\ \=\ .*/PACKAGE_NAME = \"kubernetes-client\"/" setup.py && \
    /venv/bin/pip install --disable-pip-version-check -r requirements.txt && \
    /venv/bin/python setup.py install

# Add NGC CLI
ARG NGC_CLI_URL='https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.31.0/files/ngccli_linux.zip'
RUN mkdir -p /opt/ngccli/
RUN wget --content-disposition $NGC_CLI_URL -P /opt/ngccli/ -O /opt/ngccli/ngccli_linux.zip && \
    unzip -o /opt/ngccli/ngccli_linux.zip -d /opt/ngccli/ && \
    /opt/ngccli/ngc-cli/ngc --version
ENV PATH="/opt/ngccli/ngc-cli:${PATH}" NGC_CLI_TRACE_DISABLE=1

# Set API's base workdir
WORKDIR /opt/api

# Add API's Python requirements
COPY docker/requirements.txt .
RUN /venv/bin/pip install --disable-pip-version-check -r requirements.txt

# Fetch metadata for PTMs
ARG NGC_KEY
ARG PTM_ORG_TEAMS="nvidia/tao,iasixjqzw1hj/no-team,ea-tlt/tao_ea"
ARG DEPLOYMENT_MODE=PROD
ENV DEPLOYMENT_MODE=$DEPLOYMENT_MODE
COPY api/utils.py .
COPY api/constants.py .
COPY api/pretrained_models.py .
COPY api/pretrained_models.csv .
COPY cli/version.py version.py
COPY api/handlers/network_configs/ handlers/network_configs/
RUN /venv/bin/python3 -u pretrained_models.py --org-teams=$PTM_ORG_TEAMS --ngc-api-key $NGC_KEY

# Copy project files
COPY api/ .
COPY cli/tao_cli/enum_constants.py .
COPY notebooks/ notebooks/

# Add pyarmor runtime for metrics secret
COPY docker/pyarmor-regfile-1219.zip .
RUN /venv/bin/pyarmor reg pyarmor-regfile-1219.zip && rm pyarmor-regfile-1219.zip
RUN cd auth_utils && /venv/bin/pyarmor gen runtime --output .

################ PRODUCT IMAGE #################

FROM base

# Install basic stuff
RUN apt-get update && apt-get install --no-install-suggests --no-install-recommends --yes \
      wget \
      curl \
      unzip \
      ffmpeg \
      libsm6 \
      libxext6 \
      sudo \
      gnupg && \
    curl -fsSL https://nginx.org/keys/nginx_signing.key | apt-key add - && \
    echo "deb https://nginx.org/packages/ubuntu/ focal nginx" > /etc/apt/sources.list.d/nginx.list && \
    apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends --yes \
      nginx

# Install JFrog CLI for artifactory properties.
RUN wget -qO - https://releases.jfrog.io/artifactory/api/gpg/key/public | apt-key add - && \
    echo "deb https://releases.jfrog.io/artifactory/jfrog-debs xenial contrib" | tee -a /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y jfrog-cli

# Version info
ARG TAO_VERSION=unknown GIT_BRANCH=unknown GIT_COMMIT_SHA=unknown GIT_COMMIT_TIME=unknown
ENV TAO_VERSION=$TAO_VERSION GIT_BRANCH=$GIT_BRANCH GIT_COMMIT_SHA=$GIT_COMMIT_SHA GIT_COMMIT_TIME=$GIT_COMMIT_TIME

# Install API
WORKDIR /opt/api
COPY --from=staging /opt/api .
COPY --from=staging /venv /venv
COPY --from=staging /opt/ngccli/ngccli_linux.zip /opt/ngccli/ngccli_linux.zip
ENV PATH="/opt/ngccli/ngc-cli:/venv/bin:${PATH}" NGC_CLI_TRACE_DISABLE=1
COPY api/nginx.conf /etc/nginx
EXPOSE 8000

# Default command
CMD /bin/bash app_start.sh
