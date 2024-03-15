
################# BASE IMAGE #################

FROM ubuntu:20.04 AS base
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

# Copy project files
WORKDIR /opt/api
COPY api/ .
COPY notebooks/ notebooks/
COPY cli/version.py version.py
COPY docker/requirements.txt .

# Add python kubernetes client
# NOTE: Install with different package name because NGC scan gets mixed up with OpenShift kubernetes api server
RUN wget https://github.com/kubernetes-client/python/archive/refs/tags/v23.6.0.zip && \
    unzip v23.6.0.zip && \
    cd python-23.6.0 && \
    sed -i "s/PACKAGE_NAME\ \=\ .*/PACKAGE_NAME = \"kubernetes-client\"/" setup.py && \
    /venv/bin/pip install --disable-pip-version-check -r requirements.txt && \
    /venv/bin/python setup.py install

# Add NGC CLI
RUN mkdir -p /opt/ngccli/
RUN wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.31.0/files/ngccli_linux.zip' -P /opt/ngccli/ -O /opt/ngccli/ngccli_linux.zip && \
    unzip -o /opt/ngccli/ngccli_linux.zip -d /opt/ngccli/ && \
    /opt/ngccli/ngc-cli/ngc --version
ENV PATH="/opt/ngccli/ngc-cli:${PATH}" NGC_CLI_TRACE_DISABLE=1

# Add API's Python requirements
RUN /venv/bin/pip install --disable-pip-version-check -r requirements.txt

# Fetch metadata for PTMs
ARG NGC_KEY
ARG ORG_NAME
ARG TEAM_NAME
COPY docker/ngc_config_file .
RUN if [ "$NGC_KEY" != "no_key_val" ]; then \
    sed -i 's/NGC_API_KEY_FIXME/'"$NGC_KEY"'/g' ngc_config_file && \
    mkdir -p /root/.ngc/ && cp /opt/api/ngc_config_file /root/.ngc/config && chmod -R 777 /root/.ngc && \
    mkdir -p /var/www/ && cp -r /root/.ngc /var/www/ && chmod -R 777 /var/www/.ngc && \
    rm -rf ngc_config_file; \
  fi
RUN /venv/bin/python3 -u pretrained_models.py --org-teams="nvidia/tao,nvaie" --ngc-api-key $NGC_KEY

################ PRODUCT IMAGE #################

FROM base

# Install basic stuff
RUN apt-get install --no-install-suggests --no-install-recommends --yes \
      wget \
      curl \
      unzip \
      ffmpeg \
      libsm6 \
      libxext6 \
      sudo \
      curl \
      gnupg && \
    echo "deb https://nginx.org/packages/ubuntu/ focal nginx" >> /etc/apt/sources.list && \
    curl -s http://nginx.org/packages/keys/nginx_signing.key | apt-key add - && \
    apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends --yes \
      nginx

# Install JFrog CLI for artifactory properties.
RUN wget -qO - https://releases.jfrog.io/artifactory/api/gpg/key/public | apt-key add - && \
    echo "deb https://releases.jfrog.io/artifactory/jfrog-debs xenial contrib" | tee -a /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y jfrog-cli

# Install API
WORKDIR /opt/api
ARG NGC_KEY
COPY docker/ngc_config_file .
RUN if [ "$NGC_KEY" != "no_key_val" ]; then \
    sed -i 's/NGC_API_KEY_FIXME/'"$NGC_KEY"'/g' ngc_config_file && \
    mkdir -p /root/.ngc/ && cp /opt/api/ngc_config_file /root/.ngc/config && chmod -R 777 /root/.ngc && \
    mkdir -p /var/www/ && cp -r /root/.ngc /var/www/ && chmod -R 777 /var/www/.ngc && \
    rm -rf ngc_config_file; \
  fi
COPY --from=staging /opt/api .
COPY --from=staging /venv /venv
COPY --from=staging /opt/ngccli/ngccli_linux.zip /opt/ngccli/ngccli_linux.zip
ENV PATH="/opt/ngccli/ngc-cli:/venv/bin:${PATH}" NGC_CLI_TRACE_DISABLE=1
COPY api/nginx.conf /etc/nginx
EXPOSE 8000

# Default command
CMD /bin/bash app_start.sh
