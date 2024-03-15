#!/usr/bin/env bash
umask 0
unzip -q /opt/ngccli/ngccli_linux.zip -d /opt/ngccli/ && /opt/ngccli/ngc-cli/ngc --version
if [ -f /usr/local/share/ca-certificates/my-cert.crt ]; then
  update-ca-certificates
  echo -e "\n# MY-CERT" >> /opt/ngccli/ngc-cli/certifi/cacert.pem
  cat /usr/local/share/ca-certificates/my-cert.crt >> /opt/ngccli/ngc-cli/certifi/cacert.pem
fi
if [[ "$NGC_RUNNER" == "True" ]]; then
  sudo mkdir -p /users && chmod -R 777 /users
fi
json_string="$2"
ngcApiKey=$(jq -r '.auths."nvcr.io".password' <<< "$json_string")
if [[ "$NGC_RUNNER" == "True" ]]; then
    python3 pretrained_models.py --shared-folder-path / --org-teams $1 --ngc-api-key $ngcApiKey
else
    python3 pretrained_models.py --shared-folder-path /shared --org-teams $1 --ngc-api-key $ngcApiKey
fi