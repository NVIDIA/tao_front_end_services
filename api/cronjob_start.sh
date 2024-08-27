#!/usr/bin/env bash
umask 0
unzip -q /opt/ngccli/ngccli_linux.zip -d /opt/ngccli/ && /opt/ngccli/ngc-cli/ngc --version
if [ -f /usr/local/share/ca-certificates/my-cert.crt ]; then
  update-ca-certificates
  echo -e "\n# MY-CERT" >> /opt/ngccli/ngc-cli/certifi/cacert.pem
  cat /usr/local/share/ca-certificates/my-cert.crt >> /opt/ngccli/ngc-cli/certifi/cacert.pem
fi
json_string="$2"
ngcApiKey=$(jq -r '.auths."nvcr.io".password' <<< "$json_string")
python3 pretrained_models.py --shared-folder-path /shared --org-teams $1 --ngc-api-key $ngcApiKey

## Install mongodump
apt update
wget https://fastdl.mongodb.org/tools/db/mongodb-database-tools-ubuntu2204-x86_64-100.10.0.deb ## Update this when upgrading from Ubuntu 20.04 -> 22.04 OR x86 -> ARM
apt install ./mongodb-database-tools-*-100.9.5.deb
rm -f mongodb-database-tools-*.deb
python3 mongodb_backup.py --access-key $3 --secret-key $4 --s3-bucket-name $5 --s3-bucket-region $6