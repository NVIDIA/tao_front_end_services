#!/usr/bin/env bash
umask 0
unzip -q /opt/ngccli/ngccli_linux.zip -d /opt/ngccli/ && /opt/ngccli/ngc-cli/ngc --version
if [ -f /usr/local/share/ca-certificates/my-cert.crt ]; then
  update-ca-certificates
  echo -e "\n# MY-CERT" >> /opt/ngccli/ngc-cli/certifi/cacert.pem
  cat /usr/local/share/ca-certificates/my-cert.crt >> /opt/ngccli/ngc-cli/certifi/cacert.pem
fi
if [[ "$NGC_RUNNER" == "True" ]]; then
  mkdir -p /shared/users && chmod 777 /shared/users ; true
  mkdir -p /users && chmod -R 777 /users
else
  rm -rf /shared/users/$ADMIN_UUID/*
  cp -r shared/* /shared/ ; chmod 777 /shared/users ; chmod -R 777 /shared/users/$ADMIN_UUID 2>/dev/null ; true
fi
cp -r notebooks /shared/
service nginx start
/venv/bin/uwsgi --ini uwsgi.ini
