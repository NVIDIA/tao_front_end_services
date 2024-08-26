#!/usr/bin/env bash
umask 0
unzip -q /opt/ngccli/ngccli_linux.zip -d /opt/ngccli/ && /opt/ngccli/ngc-cli/ngc --version
if [ -f /usr/local/share/ca-certificates/my-cert.crt ]; then
  update-ca-certificates
  echo -e "\n# MY-CERT" >> /opt/ngccli/ngc-cli/certifi/cacert.pem
  cat /usr/local/share/ca-certificates/my-cert.crt >> /opt/ngccli/ngc-cli/certifi/cacert.pem
fi
rm -rf /shared/orgs/00000000-0000-0000-0000-000000000000/*
cp -r shared/* /shared/ ; chmod 777 /shared/orgs ; chmod -R 777 /shared/orgs/00000000-0000-0000-0000-000000000000 2>/dev/null ; true
cp -r notebooks /shared/
service nginx start
/venv/bin/uwsgi --ini uwsgi.ini
