#!/usr/bin/env bash
umask 0
unzip -q /opt/ngccli/ngccli_linux.zip -d /opt/ngccli/ && /opt/ngccli/ngc-cli/ngc --version
if [ -f /usr/local/share/ca-certificates/my-cert.crt ]; then
  update-ca-certificates
  echo -e "\n# MY-CERT" >> /opt/ngccli/ngc-cli/certifi/cacert.pem
  cat /usr/local/share/ca-certificates/my-cert.crt >> /opt/ngccli/ngc-cli/certifi/cacert.pem
fi
/venv/bin/python3 workflow.py
