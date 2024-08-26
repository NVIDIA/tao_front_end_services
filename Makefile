SHELL := /bin/bash

-include .env

helm_install: helm_uninstall helm_push
	@sleep 90; \
	rm -rf tao-api-$(TAO_VERSION)-$(USER)-$(UUID_TAO_HELM).tgz; \
	rm -rf docker/TAO_API_REPOSITORY.txt docker/UUID_TAO_HELM.txt; \
	if ! helm status tao-api >/dev/null 2>&1; \
	then \
		helm repo update && helm install tao-api $(TAO_API_ORG)-$(TAO_API_TEAM)/tao-api --version $(TAO_VERSION)-$(USER)-$(UUID_TAO_HELM) --namespace default ;\
	fi; \
	rm .env; \

helm_push:
	if ! helm repo list | grep -q $(TAO_API_ORG)-$(TAO_API_TEAM); then \
		helm repo add $(TAO_API_ORG)-$(TAO_API_TEAM) 'https://helm.ngc.nvidia.com/$(TAO_API_ORG)/$(TAO_API_TEAM)' --username='$$oauthtoken' --password='$(PASSWORD)'; \
	fi
	helm plugin install https://github.com/chartmuseum/helm-push ;\
	OUTFILE="/tmp/helm-push-$$(date "+%Y.%m.%d-%H.%M.%S").out" ;\
	cp chart/values.yaml chart/values.yaml.default ; \
	sed -i "s#repository: nvcr.io/nvstaging/tao/tao-api#repository: $(TAO_API_REGISTRY)/$(TAO_API_ORG)/$(TAO_API_TEAM)/tao-api#" "chart/values.yaml"; \
	sed -i "s#tag: v$(TAO_VERSION)-nightly-latest#tag: $(USER)-$(UUID_TAO_HELM)#" "chart/values.yaml"; \
	echo -e "\norgName: $(TAO_API_ORG)" >> "chart/values.yaml"; \
	echo "teamName: $(TAO_API_TEAM)" >> "chart/values.yaml"; \
	HASH=$$(git rev-parse --short HEAD:./chart) ;\
	head chart/Chart.yaml ;\
	helm dependency update chart ;\
	helm lint chart ;\
	helm template chart ;\
	helm package chart --version $(TAO_VERSION)-$(USER)-$(UUID_TAO_HELM) --app-version $(TAO_VERSION) ;\
	cp chart/values.yaml.default chart/values.yaml ;\
	rm -rf chart/values.yaml.default ;\
	if ! helm cm-push tao-api-$(TAO_VERSION)-$(USER)-$(UUID_TAO_HELM).tgz $(TAO_API_ORG)-$(TAO_API_TEAM) >& $$OUTFILE ;\
	then \
		echo "Push with hash success!" ;\
		cat "$$OUTFILE" ;\
	else \
		cat "$$OUTFILE" ;\
		if ! grep -q "Error: 409: $(TAO_API_ORG)/$(TAO_API_TEAM)/tao-api-$(TAO_VERSION)-$(USER)-$(UUID_TAO_HELM).tgz already exists" $$OUTFILE ;\
		then \
			echo "Seems there is no latest update in helm! Skipping.." ;\
			exit 0 ;\
		else \
			echo "Error while pushing helm.." ;\
			exit 1 ;\
		fi ;\
	fi ;\

helm_uninstall:
	@if helm status tao-api >/dev/null 2>&1; \
	then \
		helm delete tao-api -n default && sleep 60; \
	fi

docker_build:
	@read -p "Enter the TAO version value [Enter] will use the default value (5.5.0): " TAO_VERSION; TAO_VERSION=$${TAO_VERSION:-"5.5.0"}; echo "TAO_VERSION=$$TAO_VERSION" > .env; \
	read -p "Enter the docker deployment mode value [Enter] will use the default value (PROD): " DEPLOYMENT_MODE; DEPLOYMENT_MODE=$${DEPLOYMENT_MODE:-"PROD"}; echo "DEPLOYMENT_MODE=$$DEPLOYMENT_MODE" >> .env; \
	read -p "Enter the docker registry value [Enter] will use the default value (nvcr.io): " TAO_API_REGISTRY; TAO_API_REGISTRY=$${TAO_API_REGISTRY:-"nvcr.io"}; echo "TAO_API_REGISTRY=$$TAO_API_REGISTRY" >> .env; \
	read -p "Enter the docker org value [Enter] will use the default value (nvstaging): " TAO_API_ORG; TAO_API_ORG=$${TAO_API_ORG:-"nvstaging"}; echo "TAO_API_ORG=$$TAO_API_ORG" >> .env; \
	read -p "Enter the docker team value [Enter] will use the default value (tao): " TAO_API_TEAM; TAO_API_TEAM=$${TAO_API_TEAM:-"tao"}; echo "TAO_API_TEAM=$$TAO_API_TEAM" >> .env; \
	HOME_DIR=$$(getent passwd $$SUDO_USER | cut -d: -f6); \
	APIKEY=$$(grep 'apikey' $$HOME_DIR/.ngc/config | sed 's/apikey = //'); \
	stty -echo && read -p "Enter your ngc_api_key [Enter] will use the value from ~/.ngc/config: " PASSWORD && PASSWORD=$${PASSWORD:-$$APIKEY} && echo "PASSWORD=$$PASSWORD" >> .env; \
	[ ! -z "$$TAO_VERSION" ] && BUILD_ARGS="--tao_version=$$TAO_VERSION"; \
	[ ! -z "$$TAO_API_REGISTRY" ] && BUILD_ARGS="$$BUILD_ARGS --tao_api_registry=$$TAO_API_REGISTRY"; \
	[ ! -z "$$TAO_API_ORG" ] && BUILD_ARGS="$$BUILD_ARGS --tao_api_org=$$TAO_API_ORG"; \
	[ ! -z "$$TAO_API_TEAM" ] && BUILD_ARGS="$$BUILD_ARGS --tao_api_team=$$TAO_API_TEAM"; \
	[ ! -z "$$DEPLOYMENT_MODE" ] && BUILD_ARGS="$$BUILD_ARGS --deployment_mode=$$DEPLOYMENT_MODE"; \
	[ ! -z "$$PASSWORD" ] && BUILD_ARGS="$$BUILD_ARGS --ngc_api_key=$$PASSWORD"; \
       	stty echo && echo && source scripts/envsetup.sh && cd docker && ./build.sh --build --push $$BUILD_ARGS && cd ..; \
	UUID_TAO_HELM=$$(cat docker/UUID_TAO_HELM.txt) && echo "UUID_TAO_HELM=$$UUID_TAO_HELM" >> .env; \

dependencies:
	python3 -m pip install --upgrade pip setuptools wheel --no-cache-dir

cli_build: dependencies
	cd cli && python3 setup.py bdist_wheel

cli_clean:
	rm -rf cli/dist
	rm -rf cli/build
	rm -rf cli/*.egg-info
	rm -rf cli/./**/*/__pycache__
	rm -rf cli/.**/*/__pycache__

cli_install: cli_clean cli_build
	pip3 install --force-reinstall cli/dist/nvidia_tao_client-*.whl

cli_uninstall:
	pip3 uninstall -y nvidia-tao-client
	pip3 uninstall -y -r cli/requirements-pip.txt
