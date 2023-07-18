SHELL := /bin/bash

-include .env

helm_install: helm_uninstall helm_push
	@sleep 90; \
	rm -rf tao-toolkit-api-$(TAO_VERSION)-$(USER)-$(UUID_TAO_HELM).tgz; \
	rm -rf docker/TAO_API_REPOSITORY.txt docker/UUID_TAO_HELM.txt; \
	if ! helm status tao-toolkit-api >/dev/null 2>&1; \
	then \
		helm repo update && helm install tao-toolkit-api $(TAO_API_ORG)-$(TAO_API_TEAM)/tao-toolkit-api --version $(TAO_VERSION)-$(USER)-$(UUID_TAO_HELM) --namespace default ;\
	fi; \
	rm .env; \

helm_push:
	if ! helm repo list | grep -q $(TAO_API_ORG)-$(TAO_API_TEAM); then \
		helm repo add $(TAO_API_ORG)-$(TAO_API_TEAM) 'https://helm.ngc.nvidia.com/$(TAO_API_ORG)/$(TAO_API_TEAM)' --username='$$oauthtoken' --password='$(PASSWORD)'; \
	fi
	helm plugin install https://github.com/chartmuseum/helm-push ;\
	OUTFILE="/tmp/helm-push-$$(date "+%Y.%m.%d-%H.%M.%S").out" ;\
	cp chart/values.yaml chart/values.yaml.default ; \
	sed -i "s#repository: nvcr.io/nvidia/tao/tao-toolkit#repository: $(TAO_API_REGISTRY)/$(TAO_API_ORG)/$(TAO_API_TEAM)/tao-toolkit#" "chart/values.yaml"; \
	sed -i "s#tag: $(TAO_VERSION)-api#tag: $(USER)-$(UUID_TAO_HELM)#" "chart/values.yaml"; \
	HASH=$$(git rev-parse --short HEAD:./chart) ;\
	head chart/Chart.yaml ;\
	helm lint chart ;\
	helm template chart ;\
	helm package chart --version $(TAO_VERSION)-$(USER)-$(UUID_TAO_HELM) --app-version $(TAO_VERSION) ;\
	cp chart/values.yaml.default chart/values.yaml ;\
	rm -rf chart/values.yaml.default ;\
	if ! helm cm-push tao-toolkit-api-$(TAO_VERSION)-$(USER)-$(UUID_TAO_HELM).tgz $(TAO_API_ORG)-$(TAO_API_TEAM) >& $$OUTFILE ;\
	then \
		echo "Push with hash success!" ;\
		cat "$$OUTFILE" ;\
	else \
		cat "$$OUTFILE" ;\
		if ! grep -q "Error: 409: $(TAO_API_ORG)/$(TAO_API_TEAM)/tao-toolkit-api-$(TAO_VERSION)-$(USER)-$(UUID_TAO_HELM).tgz already exists" $$OUTFILE ;\
		then \
			echo "Seems there is no latest update in helm! Skipping.." ;\
			exit 0 ;\
		else \
			echo "Error while pushing helm.." ;\
			exit 1 ;\
		fi ;\
	fi ;\

helm_uninstall:
	@if helm status tao-toolkit-api >/dev/null 2>&1; \
	then \
		helm delete tao-toolkit-api -n default; \
	fi
	@sleep 60

docker_build:
	@read -p "Enter the docker registry value [nvcr.io]: " TAO_API_REGISTRY && echo "TAO_API_REGISTRY=$$TAO_API_REGISTRY" > .env; \
	read -p "Enter the docker org value [nvidia]: " TAO_API_ORG && echo "TAO_API_ORG=$$TAO_API_ORG" >> .env; \
	read -p "Enter the docker team value [tao]: " TAO_API_TEAM && echo "TAO_API_TEAM=$$TAO_API_TEAM" >> .env; \
	echo "TAO_VERSION=5.0.0" >> .env; \
	stty -echo && read -p "Enter your ngc_api_key [from ~/.ngc/config]: " PASSWORD && echo "PASSWORD=$$PASSWORD" >> .env; \
	[ ! -z "$$TAO_API_REGISTRY" ] && BUILD_ARGS="--tao_api_registry=$$TAO_API_REGISTRY"; \
	[ ! -z "$$TAO_API_ORG" ] && BUILD_ARGS="$$BUILD_ARGS --tao_api_org=$$TAO_API_ORG"; \
	[ ! -z "$$TAO_API_TEAM" ] && BUILD_ARGS="$$BUILD_ARGS --tao_api_team=$$TAO_API_TEAM"; \
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
	pip3 uninstall -y -r requirements-pip.txt
