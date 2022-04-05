SHELL = /bin/sh

# Project variables
image_name := order-estimator
container_name = $(image_name)-container

cwd := $(shell pwd)
user := $(shell whoami)
uid := $(shell id -u)
gid := $(shell id -g)
user_home := /home/$(user)
container_app_dir := $(user_home)/app
ports := 8080-8080

# Directories & files
# Replace - with _ for Python package name.
package_dir := $(subst -,_,$(image_name))
scripts_dir := scripts
tests_dir := tests
artifacts_dir := artifacts

is_container_running := $(shell docker container list | grep $(container_name))

# Container mounts
mounts := -v $(cwd)/$(artifacts_dir):$(container_app_dir)/$(artifacts_dir)

# Build container
docker_build_options := -t $(image_name) --build-arg USER=$(user) \
	--build-arg UID=$(uid) --build-arg GID=$(gid) \
	--build-arg APPDIR=$(container_app_dir)

# Run options
docker_run_options := --rm -d -t --name $(container_name) \
	-h $(container_name) $(mounts) -p $(ports):$(ports) \
	-u $(uid):$(gid)


# * Targets

## Build container and install poetry
.PHONY: all
all: build-image poetry.lock


## Build docker image.
build-image:
	docker build $(docker_build_options) .
	@echo "\nEnvironment is built! A Docker image was created: $(image_name)"


## Build without cache
build-image-nochache:
	@docker build $(docker_build_options) --no-cache .
	@echo "\nEnvironment is built! A Docker image was created: $(image_name)"


## Install poetry locally
poetry.lock:
	@poetry install

.PHONY: poetry-install
poetry-install: poetry.lock
	@echo "Done"


## Start container
container-start:
	@docker run $(docker_run_options) $(image_name)


## Stop running container
container-stop:
	@echo 'Stopping container.'
	@docker container stop $(container_name)


## Force remove container
container-rm:
	@echo 'Stopping container.'
	@docker container rm $(container_name) --force


## Enter interactive container shell
container-shell:
ifeq ($(is_container_running),)
	@echo 'Starting container.'
	@docker run $(docker_run_options) $(image_name)
endif
	@docker exec -it $(container_name) bash



## Start API
.PHONY: run-api
run-api:
	@poetry run python $(scripts_dir)/run_api.py


## Run linter
.PHONY: lint
lint:
	@poetry run flake8 $(package_dir)/


## Run tests
.PHONY: tests
tests:
	@poetry run pytest $(tests_dir)/
