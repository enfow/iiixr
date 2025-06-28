DOCKER_IMAGE_NAME := iiixr-2-dashboard
DOCKER_CONTAINER_NAME := iiixr-2-dashboard-container

dev:
	npm run dev

docker-build:
	docker build -t $(DOCKER_IMAGE_NAME) -f docker/Dockerfile .

docker-run:
	docker run -d \
		--name $(DOCKER_CONTAINER_NAME) \
		-p 3000:3000 \
		$(DOCKER_IMAGE_NAME)

docker-stop:
	docker stop $(DOCKER_CONTAINER_NAME)

docker-remove:
	docker rm $(DOCKER_CONTAINER_NAME)
