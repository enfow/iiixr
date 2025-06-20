DEVICE ?= cpu
DOCKER_FILE := docker/Dockerfile
DOCKER_IMAGE_NAME := iiixr-2
DOCKER_CONTAINER_NAME := iiixr-2-container

ifeq ($(DEVICE),cuda)
	DOCKER_FILE	:= docker/Dockerfile.gpu
	DOCKER_IMAGE_NAME := $(DOCKER_IMAGE_NAME)-gpu
	DOCKER_CONTAINER_NAME := iiixr-2-gpu-container
endif

# Environment variables
PYTHON := python
SRC_DIR := src
RESULTS_DIR := results

# Default values for training parameters
MODEL ?= ppo  # ppo, sac, rainbow_dqn, discrete_sac
ENV ?= LunarLander-v3  # LunarLander-v3, BipedalWalker-v3
N_EPISODES ?= 1000
MAX_STEPS ?= 1000
BATCH_SIZE ?= 64
LEARNING_RATE ?= 0.001
GAMMA ?= 0.99
BUFFER_SIZE ?= 1000000
SEED ?= 42

# rainbow_dqn
ALPHA ?= 0.6
BETA_START ?= 0.4
BETA_FRAMES ?= 100000
TARGET_UPDATE ?= 10

# sac
TAU ?= 0.005
ENTROPY_COEF ?= 1.0
START_STEPS ?= 1000

# ppo
PPO_EPOCHS ?= 10
CLIP_EPS ?= 0.2

# Default values for evaluation parameters
EVAL_RESULT_PATH ?= results/ppo
EVAL_EPISODES ?= 10
EVAL_PERIOD ?= 10
EVAL ?= false

.PHONY: train docker-build docker-run docker-stop docker-remove train-all eval clean help install

format:
	ruff format src
	isort src

docker-build:
	docker build -t $(DOCKER_IMAGE_NAME) -f $(DOCKER_FILE) .
	
docker-run:
	docker run -it --rm \
		$(if $(filter cuda,$(DEVICE)),--gpus all,) \
		--name $(DOCKER_CONTAINER_NAME) \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/results:/app/results \
		-e MODEL=$(MODEL) \
		-e ENV=$(ENV) \
		-e N_EPISODES=$(N_EPISODES) \
		-e MAX_STEPS=$(MAX_STEPS) \
		-e BATCH_SIZE=$(BATCH_SIZE) \
		-e LEARNING_RATE=$(LEARNING_RATE) \
		-e GAMMA=$(GAMMA) \
		-e EVAL=$(EVAL) \
		-e EVAL_PERIOD=$(EVAL_PERIOD) \
		-e EVAL_EPISODES=$(EVAL_EPISODES) \
		-e DEVICE=$(DEVICE) \
		-e BUFFER_SIZE=$(BUFFER_SIZE) \
		-e SEED=$(SEED) \
		-e ALPHA=$(ALPHA) \
		-e BETA_START=$(BETA_START) \
		-e BETA_FRAMES=$(BETA_FRAMES) \
		-e TARGET_UPDATE=$(TARGET_UPDATE) \
		-e TAU=$(TAU) \
		-e ENTROPY_COEF=$(ENTROPY_COEF) \
		-e START_STEPS=$(START_STEPS) \
		-e PPO_EPOCHS=$(PPO_EPOCHS) \
		-e CLIP_EPS=$(CLIP_EPS) \
		$(DOCKER_IMAGE_NAME)

docker-run-eval:
	docker run -it --rm \
		--name $(DOCKER_CONTAINER_NAME) \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/results:/app/results \
		-e EVAL_RESULT_PATH=$(EVAL_RESULT_PATH) \
		-e EVAL_EPISODES=$(EVAL_EPISODES) \
		-e DEVICE=$(DEVICE) \
		$(DOCKER_IMAGE_NAME) make eval

docker-stop:
	docker stop $(DOCKER_CONTAINER_NAME)

docker-remove:
	docker rm $(DOCKER_CONTAINER_NAME)

# Training targets
train:
	$(PYTHON) $(SRC_DIR)/train.py \
		--model $(MODEL) \
		--env $(ENV) \
		--episodes $(N_EPISODES) \
		--max_steps $(MAX_STEPS) \
		--save_dir $(RESULTS_DIR) \
		--lr $(LEARNING_RATE) \
		--batch_size $(BATCH_SIZE) \
		--gamma $(GAMMA) \
		--eval $(EVAL) \
		--eval_period $(EVAL_PERIOD) \
		--eval_episodes $(EVAL_EPISODES) \
		--device $(DEVICE) \
		--buffer_size $(BUFFER_SIZE) \
		--seed $(SEED) \
		--alpha $(ALPHA) \
		--beta_start $(BETA_START) \
		--beta_frames $(BETA_FRAMES) \
		--target_update $(TARGET_UPDATE) \
		--tau $(TAU) \
		--entropy_coef $(ENTROPY_COEF) \
		--start_steps $(START_STEPS) \
		--ppo_epochs $(PPO_EPOCHS) \
		--clip_eps $(CLIP_EPS)

# Train all algorithms sequentially
train-all:
	make train MODEL=ppo
	make train MODEL=sac
	make train MODEL=rainbow_dqn
	make train MODEL=discrete_sac

# Evaluation targets
eval:
	$(PYTHON) $(SRC_DIR)/eval.py \
		--result_path $(EVAL_RESULT_PATH) \
		--episodes $(EVAL_EPISODES)

clean:
	rm -rf models/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

help:
	@echo "Available targets:"
	@echo "  install        Install dependencies"
	@echo "  clean          Clean results directory"
	@echo "  train          Train a model (default: PPO)"
	@echo "  train-all      Train all agents sequentially"
	@echo "  eval           Evaluate a trained model"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-run     Run training in Docker container"
	@echo "  docker-run-eval Run evaluation in Docker container"
	@echo ""
	@echo "Training example usage:"
	@echo "  make train MODEL=ppo ENV=LunarLander-v2 N_EPISODES=1000"
	@echo "  make train MODEL=sac ENV=Pendulum-v1 LEARNING_RATE=0.0003"
	@echo "  make train MODEL=rainbow_dqn ENV=CartPole-v1"
	@echo ""
	@echo "Docker example usage:"
	@echo "  make docker-run MODEL=ppo ENV=LunarLander-v2"
	@echo "  make docker-run-eval MODEL=ppo EVAL_RESULT_PATH=results/ppo"
	@echo ""
	@echo "Evaluation example usage:"
	@echo "  make eval EVAL_RESULT_PATH=results/ppo EVAL_EPISODES=20"
