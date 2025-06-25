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
MODEL ?= ppo  # ppo, sac, rainbow_dqn, discrete_sac, sac_v2, td3
ENV ?= LunarLander-v3  # LunarLander-v3, BipedalWalkerHardcore-v3
N_EPISODES ?= 1000
MAX_STEPS ?= 1000
BATCH_SIZE ?= 64
LEARNING_RATE ?= 0.001
GAMMA ?= 0.99
BUFFER_SIZE ?= 1000000
SEED ?= 42
HIDDEN_DIM ?= 256
N_LAYERS ?= 3

# rainbow_dqn
ALPHA ?= 0.6
BETA_START ?= 0.4
BETA_FRAMES ?= 1000000
TARGET_UPDATE_INTERVAL ?= 32000
N_STEPS ?= 3
N_ATOMS ?= 51
V_MIN ?= -10.0
V_MAX ?= 10.0

# sac
TAU ?= 0.005
ENTROPY_COEF ?= 1.0
START_STEPS ?= 10000

# ppo
PPO_EPOCHS ?= 10
CLIP_EPS ?= 0.2
N_TRANSACTIONS ?= 1000
NORMALIZE_ADVANTAGES ?= false

# td3
POLICY_DELAY ?= 2
POLICY_NOISE ?= 0.2
NOISE_CLIP ?= 0.5
EXPLORATION_NOISE ?= 0.1

# Default values for evaluation parameters
EVAL_RESULT_PATH ?= results/ppo
EVAL_EPISODES ?= 10
EVAL_PERIOD ?= 10
EVAL ?= false

# Default values for HPO parameters
HPO_N_TRIALS ?= 10
HPO_STUDY_NAME ?= hpo_experiment
HPO_CONFIG ?= src/hpo_config.yaml
HPO_TIMEOUT ?= 

# HPO searchable hyperparameters (comma-separated values)
HPO_BATCH_SIZE ?= 
HPO_GAMMA ?= 
HPO_HIDDEN_DIM ?= 
HPO_N_LAYERS ?= 
HPO_PPO_EPOCHS ?= 
HPO_CLIP_EPS ?= 
HPO_START_STEPS ?= 
HPO_ENTROPY_COEF ?= 
HPO_TARGET_UPDATE ?=
HPO_TARGET_UPDATE_INTERVAL ?=
HPO_N_STEPS ?=
HPO_N_ATOMS ?=
HPO_V_MIN ?=
HPO_V_MAX ?=
HPO_N_TRANSACTIONS ?= 
HPO_NORMALIZE_ADVANTAGES ?=

# TD3 specific parameters
HPO_POLICY_DELAY ?=
HPO_POLICY_NOISE ?=
HPO_NOISE_CLIP ?= 
HPO_EXPLORATION_NOISE ?= 

# YAML configuration
TRAIN_CONFIG ?= config/train_config.yaml

.PHONY: train docker-build docker-run docker-stop docker-remove train-all eval clean help install hpo hpo-basic hpo-ppo hpo-sac hpo-rainbow-dqn hpo-custom docker-run-eval docker-run-hpo train-yaml train-yaml-ppo train-yaml-sac train-yaml-rainbow-dqn train-yaml-discrete-sac train-yaml-td3 train-yaml-custom

format:
	ruff format src
	isort src

docker-build:
	docker build -t $(DOCKER_IMAGE_NAME) -f $(DOCKER_FILE) .
	
docker-run:
	docker run -it --rm \
		$(if $(filter cuda,$(DEVICE)),--gpus all,) \
		--name $(DOCKER_CONTAINER_NAME)-$(MODEL)-$(ENV) \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/results:/app/results \
		-v $(PWD)/config:/app/config \
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
		-e HIDDEN_DIM=$(HIDDEN_DIM) \
		-e N_LAYERS=$(N_LAYERS) \
		-e ALPHA=$(ALPHA) \
		-e BETA_START=$(BETA_START) \
		-e BETA_FRAMES=$(BETA_FRAMES) \
		-e TARGET_UPDATE_INTERVAL=$(TARGET_UPDATE_INTERVAL) \
		-e N_STEPS=$(N_STEPS) \
		-e N_ATOMS=$(N_ATOMS) \
		-e V_MIN=$(V_MIN) \
		-e V_MAX=$(V_MAX) \
		-e TAU=$(TAU) \
		-e ENTROPY_COEF=$(ENTROPY_COEF) \
		-e START_STEPS=$(START_STEPS) \
		-e PPO_EPOCHS=$(PPO_EPOCHS) \
		-e CLIP_EPS=$(CLIP_EPS) \
		-e N_TRANSACTIONS=$(N_TRANSACTIONS) \
		-e NORMALIZE_ADVANTAGES=$(NORMALIZE_ADVANTAGES) \
		-e POLICY_DELAY=$(POLICY_DELAY) \
		-e POLICY_NOISE=$(POLICY_NOISE) \
		-e NOISE_CLIP=$(NOISE_CLIP) \
		-e EXPLORATION_NOISE=$(EXPLORATION_NOISE) \
		$(DOCKER_IMAGE_NAME)

docker-run-eval:
	docker run -it --rm \
		--name $(DOCKER_CONTAINER_NAME) \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/results:/app/results \
		-v $(PWD)/config:/app/config \
		-e EVAL_RESULT_PATH=$(EVAL_RESULT_PATH) \
		-e EVAL_EPISODES=$(EVAL_EPOCHS) \
		-e DEVICE=$(DEVICE) \
		$(DOCKER_IMAGE_NAME) make eval

docker-run-hpo:
	docker run -it --rm \
		$(if $(filter cuda,$(DEVICE)),--gpus all,) \
		--name $(DOCKER_CONTAINER_NAME)-hpo-$(MODEL)-$(ENV) \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/results:/app/results \
		-v $(PWD)/config:/app/config \
		-e MODEL=$(MODEL) \
		-e ENV=$(ENV) \
		-e N_EPISODES=$(N_EPISODES) \
		-e MAX_STEPS=$(MAX_STEPS) \
		-e DEVICE=$(DEVICE) \
		-e HPO_N_TRIALS=$(HPO_N_TRIALS) \
		-e HPO_STUDY_NAME=$(HPO_STUDY_NAME) \
		-e HPO_CONFIG=$(HPO_CONFIG) \
		-e HPO_TIMEOUT=$(HPO_TIMEOUT) \
		-e HPO_BATCH_SIZE=$(HPO_BATCH_SIZE) \
		-e HPO_GAMMA=$(HPO_GAMMA) \
		-e HPO_HIDDEN_DIM=$(HPO_HIDDEN_DIM) \
		-e HPO_N_LAYERS=$(HPO_N_LAYERS) \
		-e HPO_PPO_EPOCHS=$(HPO_PPO_EPOCHS) \
		-e HPO_CLIP_EPS=$(HPO_CLIP_EPS) \
		-e HPO_START_STEPS=$(HPO_START_STEPS) \
		-e HPO_ENTROPY_COEF=$(HPO_ENTROPY_COEF) \
		-e HPO_TARGET_UPDATE=$(HPO_TARGET_UPDATE) \
		-e HPO_TARGET_UPDATE_INTERVAL=$(HPO_TARGET_UPDATE_INTERVAL) \
		-e HPO_N_STEPS=$(HPO_N_STEPS) \
		-e HPO_N_ATOMS=$(HPO_N_ATOMS) \
		-e HPO_V_MIN=$(HPO_V_MIN) \
		-e HPO_V_MAX=$(HPO_V_MAX) \
		-e HPO_N_TRANSACTIONS=$(HPO_N_TRANSACTIONS) \
		-e HPO_NORMALIZE_ADVANTAGES=$(HPO_NORMALIZE_ADVANTAGES) \
		$(DOCKER_IMAGE_NAME) make hpo

docker-stop:
	docker stop $(DOCKER_CONTAINER_NAME)

docker-remove:
	docker rm $(DOCKER_CONTAINER_NAME)

# Training targets
train:
	$(PYTHON) $(SRC_DIR)/train.py \
		--model $(MODEL) \
		--env $(ENV) \
		--config $(TRAIN_CONFIG) \
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
		--hidden_dim $(HIDDEN_DIM) \
		--n_layers $(N_LAYERS) \
		--alpha $(ALPHA) \
		--beta_start $(BETA_START) \
		--beta_frames $(BETA_FRAMES) \
		--target_update_interval $(TARGET_UPDATE_INTERVAL) \
		--n_steps $(N_STEPS) \
		--n_atoms $(N_ATOMS) \
		--v_min $(V_MIN) \
		--v_max $(V_MAX) \
		--tau $(TAU) \
		--entropy_coef $(ENTROPY_COEF) \
		--start_steps $(START_STEPS) \
		--ppo_epochs $(PPO_EPOCHS) \
		--clip_eps $(CLIP_EPS) \
		--n_transactions $(N_TRANSACTIONS) \
		--normalize_advantages $(NORMALIZE_ADVANTAGES) \
		--policy_delay $(POLICY_DELAY) \
		--policy_noise $(POLICY_NOISE) \
		--noise_clip $(NOISE_CLIP) \
		--exploration_noise $(EXPLORATION_NOISE)


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
		--episodes $(EVAL_EPOCHS)

# HPO targets
hpo:
	$(PYTHON) $(SRC_DIR)/run_hpo.py \
		--config $(HPO_CONFIG) \
		--model $(MODEL) \
		--env $(ENV) \
		--device $(DEVICE) \
		--hpo_n_trials $(HPO_N_TRIALS) \
		--hpo_study_name $(HPO_STUDY_NAME) \
		$(if $(N_EPISODES),--episodes $(N_EPISODES),) \
		$(if $(MAX_STEPS),--max_steps $(MAX_STEPS),) \
		$(if $(HPO_TIMEOUT),--timeout $(HPO_TIMEOUT),) \
		$(if $(HPO_BATCH_SIZE),--batch_size $(HPO_BATCH_SIZE),) \
		$(if $(HPO_GAMMA),--gamma $(HPO_GAMMA),) \
		$(if $(HPO_HIDDEN_DIM),--hidden_dim $(HPO_HIDDEN_DIM),) \
		$(if $(HPO_N_LAYERS),--n_layers $(HPO_N_LAYERS),) \
		$(if $(HPO_PPO_EPOCHS),--ppo_epochs $(HPO_PPO_EPOCHS),) \
		$(if $(HPO_CLIP_EPS),--clip_eps $(HPO_CLIP_EPS),) \
		$(if $(HPO_START_STEPS),--start_steps $(HPO_START_STEPS),) \
		$(if $(HPO_ENTROPY_COEF),--entropy_coef $(HPO_ENTROPY_COEF),) \
		$(if $(HPO_TARGET_UPDATE),--target_update $(HPO_TARGET_UPDATE),) \
		$(if $(HPO_TARGET_UPDATE_INTERVAL),--target_update_interval $(HPO_TARGET_UPDATE_INTERVAL),) \
		$(if $(HPO_N_STEPS),--n_steps $(HPO_N_STEPS),) \
		$(if $(HPO_N_ATOMS),--n_atoms $(HPO_N_ATOMS),) \
		$(if $(HPO_V_MIN),--v_min $(HPO_V_MIN),) \
		$(if $(HPO_V_MAX),--v_max $(HPO_V_MAX),) \
		$(if $(HPO_N_TRANSACTIONS),--n_transactions $(HPO_N_TRANSACTIONS),) \
		$(if $(HPO_NORMALIZE_ADVANTAGES),--normalize_advantages $(HPO_NORMALIZE_ADVANTAGES),) \
		$(if $(HPO_POLICY_DELAY),--policy_delay $(HPO_POLICY_DELAY),) \
		$(if $(HPO_POLICY_NOISE),--policy_noise $(HPO_POLICY_NOISE),) \
		$(if $(HPO_NOISE_CLIP),--noise_clip $(HPO_NOISE_CLIP),) \
		$(if $(HPO_EXPLORATION_NOISE),--exploration_noise $(HPO_EXPLORATION_NOISE),)