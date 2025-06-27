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
MODEL ?=
ENV ?=
N_EPISODES ?=
MAX_STEPS ?=
BATCH_SIZE ?=
LEARNING_RATE ?=
GAMMA ?=
BUFFER_SIZE ?=
SEED ?=
HIDDEN_DIM ?=
N_LAYERS ?=
EMBEDDING_TYPE ?=

# rainbow_dqn
ALPHA ?=
BETA_START ?=
BETA_FRAMES ?=
TARGET_UPDATE_INTERVAL ?=
N_STEPS ?=
N_ATOMS ?=
V_MIN ?=
V_MAX ?=

# sac
TAU ?=
ENTROPY_COEF ?=
START_STEPS ?=

# ppo
PPO_EPOCHS ?=
CLIP_EPS ?=
N_TRANSACTIONS ?=
NORMALIZE_ADVANTAGES ?=

# td3
POLICY_DELAY ?=
POLICY_NOISE ?=
NOISE_CLIP ?=
EXPLORATION_NOISE ?=

# Default values for evaluation parameters
EVAL_RESULT_PATH ?=
EVAL_EPISODES ?=
EVAL_PERIOD ?=
EVAL ?=

# Default values for HPO parameters
HPO_N_TRIALS ?=
HPO_STUDY_NAME ?=
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
HPO_CONFIG ?= config/hpo_config.yaml

# Plot target parameters
PLOT_RESULT_DIR ?=

.PHONY: train docker-build docker-run docker-stop docker-remove train-all eval clean help install hpo hpo-basic hpo-ppo hpo-sac hpo-rainbow-dqn hpo-custom docker-run-eval docker-run-hpo train-yaml train-yaml-ppo train-yaml-sac train-yaml-rainbow-dqn train-yaml-discrete-sac train-yaml-td3 train-yaml-custom plot

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
		$(if $(ENV),-e ENV=$(ENV),) \
		$(if $(MODEL),-e MODEL=$(MODEL),) \
		$(if $(TRAIN_CONFIG),-e TRAIN_CONFIG=$(TRAIN_CONFIG),) \
		$(if $(N_EPISODES),-e N_EPISODES=$(N_EPISODES),) \
		$(if $(MAX_STEPS),-e MAX_STEPS=$(MAX_STEPS),) \
		$(if $(RESULTS_DIR),-e RESULTS_DIR=$(RESULTS_DIR),) \
		$(if $(LEARNING_RATE),-e LEARNING_RATE=$(LEARNING_RATE),) \
		$(if $(BATCH_SIZE),-e BATCH_SIZE=$(BATCH_SIZE),) \
		$(if $(GAMMA),-e GAMMA=$(GAMMA),) \
		$(if $(EVAL),-e EVAL=$(EVAL),) \
		$(if $(EVAL_PERIOD),-e EVAL_PERIOD=$(EVAL_PERIOD),) \
		$(if $(EVAL_EPISODES),-e EVAL_EPISODES=$(EVAL_EPISODES),) \
		$(if $(DEVICE),-e DEVICE=$(DEVICE),) \
		$(if $(BUFFER_SIZE),-e BUFFER_SIZE=$(BUFFER_SIZE),) \
		$(if $(SEED),-e SEED=$(SEED),) \
		$(if $(HIDDEN_DIM),-e HIDDEN_DIM=$(HIDDEN_DIM),) \
		$(if $(N_LAYERS),-e N_LAYERS=$(N_LAYERS),) \
		$(if $(ALPHA),-e ALPHA=$(ALPHA),) \
		$(if $(BETA_START),-e BETA_START=$(BETA_START),) \
		$(if $(BETA_FRAMES),-e BETA_FRAMES=$(BETA_FRAMES),) \
		$(if $(TARGET_UPDATE_INTERVAL),-e TARGET_UPDATE_INTERVAL=$(TARGET_UPDATE_INTERVAL),) \
		$(if $(N_STEPS),-e N_STEPS=$(N_STEPS),) \
		$(if $(N_ATOMS),-e N_ATOMS=$(N_ATOMS),) \
		$(if $(V_MIN),-e V_MIN=$(V_MIN),) \
		$(if $(V_MAX),-e V_MAX=$(V_MAX),) \
		$(if $(TAU),-e TAU=$(TAU),) \
		$(if $(ENTROPY_COEF),-e ENTROPY_COEF=$(ENTROPY_COEF),) \
		$(if $(START_STEPS),-e START_STEPS=$(START_STEPS),) \
		$(if $(PPO_EPOCHS),-e PPO_EPOCHS=$(PPO_EPOCHS),) \
		$(if $(CLIP_EPS),-e CLIP_EPS=$(CLIP_EPS),) \
		$(if $(N_TRANSACTIONS),-e N_TRANSACTIONS=$(N_TRANSACTIONS),) \
		$(if $(NORMALIZE_ADVANTAGES),-e NORMALIZE_ADVANTAGES=$(NORMALIZE_ADVANTAGES),) \
		$(if $(POLICY_DELAY),-e POLICY_DELAY=$(POLICY_DELAY),) \
		$(if $(POLICY_NOISE),-e POLICY_NOISE=$(POLICY_NOISE),) \
		$(if $(NOISE_CLIP),-e NOISE_CLIP=$(NOISE_CLIP),) \
		$(if $(EXPLORATION_NOISE),-e EXPLORATION_NOISE=$(EXPLORATION_NOISE),) \
		$(if $(EMBEDDING_TYPE),-e EMBEDDING_TYPE=$(EMBEDDING_TYPE),) \
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
		$(if $(ENV),--env $(ENV),) \
		$(if $(MODEL),--model $(MODEL),) \
		$(if $(TRAIN_CONFIG),--config $(TRAIN_CONFIG),) \
		$(if $(N_EPISODES),--episodes $(N_EPISODES),) \
		$(if $(MAX_STEPS),--max_steps $(MAX_STEPS),) \
		$(if $(RESULTS_DIR),--save_dir $(RESULTS_DIR),) \
		$(if $(LEARNING_RATE),--lr $(LEARNING_RATE),) \
		$(if $(BATCH_SIZE),--batch_size $(BATCH_SIZE),) \
		$(if $(GAMMA),--gamma $(GAMMA),) \
		$(if $(EVAL),--eval $(EVAL),) \
		$(if $(EVAL_PERIOD),--eval_period $(EVAL_PERIOD),) \
		$(if $(EVAL_EPISODES),--eval_episodes $(EVAL_EPISODES),) \
		$(if $(DEVICE),--device $(DEVICE),) \
		$(if $(BUFFER_SIZE),--buffer_size $(BUFFER_SIZE),) \
		$(if $(SEED),--seed $(SEED),) \
		$(if $(HIDDEN_DIM),--hidden_dim $(HIDDEN_DIM),) \
		$(if $(N_LAYERS),--n_layers $(N_LAYERS),) \
		$(if $(ALPHA),--alpha $(ALPHA),) \
		$(if $(BETA_START),--beta_start $(BETA_START),) \
		$(if $(BETA_FRAMES),--beta_frames $(BETA_FRAMES),) \
		$(if $(TARGET_UPDATE_INTERVAL),--target_update_interval $(TARGET_UPDATE_INTERVAL),) \
		$(if $(N_STEPS),--n_steps $(N_STEPS),) \
		$(if $(N_ATOMS),--n_atoms $(N_ATOMS),) \
		$(if $(V_MIN),--v_min $(V_MIN),) \
		$(if $(V_MAX),--v_max $(V_MAX),) \
		$(if $(TAU),--tau $(TAU),) \
		$(if $(ENTROPY_COEF),--entropy_coef $(ENTROPY_COEF),) \
		$(if $(START_STEPS),--start_steps $(START_STEPS),) \
		$(if $(PPO_EPOCHS),--ppo_epochs $(PPO_EPOCHS),) \
		$(if $(CLIP_EPS),--clip_eps $(CLIP_EPS),) \
		$(if $(N_TRANSACTIONS),--n_transactions $(N_TRANSACTIONS),) \
		$(if $(NORMALIZE_ADVANTAGES),--normalize_advantages $(NORMALIZE_ADVANTAGES),) \
		$(if $(POLICY_DELAY),--policy_delay $(POLICY_DELAY),) \
		$(if $(POLICY_NOISE),--policy_noise $(POLICY_NOISE),) \
		$(if $(NOISE_CLIP),--noise_clip $(NOISE_CLIP),) \
		$(if $(EXPLORATION_NOISE),--exploration_noise $(EXPLORATION_NOISE),) \
		$(if $(EMBEDDING_TYPE),--embedding_type $(EMBEDDING_TYPE),) \


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
		$(if $(MODEL),--model $(MODEL),) \
		$(if $(ENV),--env $(ENV),) \
		$(if $(DEVICE),--device $(DEVICE),) \
		$(if $(HPO_N_TRIALS),--hpo_n_trials $(HPO_N_TRIALS),) \
		$(if $(HPO_STUDY_NAME),--hpo_study_name $(HPO_STUDY_NAME),) \
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

# Plot target
plot:
	PYTHONPATH=./src $(PYTHON) $(SRC_DIR)/result_parser/plot.py $(PLOT_RESULT_DIR)