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
BETA_FRAMES ?= 100000
TARGET_UPDATE ?= 10

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
		-e TARGET_UPDATE=$(TARGET_UPDATE) \
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
		--target_update $(TARGET_UPDATE) \
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
		$(if $(HPO_N_TRANSACTIONS),--n_transactions $(HPO_N_TRANSACTIONS),) \
		$(if $(HPO_NORMALIZE_ADVANTAGES),--normalize_advantages $(HPO_NORMALIZE_ADVANTAGES),) \
		$(if $(HPO_POLICY_DELAY),--policy_delay $(HPO_POLICY_DELAY),) \
		$(if $(HPO_POLICY_NOISE),--policy_noise $(HPO_POLICY_NOISE),) \
		$(if $(HPO_NOISE_CLIP),--noise_clip $(HPO_NOISE_CLIP),) \
		$(if $(HPO_EXPLORATION_NOISE),--exploration_noise $(HPO_EXPLORATION_NOISE),)

# Specific HPO targets
hpo-basic:
	make hpo HPO_N_TRIALS=10 HPO_STUDY_NAME=basic_hpo

hpo-ppo:
	make hpo MODEL=ppo HPO_N_TRIALS=20 HPO_STUDY_NAME=ppo_optimization \
		HPO_ENTROPY_COEF='0.01,0.1,0.5,1.0' \
		HPO_CLIP_EPS='0.1,0.15,0.2' \
		HPO_BATCH_SIZE='64,128,256' \
		HPO_GAMMA='0.95,0.97,0.99' \
		HPO_N_TRANSACTIONS='500,1000,2000' \
		HPO_NORMALIZE_ADVANTAGES='true,false'

hpo-sac:
	make hpo MODEL=sac HPO_N_TRIALS=20 HPO_STUDY_NAME=sac_optimization \
		HPO_START_STEPS='500,1000,2000' \
		HPO_ENTROPY_COEF='0.01,0.1,0.5,1.0' \
		HPO_BATCH_SIZE='64,128,256' \
		HPO_GAMMA='0.95,0.97,0.99'

hpo-rainbow-dqn:
	make hpo MODEL=rainbow_dqn HPO_N_TRIALS=20 HPO_STUDY_NAME=rainbow_dqn_optimization \
		HPO_TARGET_UPDATE='5,10,20,50' \
		HPO_BATCH_SIZE='64,128,256' \
		HPO_GAMMA='0.95,0.97,0.99'

hpo-custom:
	make hpo HPO_N_TRIALS=50 HPO_STUDY_NAME=custom_hpo \
		HPO_BATCH_SIZE='64,128,256,512' \
		HPO_GAMMA='0.95,0.97,0.99,0.995' \
		HPO_HIDDEN_DIM='128,256,512' \
		HPO_N_LAYERS='2,3,4' \
		HPO_PPO_EPOCHS='4,6,8,10' \
		HPO_CLIP_EPS='0.1,0.15,0.2,0.25' \
		HPO_START_STEPS='500,1000,2000' \
		HPO_ENTROPY_COEF='0.01,0.1,0.5,1.0' \
		HPO_TARGET_UPDATE='5,10,20,50' \
		HPO_N_TRANSACTIONS='500,1000,2000' \
		HPO_NORMALIZE_ADVANTAGES='true,false'

clean:
	rm -rf models/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

help:
	@echo "Available targets:"
	@echo "  install        Install dependencies"
	@echo "  clean          Clean results directory"
	@echo "  train          Train a model (default: PPO) - CLI arguments"
	@echo "  train-all      Train all agents sequentially - CLI arguments"
	@echo "  train-yaml     Train a model using YAML config file"
	@echo "  train-yaml-ppo Train PPO using YAML config file"
	@echo "  train-yaml-sac Train SAC using YAML config file"
	@echo "  train-yaml-rainbow-dqn Train Rainbow DQN using YAML config file"
	@echo "  train-yaml-discrete-sac Train Discrete SAC using YAML config file"
	@echo "  train-yaml-td3 Train TD3 using YAML config file"
	@echo "  train-yaml-all Train all agents sequentially using YAML config"
	@echo "  eval           Evaluate a trained model"
	@echo "  hpo            Run HPO with custom parameters"
	@echo "  hpo-basic      Run basic HPO with default config"
	@echo "  hpo-ppo        Run PPO-specific HPO optimization"
	@echo "  hpo-sac        Run SAC-specific HPO optimization"
	@echo "  hpo-rainbow-dqn Run Rainbow DQN-specific HPO optimization"
	@echo "  hpo-custom     Run custom HPO with specific search spaces"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-run     Run training in Docker container"
	@echo "  docker-run-eval Run evaluation in Docker container"
	@echo "  docker-run-hpo Run HPO in Docker container"
	@echo ""
	@echo "YAML Configuration Training:"
	@echo "  make train-yaml                    # Use default config/train_config.yaml"
	@echo "  make train-yaml TRAIN_CONFIG=config/my_config.yaml"
	@echo "  make train-yaml-ppo               # Train PPO with YAML config"
	@echo "  make train-yaml-sac               # Train SAC with YAML config"
	@echo "  make train-yaml-rainbow-dqn       # Train Rainbow DQN with YAML config"
	@echo "  make train-yaml-discrete-sac      # Train Discrete SAC with YAML config"
	@echo "  make train-yaml-td3               # Train TD3 with YAML config"
	@echo "  make train-yaml-all               # Train all models with YAML config"
	@echo ""
	@echo "CLI Training example usage:"
	@echo "  make train MODEL=ppo ENV=LunarLander-v3 N_EPISODES=1000"
	@echo "  make train MODEL=sac ENV=Pendulum-v1 LEARNING_RATE=0.0003"
	@echo "  make train MODEL=rainbow_dqn ENV=CartPole-v1"
	@echo ""
	@echo "YAML Training example usage:"
	@echo "  make train-yaml                    # Use default config"
	@echo "  make train-yaml-ppo               # Train PPO with YAML"
	@echo "  make train-yaml-sac ENV=BipedalWalker-v3  # Override environment"
	@echo "  make train-yaml-rainbow-dqn --episodes 2000  # Override episodes"
	@echo "  python src/train.py --config config/train_config.yaml --seed 42  # CLI overrides YAML"
	@echo ""
	@echo "HPO example usage:"
	@echo "  make hpo HPO_STUDY_NAME=my_experiment HPO_N_TRIALS=100"
	@echo "  make hpo ENV=CartPole-v1 HPO_BATCH_SIZE='64,128,256' HPO_HIDDEN_DIM='128,256,512'"
	@echo "  make hpo MODEL=sac HPO_START_STEPS='500,1000,2000' HPO_ENTROPY_COEF='0.01,0.1,0.5,1.0'"
	@echo "  make hpo MODEL=rainbow_dqn HPO_TARGET_UPDATE='5,10,20,50'"
	@echo "  make hpo MODEL=ppo HPO_ENTROPY_COEF='0.01,0.1,0.5,1.0' HPO_CLIP_EPS='0.1,0.15,0.2'"
	@echo "  make hpo MODEL=ppo HPO_N_TRANSACTIONS='500,1000,2000' HPO_NORMALIZE_ADVANTAGES='true,false'"
	@echo "  make hpo-ppo ENV=LunarLander-v3"
	@echo "  make hpo-custom HPO_N_TRIALS=200"
	@echo ""
	@echo "Docker example usage:"
	@echo "  make docker-run MODEL=ppo ENV=LunarLander-v3"
	@echo "  make docker-run-eval MODEL=ppo EVAL_RESULT_PATH=results/ppo"
	@echo "  make docker-run-hpo MODEL=ppo ENV=LunarLander-v3 HPO_N_TRIALS=100"
	@echo ""
	@echo "Evaluation example usage:"
	@echo "  make eval EVAL_RESULT_PATH=results/ppo EVAL_EPISODES=20"
