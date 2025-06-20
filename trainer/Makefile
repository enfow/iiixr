DOCKER_IMAGE_NAME := iiixr-2-trainer
DOCKER_CONTAINER_NAME := iiixr-2-container

# Environment variables
PYTHON := python
SRC_DIR := src
RESULTS_DIR := results

# Default values for training parameters
MODEL := ppo  # ppo, sac, rainbow_dqn, discrete_sac
ENV := LunarLander-v3  # LunarLander-v3, BipedalWalker-v3
N_EPISODES := 1000
MAX_STEPS := 1000
BATCH_SIZE := 64
LEARNING_RATE := 0.001
GAMMA := 0.99
LOG_INTERVAL := 10

# Algorithm-specific default parameters
# PPO
CLIP_RATIO := 0.2
GAE_LAMBDA := 0.95
POLICY_EPOCHS := 10

# SAC
ALPHA := 0.2
TARGET_UPDATE_INTERVAL := 1
AUTO_ENTROPY := true

# Rainbow DQN
N_ATOMS := 51
V_MIN := -10.0
V_MAX := 10.0
N_STEP := 3
MEMORY_SIZE := 100000

# Default values for evaluation parameters
EVAL_RESULTS_PATH := results/ppo
EVAL_EPISODES := 10
EVAL_PERIOD := 10
EVAL := false

.PHONY: train docker-build docker-run docker-stop docker-remove train-ppo train-sac train-rainbow-dqn run-server clean compose-up compose-down compose-logs help install eval-ppo eval-sac eval-rainbow eval-discrete-sac

format:
	ruff format src
	isort src

docker-build:
	docker build -t $(DOCKER_IMAGE_NAME) -f docker/Dockerfile .

docker-run:
	docker run -it --rm \
		--name $(DOCKER_CONTAINER_NAME) \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/models:/app/models \
		$(DOCKER_IMAGE_NAME)

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
		--eval_episodes $(EVAL_EPISODES)

# Train all algorithms sequentially
train-all:
	make train MODEL=ppo
	make train MODEL=sac
	make train MODEL=rainbow_dqn
	make train MODEL=discrete_sac

# Evaluation targets
eval:
	$(PYTHON) $(SRC_DIR)/eval.py \
		--results_path $(EVAL_RESULTS_PATH) \
		--episodes $(EVAL_EPISODES)

# Clean up generated files
clean:
	rm -rf models/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete