DOCKER_IMAGE_NAME := iiixr-2-trainer
DOCKER_CONTAINER_NAME := iiixr-2-container

# Environment variables
PYTHON := python
SRC_DIR := src
RESULTS_DIR := results

# Default values for training parameters
ENV := LunarLander-v3
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

.PHONY: train docker-build docker-run docker-stop docker-remove train-ppo train-sac train-rainbow-dqn run-server clean compose-up compose-down compose-logs help install

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

help:
	@echo "Available targets:"
	@echo "  install        Install dependencies"
	@echo "  clean          Clean results directory"
	@echo "  train-ppo      Train PPO agent"
	@echo "  train-sac      Train SAC agent"
	@echo "  train-rainbow  Train Rainbow DQN agent"
	@echo "  train-all      Train all agents sequentially"
	@echo ""
	@echo "Example usage:"
	@echo "  make train-ppo ENV=LunarLander-v2 N_EPISODES=1000"
	@echo "  make train-sac ENV=Pendulum-v1 LEARNING_RATE=0.0003"
	@echo "  make train-rainbow ENV=CartPole-v1 MEMORY_SIZE=50000"

install:
	pip install -r requirements.txt

clean:
	rm -rf $(RESULTS_DIR)/*

# Training targets
train-ppo:
	$(PYTHON) $(SRC_DIR)/train.py \
		--algorithm ppo \
		--env $(ENV) \
		--n_episodes $(N_EPISODES) \
		--max_steps $(MAX_STEPS) \
		--results_dir $(RESULTS_DIR) \
		--learning_rate $(LEARNING_RATE) \
		--batch_size $(BATCH_SIZE) \
		--gamma $(GAMMA) \
		--log_interval $(LOG_INTERVAL) \
		--clip_ratio $(CLIP_RATIO) \
		--gae_lambda $(GAE_LAMBDA) \
		--policy_epochs $(POLICY_EPOCHS)

train-sac:
	$(PYTHON) $(SRC_DIR)/train.py \
		--algorithm sac \
		--env $(ENV) \
		--n_episodes $(N_EPISODES) \
		--max_steps $(MAX_STEPS) \
		--results_dir $(RESULTS_DIR) \
		--learning_rate $(LEARNING_RATE) \
		--batch_size $(BATCH_SIZE) \
		--gamma $(GAMMA) \
		--log_interval $(LOG_INTERVAL) \
		--alpha $(ALPHA) \
		--target_update_interval $(TARGET_UPDATE_INTERVAL) \
		--automatic_entropy_tuning $(AUTO_ENTROPY)

train-rainbow:
	$(PYTHON) $(SRC_DIR)/train.py \
		--algorithm rainbow_dqn \
		--env $(ENV) \
		--n_episodes $(N_EPISODES) \
		--max_steps $(MAX_STEPS) \
		--results_dir $(RESULTS_DIR) \
		--learning_rate $(LEARNING_RATE) \
		--batch_size $(BATCH_SIZE) \
		--gamma $(GAMMA) \
		--log_interval $(LOG_INTERVAL) \
		--n_atoms $(N_ATOMS) \
		--v_min $(V_MIN) \
		--v_max $(V_MAX) \
		--n_step $(N_STEP) \
		--memory_size $(MEMORY_SIZE)

# Train all algorithms sequentially
train-all: train-ppo train-sac train-rainbow

# Clean up generated files
clean:
	rm -rf models/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete