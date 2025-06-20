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

# Default values for evaluation parameters
EVAL_EPISODES := 10
EVAL_RESULTS_PATH := results/ppo

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

help:
	@echo "Available targets:"
	@echo "  install        Install dependencies"
	@echo "  clean          Clean results directory"
	@echo "  train-ppo      Train PPO agent"
	@echo "  train-sac      Train SAC agent"
	@echo "  train-rainbow  Train Rainbow DQN agent"
	@echo "  train-all      Train all agents sequentially"
	@echo "  eval-ppo       Evaluate PPO agent"
	@echo "  eval-sac       Evaluate SAC agent"
	@echo "  eval-rainbow   Evaluate Rainbow DQN agent"
	@echo "  eval-discrete-sac Evaluate Discrete SAC agent"
	@echo ""
	@echo "Training example usage:"
	@echo "  make train-ppo ENV=LunarLander-v2 N_EPISODES=1000"
	@echo "  make train-sac ENV=Pendulum-v1 LEARNING_RATE=0.0003"
	@echo "  make train-rainbow ENV=CartPole-v1 MEMORY_SIZE=50000"
	@echo ""
	@echo "Evaluation example usage:"
	@echo "  make eval-ppo EVAL_RESULTS_PATH=results/ppo ENV=LunarLander-v2"
	@echo "  make eval-sac EVAL_RESULTS_PATH=results/sac ENV=Pendulum-v1 EVAL_EPISODES=20"

install:
	pip install -r requirements.txt

clean:
	rm -rf $(RESULTS_DIR)/*

# Training targets
train-ppo:
	$(PYTHON) $(SRC_DIR)/train.py \
		--model ppo \
		--env $(ENV) \
		--episodes $(N_EPISODES) \
		--max_steps $(MAX_STEPS) \
		--save_dir $(RESULTS_DIR) \
		--lr $(LEARNING_RATE) \
		--batch_size $(BATCH_SIZE) \
		--gamma $(GAMMA)

train-sac:
	$(PYTHON) $(SRC_DIR)/train.py \
		--model sac \
		--env $(ENV) \
		--episodes $(N_EPISODES) \
		--max_steps $(MAX_STEPS) \
		--save_dir $(RESULTS_DIR) \
		--lr $(LEARNING_RATE) \
		--batch_size $(BATCH_SIZE) \
		--gamma $(GAMMA) \
		--target_update $(TARGET_UPDATE_INTERVAL)

train-discrete-sac:
	$(PYTHON) $(SRC_DIR)/train.py \
		--model discrete_sac \
		--env $(ENV) \
		--episodes $(N_EPISODES) \
		--max_steps $(MAX_STEPS) \
		--save_dir $(RESULTS_DIR) \
		--lr $(LEARNING_RATE) \
		--batch_size $(BATCH_SIZE) \
		--gamma $(GAMMA) \
		--target_update $(TARGET_UPDATE_INTERVAL)

train-rainbow:
	$(PYTHON) $(SRC_DIR)/train.py \
		--model rainbow_dqn \
		--env $(ENV) \
		--episodes $(N_EPISODES) \
		--max_steps $(MAX_STEPS) \
		--save_dir $(RESULTS_DIR) \
		--lr $(LEARNING_RATE) \
		--batch_size $(BATCH_SIZE) \
		--gamma $(GAMMA) \
		--buffer_size $(MEMORY_SIZE) \
		--target_update $(TARGET_UPDATE_INTERVAL)

# Train all algorithms sequentially
train-all: train-ppo train-sac train-rainbow

# Clean up generated files
clean:
	rm -rf models/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Evaluation targets
eval-ppo:
	$(PYTHON) $(SRC_DIR)/eval.py \
		--results_path $(EVAL_RESULTS_PATH) \
		--env $(ENV) \
		--episodes $(EVAL_EPISODES)

eval-sac:
	$(PYTHON) $(SRC_DIR)/eval.py \
		--results_path $(EVAL_RESULTS_PATH) \
		--env $(ENV) \
		--episodes $(EVAL_EPISODES)

eval-rainbow:
	$(PYTHON) $(SRC_DIR)/eval.py \
		--results_path $(EVAL_RESULTS_PATH) \
		--env $(ENV) \
		--episodes $(EVAL_EPISODES)

eval-discrete-sac:
	$(PYTHON) $(SRC_DIR)/eval.py \
		--results_path $(EVAL_RESULTS_PATH) \
		--env $(ENV) \
		--episodes $(EVAL_EPISODES)