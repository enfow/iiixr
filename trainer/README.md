# Reinforcement Learning Trainer

## Quick Start

### 1. Build Docker Image

```bash
# Build CPU version
make docker-build

# Build GPU version (CUDA support)
make docker-build DEVICE=cuda
```

### 2. Start Training

#### Basic Training Commands

```bash
# Basic training (runs inside Docker container)
make docker-run MODEL=<model_name> ENV=<environment_name>

# Example: Train PPO on LunarLander-v3
make docker-run MODEL=ppo ENV=LunarLander-v3

# Example: Train Rainbow DQN on BipedalWalker-v3 Hardcore
make docker-run MODEL=rainbow_dqn ENV=BipedalWalkerHardcore-v3

# Example: use gpu
make docker-run MODEL=ppo ENV=LunarLander-v3 DEVICE=cuda
```

#### Key Parameters

| Parameter | Description | Default | Examples |
|-----------|-------------|---------|----------|
| `MODEL` | Model to train | - | `ppo`, `sac`, `rainbow_dqn`, `ddqn`, `c51`, `td3` |
| `ENV` | Environment name | - | `LunarLander-v3`, `BipedalWalker-v3`,  `BipedalWalkerHardcore-v3`, |
| `N_EPISODES` | Number of training episodes | - | `1000`, `5000` |
| `MAX_STEPS` | Maximum steps per episode | - | `1000` |
| `BATCH_SIZE` | Batch size | - | `256` |
| `LEARNING_RATE` | Learning rate | - | `0.0003` |
| `DEVICE` | Device to use | `cpu` | `cpu`, `cuda` |

#### Using YAML Configuration Files

```bash
make docker-run TRAIN_CONFIG=config/train_config.yaml
```

## Evaluation

### Run Evaluation

```bash
# Evaluate trained model
make docker-run-eval EVAL_RESULT_PATH=results/[EXP_RESULT_DIR] EVAL_EPISODES=100

# Or run evaluation directly
make eval EVAL_RESULT_PATH=results/[EXP_RESULt_DIR] EVAL_EPISODES=100
```

### Evaluation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `EVAL_RESULT_PATH` | Path to result directory to evaluate | - |
| `EVAL_EPISODES` | Number of evaluation episodes | - |
| `DEVICE` | Device to use | `cpu` |

## Hyperparameter Optimization (HPO)

### Run HPO

```bash
# Basic HPO execution
make docker-run-hpo MODEL=ppo ENV=LunarLander-v3 HPO_N_TRIALS=50

# Or run HPO directly
make hpo MODEL=ppo ENV=LunarLander-v3 HPO_N_TRIALS=50
```

### HPO Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `HPO_N_TRIALS` | Number of optimization trials | - |
| `HPO_STUDY_NAME` | Study name | - |
| `HPO_TIMEOUT` | Maximum execution time (seconds) | - |
| `HPO_CONFIG` | HPO configuration file | `config/hpo_config.yaml` |

### Setting HPO Search Ranges

```bash
# Set batch size search range
make docker-run-hpo MODEL=ppo ENV=LunarLander-v3 \
  HPO_N_TRIALS=30 \
  HPO_BATCH_SIZE="64,128,256" \
  HPO_HIDDEN_DIM="128,256,512" \
  HPO_LEARNING_RATE="0.0001,0.0003,0.001"

# PPO specific parameter search
make docker-run-hpo MODEL=ppo ENV=LunarLander-v3 \
  HPO_N_TRIALS=20 \
  HPO_PPO_EPOCHS="3,5,10" \
  HPO_CLIP_EPS="0.1,0.2,0.3" \
  HPO_ENTROPY_COEF="0.01,0.1,0.2"
```

### Using HPO Configuration Files

```bash
# Use HPO configuration file
make docker-run-hpo MODEL=ppo ENV=LunarLander-v3 HPO_CONFIG=config/hpo_config.yaml
```

## Result Analysis

### Generate Plots

```bash
# Generate training result plots
make plot PLOT_RESULT_DIR=results/[EXP_RESULT_DIR]
```

### Generate GIFs

```bash
# Generate GIF of trained agent behavior
make gif GIF_RESULT_DIR=results/[EXP_RESULT_DIR]

# Advanced GIF options
make gif GIF_RESULT_DIR=results/[EXP_RESUlT_DIR] \
  GIF_MAX_STEPS=500 \
  GIF_FPS=30 \
  GIF_EPISODES=3 \
  GIF_RENDER_MODE=rgb_array
```

## Code Formatting

```bash
# Format code
make format
```

## Project Structure

```
trainer/
├── src/                    # Source code
│   ├── train.py           # Training script
│   ├── eval.py            # Evaluation script
│   ├── run_hpo.py         # HPO script
│   └── result_handler/    # Result processing tools
├── config/                # Configuration files
│   ├── train_config.yaml  # Default training config
│   ├── hpo_config.yaml    # HPO configuration
│   └── *.yaml            # Model-specific configs
├── results/               # Training results storage
├── docker/               # Docker configuration
└── Makefile              # Make commands
```
