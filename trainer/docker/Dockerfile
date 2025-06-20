FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    swig \
    build-essential \
    python3-dev \
    make \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

COPY pyproject.toml uv.lock ./

RUN uv pip install --system .

COPY src/ ./src/
COPY Makefile .

RUN mkdir -p /app/models

# Set default environment variables
ENV MODEL=ppo
ENV ENV=LunarLander-v3
ENV N_EPISODES=1000
ENV MAX_STEPS=1000
ENV BATCH_SIZE=64
ENV LEARNING_RATE=0.001
ENV GAMMA=0.99
ENV EVAL=false
ENV EVAL_PERIOD=10
ENV EVAL_EPISODES=10

CMD ["make", "train"]