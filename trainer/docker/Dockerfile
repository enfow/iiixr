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

CMD ["make", "train"] 