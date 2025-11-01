FROM rust:1.82.0-slim-bullseye

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    ca-certificates \
    pkg-config \
    libssl-dev \
    libpq-dev \
    build-essential \
    clang \
    libclang-dev \
    capnproto \
 && rm -rf /var/lib/apt/lists/*

RUN rustup set profile minimal \
 && rustup toolchain install nightly --no-self-update \
 && rustup default nightly

WORKDIR /workspace
