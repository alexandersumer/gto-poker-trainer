# syntax=docker/dockerfile:1

FROM rust:1.90 as builder
WORKDIR /app

# Cache dependencies
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY public ./public
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/gto-trainer /usr/local/bin/gto-trainer
COPY public ./public

ENV RUST_LOG=info
EXPOSE 8080

CMD ["gto-trainer", "serve", "--addr", "0.0.0.0:8080"]
