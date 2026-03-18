FROM postgres:16

# install system deps
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY postgres/init /docker-entrypoint-initdb.d/