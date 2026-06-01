#!/bin/bash
#
# Build DrugCLIP Docker Image
#
# Usage:
#   bash build.sh                    # Build without proxy
#   HTTP_PROXY=... bash build.sh     # Build with proxy from environment
#

IMAGE_NAME="drugclip:latest"

# Check if proxy is set
if [ -n "$http_proxy" ] || [ -n "$HTTP_PROXY" ]; then
    PROXY_ARG="--build-arg http_proxy=${http_proxy:-$HTTP_PROXY} --build-arg https_proxy=${https_proxy:-$HTTPS_PROXY}"
    echo "Building with proxy: ${http_proxy:-$HTTP_PROXY}"
else
    PROXY_ARG=""
    echo "Building without proxy"
fi

# Build Docker image
docker build -t ${IMAGE_NAME} . ${PROXY_ARG}

if [ $? -eq 0 ]; then
    echo "✓ Docker image '${IMAGE_NAME}' built successfully"
else
    echo "✗ Failed to build Docker image"
    exit 1
fi