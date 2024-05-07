#!/bin/bash

set -eo pipefail

docker run --rm -it --hostname=cutout --name=des-cutout \
    -v $(pwd)/input:/home/worker/input:ro \
    # -v $(pwd)/src/bulkthumbs.py:/home/worker/bulkthumbs.py:ro \
    # -v $(pwd)/src/task.py:/home/worker/task.py:ro \
    -v $(pwd)/output:/home/worker/output \
    local-des-cutouts:dev \
    python3 task.py
