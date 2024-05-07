# Local DES cutouts

This is a set of scripts to generate cutout images from the Dark Energy Survey data based on a set of input positions and configuration options.

## Build the Docker image

```bash
docker build . --build-arg="UID=$UID" --build-arg="GID=$UID" -t local-des-cutouts:dev
```

## Prepare configuration files

Populate `input/positions.csv` and customize `input/cutout_config.yaml`.

## Generate the cutouts

```bash
docker run --rm -it --hostname=cutout --name=des-cutout \
    -v $(pwd)/input:/home/worker/input:ro \
    -v $(pwd)/output:/home/worker/output \
    local-des-cutouts:dev \
    python3 task.py
```
