FROM registry.gitlab.com/des-labs/kubernetes/des-task-cutout:latest

USER root
ARG UID=68586
ARG GID=2402
RUN groupmod -g ${GID} worker && usermod -u ${UID} -g ${GID} worker

USER worker
COPY --chown=worker:worker ./src/* ./
