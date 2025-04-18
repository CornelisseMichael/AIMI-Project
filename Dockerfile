FROM --platform=linux/amd64 pytorch/pytorch AS example-algorithm-amd64
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Copy files to the container
COPY --chown=user:user requirements.txt processor.py dataloader.py experiment_config.py /opt/app/
COPY --chown=user:user models /opt/app/models
COPY --chown=user:user results /opt/app/resources
# COPY --chown=user:user test /opt/app/test

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
