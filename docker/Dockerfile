# ============================================
# Build arguments configuration
# ============================================
# Base image
ARG BASE_IMAGE=inseefrlab/onyxia-python-minimal:py3.13.7-gpu

# Full Python version used for micromamba environment creation
ARG MAMBA_PYTHON_VERSION=3.13.7

# Python version used for site-packages path
ARG MAJOR_PYTHON_VERSION=3.13

# Name of the subfolder of the package to install
ARG DC_NAME=dc2

# ============================================
# Stage 1: Builder - Compilation and installation of dependencies
# ============================================
FROM ${BASE_IMAGE} as builder

# Docker doesn't automate ARG propagation, and mamba_shell require ENV variable
ARG DC_NAME
ARG MAJOR_PYTHON_VERSION
ARG MAMBA_PYTHON_VERSION
ENV DC_NAME=${DC_NAME}
ENV MAJOR_PYTHON_VERSION=${MAJOR_PYTHON_VERSION}
ENV MAMBA_PYTHON_VERSION=${MAMBA_PYTHON_VERSION}

# Environment variables inherited from the base image : 
# USERNAME, GROUPNAME, WORKSPACE_DIR

# Switch to root to install system dependencies
USER root

# Clean BASE Image
RUN \
    # Clean workspace
    rm -rf /home/onyxia/work/* && \
    # Remove ubuntugis PPA, which is not available for Ubuntu Noble
    rm -f /etc/apt/sources.list.d/ubuntugis-ubuntu-ppa-noble.sources

# Installation of system dependencies needed for compilation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libboost-dev \
        libboost-filesystem-dev \
        libboost-system-dev \
        libeigen3-dev \
        # Package for BLAS library (if auto-detection fails)
        libblas-dev \
        liblapack-dev \
        cmake \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Micromamba installation
ENV MAMBA_BIN_PATH=/usr/local/bin
RUN mkdir -p ${MAMBA_BIN_PATH} &&\
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C ${MAMBA_BIN_PATH} --strip-components=1 &&\
    chmod +x ${MAMBA_BIN_PATH}/micromamba && \
    ln -s /usr/local/bin/micromamba /usr/local/bin/mamba
    
# Micromamba configuration to use a shared environment
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV MAMBA_EXE=/usr/local/bin/micromamba


# Poetry and pip are installed in environment.yml via micromamba
# Configuration of pip and poetry cache location
ENV PIP_CACHE_DIR=/root/.cache/pip \
    POETRY_CACHE_DIR=/root/.cache/pypoetry


    # Creation of /${DC_NAME} directory
RUN mkdir -p /${DC_NAME}

# ============
# Python dependencies installation in a dedicated environment
# ============
# Micromamba dependencies installation from the environment.yml file
# Jupyterlab and ipykernel are added to be able to use the environment in Jupyterlab
# Cleanup in the same layer
COPY docker/environment.yml /${DC_NAME}/environment.yml
RUN micromamba create -y -n ${DC_NAME} -f /${DC_NAME}/environment.yml  python=${MAMBA_PYTHON_VERSION} && \
    # Add Jupyterlab and ipykernel to be able to use the environment in Jupyterlab
    micromamba install -n ${DC_NAME} -c conda-forge jupyterlab ipykernel pip -y && \
    micromamba clean --all --yes && \
    rm -rf /root/.cache /home/*/.cache 2>/dev/null || true

# Environment activation for following commands
COPY docker/mamba-shell /usr/local/bin/mamba-shell
RUN chmod +x /usr/local/bin/mamba-shell
SHELL ["/usr/local/bin/mamba-shell"]
ENV PATH=/opt/conda/envs/${DC_NAME}/bin:$PATH




# Poetry configuration to not create virtualenv
# and install in the system environment (which is already the micromamba environment)
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_HOME=/opt/poetry

# Copy Poetry configuration files
WORKDIR /${DC_NAME}
COPY pyproject.toml ./
COPY ${DC_NAME} ./${DC_NAME}
COPY README.md ./README.md
COPY LICENSE ./LICENSE


# Dependencies installation with Poetry
#   - use --no-root to not install the package itself at this stage
#   - use --without dev to not install dev dependencies
#   - use --verbose to have more details during installation
# Cache cleanup in the same layer
RUN \
    # Build perf
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    # Install dependencies without dev, and install the package in editable mode
    poetry install --no-root  --without dev --verbose && \
    # Cleanup
    poetry cache clear pypi --all -n || true && \
    poetry cache clear _default_cache --all -n || true && \
    pip cache purge || true && \
    rm -rf /root/.cache /home/*/.cache 2>/dev/null || true

# Package installation as "non-editable"
#  - Separate layer to be able to rebuild only this layer if needed
RUN \
    # Build perf
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    # Build the package as a wheel
    poetry build -f wheel && \
    # Install the wheel (ensures a non-editable installation)
    pip install --no-deps --verbose dist/*.whl && \
    # Cleanup
    poetry cache clear pypi --all -n || true && \
    poetry cache clear _default_cache --all -n || true && \
    rm -rf dist/ build/ && \
    pip cache purge || true && \
    rm -rf /root/.cache /home/*/.cache 2>/dev/null || true



# =======================
# FIX : Give user permission on all files in HOME
# =======================
RUN find "${HOME}" -not -user "${USERNAME}" -execdir chown --no-dereference "${USERNAME}:${GROUPNAME}" {} \+



# --- 
# Aggressive cleanup to reduce image size
# --- 
# remove Micromamba cache 
RUN micromamba clean --all --yes && \
    rm -rf /opt/conda/pkgs /opt/conda/conda-meta

# Remove Poetry and build dependencies which are not needed at runtime
RUN pip uninstall -y poetry setuptools wheel && \
    rm -rf /opt/poetry

# DCtools environment cleanup
RUN \
    # Python cache files
    find /opt/conda/envs/${DC_NAME} -name "*.pyc" -delete && \
    find /opt/conda/envs/${DC_NAME} -name "*.pyo" -delete && \
    find /opt/conda/envs/${DC_NAME} -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    # Remove tests
    find /opt/conda/envs/${DC_NAME} -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/conda/envs/${DC_NAME} -name "test" -type d -exec rm -rf {} + 2>/dev/null || true && \
    # Cleanup problematic .pth file
    rm -f /opt/conda/envs/${DC_NAME}/lib/python${MAJOR_PYTHON_VERSION}/site-packages/distutils-precedence.pth && \
    # Remove documentation files
    rm -rf /opt/conda/envs/${DC_NAME}/share/doc \
           /opt/conda/envs/${DC_NAME}/share/man \
           /opt/conda/envs/${DC_NAME}/share/gtk-doc && \
    # Remove headers and static libraries (if no compilation needed)
    rm -rf /opt/conda/envs/${DC_NAME}/include \
           /opt/conda/envs/${DC_NAME}/lib/*.a && \
    # Strip binaries
    find /opt/conda/envs/${DC_NAME} -name "*.so" -exec strip --strip-unneeded {} \; 2>/dev/null || true && \
    # DO NOT remove .dist-info and .egg-info (needed for package introspection)
    # find /opt/conda/envs/${DC_NAME} -name "*.a" -delete 2>/dev/null || true && \
    # find /opt/conda/envs/${DC_NAME} -type d -name "*.dist-info" -exec rm -rf {}/RECORD {} + && \
    # find /opt/conda/envs/${DC_NAME} -type d -name "*.egg-info" -exec rm -rf {} + && \
    rm -rf /tmp/* /var/tmp/*




# ============================================
# Stage 2: Runtime - Minimal final image
# ============================================
FROM ${BASE_IMAGE} as runtime

# Docker doesn't automate ARG propagation, and mamba_shell require ENV variable
ARG DC_NAME
ARG MAJOR_PYTHON_VERSION
ARG MAMBA_PYTHON_VERSION
ENV DC_NAME=${DC_NAME}
ENV MAJOR_PYTHON_VERSION=${MAJOR_PYTHON_VERSION}
ENV MAMBA_PYTHON_VERSION=${MAMBA_PYTHON_VERSION}

# Environment variables inherited from the base image : 
#   USERNAME, GROUPNAME, WORKSPACE_DIR


USER root

# Remove ubuntugis PPA, which is not available for Ubuntu Noble
RUN rm -f /etc/apt/sources.list.d/ubuntugis-ubuntu-ppa-noble.sources

# Install only runtime dependencies (not dev headers)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libboost-filesystem1.83.0 \
        libboost-system1.83.0 \
        libblas3 \
        liblapack3 \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* \
        /tmp/* \
        /var/tmp/* \
        /usr/share/man \
        /usr/share/doc \
        /usr/share/doc-base

# Copy ONLY the installed Python environment (not micromamba, not the source code)
COPY --from=builder /opt/conda/envs/${DC_NAME} /opt/conda/envs/${DC_NAME}

# Full Python environment configuration
ENV PATH=/opt/conda/envs/${DC_NAME}/bin:$PATH \
    PYTHONPATH=/opt/conda/envs/${DC_NAME}/lib/python${MAJOR_PYTHON_VERSION}/site-packages:$PYTHONPATH \
    LD_LIBRARY_PATH=/opt/conda/envs/${DC_NAME}/lib:$LD_LIBRARY_PATH
    
# Permissions on HOME files
RUN find "${HOME}" -not -user "${USERNAME}" -execdir chown --no-dereference "${USERNAME}:${GROUPNAME}" {} \+ 2>/dev/null || true


# For security reasons, we do not want to run the container as root
USER ${USERNAME}
WORKDIR ${WORKSPACE_DIR}

# Copy ${DC_NAME} project files
COPY --chown=${USERNAME}:${GROUPNAME} ${DC_NAME} ./${DC_NAME}
COPY --chown=${USERNAME}:${GROUPNAME} notebooks ./notebooks

# =======================
# Manage Jupyterlab launch
# =======================
# CMD on onyxia-jupyter-python is buggy with WORKSPACE_DIR
#  - ${WORKSPACE_DIR} is not expanded if used directly in CMD
# On Edito datalab Edito, the launch command is overridden anyway in the k8s deployment.yaml file
#     command: ["/bin/sh","-c"]
#     args: ["{{ .Values.init.standardInitPath }} jupyter lab --no-browser --ip '0.0.0.0' --LabApp.token='$(PASSWORD)' --ContentsManager.allow_hidden=True"]
#   "standardInitPath" is defined in values.yaml file : `standardInitPath: /opt/onyxia-init.sh`
# So the command is at last ["/bin/sh","-c", jupyter lab --no-browser --ip '0.0.0.0' --LabApp.token='$(PASSWORD)' --ContentsManager.allow_hidden=True"]
CMD ["/bin/sh", "-c", "jupyter lab --no-browser --ip 0.0.0.0"]
# --NotebookApp.websocket_ping_interval=90000 --NotebookApp.websocket_ping_timeout=90000
