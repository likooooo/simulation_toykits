FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:/usr/local/cuda/bin:$PATH" \
    APP_LIB_PATH=/app/assets/lib

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gpg software-properties-common git git-lfs openssh-client curl \
    python3.12 python3.12-dev python3.12-venv \
    libboost-all-dev libfftw3-dev libeigen3-dev \
    build-essential cmake pkg-config && \
    \
    # 安装 Intel oneAPI 源
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && apt-get install -y --no-install-recommends intel-oneapi-mkl-devel && \
    \
    # 初始化虚拟环境
    python3.12 -m venv /opt/venv && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

ARG REPO_URL=git@github.com:likooooo/simulation_toykits.git
ARG ASSETS_URL=git@github.com:likooooo/simulation_toykits_web_assets.git

RUN --mount=type=secret,id=SSH_PRIVATE_KEY \
    mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts && \
    cat /run/secrets/SSH_PRIVATE_KEY | tr -d '\r' > ~/.ssh/id_ed25519 && \
    chmod 600 ~/.ssh/id_ed25519 && \
    git config --global url."git@github.com:".insteadOf https://github.com/ && \
    git clone --depth 1 "${REPO_URL}" . && \
    if [ -n "$ASSETS_URL" ]; then \
        git config submodule.assets.url "${ASSETS_URL}" && \
        git submodule sync assets; \
    fi && \
    git submodule update --init --recursive --depth 1 && \
    rm -rf ~/.ssh

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="${APP_LIB_PATH}:${PYTHONPATH}" \
    LD_LIBRARY_PATH="${APP_LIB_PATH}:${LD_LIBRARY_PATH}"

RUN echo "${APP_LIB_PATH}" > /etc/ld.so.conf.d/99-app-libs.conf && ldconfig

EXPOSE 7860

CMD ["sh", "-c", ". /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1 && exec streamlit run app.py --server.port=7860 --server.address=0.0.0.0"]