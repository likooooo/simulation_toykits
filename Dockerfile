# -----------------------------------------------------------------------------
# 阶段 1：拉取代码并安装依赖（含 git，体积大）
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true \
    && sed -i 's|security.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true \
    && sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list 2>/dev/null || true \
    && sed -i 's|security.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list 2>/dev/null || true \
    && apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG REPO_URL=https://github.com/likooooo/simulation_toykits.git
ARG ASSETS_URL=https://github.com/likooooo/simulation_toykits_assets.git
RUN git clone --depth 1 "${REPO_URL}" . \
    && git config submodule.assets.url "${ASSETS_URL}" \
    && git submodule update --init --recursive --depth 1

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y pip setuptools wheel 2>/dev/null || true \
    && find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local -type f -name '*.pyc' -delete 2>/dev/null || true

# -----------------------------------------------------------------------------
# 阶段 2：仅保留 Python 运行时与依赖，不含 git 与构建工具
# -----------------------------------------------------------------------------
FROM python:3.12-slim

# 不再安装 git，仅保留运行时所需
WORKDIR /app

# 复制 site-packages 与 pip 安装的入口脚本（如 streamlit）
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# simulation.so 与依赖 .so 位于 assets/lib（submodule 已包含）
RUN echo '/app/assets/lib' > /etc/ld.so.conf.d/99-app-libs.conf && ldconfig

ENV LD_LIBRARY_PATH=/app/assets/lib
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8052

EXPOSE ${PORT}
CMD ["sh", "-c", "export LD_LIBRARY_PATH=/app/assets/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} && exec streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
