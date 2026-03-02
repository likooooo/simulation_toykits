FROM python:3.12-slim

# 使用清华 apt 镜像源加速（与 pip 一致）
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true \
    && sed -i 's|security.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true \
    && sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list 2>/dev/null || true \
    && sed -i 's|security.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list 2>/dev/null || true \
    && apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG REPO_URL=https://github.com/likooooo/simulation_toykits.git
ARG ASSETS_URL=https://github.com/likooooo/simulation_toykits_assets.git
RUN git clone "${REPO_URL}" . \
    && git config submodule.assets.url "${ASSETS_URL}" \
    && git submodule update --init --recursive

# 使用国内镜像源加速 pip 安装
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -r requirements.txt

COPY docker_artifacts/ /tmp/docker_artifacts/
RUN mkdir -p /app/assets /app/libs \
  && if [ -f /tmp/docker_artifacts/simulation.so ]; then \
       cp /tmp/docker_artifacts/simulation.so /app/assets/ \
       && [ -d /tmp/docker_artifacts/core_plugins ] && cp -r /tmp/docker_artifacts/core_plugins /app/assets/ \
       && [ -d /tmp/docker_artifacts/plugin ] && cp -r /tmp/docker_artifacts/plugin /app/assets/ \
       && [ -d /tmp/docker_artifacts/libs ] && cp -r /tmp/docker_artifacts/libs/. /app/libs/; \
     fi \
  && rm -rf /tmp/docker_artifacts

RUN echo '/app/libs' > /etc/ld.so.conf.d/99-app-libs.conf && ldconfig

ENV LD_LIBRARY_PATH=/app/libs
# 默认 8052；设为 80 可用阿里云安全组只放行 80，访问时无需写端口
ENV PORT=8052

EXPOSE ${PORT}
# 使用 shell 显式导出 LD_LIBRARY_PATH，确保 streamlit 及其子进程都能找到 .so
CMD ["sh", "-c", "export LD_LIBRARY_PATH=/app/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} && exec streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
