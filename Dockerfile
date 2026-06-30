# 本地构建：直接使用当前目录代码 + 预构建 artifact
# 用法：
#   export SIMULATION_DATABASE_KEY=...   # docker run 验证时需要
#   python scripts/build_toykits.py docker
# 构建前请确保 .simulation_toolkits/ 含 simulation.so（build 或 --download_toolkits）

# -----------------------------------------------------------------------------
# 阶段 1：安装 Python 依赖（runtime only）
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt ./
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y pip setuptools wheel pytest 2>/dev/null || true \
    && find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local -type f -name '*.pyc' -delete 2>/dev/null || true

# -----------------------------------------------------------------------------
# 阶段 2：.simulation_toolkits（含 libuca 等）拷贝到 /tmp，再用 ldd 补充镜像内系统库
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS so-artifacts

COPY .simulation_toolkits/ /tmp/
RUN for f in $(ldd /tmp/simulation.so 2>/dev/null | sed -n 's/.*=> \s*\([^ ]*\) .*/\1/p'); do \
      [ -f "$f" ] && cp -L "$f" /tmp/ 2>/dev/null || true; \
    done

# -----------------------------------------------------------------------------
# 阶段 3：最小运行时镜像
# -----------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY app.py common.py ./
COPY core/ core/
COPY ui/ ui/
COPY filmstack_simulation/ filmstack_simulation/
COPY simulation_database/ simulation_database/
COPY pages/ pages/
COPY docs/ docs/

RUN mkdir -p /app/.simulation_toolkits
COPY --from=so-artifacts /tmp/ /app/.simulation_toolkits/

RUN echo '/app/.simulation_toolkits' > /etc/ld.so.conf.d/99-app-libs.conf && ldconfig

ENV LD_LIBRARY_PATH=/app/.simulation_toolkits
ENV PYTHONPATH=/app:/app/.simulation_toolkits
ENV SIMULATION_ARTIFACTS_DIR=/app/.simulation_toolkits
ENV SIMULATION_DATABASE_DIR=/app/.simulation_toolkits/assets
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8052

EXPOSE ${PORT}
CMD ["sh", "-c", "export LD_LIBRARY_PATH=/app/.simulation_toolkits${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} && exec streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
