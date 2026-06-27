# 本地构建：直接使用当前目录代码，无需从 GitHub 拉取
# 用法：在仓库根目录执行
#   docker build -t simulation-toykits:v1 .
# 或：python scripts/build_toykits.py docker
# 构建前请确保：git submodule update --init --recursive simulation_core
# 若 .simulation_core/ 无 simulation.so，请先执行 build_toykits

# -----------------------------------------------------------------------------
# 阶段 1：安装 Python 依赖
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt ./
COPY simulation_core/3rdparty/infrastructure/requirements.txt simulation_core/3rdparty/infrastructure/requirements.txt
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y pip setuptools wheel pytest pymoo cmake 2>/dev/null || true \
    && find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local -type f -name '*.pyc' -delete 2>/dev/null || true

# -----------------------------------------------------------------------------
# 阶段 2：.simulation_core（含 libuca 等）拷贝到 /tmp，再用 ldd 补充镜像内系统库
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS so-artifacts

COPY .simulation_core/ /tmp/
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
COPY simulation_core/assets/database/ simulation_core/assets/database/

RUN mkdir -p /app/.simulation_core
COPY --from=so-artifacts /tmp/ /app/.simulation_core/

RUN echo '/app/.simulation_core' > /etc/ld.so.conf.d/99-app-libs.conf && ldconfig

ENV LD_LIBRARY_PATH=/app/.simulation_core
ENV PYTHONPATH=/app:/app/.simulation_core
ENV SIMULATION_ARTIFACTS_DIR=/app/.simulation_core
ENV SIMULATION_DATABASE_DIR=/app/simulation_core/assets/database
ENV SIMULATION_TMM_ASSETS_DIR=/app/simulation_core/assets/ipynb/simulation/TMM
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8052

EXPOSE ${PORT}
CMD ["sh", "-c", "export LD_LIBRARY_PATH=/app/.simulation_core${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} && exec streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
