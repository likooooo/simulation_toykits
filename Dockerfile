# 1. 选择基础镜像 (Python 3.9 Slim 版本体积较小)
FROM python:3.9-slim
# 2. 设置工作目录
WORKDIR /app

# 3. 安装系统依赖
# 如果你的 C++ 代码编译需要 gcc/g++ 或者其他库 (如 libgl1 用于 opencv/matplotlib)
# 即使是直接运行编译好的 C++，也通常需要基本的动态库
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

# 4. 复制 Python 依赖并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制你的源代码 (包括 Python 和 C++ 源码/可执行文件)
COPY . .

# (可选) 如果需要在构建镜像时编译 C++ 代码
# RUN cd cpp_engine && make

# 6. 暴露 Streamlit 的默认端口
EXPOSE 8501

# 7. 启动命令
# address=0.0.0.0 允许从 Docker 外部访问
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]