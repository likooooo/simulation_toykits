# Fresnel 智能体 · Ollama 推理服务

为 Fresnel 多层膜设计/分析专家智能体提供本地大模型推理能力。通过 Docker 一键部署 Ollama，无需在宿主机安装 Ollama 或手动下载模型；模型与配置持久化在 volume 中，重启不丢失。

---

## 环境要求

- **Docker** 与 **Docker Compose**（或 `docker compose` 插件）
- **NVIDIA GPU**，驱动已安装
- **NVIDIA Container Toolkit** 已安装并已执行 `nvidia-ctk runtime configure --runtime=docker`（使 Docker 可使用 GPU）

---

## 快速开始

**1. 启动服务**

```bash
cd agent/fresnel_caculator/docker
docker compose up -d
```

**2. 拉取默认模型**

首次使用需拉取模型（约数 GB，仅一次）：

```bash
chmod +x pull_model.sh
./pull_model.sh
```

默认拉取 `qwen2.5:7b`。如需更换模型，可复制 `.env.example` 为 `.env`，修改 `OLLAMA_MODEL` 后再次执行 `./pull_model.sh`。

**3. 运行智能体**

在**仓库根目录**执行：

```bash
python -m agent.fresnel_caculator.run_agent "请计算 Vacuum 0 SiO2 0.1 Vacuum 0 在 532nm、0 度下的 R 和 T"
```

智能体默认连接 `http://localhost:11434`，无需额外配置。

---

## 运维与参考

| 操作 | 命令 |
|------|------|
| 查看服务状态 | `docker compose ps` |
| 查看推理日志 | `docker compose logs -f ollama` |
| 停止服务 | `docker compose down`（模型数据保留在 volume 中） |
| 查看已拉取模型 | `docker exec fresnel-ollama ollama list` |
| 拉取其他模型 | `docker exec fresnel-ollama ollama pull <模型名>` |

**数据持久化**：模型与配置保存在 volume `fresnel_ollama_data`。需要彻底清空时执行：`docker compose down -v`。

**远程访问**：其他机器或容器需访问本服务时，使用 `http://<本机IP>:11434`，并在运行智能体时加上 `--base-url http://<本机IP>:11434`。
