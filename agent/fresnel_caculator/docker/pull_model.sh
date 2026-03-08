#!/usr/bin/env bash
# 在 Ollama 容器启动后拉取默认模型，供 Fresnel 智能体使用。
# 用法：在 docker 目录下执行 ./pull_model.sh

set -e

CONTAINER_NAME="${OLLAMA_CONTAINER_NAME:-fresnel-ollama}"
MODEL="${OLLAMA_MODEL:-qwen3.5:9b}" # qwen2.5:7b

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

echo "Waiting for Ollama (${CONTAINER_NAME}) to be ready..."
for i in 1 2 3 4 5 6 7 8 9 10; do
  if docker exec "$CONTAINER_NAME" ollama list 2>/dev/null; then
    echo "Ollama is up."
    break
  fi
  if [ "$i" -eq 10 ]; then
    echo "Ollama did not become ready. Is the container running? (docker compose up -d)" >&2
    exit 1
  fi
  sleep 2
done

echo "Pulling model: ${MODEL} (this may take a while)..."
docker exec "$CONTAINER_NAME" ollama pull "$MODEL"
echo "Done. Run the agent with: python -m agent.fresnel_caculator.run_agent 'your prompt' --model $MODEL"
