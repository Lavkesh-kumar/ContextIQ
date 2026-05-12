FROM python:3.11-slim

RUN useradd -m -u 1000 appuser

ENV PORT=7860 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    CONTEXTIQ_MODEL_PATH=/app/model/phi-3-openvino \
    CONTEXTIQ_DEVICE=CPU \
    CONTEXTIQ_MAX_NEW_TOKENS=256 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "huggingface_hub[hf_transfer]>=0.22" \
    && python -c "import openvino, os; print(os.path.join(os.path.dirname(openvino.__file__), 'libs'))" > /openvino_libs_path \
    && cat /openvino_libs_path

ARG HF_MODEL_REPO

RUN HF_MODEL_REPO="${HF_MODEL_REPO}" python -c "import os, sys; from huggingface_hub import snapshot_download; repo = os.environ.get('HF_MODEL_REPO', '').strip(); sys.exit(1) if not repo else snapshot_download(repo_id=repo, local_dir='/app/model/phi-3-openvino', ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*'])"

COPY --chown=appuser:appuser . .

RUN mkdir -p /app/data /app/.cache/huggingface \
    && chown -R appuser:appuser /app/data /app/.cache /app/model \
    && chmod +x /app/start.sh

USER appuser

EXPOSE 7860

CMD ["sh", "/app/start.sh"]