FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=8080

WORKDIR /app

COPY server/requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY server/app.py .

# モデル重みをビルド時にダウンロードしてイメージに焼き込む（起動時のダウンロードを無くす）
RUN python -c "from app import MODELS, get_model; [get_model(n, r) for n, r in MODELS]"

EXPOSE 8080
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
