FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PSSP_MODEL_PATH=/app/best_model_v2.keras

WORKDIR /app

COPY pyproject.toml README.md ./
COPY python_service ./python_service
RUN python -m pip install --no-cache-dir ".[ml,server]"

COPY public ./public
COPY best_model_v2.keras ./best_model_v2.keras

EXPOSE 5001
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5001/health')"

CMD ["gunicorn", "--chdir", "python_service", "--bind", "0.0.0.0:5001", "--workers", "1", "--threads", "4", "app:app"]
