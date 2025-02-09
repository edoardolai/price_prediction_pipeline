FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/trainer/src/

ENV PYTHONPATH=/app
ENV INPUT_DIR=/app/data/processed
ENV ARTIFACTS_DIR=/app/data/artifacts
ENV CONFIG_PATH=/app/airflow/dags/config/etl_config.json

CMD ["python", "-m", "trainer.src.main"]