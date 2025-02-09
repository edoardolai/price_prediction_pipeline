FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY src/ /app/cleaner/src/

ENV PYTHONPATH=/app
ENV INPUT_DIR=/app/data/raw
ENV OUTPUT_DIR=/app/data/processed
ENV ARTIFACTS_DIR=/app/data/artifacts

CMD ["python", "-m", "cleaner.src.main"]