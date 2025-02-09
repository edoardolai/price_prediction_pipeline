FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/scraper/src/

ENV PYTHONPATH=/app
ENV OUTPUT_DIR=/app/data/raw

CMD ["python", "-m", "scraper.src.main"]