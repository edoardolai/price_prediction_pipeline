FROM apache/airflow:2.10.4-python3.9

# Create directories for requirements
USER root
RUN mkdir -p /requirements /opt/airflow/data && \
    chown -R airflow: /requirements /opt/airflow/data
# Switch to airflow user for pip installations
USER airflow

# Copy requirements files
COPY --chown=airflow:root services/scraper/requirements.txt /requirements/scraper.txt
COPY --chown=airflow:root services/cleaner/requirements.txt /requirements/cleaner.txt
COPY --chown=airflow:root services/trainer/requirements.txt /requirements/trainer.txt

# Install dependencies
RUN pip install --no-cache-dir -r /requirements/scraper.txt \
    && pip install --no-cache-dir -r /requirements/cleaner.txt \
    && pip install --no-cache-dir -r /requirements/trainer.txt

# Verify installations
RUN pip list