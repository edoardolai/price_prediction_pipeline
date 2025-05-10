# Property Price Prediction Pipeline

A proof of concept project showcasing Apache Airflow orchestration and Docker containerization concepts. The pipeline scrapes real estate data, processes it, and trains a machine learning model to predict property prices.

## Workflow
The pipeline follows this process:
[![](https://mermaid.ink/img/pako:eNqdlFGPojAUhf9K03lVY1tA5WGTWckmm6zZ2ZX4sDgPXbggEahpSxzXmf8-BVFxJTPGPpVyvnNPm9vucSgiwC5OJN-skP91WSAzVPn3sLDEngjXINH3IpZcaVmGupSwxAddNbxp0GimIt8IBc-o3_-CXoUMV2AIrkG9Io-cVYXmaQHSRY-pjDOxfW6bdcK0A34SSicS5r9-fM6zDn4emh2C_By2OuBpBry4BbY7YF_WkwaGIloWV8feHA36ebJLRXFx7ORQT5n_UZlVxXwSNJtCPlfrVji_EWuZJgnISkuDZg9XWnqlZUEdOS2Strg7uMc1R99M9HbYY_2tTOtjWZBgIbIyBxf95ltUMa0Ei0YtgUd11OtwJyN6MnqSIgSlILqyo5d2rGXH_rNjJ7uZuRcZepQ6jXmo1Yd7nmsheXJxKxakHaA1b1X3mmDK4FX1GQlmoHl0jt8up3cZVF0Wp1nmPgwncdwzfSHW4D4wxpp5f5tGeuVam5cLihwpy_qAopeUfx9F76LYPdTimHA0psSit3P0To7dzuEezkHmPI3M67qvXJZYryA3XeKaacTluuqXN6PjpRbzXRFi17yv0MNSlMkKuzHPlPkqN6YhwEu56bb8tLrhxR8hzt8QpaaLZofHvH7Taw129_gFu8RyBmToTOzxkDjj8cR2enhXLVsDyxoxymyz7NjEeuvhf7XtcOAQYtPxaGzRCR05zuTtHQ7Q6-o?type=png)](https://mermaid.live/edit#pako:eNqdlFGPojAUhf9K03lVY1tA5WGTWckmm6zZ2ZX4sDgPXbggEahpSxzXmf8-BVFxJTPGPpVyvnNPm9vucSgiwC5OJN-skP91WSAzVPn3sLDEngjXINH3IpZcaVmGupSwxAddNbxp0GimIt8IBc-o3_-CXoUMV2AIrkG9Io-cVYXmaQHSRY-pjDOxfW6bdcK0A34SSicS5r9-fM6zDn4emh2C_By2OuBpBry4BbY7YF_WkwaGIloWV8feHA36ebJLRXFx7ORQT5n_UZlVxXwSNJtCPlfrVji_EWuZJgnISkuDZg9XWnqlZUEdOS2Strg7uMc1R99M9HbYY_2tTOtjWZBgIbIyBxf95ltUMa0Ei0YtgUd11OtwJyN6MnqSIgSlILqyo5d2rGXH_rNjJ7uZuRcZepQ6jXmo1Yd7nmsheXJxKxakHaA1b1X3mmDK4FX1GQlmoHl0jt8up3cZVF0Wp1nmPgwncdwzfSHW4D4wxpp5f5tGeuVam5cLihwpy_qAopeUfx9F76LYPdTimHA0psSit3P0To7dzuEezkHmPI3M67qvXJZYryA3XeKaacTluuqXN6PjpRbzXRFi17yv0MNSlMkKuzHPlPkqN6YhwEu56bb8tLrhxR8hzt8QpaaLZofHvH7Taw129_gFu8RyBmToTOzxkDjj8cR2enhXLVsDyxoxymyz7NjEeuvhf7XtcOAQYtPxaGzRCR05zuTtHQ7Q6-o)

## Overview

- **ETL Pipeline** using Apache Airflow
- **Containerized Services** using Docker
- **ML Model Training** for property price prediction
- **Web Interface** for price predictions

## Prerequisites

- Docker and Docker Compose
- Git
- Python 3.8+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/price_prediction_pipeline.git
cd price_prediction_pipeline
```

2. Copy configuration templates:
```bash
cp .env.template .env
cp docker-compose.template.yaml docker-compose.yaml
cp services/scraper/src/config.template.json services/scraper/src/config.json
```
3. Configure environment <br>
### .env
```bash
AIRFLOW_UID=50000
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_DB=airflow
AIRFLOW_IMAGE_NAME=apache/airflow:2.10.4
```
 ## Airflow Configuration
  The docker-compose.yaml configures Airflow with LocalExecutor
  Mounts necessary volumes for dags, logs, and configuration.
 ## Scraper Configuration
  Edit config.json with required tokens:
``` bash
{
  "immoweb_session": "your_session_token",
  "XSRF-TOKEN": "your_xsrf_token",
  "__cf_bm": "your_cf_token"
}
```
# Running the Pipeline

1. Start the containers
```
docker-compose build
docker-compose up -d
```
2. Access Airflow UI:
  Navigate to http://localhost:8080
  Default credentials:
  Username: airflow
  Password: airflow
3. Enable the DAG:
  Find property_etl_pipeline in the DAGs list
  Toggle the switch to enable it

## Services
<ul>
  <li>Scraper: Collects property data</li>
  <li>Cleaner: Processes and cleanses data</li>
  <li>Trainer: Trains ML model</li>
  <li>Deployment: Web interface for predictions (Not added to the dag yet, needs some rework)</li>


</ul>

## Volumes
The pipeline uses Docker volumes for:

Data persistence
Configuration storage
Model artifacts

## Troubleshooting
1. ### Permission Issues:

Ensure correct AIRFLOW_UID in .env
Check volume permissions

2. ### Container Errors:

Check logs: docker compose logs -f
Verify configuration files

## Development
To modify the pipeline:
1. Stop containers: docker compose down (-v and --remove-orphans flag optional but suggested to clean up)
2. Make changes
3. Rebuild: 
```bash
docker compose build
docker compose up
```

## Future improvements

### 1. Deploy Streamlit frontend for interactive predictions

### 2. Add model versioning
  - Expand to multiple data sources
### 3. Deployment
- Deploy on cloud infrastructure
- Implement monitoring and alerting




