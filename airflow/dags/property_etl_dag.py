from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_alerts": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "property_etl_pipeline",
    default_args=default_args,
    description="End-to-end ETL pipeline for property price prediction",
    schedule="0 0 * * *",
    start_date=datetime(2024, 2, 1),
    catchup=False,
)

mounts = [
    Mount(
        source="price_prediction_pipeline_airflow_data",
        target="/app/data",
        type="volume",
    ),
    Mount(
        source="price_prediction_pipeline_airflow_config",
        target="/app/airflow/dags/config",
        type="volume",
    ),
]

scrape_task = DockerOperator(
    task_id="scrape_properties",
    image="price_prediction_pipeline-scraper:latest",
    command="python -m scraper.src.main",
    docker_url="unix://var/run/docker.sock",
    network_mode="bridge",
    api_version="auto",
    mount_tmp_dir=False,
    auto_remove=True,
    mounts=mounts,
    container_name=f"scraper_container_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    dag=dag,
)

clean_task = DockerOperator(
    task_id="clean_properties",
    image="price_prediction_pipeline-cleaner:latest",
    command="python -m cleaner.src.main",
    docker_url="unix://var/run/docker.sock",
    network_mode="bridge",
    api_version="auto",
    mount_tmp_dir=False,
    auto_remove=True,
    mounts=mounts,
    container_name=f"cleaner_container_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    dag=dag,
)

train_task = DockerOperator(
    task_id="train_model",
    image="price_prediction_pipeline-trainer:latest",
    command="python -m trainer.src.main",
    docker_url="unix://var/run/docker.sock",
    network_mode="bridge",
    api_version="auto",
    mount_tmp_dir=False,
    auto_remove=True,
    mounts=mounts,
    container_name=f"trainer_container_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    dag=dag,
)

scrape_task >> clean_task >> train_task
