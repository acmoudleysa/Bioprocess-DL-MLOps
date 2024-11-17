from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.exceptions import AirflowException
import logging
import sys
from typing import Callable

from bioprocess_mlops.pipeline import TrainingPipeline
from bioprocess_mlops.utils import setup_logging


setup_logging()

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'BIOPROCESS',
    'depends_on_past': False,
    'email': ['ronaldoamulya@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=4)
}


def create_pipeline_step(method: Callable) -> Callable:
    def execute_step(**context):
        try:
            logger.info(f"Starting {method.__name__}")
            pipeline = TrainingPipeline()
            method(pipeline)
            logger.info(f"Successfully completed {method.__name__}")
        except Exception:
            logger.exception(f"Failed to execute {method.__name__}")
            raise AirflowException(f"Pipeline step {method.__name__} failed")

    return execute_step


with DAG(
    "bioproces_training_pipeline",
    default_args=default_args,
    description="Bioprocess Training Pipeline",
    schedule_interval="@weekly",
    start_date=datetime(2024, 11, 17),
    catchup=False,
    tags=['bioprocess', 'training', 'ml'],
    max_active_runs=1
) as dag:
    start = EmptyOperator(
        task_id='start',
        dag=dag
    )

    data_ingestion = PythonOperator(
        task_id="data_ingestion",
        python_callable=create_pipeline_step(TrainingPipeline.start_data_ingestion),
        provide_context=True,
        dag=dag
    )

    data_preprocessing = PythonOperator(
        task_id="data_preprocessing",
        python_callable=create_pipeline_step(TrainingPipeline.start_data_transformation),
        provide_context=True,
        dag=dag
    )

    model_training = PythonOperator(
        task_id='model_training',
        python_callable=create_pipeline_step(
            TrainingPipeline.start_model_training
        ),
        provide_context=True,
        dag=dag
    )

    model_evaluation = PythonOperator(
        task_id='model_evaluation',
        python_callable=create_pipeline_step(TrainingPipeline.start_model_evaluation),
        provide_context=True,
        dag=dag
    )

    end = EmptyOperator(
        task_id='end',
        dag=dag
    )


start >> data_ingestion >> data_preprocessing >> model_training >> model_evaluation >> end