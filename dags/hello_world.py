from datetime import datetime
from airflow import DAG
from airflow.operators.dummy import DummyOperator

with DAG(
    dag_id='hello_world',
    description='Un DAG de test simple',
    schedule_interval='@once',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['test'],
) as dag:
    
    task1 = DummyOperator(task_id='start')
    task2 = DummyOperator(task_id='process')
    task3 = DummyOperator(task_id='end')
    
    task1 >> task2 >> task3
