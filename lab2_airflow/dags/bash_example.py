# dags/bash_example.py
from datetime import datetime
import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

local_tz = pendulum.timezone("America/New_York")

with DAG(
    dag_id="bash",
    start_date=datetime(2024, 1, 1, tzinfo=local_tz),
    schedule="@daily",
    catchup=False,
    default_args={"owner": "you"},
    tags=["lab1", "bash"],
) as dag:

    hello = BashOperator(
        task_id="hello",
        bash_command='echo "Hello from Bash at $(date)"'
    )

    list_dir = BashOperator(
        task_id="list_dir",
        bash_command="ls -la /opt/airflow/dags"
    )

    hello >> list_dir
# End of file