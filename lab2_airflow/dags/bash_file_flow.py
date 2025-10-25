# dags/bash_file_flow.py
from datetime import datetime
import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule

# Use the folder you mounted in docker-compose:
#   - ${AIRFLOW_PROJ_DIR:-.}/working_data:/opt/airflow/working_data
DATA_DIR = "/opt/airflow/working_data"

local_tz = pendulum.timezone("America/New_York")

with DAG(
    dag_id="bash_file_flow",
    start_date=datetime(2024, 1, 1, tzinfo=local_tz),
    schedule=None,          # trigger manually
    catchup=False,
    default_args={"owner": "you"},
    tags=["bash", "files", "cleanup"],
    params={
        "filename": "demo.txt",
        "content": "Hello from Airflow!"
    }
) as dag:

    make_dir = BashOperator(
        task_id="make_dir",
        bash_command=f"mkdir -p {DATA_DIR}",
    )

    create_file = BashOperator(
        task_id="create_file",
        bash_command=(
            f'echo "{{{{ params.content }}}}" '
            f'> {DATA_DIR}/{{{{ params.filename }}}}'
        ),
    )

    list_files = BashOperator(
        task_id="list_files",
        bash_command=f"ls -la {DATA_DIR}",
    )

    show_contents = BashOperator(
        task_id="show_contents",
        bash_command=f"cat {DATA_DIR}/{{{{ params.filename }}}}",
    )

    cleanup = BashOperator(
        task_id="cleanup",
        bash_command=f"rm -f {DATA_DIR}/{{{{ params.filename }}}}",
        trigger_rule=TriggerRule.ALL_DONE,  # run even if upstream failed
    )

    # Flow: make dir -> create -> list -> show -> cleanup
    make_dir >> create_file >> list_files >> show_contents >> cleanup
