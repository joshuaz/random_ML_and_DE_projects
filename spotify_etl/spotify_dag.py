from datetime import datetime, timedelta

from airflow.models import DAG
from airflow.operators.bash import BashOperator

args = {
    'owner': 'Joshua',
    'start_date': datetime(2024,2,22),
    'email': ['EMAIL'],
    'email_on_failure': True,
    'retry_delay' : timedelta(seconds=60),
    'retries' : 3
}

dag = DAG(
    dag_id='spotify_dag',
    default_args=args,
    schedule_interval='12 6 * * *',
    tags = ['spotify', 'practice', 'etl']
)

# Install Selenium task
install_packages = BashOperator(
    task_id='install_packages',
    bash_command='pip3 install -r /opt/airflow/data/requirements.txt',
    dag=dag,
)


update_data = BashOperator(
    task_id='spotify_dag',
    bash_command='python3 /opt/airflow/data/authorization_code_flow_w_credentials.py',
    dag=dag
)

# Set up task dependencies
install_packages >> update_data