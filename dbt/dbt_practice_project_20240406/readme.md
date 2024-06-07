1. install dbt. I used a virtual environment
`python3 -m venv dbt_venv`
`source dbt_venv/bin/activate`
`pip install dbt-core dbt-duckdb dbt-bigquery`

2. download data from Kaggle (I downloaded the https://www.kaggle.com/datasets/prishasawhney/mushroom-dataset)
- Create data set `mushrooms`
- create table `mushrooms`

3. create project in BigQuery
- Go to the BigQuery Console to create project 
- go to credential wizard (https://console.cloud.google.com/apis/credentials)
    - Which API are you using? BigQuery API
    - What data will you be accessing? Application data
    - Service account details — Service account name: dbt-user
    - Grant this service account access to the project — Role: BigQuery Admin
    - Grant users access to this service account — Service account admin roles: I put my other gmails
- click on your service account
- Keys
    - Add Key
    - download json and store it somewhere safe

4. Initialize Airflow via Docker

5. In terminal, `dbt init dbt_practice_project_20240406` 

6. I saved my data in the seeds folder. THen I ran `dbt seed` to ensure it was in there.

7. Then I ran `dbt build`, and I saw my data in GCP

8. Or you can run `dbt run --select=mushroom_counts` to run just the one model
