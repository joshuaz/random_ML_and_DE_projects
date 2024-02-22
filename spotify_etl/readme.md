This project creates a simple ETL(Extract, Transform, Load) pipeline of my Spotify listening history and sends the data to SQLite.

## Files
`authorization_code_flow.py`
- this python script calls from Spotify’s API to read the data
- It uses Selenium to open a web broswer with my Spotify username/password to grant access permitting an authorization code, which is then used to an access token for the API
- Once it receives the data, it performs some basic data quality checks before sending it to SQLite (or mysql if you choose to uncomment it)
 
`docker-compose.yaml`
- To download docker, I opened a terminal and ran `brew install --cask docker`
  - `brew install`: This is the standard Homebrew command to install packages.
  - `cask`: This option indicates that you want to install a GUI application using Homebrew Cask.
- In my root directory, I created a folder called `docker` (i.e., `mkdir docker`)
- Within `docker`, I created three folders: `dags`, `logs`, and `plugins`
- I placed `docker-compose.yaml` in `docker` (obtained from `https://airflow.apache.org/docs/apache-airflow/2.0.2/docker-compose.yaml`)
- to activate airflow I ran `docker-compose up airflow-init` and `docker-compose up`
  
`spotify_dag.py`