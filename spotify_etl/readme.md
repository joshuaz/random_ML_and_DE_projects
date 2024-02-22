This project creates a simple ETL(Extract, Transform, Load) pipeline of my Spotify listening history and sends the data to SQLite.

## Files
`authorization_code_flow.py`
- this python script calls from Spotify’s API to read the data
- It uses Selenium to open a web broswer with my Spotify username/password to grant access permitting an authorization code, which is then used to an access token for the API
- Once it receives the data, it performs some basic data quality checks before sending it to SQLite (or mysql if you choose to uncomment it)

`docker-compose.yaml`
- 
`spotify_dag.py`
