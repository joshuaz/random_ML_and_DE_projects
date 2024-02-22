import requests
from base64 import b64encode
from urllib.parse import urlencode, urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from queue import Queue
import pandas as pd
from datetime import datetime, timedelta
import time
# Selenium for opening web browser to grant access
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
# sql
from sqlalchemy import create_engine
import sqlite3

# Values
CLIENT_ID = "INSERT YOUR CLIENT ID HERE"
CLIENT_SECRET = "INSERT YOUR CLIENT SECRET HERE"
REDIRECT_URI = "http://localhost:8080/callback"
SCOPE = "user-read-recently-played"
TOKEN_ENDPOINT = "https://accounts.spotify.com/api/token"
API_ENDPOINT = "https://api.spotify.com/v1/me/player/recently-played?limit=50"
SPOTIFY_USERNAME = "INSERT YOUR SPOTIFY USER NAME HERE"
SPOTIFY_PASSWORD = "INSERT YOUR SPOTIFY PASSWORD HERE"
db_username = ""
db_password = ""
db_host = "localhost"
db_name = "spotify"
sqllite_path = "spotify.db"

# Use a simple web server to handle the callback
class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.server.queue.put(self.path)
        print(f"Callback received: {self.path}")
        self.wfile.write(b"<html><body><h1>Authorization successful. You can close this tab now.</h1></body></html>")


def run_server(queue):
    server_address = ("", 8080)
    httpd = HTTPServer(server_address, CallbackHandler)
    httpd.queue = queue
    print("Starting server on http://localhost:8080")
    httpd.serve_forever()


def get_authorization_code(client_id, redirect_uri, scope):
    # Construct the authorization URL
    authorize_url = "https://accounts.spotify.com/authorize"
    params = {
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'scope': scope,
        'response_type': 'code',
    }
    authorization_url = f"{authorize_url}?{urlencode(params)}"

    # Set up a queue to communicate with the callback server
    queue = Queue()

    # Start a daemon thread to run the callback server
    server_thread = Thread(target=run_server, args=(queue,), daemon=True)

    try:
        # Start the server thread before opening the URL
        server_thread.start()

        # Open the URL in a browser using Selenium
        # Ensure you have ChromeDriver installed
        try:
            print(f"Automatically opening the following URL in a browser:\n{authorization_url}")
            open_url_and_grant_permission(authorization_url, SPOTIFY_USERNAME, SPOTIFY_PASSWORD)
        except Exception as e:
            print(f"Error granting permission automatically using Selenium: {e}")
            print(f"Please go to the following URL and grant permission:\n{authorization_url}")

        # Wait for the callback URL from the user's interaction
        path = queue.get()
        print("Received callback path:", path)

        # Parse the callback URL to extract the authorization code
        query_params = parse_qs(urlparse(path).query)
        authorization_code = query_params.get('code', [None])[0]

        # Check if authorization code is obtained successfully
        if authorization_code:
            print(f"Authorization code obtained in get_authorization_code function: {authorization_code}")
            return authorization_code
        else:
            print("Error: Authorization code not found in the callback URL.")
            return None
    except Exception as e:
        print(f"Error during authorization code retrieval: {e}")
        return None
#    finally:
# # Ensure the server thread is joined even if there's an exception
# # Join the server thread to clean up resources
#         print("Trying to join server thread...")
#         server_thread.join()
#         print("Server thread joined.")


def get_access_token(client_id, client_secret, redirect_uri, authorization_code):
    credentials = f"{client_id}:{client_secret}"
    encoded_credentials = b64encode(credentials.encode()).decode('utf-8')

    headers = {
        'Authorization': f'Basic {encoded_credentials}',
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    data = {
        'grant_type': 'authorization_code',
        'code': authorization_code,
        'redirect_uri': redirect_uri,
    }

    print("Requesting access token...")

    try:
        response = requests.post(TOKEN_ENDPOINT, headers=headers, data=data)
        response.raise_for_status()  # Raise an HTTPError for bad responses
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error during token retrieval: {errh}")
        print(response.text)
        return None
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting during token retrieval: {errc}")
        return None
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error during token retrieval: {errt}")
        return None
    except requests.exceptions.RequestException as err:
        print(f"Request Error during token retrieval: {err}")
        return None

    # Check the response
    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data['access_token']
        print(f"Access token obtained: {access_token}")
        return access_token
    else:
        print(f"Error during token retrieval: {response.status_code}, {response.text}")
        return None


def open_url_and_grant_permission(authorization_url, spotify_username, spotify_password):
    try:
        # Open the URL in a new Chrome browser window
        browser = webdriver.Chrome()  # You need to have ChromeDriver installed
        browser.get(authorization_url)

        # Enter Spotify username
        # username_input = browser.find_element_by_name("username") # for Selenium version 4.2 or older
        username_input = browser.find_element(By.ID, "login-username") # for Selenium version 4.3 or later
        username_input.send_keys(spotify_username)

        # Enter Spotify password
        # password_input = browser.find_element_by_name("password") # for Selenium version 4.2 or older
        password_input = browser.find_element(By.ID, "login-password") # for Selenium version 4.3 or later
        password_input.send_keys(spotify_password)

        # Press Enter to submit the login form
        password_input.send_keys(Keys.RETURN)

        # Wait for some time to allow user interaction
        time.sleep(10)

        # Close the browser window
        #browser.quit()

    except Exception as e:
        print(f"Error during browser automation: {e}")
        if browser:
            browser.quit()


def return_dataframe(access_token):
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    # Retrieve most recently played tracks
    print("Retrieving data...")
    try:
        # add timeline
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        yesterday_unix_timestamp = int(yesterday.timestamp()) * 1000
        API_ENDPOINT_YESTERDAY = f"{API_ENDPOINT}&after={yesterday_unix_timestamp}"
        # make request
        r = requests.get(API_ENDPOINT_YESTERDAY, headers=headers)
        r.raise_for_status()  # Raise an HTTPError for bad responses
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error during data retrieval: {errh}")
        return None
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting during data retrieval: {errc}")
        return None
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error during data retrieval: {errt}")
        return None
    except requests.exceptions.RequestException as err:
        print(f"Request Error during data retrieval: {err}")
        return None
    else:
        # Check the response
        if r.status_code == 200:
            data = r.json()
            song_names = []
            artist_names = []
            played_at_list = []
            timestamps = []

            for song in data["items"]:
                song_names.append(song["track"]["name"])
                artist_names.append(song["track"]["album"]["artists"][0]["name"])
                played_at_list.append(song["played_at"])
                timestamps.append(song["played_at"][0:10])

            song_dict = {
                "song_name": song_names,
                "artist_name": artist_names,
                "played_at": played_at_list,
                "timestamp": timestamps
            }

            song_df = pd.DataFrame(song_dict, columns=["song_name", "artist_name", "played_at", "timestamp"])
            print("Data retrieved successfully.")
            return song_df
        else:
            print(f"Error during data retrieval: {r.status_code}")
            print(r.text)
            return None


def Data_Quality(load_df):
    # Checking if the DataFrame is empty
    if load_df.empty:
        print('No Songs Extracted')
        return False
    
    #Enforcing Primary keys since we do not want duplicates
    if pd.Series(load_df['played_at']).is_unique:
       pass
    else:
        #The Reason for using exception is to immediately terminate the program and avoid further processing
        raise Exception("Primary Key Exception,Data Might Contain duplicates")
    
    #Checking for Nulls in our data frame 
    if load_df.isnull().values.any():
        raise Exception("Null values found")


def load_data():
    # Authorization process
    authorization_code = get_authorization_code(CLIENT_ID, REDIRECT_URI, SCOPE)

    # Retrieve access token
    if authorization_code:
        print(f"Authorization code obtained: {authorization_code}")

        # Retrieve access token
        access_token = get_access_token(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, authorization_code)

        if access_token:
            print(f"Access Token: {access_token}")

            # Now, proceed to fetch user data or perform any other necessary operations.
            df = return_dataframe(access_token)
            print(df)
            return df
        else:
            print("Failed to obtain access token.")
            return None
    else:
        print("Failed to obtain authorization code.")
        return None


def send_to_mysql(db_username, db_password, db_host, db_name, df, table_name):
    con_string = f"mysql+pymysql://{db_username}:{db_password}@{db_host}/{db_name}"
    # initialize enginer and connection
    engine = create_engine(con_string, echo=False)
    try:
        # Try to connect to MySQL
        with engine.connect() as con:
            print("Connected to MySQL")
            # Try to send DataFrame to MySQL
            try:
                df.to_sql(table_name, con=engine, if_exists='append', index=False)
                print("DataFrame successfully sent to MySQL")
            except Exception as df_error:
                print(f"Error sending DataFrame to MySQL: {df_error}")
    except Exception as connection_error:
        print(f"Could not connect to MySQL. Error: {connection_error}")
    

def send_to_sqllite(sqllite_path, df):
    try:
        with sqlite3.connect(sqllite_path) as conn:
            df.to_sql('spotify_songs', conn, index=False, if_exists='append')
        print("Data successfully sent to SQLite")
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    df = load_data()
    print("data are loaded")
    Data_Quality(df)
    print("quality check complete")
    #send_to_mysql(db_username, db_password, db_host, db_name, df, 'spotify_songs')
    #print("sent to mysql")
    send_to_sqllite(sqllite_path, df)
    print("sent to sqllite")
