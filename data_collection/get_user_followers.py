import requests
import os
import json
import time

load_path = "XXX"

def auth():
    bearer_token = "<your_bearer_token>"
    return bearer_token

def get_news_files(news_source, label):
    path = load_path + news_source + '//' + label
    os.chdir(path)
    list_news_files = os.listdir(path)
    return list_news_files

def get_ids(news_source, filename, label):
    path = load_path + news_source + '//' + label + '//' + filename + '//' + 'tweets'
    os.chdir(path)
    list_tweets = os.listdir(path)
    user_ids = []
    for tweet_id in list_tweets:
        f = open(tweet_id,)
        tweet = json.load(f)
        user_ids.append(str(tweet['user']['id']))
    return user_ids, list_tweets

def create_url(user_id):
    url = 'https://api.twitter.com/2/users/{}/followers'.format(user_id)
    return url

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

def connect_to_endpoint(url, headers):
    connection_timeout = 30
    start_time = time.time()
    while True:
        try:
            response = requests.request("GET", url, headers=headers)
            
            if response.status_code == 403:
                response = []
                return response
            
            if response.status_code == 404:       
                response = []
                return response
            
            if response.status_code == 429:
                print('sleeping for 15 minutes')        
                time.sleep(900)
                response = requests.request("GET", url, headers=headers)

                if response.status_code == 403:
                    response = []
                    return response

                if response.status_code == 404:
                    response = []
                    return response
            else:
                return response.json()
            
        except requests.ConnectionError as ce:
            if time.time() > start_time + connection_timeout:
                raise ('ConnectionError: Could not connect within %s seconds')
            else:
                time.sleep(1)
    
def save_to_json_file(json_response, news_source, label, filename, tweet_id):
    if len(json_response) !=0:
        if not os.path.exists(load_path + news_source + '//' + label + '//' + filename + '//' + 'user_followers'):
            os.mkdir(load_path + news_source + '//' + label + '//' + filename + '//' + 'user_followers')
        with open(load_path + news_source + '//' + label + '//' + filename + '//' + 'user_followers' +'//' + tweet_id, 'w') as outfile:
                        json.dump(json_response, outfile)
    else:
        pass
    
def main():
    bearer_token = auth()
    labels = ['fake', 'real']
    news_sources = ['politifact']
    for source in news_sources:
        for label in labels:
            list_news_files = get_news_files(source, label)
            for filename in list_news_files: #[64:]
                print(filename)
                try:
                    ids, list_tweets = get_ids(source, filename, label)
                    for i, ID in enumerate(ids):
                        url = create_url(ID)
                        headers = create_headers(bearer_token)
                        json_response = connect_to_endpoint(url, headers)
                        save_to_json_file(json_response, source, label, filename, list_tweets[i])
                except:
                    pass
            print(f'user followers related to {label} news of {source} collected')
        
if __name__ == "__main__":
    main()