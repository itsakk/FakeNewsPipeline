import json
import re 
import networkx as nx
import pandas as pd
import numpy as np
import os
from PIL import Image

def get_path():
    load_path = "XXX"
    return load_path

def get_news_files(path, news_source, label):
    path = path + news_source + '\\' + label
    os.chdir(path)
    list_news_files = os.listdir(path)
    return list_news_files

def get_news_content(path, news_source, label, filename):
    path = path + '\\' + news_source + '\\' + label + '\\' + filename + '\\'
    os.chdir(path)
    if os.path.isfile('news content.json') == True:
        f = open(f'news content.json',)
        data = json.load(f)
        keys = ['url', 'images', 'top_img', 'keywords', 'canonical_link', 'meta_data', 'movies', 'publish_date', 'summary']
        for key in keys:
            data.pop(key)
        news_content_df = pd.DataFrame.from_dict([data])
        news_content_df['filename'] = filename
        return news_content_df[['text', 'filename']]
    
def get_tweets(path, news_source, label, filename):
    path = path + '\\' + news_source + '\\' + label + '\\' + filename + '\\' + 'tweets'
    os.chdir(path)
    list_tweets = os.listdir(path)
    tweet = pd.read_json(list_tweets[0], typ = 'series')
    tweet = tweet[['created_at', 'id', 'text']]
    df = pd.DataFrame([tweet])
    for tweet_id in list_tweets[1:]:
        tweet = pd.read_json(tweet_id, typ = 'series')
        df = df.append(tweet, ignore_index = True)
    return df[['id', 'text']]

def get_retweets(path, news_source, label, filename):
    path = path + '\\' + news_source + '\\' + label + '\\' + filename + '\\' +'retweet'
    os.chdir(path)
    list_retweets = os.listdir(path)
    retweet = pd.read_json(list_retweets[0], typ = 'series')
    rts = pd.DataFrame([retweet['retweeted_status']['user']])
    rts['tweet_id'] = list_retweets[0][:-5]
    for tweet_id in list_retweets[1:]:
        retweet = pd.read_json(tweet_id, typ = 'series')
        rt = pd.DataFrame([retweet['retweeted_status']['user']])
        rt['tweet_id'] = tweet_id[:-5]
        rts = rts.append(rt, ignore_index = True)
    return rts[['description', 'tweet_id']]

def get_user_profiles(path, news_source, label, filename):
    path = path + '\\' + news_source + '\\' + label + '\\' + filename + '\\' + 'user_profile'
    if os.path.isdir(path) == True:
        os.chdir(path)
        list_users = os.listdir(path)
        users = pd.read_json(list_users[0], typ = 'series')
        users = pd.DataFrame([users.data])
        users['tweet_id'] = list_users[0][:-5]
        for tweet_id in list_users:
            try:
                user_profile = pd.read_json(tweet_id, typ = 'series')
                user_profile = pd.DataFrame([user_profile.data])
                user_profile['tweet_id'] = tweet_id[:-5]
                users = users.append(user_profile, ignore_index = True)
            except:
                pass
        return users[['description', 'tweet_id']]

def get_all_images(news_source, label, filename):
    list_images = []
    for name in filename:
        try:
            path = get_path()
            path = path + news_source + '\\' + label + '\\' + name + '\\'
            os.chdir(path)
            img = Image.open(f'{re.sub("[^0-9]", "", name)}.png')
            list_images.append((img, name))
        except:
            list_images.append((None, name))
    return list_images