from query_data import *
from bert_embedding import word_embeddings
import os
import json
import torch
import numpy as np
import pandas as pd

def get_news_one_hot_encoding(path, source, label, list_news_files):
    news_df = get_news_content(path, source, label, list_news_files[0])
    
    if isinstance(news_df, pd.DataFrame) != True:
        news_df = pd.DataFrame()
    else:
        news_df['text'] = word_embeddings(news_df.text.values, 512).tolist()
        
    for news in list_news_files[1:]:
        news_content = get_news_content(path, source, label, news)
        if isinstance(news_content, pd.DataFrame) == True:
            news_content['text'] = word_embeddings(news_content.text.values, 512).tolist()
            news_df = news_df.append(news_content)

    return news_df

def get_tweets_encoded(path, source, label, list_news_files):
    tweets_df = get_tweets(path, source, label, list_news_files[0])
    tweets_df['filename'] = list_news_files[0]
    text_encoded = word_embeddings(tweets_df.text.values, 30).tolist()
    for news in list_news_files[1:]:
        try: 
            tweet = get_tweets(path, source, label, news)
            tweet['filename'] = news
            tweet_desc = word_embeddings(tweet.text.values, 30)
            text_encoded = np.concatenate((text_encoded, tweet_desc), axis=0)
            tweets_df = tweets_df.append(tweet)
        except:
            pass
    tweets_df['text'] = text_encoded.tolist()
    return tweets_df

def get_retweets_encoded(path, source, label, list_news_files):
    retweets_df = get_retweets(path, source, label, list_news_files[0])
    retweets_df['filename'] = list_news_files[0]
    description_encoded = word_embeddings(retweets_df.description.values, 30)
    for news in list_news_files[1:]:
        try: 
            rt = get_retweets(path, source, label, news)
            rt['filename'] = news
            rt_desc = word_embeddings(rt.description.values, 30)
            description_encoded = np.concatenate((description_encoded, rt_desc), axis=0)
            retweets_df = retweets_df.append(rt)
        except:
            pass
        
    retweets_df['description_encoded'] = description_encoded.tolist()    
    return retweets_df

def get_user_profiles_encoded(path, source, label, list_news_files):
    users_df = get_user_profiles(path, source, label, list_news_files[0])
    users_df['filename'] = list_news_files[0]
    description_encoded = word_embeddings(users_df.description.values, 20).tolist()
    for news in list_news_files[1:]:
        try: 
            user = get_user_profiles(path, source, label, news)
            user['filename'] = news
            user_desc = word_embeddings(user.description.values, 20)
            description_encoded = np.concatenate((description_encoded, user_desc), axis=0)
            users_df = users_df.append(user)
        except:
            pass

    users_df['description'] = description_encoded.tolist()    
    return users_df

if __name__ == "__main__":
    
    source = 'politifact'
    os.chdir('C:\\Users\\arman\\Documents\\Bristol_DataScience_MSc\\MSc_Thesis\\Data\\')
    if not os.path.isdir('processed_'+source +'_3'):
        os.mkdir('processed_' + source) 
        os.mkdir('processed_' + source + '//' + 'fake')
        os.mkdir('processed_' + source + '//' + 'real')
        
    labels = ['fake', 'real']
    
    for label in labels:
        path = get_path()
        list_news_files = get_news_files(path, source, label)
        
        
        news_encoded = get_news_one_hot_encoding(path, source, label, list_news_files)
        users_encoded = get_user_profiles_encoded(path, source, label, list_news_files)
        retweets_encoded = get_retweets_encoded(path, source, label, list_news_files)
        users_encoded = get_user_profiles_encoded(path, source, label, list_news_files)

        save_path = path + source + '//' + label + '//'
        for news in list_news_files:
            os.chdir(save_path)
            os.mkdir(f'{news}')
            os.chdir(f'{news}')
            news_encoded[news_encoded['filename'] == news].to_json(r'news_embedding.json', index = 'true')
            tweets_encoded[tweets_encoded['filename'] == news].to_json(r'tweets_embedding.json', index = 'true')
            retweets_encoded[retweets_encoded['filename'] == news].to_json(r'retweets_embedding.json', index = 'true')
            users_encoded[users_encoded['filename'] == news].to_json(r'users_embedding.json', index = 'true')
