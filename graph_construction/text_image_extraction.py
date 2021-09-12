from query_data import *
from bert_embedding import word_embeddings
from image_extraction import *
import os
import json
import torch

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
    news_df = news_df.reset_index()
    images = get_images_features(path, source, label, list_news_files)
    news_df = news_df.merge(images, on="filename")
    news_df['image_text_features'] = news_df[['text', 'image_feature']].apply(lambda row: np.concatenate((row['text'], row['image_feature'])), axis = 1)
    news_df.drop(news_df.columns.difference(['image_text_features', 'filename']), 1, inplace=True)
    
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
    tweets_df = tweets_df.reset_index()
    tweets_df.drop(tweets_df.columns.difference(['text', 'filename']), 1, inplace=True)
    tweets_df['text'] = tweets_df['text'].apply(lambda x: np.concatenate((x, [0]*4096)))
    return tweets_df

def get_retweets_encoded(path, source, label, list_news_files):
    retweets_df = get_retweets(path, source, label, list_news_files[0])
    retweets_df['filename'] = list_news_files[0]
    description_encoded = word_embeddings(retweets_df.description.values)
    for news in list_news_files[1:]:
        try: 
            rt = get_retweets(path, source, label, news)
            rt['filename'] = news
            rt_desc = word_embeddings(rt.description.values, 20)
            description_encoded = np.concatenate((description_encoded, rt_desc), axis=0)
            retweets_df = retweets_df.append(rt)
        except:
            pass
    
    retweets_df['description_encoded'] = description_encoded.tolist()
    retweets_df = retweets_df.reset_index()
    retweets_df.drop(retweets_df.columns.difference(['description_encoded', 'filename']), 1, inplace=True)
    retweets_df['description_encoded'] = retweets_df['description_encoded'].apply(lambda x: np.concatenate((x, [0]*4096)))
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
    users_df = users_df.reset_index()
    users_df.drop(users_df.columns.difference(['description', 'filename']), 1, inplace=True)
    users_df['description'] = users_df['description'].apply(lambda x: np.concatenate((x, [0]*4096)))    
    return users_df

if __name__ == "__main__":
    
    source = "" # politifact or gossipcop
    path = "XXX"
    os.chdir('//media//spinner//armand//')
    if not os.path.isdir('processed_'+source):
        os.mkdir('processed_' + source) 
        os.mkdir('processed_' + source  + '//fake')
        os.mkdir('processed_' + source + '//real')
        
    labels = ['fake', 'real']
    
    for label in labels:
        path = get_path()
        list_news_files = get_news_files(path, source, label)
        
        
        news_encoded = get_news_one_hot_encoding(path, source, label, list_news_files)
        users_encoded = get_user_profiles_encoded(path, source, label, list_news_files)
        retweets_encoded = get_retweets_encoded(path, source, label, list_news_files)
        users_encoded = get_user_profiles_encoded(path, source, label, list_news_files)

        save_path = path + source + '_2' + '//' + label + '//'
        for news in list_news_files:
            os.chdir(save_path)
            os.mkdir(f'{news}')
            os.chdir(f'{news}')
            news_encoded[news_encoded['filename'] == news].to_json(r'news_embedding.json', index = 'true')
            tweets_encoded[tweets_encoded['filename'] == news].to_json(r'tweets_embedding.json', index = 'true')
            retweets_encoded[retweets_encoded['filename'] == news].to_json(r'retweets_embedding.json', index = 'true')
            users_encoded[users_encoded['filename'] == news].to_json(r'users_embedding.json', index = 'true')