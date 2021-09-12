import torch
import pandas as pd
import networkx as nx
import torch_geometric
from torch_geometric.utils.convert import from_networkx
import os
import json
from query_data import *
from torch_geometric.data import InMemoryDataset

def build_graph_data(news_content, tweets, retweets, users, lab):
    
    if lab == 'fake':
        lab = 0
    else:
        lab = 1
        
    if len(news_content['filename']) != 0:
        G = nx.Graph()

        G.add_node(0, x = news_content['text']['0'], y = lab)

        for i in range(len(tweets['filename'])):

            G.add_node(f'tweet_{i}', x = tweets['text'][f'{i}'])
            G.add_edge(0, f'tweet_{i}')

            if (str(tweets['id'][f'{i}']) in retweets['tweet_id'].values()) == True:

                ind = list(retweets['tweet_id'].keys())[list(retweets['tweet_id'].values()).index(str(tweets['id'][f'{i}']))]

                G.add_node(f'retweet_'+ind, x = retweets['description_encoded'][ind])
                G.add_edge(f'tweet_{i}', f'retweet_'+ind)

            if (str(tweets['id'][f'{i}']) in users['tweet_id'].values()) == True:

                ind = list(users['tweet_id'].keys())[list(users['tweet_id'].values()).index(str(tweets['id'][f'{i}']))]

                G.add_node(f'user_profile_'+ind, x = users['description'][ind])
                G.add_edge(f'tweet_{i}', f'user_profile_'+ind)

        return from_networkx(G) 
        
def load_data(label):
    
    f = open(f'retweets_embedding.json',)
    retweets = json.load(f)

    f = open(f'tweets_embedding.json',)
    tweets = json.load(f)
    
    f = open(f'news_embedding.json',)
    news = json.load(f)

    f = open(f'users_embedding.json',)
    users = json.load(f)
    
    return news, tweets, retweets, users
    
    
def get_all_graphs():
    
    labels = ['fake', 'real']
    source = 'politifact'
    for label in labels:
        
        path = get_path()
        list_news_files = get_news_files(path, source, label)
        load_path_data = "XXX"
        save_path_data = "XXX"
        os.chdir(save_path_data + source + '//' + label + '//')

        for news in list_news_files:
            print(news)
            os.chdir(load_path_data + source + '//' + label + '//' + news)
            news_content, tweets, retweets, users = load_data(label)

            data = build_graph_data(news_content, tweets, retweets, users, label)
            
            os.chdir(save_path_data + source + '//')

            if data != None:
                torch.save(data, f'{news}.pt')
    
if __name__ == '__main__':
    data = get_all_graphs()