import torch
from torch_geometric.data import Data, DataLoader
import json
import os
from query_data import *
import dgl
import pygraphviz as pgv

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

def build_hetero_graph(news, tweets, retweets, users, lab):
    
    if lab == 'fake':
        lab = 0
    else:
        lab = 1
    
    if (len(news['filename']) != 0) and (len(tweets['filename']) !=0) and (len(retweets['filename']) !=0) and (len(users['filename']) !=0):
        for cle in news['filename'].keys():
            val = cle

        News_to_Tweets = [0]*len(tweets['filename'])
        Tweets_to_News = []
        Tweets_to_Retweets = []
        Retweets_to_Tweets = []
        Tweets_to_Users = []
        Users_to_Tweets = []
        x_dict = {'News': [], 'Tweets': [], 'Retweets': [], 'Users': []}
        x_dict['News'].append(news['image_text_features'][str(val)])
        i = 0
        l = 0
        k = 0
        for j in range(len(tweets['filename'])):
            x_dict['Tweets'].append(tweets['text'][f'{j}'])
            Tweets_to_News.append(i)
            if (str(tweets['id'][f'{j}']) in retweets['tweet_id'].values()) == True:
                ind = list(retweets['tweet_id'].keys())[list(retweets['tweet_id'].values()).index(str(tweets['id'][f'{j}']))]
                x_dict['Retweets'].append(retweets['description_encoded'][f'{ind}'])
                Tweets_to_Retweets.append(i)
                Retweets_to_Tweets.append(l)
                l = l + 1
                
            if (str(tweets['id'][f'{j}']) in users['tweet_id'].values()) == True:
                ind = list(users['tweet_id'].keys())[list(users['tweet_id'].values()).index(str(tweets['id'][f'{j}']))]
                x_dict['Users'].append(users['description'][f'{ind}'])
                Tweets_to_Users.append(i)
                Users_to_Tweets.append(k)
                k = k + 1
            i = i + 1
    
        graph = dgl.heterograph({
           ('News', 'mentionned by', 'Tweets'): (torch.tensor(News_to_Tweets), torch.tensor(Tweets_to_News)),
           ('Tweets', 'spread by', 'Retweets'): (torch.tensor(Tweets_to_Retweets), torch.tensor(Retweets_to_Tweets)),
           ('Tweets','tweeted by', 'Users'): (torch.tensor(Tweets_to_Users), torch.tensor(Users_to_Tweets)),
           ('Tweets', 'mentionned', 'News'): (torch.tensor(Tweets_to_News), torch.tensor(News_to_Tweets)),
           ('Retweets', 'spread', 'Tweets'): (torch.tensor(Retweets_to_Tweets), torch.tensor(Tweets_to_Retweets)),
           ('Users','tweeted', 'Tweets'): (torch.tensor(Users_to_Tweets), torch.tensor(Tweets_to_Users)),
        })
        
        graph.nodes['News'].data['Text'] = torch.tensor(x_dict['News'])
        graph.nodes['Tweets'].data['Text'] = torch.tensor(x_dict['Tweets'])
        graph.nodes['Retweets'].data['Text'] = torch.tensor(x_dict['Retweets'])
        graph.nodes['Users'].data['Text'] = torch.tensor(x_dict['Users'])


        return (graph, lab)
    else:
        return None, None

def get_all_graphs(source):
    
    graphs = []
    labels = []
    for label in ['fake', 'real']:
        
        path = get_path()
        load_path = "XXX" # processed_data_path
        list_news_files = get_news_files(path, source, label)
        os.chdir(load_path + source + '//' + label + '//')
        for news in list_news_files:
            print(news)
            os.chdir(load_path + source + '//' + label + '//' + news)

            news_content, tweets, retweets, users = load_data(label)

            graph, lab = build_hetero_graph(news_content, tweets, retweets, users, label)
            
            if graph != None:
                graphs.append(graph)
                labels.append(lab)
                
    return graphs, labels