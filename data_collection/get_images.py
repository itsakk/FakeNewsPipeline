import json
import requests
import os
import re

load_path = "XXX"

def get_news_files(news_source, label):
    path = load_path + news_source + '//' +label
    os.chdir(path)
    list_news_files = os.listdir(path)
    return list_news_files

def get_data(news_source, label, filename):
    try:
        path = load_path + news_source + '//' + label + '//' + filename
        os.chdir(path)
        f = open('news content.json',)
        data = json.load(f)
        return data
    except:
        pass
    
def save_image(data, filename):
    try:
        response = requests.get(data['top_img'], timeout = 3)
        file = open(re.sub('[a-z-]', '', filename)+'.png', "wb")
        file.write(response.content)
        file.close()
        return 
    except:
        return
    
def main():
    news_sources = ['politifact', 'gossipcop']
    labels = ['fake', 'real']
    for source in news_sources:
        for label in labels:
            list_news_files = get_news_files(source, label)
            for filename in list_news_files:
                print(filename)
                try:
                    data = get_data(source, label, filename)
                    save_image(data, filename)
                except:
                    pass
            print(f'images related to {label} news of {source} collected')

if __name__ == "__main__":
    main()