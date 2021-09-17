# FakeNewsPipeline

In the recent years, social networks became one of the biggest sources of news consumption; people get information on social networks and share what they read with their network, which leads to the wide dissemination of fake news, i.e., news with false information, causing negative effects on the society. The detection, evolution and mitigation of fake news quickly became an important area of research – particularly because of recent events such as US 2016 elections and Covid-19.

This project aims to use the FakeNewsNet dataset to build a model that could classify news, namely predict if a news is fake or true. The dataset is composed of diverse features such as the news content (text, images, videos), social context (user profile, tweets, likes, retweets, ...) and spatio-temporal information (users spatial and temporal information). Different approaches have been explored for fake news detection such as uni-modal and multi-modal machine learning, using state-of-art machine learning models such as graph neural networks.

This project deals with all the steps of the data science pipeline: starting with data collection of fake news data using Twitter API, we then pre-process the data using pre-trained models to extract text and image features. Once the features extracted, we build an advanced machine learning model to classify fake/real news present in social media and train it with the processed features and evaluate its accuracy. The purpose of this thesis is to build a pipeline that could outperform the performance of state-of-art models, such as the UPFD framework

## Dataset

The dataset used is the FakeNewsNet dataset, which can be rietreve using the [Github](https://github.com/KaiDMML/FakeNewsNet) repository.

## Results
![upfd-results](https://user-images.githubusercontent.com/66783741/133689112-517f2b30-bc20-49bd-954e-9bb91e63e52b.PNG)
