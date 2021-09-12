import torch
import pandas as pd
from transformers import BertTokenizer
from sklearn.decomposition import PCA
from transformers import BertModel

def use_gpu():
    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device

def load_bert_tokenizer():
    # Load Bert tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    return tokenizer

def get_inputs(sentences, tokenizer, device, max_length):
    # Tokenize all of the sentences and map the tokens to their segments IDs.
    input_ids = []
    attention_masks = []

    # For every tweet
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_length,           # Pad & truncate all sentences.
                            padding = 'max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                            truncation = True
                       ).to(device)

        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])


    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    segment_ids = torch.ones_like(input_ids)

    return input_ids, segment_ids

def get_model(device):
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states = True, # Whether the model returns all hidden-states.
                                      )
    model.to(device)

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    return model

def get_embeddings(model, input_ids, segment_ids):
    
    with torch.no_grad():
        outputs = model(input_ids, segment_ids)
        hidden_states = outputs[2]

    token_vecs = hidden_states[-2]
    # Calculate the average of all 64 token vectors.
    sentences_embedding = torch.mean(token_vecs, dim=1)
    return sentences_embedding

def word_embeddings(text_content, max_length):
    
    # change device to gpu
    device = use_gpu()
    
    # get bert tokenizer
    tokenizer = load_bert_tokenizer()
    
    # load model
    
    model = get_model(device)

    input_ids, segment_ids = get_inputs(text_content, tokenizer, device, max_length)

    # get embeddings
    
    sentences_embedding = get_embeddings(model, input_ids, segment_ids)

    return sentences_embedding.cpu().numpy()

def visualize_embeddings(sentences_embedding, text):
    # function to visualize embeddings using dimensionnality reduction
    pca = PCA(n_components= 10)
    y = pca.fit_transform(sentences_embedding.cpu())

    from sklearn.manifold import TSNE

    y = TSNE(n_components=2).fit_transform(y)
    import plotly as py
    import plotly.graph_objs as go
    data = [
        go.Scatter(
            x=[i[0] for i in y],
            y=[i[1] for i in y],
            mode='markers',
            text=[i for i in text],
        marker=dict(
            size=16,
            color = [len(i) for i in text], #set color equal to a variable
            opacity= 0.8,
            colorscale='Viridis',
            showscale=False
        )
        )
    ]
    layout = go.Layout()
    layout = dict(
                  yaxis = dict(zeroline = False),
                  xaxis = dict(zeroline = False)
                 )
    fig = go.Figure(data=data, layout=layout)
    fig.show()