import argparse
from FakeNewsDataset import FakeNewsDataset
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import torch.nn as nn
from dgl.dataloading import GraphDataLoader
import dgl
import torch
from dgl.data.utils import split_dataset
from tqdm import tqdm
from eval_helper import eval_deep
import matplotlib.pyplot as plt
import numpy as np
torch.cuda.empty_cache()
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
# hyper-parameters
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--epochs', type=int, default=30, help='maximum number of epochs')

args = parser.parse_args()    
torch.manual_seed(args.seed)

class RGCN(nn.Module):
    def __init__(self, in_feats_list, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(in_feats, hid_feats, aggregator_type= 'pool')
             for rel, in_feats in zip(rel_names, in_feats_list)}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(hid_feats, hid_feats, aggregator_type= 'pool')
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs is features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, drop_ratio, n_classes, rel_names):
        super().__init__()
        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(in_dim[0][0], hidden_dim)
        self.lin3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.dropout = nn.Dropout(p = drop_ratio)
        self.classify = nn.Linear(hidden_dim*2, n_classes)
        
    def forward(self, g):
        h = g.ndata['Text']
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = 0
            for ntype in h.keys():
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
            hg = F.relu(self.lin1(hg))
            hg = self.dropout(hg)
            news = torch.stack([news for news in g.ndata['Text']['News']])
            news = F.relu(self.lin2(news))
            news = self.dropout(news)
            hg = torch.cat([hg, news], dim=1)
            hg = self.dropout(hg)
        return F.log_softmax(self.classify(hg), dim=-1)

@torch.no_grad()
def compute_test(loader, verbose=False):
    model.eval()
    loss_test = 0.0
    out_log = []
    for data, labels in loader:
        data = data.to(args.device)
        out = model(data).cpu()
        if verbose:
            print(F.softmax(out, dim=1).cpu().numpy())
        out_log.append([F.softmax(out, dim=1), labels])
        loss_test += F.nll_loss(out, labels).item()
    return eval_deep(out_log, loader, hetero = True), loss_test

dataset = FakeNewsDataset(raw_dir = 'gossipcop')

train_set, val_set, test_set = split_dataset(dataset, frac_list = [0.7, 0.1, 0.2], shuffle = True, random_state = args.seed)

train_loader = GraphDataLoader(
    train_set,
    batch_size=args.batch_size,
    drop_last=False,
    shuffle=True)

valid_loader = GraphDataLoader(
    val_set,
    batch_size=args.batch_size,
    drop_last=False,
    shuffle=True)

test_loader = GraphDataLoader(
    test_set,
    batch_size=args.batch_size,
    drop_last=False,
    shuffle=True)
    
args.features_size = [(4864, 768), (768, 768), (768, 4864), (768, 768), (768, 768), (768, 768)]
args.num_classes = 2
args.rel_names = ['mentionned by', 'spread', 'mentionned', 'spread by', 'tweeted by', 'tweeted']

model = HeteroClassifier(args.features_size, args.nhid, args.drop_ratio, args.num_classes, args.rel_names)
model = model.to(args.device)  
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Model training   
if __name__ == '__main__':
    
    min_loss = 1e10
    val_loss_values = []
    best_epoch = 0
    acc_train_values = []
    acc_val_values = []
    train_loss_values = []
    val_loss_values = []
    
    model.train()
    for epoch in tqdm(range(args.epochs)):
        loss_train = 0.0
        out_log = []
        for i, (batched_graph, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            batched_graph = batched_graph.to(args.device)
            out = model(batched_graph)
            out = out.cpu()
            loss = F.nll_loss(out, labels)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            out_log.append([F.softmax(out, dim=1), labels])
        acc_train, _, _, _, recall_train, _ = eval_deep(out_log, train_loader, hetero = True)
        [acc_val, _, _, _, recall_val, _], loss_val = compute_test(valid_loader)
        print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
    			  f' recall_train: {recall_train:.4f},'
    			  f' loss_val: {loss_val:.4f},'
    			  f' recall_val: {recall_val:.4f},'
                  f' acc_val: {acc_val:.4f}')
        train_loss_values.append(loss_train)
        val_loss_values.append(loss_val)
        acc_train_values.append(acc_train)
        acc_val_values.append(acc_val)
    plt.figure(figsize = [15, 8])
    plt.plot(np.arange(args.epochs), train_loss_values, color = 'b', label = 'Train')
    plt.plot(np.arange(args.epochs), val_loss_values, color = 'r', label = 'Validation')
    plt.xlabel('epochs')
    plt.ylabel('Loss function')   
    plt.legend()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.figure(figsize = [15, 8])
    plt.plot(np.arange(args.epochs), acc_train_values, color = 'b', label = 'Train')
    plt.plot(np.arange(args.epochs), acc_val_values, color = 'r', label = 'Validation')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    [acc, f1_macro, f1_micro, precision, recall, ap], test_loss = compute_test(test_loader, verbose = False)
    print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f}, '
    		  f'precision: {precision:.4f}, recall: {recall:.4f}')
