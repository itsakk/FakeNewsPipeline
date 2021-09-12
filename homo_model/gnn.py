import argparse
import os
import torch
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, DataListLoader
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, DataParallel
from eval_helper import eval_deep
import time
import matplotlib.pyplot as plt
import numpy as np

class Model(torch.nn.Module):
    def __init__(self, args, concat=False):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.model = args.model
        self.concat = concat
        
        if self.model == 'gcn':
            self.conv1 = GCNConv(self.num_features, self.nhid)
        elif self.model == 'sage':
            self.conv1 = SAGEConv(self.num_features, self.nhid)
        elif self.model == 'gat':
            self.conv1 = GATConv(self.num_features, self.nhid)

        if self.concat:
            self.lin0 = torch.nn.Linear(self.num_features, self.nhid)
            self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)

        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        edge_attr = None
        
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = gmp(x, batch)
        
        if self.concat:
            news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])	
            news = F.relu(self.lin0(news))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.lin1(x))
            
        x = F.log_softmax(self.lin2(x), dim=-1)

        return x


@torch.no_grad()
def compute_test(loader, verbose=False):
	model.eval()
	loss_test = 0.0
	out_log = []
	for data in loader:
		if not args.multi_gpu:
			data = data.to(args.device)
		out = model(data)
		if args.multi_gpu:
			y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
		else:
			y = data.y
		if verbose:
			print(F.softmax(out, dim=1).cpu().numpy())
		out_log.append([F.softmax(out, dim=1), y])
		loss_test += F.nll_loss(out, y).item()
	return eval_deep(out_log, loader), loss_test

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')

# hyper-parameters
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0, help='dropout ratio')
parser.add_argument('--epochs', type=int, default=25, help='maximum number of epochs')
parser.add_argument('--concat', type=bool, default=True, help='whether concat news embedding and graph embedding')
parser.add_argument('--model', type=str, default='sage', help='model type, [gcn, gat, sage]')
args = parser.parse_args()


args.num_classes = 2
args.num_features = 768

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    
data_list = []

os.chdir('C:\\Users\\arman\\Documents\\Bristol_DataScience_MSc\\MSc_Thesis\\Data\\processed_politifact\\')
filenames = os.listdir()

for filename in filenames:
    data = torch.load(filename)
    if data.edge_index.size()[1] != 0:
        data_list.append(torch.load(filename))

num_training = int(len(data_list) * 0.7)
num_val = int(len(data_list) * 0.1)
num_test = len(data_list) - (num_training + num_val)

training_set, validation_set, test_set = random_split(data_list, [num_training, num_val, num_test])

if args.multi_gpu:
	loader = DataListLoader
else:
	loader = DataLoader

train_loader = loader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = loader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = loader(test_set, batch_size=args.batch_size, shuffle=False)

model = Model(args, concat=args.concat)

if args.multi_gpu:
	model = DataParallel(model)
    
model = model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


if __name__ == '__main__':
	# Model training
    min_loss = 1e10
    acc_train_values = []
    acc_val_values = []
    train_loss_values = []
    val_loss_values = []
    best_epoch = 0
    t = time.time()
    model.train()
    for epoch in tqdm(range(args.epochs)):
        loss_train = 0.0
        out_log = []
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            if not args.multi_gpu:
                data = data.to(args.device)
                out = model(data)
            if args.multi_gpu:
                y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
            else:
                y = data.y
                loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            out_log.append([F.softmax(out, dim=1), y])
        acc_train, _, _, _, recall_train, auc_train, _ = eval_deep(out_log, train_loader)
        [acc_val, _, _, _, recall_val, auc_val, _], loss_val = compute_test(val_loader)
        print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
			  f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
			  f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
			  f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')
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
    [acc, f1_macro, f1_micro, precision, recall, auc, ap], test_loss = compute_test(test_loader, verbose=False)
    print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f}, '
		  f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')
    
