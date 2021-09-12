from dgl.data import DGLDataset
from build_hetero_graph import get_all_graphs
import os
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
import pickle

class FakeNewsDataset(DGLDataset):

    def __init__(self, raw_dir=None, 
                 force_reload=False, verbose=False):
        super(FakeNewsDataset, self).__init__(name='fakenews',
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)
    def process(self):
        self.graphs, self.label = get_all_graphs(self.raw_dir)

    def __getitem__(self, idx):
        return self.graphs[idx], self.label[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.graphs)