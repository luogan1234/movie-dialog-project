import numpy as np
import torch

class Config:
    def __init__(self, store, use_cpu, epochs):
        self.model_name = store.model_name
        task = store.task
        if task in ['IMDB']:
            self.task = 'regression'
        if task in ['genre', 'gender']:
            self.task = 'classification'
        assert self.task
        
        self.vocabs = store.vocabs
        self.vocab_dim = 128
        self.dialog_dim = 768
        self.feature_dim = 256
        self.filter_sizes = (2, 3, 4, 5)
        self.epochs = epochs
        self.batch_size = 32
        self.lr = 1e-3
        self.num_classes = store.num_classes if self.task =='classification' else 1
        self.use_gpu = not use_cpu
        self.save_dir = 'result/'
    
    def to_torch(self, x):
        if self.use_gpu:
            return torch.from_numpy(x).cuda()
        else:
            return torch.from_numpy(x)