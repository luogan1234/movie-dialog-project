import numpy as np
import torch

class Config:
    def __init__(self, store, use_cpu, epochs):
        self.model_name = store.model_name
        self.task = store.task
        assert self.task
        
        self.vocabs = store.vocabs
        self.vocab_dim = 128
        self.dialog_dim = 768
        self.feature_dim = 256
        self.filter_sizes = (3, 5, 7, 9)
        self.epochs = epochs
        self.batch_size = 32
        self.lr = 1e-3
        if self.task == 'IMDB':
            self.num_classes = 10
        if self.task == 'genre':
            self.num_classes = 24
        if self.task != 'gender':
            self.num_class = 2
        self.use_gpu = not use_cpu
        self.save_dir = 'result/'
    
    def to_torch(self, x):
        if self.use_gpu:
            return torch.from_numpy(x).cuda()
        else:
            return torch.from_numpy(x)