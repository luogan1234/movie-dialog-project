import numpy as np
import torch

class Config:
    def __init__(self, store, use_cpu, epochs, log_interval):
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
        self.log_interval = log_interval
    
    def to_torch(self, x):
        if self.use_gpu:
            return torch.from_numpy(x).cuda()
        else:
            return torch.from_numpy(x)
    
    def to_str(self):
        return '_'.join([self.model_name, self.task, 'vd'+str(self.vocab_dim), 'dd'+str(self.dialog_dim), 'fd'+str(self.feature_dim),
                         'e'+str(self.epochs), 'bs'+str(self.batch_size), 'lr'+str(self.lr), 'fs'+'-'.join([str(x) for x in self.filter_sizes])])