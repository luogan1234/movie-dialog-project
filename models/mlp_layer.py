import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, in_features // 2)
        self.fc2 = nn.Linear(in_features // 2, in_features // 4)
        self.fc3 = nn.Linear(in_features // 4, in_features // 8)
        self.fc4 = nn.Linear(in_features // 8, out_features)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input):
        input = self.dropout(input)
        x = F.leaky_relu(self.fc1(input))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def __repr__(self):
        return '{} ({} -> {})'.format(self.__class__.__name__, self.in_features, self.out_features)