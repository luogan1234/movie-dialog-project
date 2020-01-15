import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.text_cnn_layer import TextCNN
from models.mlp_layer import MLP

class PretrainModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.textcnn = TextCNN(config.feature_dim // len(config.filter_sizes), config.filter_sizes, config.dialog_dim)
        self.fc = MLP(config.feature_dim, config.num_classes)

    def forward(self, input):
        out = self.textcnn(input)
        out = self.fc(out)
        return out