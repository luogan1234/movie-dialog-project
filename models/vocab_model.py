import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.text_cnn_layer import TextCNN
from models.mlp_layer import MLP

class VocabModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_embeddings = nn.Embedding(len(config.vocabs)+1, config.vocab_dim)
        self.vocab_embeddings.weight.data[0].zero_()
        self.textcnn1 = TextCNN(config.dialog_dim // len(config.filter_sizes), config.filter_sizes, config.vocab_dim)
        self.textcnn2 = TextCNN(config.feature_dim // len(config.filter_sizes), config.filter_sizes, config.dialog_dim)
        self.fc = MLP(config.feature_dim, config.num_classes)

    def forward(self, input):
        xs = self.vocab_embeddings(input)
        outs = []
        for x in xs:
            outs.append(self.textcnn1(x))
        out = torch.stack(outs)
        out = self.textcnn2(out)
        out = self.fc(out)
        return out