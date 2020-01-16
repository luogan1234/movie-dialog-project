import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.text_cnn_layer import TextCNN
from models.mlp_layer import MLP

class PretrainModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed = np.load('dataset/embed_pca.npz', allow_pickle=True)['emb'][:, :config.dialog_dim]
        self.dialog_embeddings = nn.Embedding.from_pretrained(torch.tensor(embed, dtype=torch.float32), freeze=True)
        self.textcnn = TextCNN(config.feature_dim // len(config.filter_sizes), config.filter_sizes, config.dialog_dim)
        self.fc = MLP(config.feature_dim, config.num_classes)

    def forward(self, input):
        x = self.dialog_embeddings(input)
        out = self.textcnn(x)
        out = self.fc(out)
        return out