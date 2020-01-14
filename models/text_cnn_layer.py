import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TextCNN(nn.Module):
    def __init__(self, num_filters, filter_sizes, input_dim):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (k, input_dim)) for k in filter_sizes])

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input):
        x = input.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(input, conv) for conv in self.convs], 1)
        out = self.fc(out)
        return out
