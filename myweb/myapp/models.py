from __future__ import unicode_literals
from django.db import models
import torch
import torch.nn.functional as Fun

# Create your models here.

class message(models.Model):
    username = models.CharField(max_length=1000)
    password = models.CharField(max_length=1000)
    predict_result = models.CharField(max_length=1000)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = Fun.relu(self.hidden(x))      # activation function for hidden layer we choose sigmoid
        x = self.out(x)
        return x
