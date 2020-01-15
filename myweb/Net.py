import torch
import torch.nn.functional as Fun


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = Fun.relu(self.hidden(x))      # activation function for hidden layer we choose sigmoid
        x = self.out(x)
        return x