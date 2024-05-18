import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers):
        super(DNN, self).__init__()

        self.input_dim = input_dim
        
        seq = []
        seq.append(nn.Linear(input_dim, hidden_dim))
        seq.append(nn.ReLU())
        for _ in range(num_hidden_layers):
            seq.append(nn.Linear(hidden_dim, hidden_dim))
            seq.append(nn.ReLU())
        seq.append(nn.Linear(hidden_dim, output_dim))
        
        self.seq = nn.Sequential(*seq)
        
        
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.seq(x)
        x = F.softmax(x, dim=1)
        return x