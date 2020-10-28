import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from attractiveembedding import AttractiveEmbedding
from transformermodel import TransformerModel


class AttractiveNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, dropout, num_layers):
        super().__init__()
        
        self.embedding = AttractiveEmbedding(input_dim, embedding_dim)
        # self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, dropout=dropout)
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)

        # self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size), torch.zeros(1, 1, self.hidden_layer_size))
        
    def forward(self, x):

        #x = [sent len, batch size]
        
        embedded = self.embedding(x)
        
        #embedded = [sent len, batch size, emb dim]
        
        # output, self.hidden_cell = self.rnn(embedded, self.hidden_cell)
        output, hidden = self.rnn(embedded)
        
        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        
        # assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        
        # x = self.linear(output.view(len(x), -1))
        x = self.linear(hidden)
        return x