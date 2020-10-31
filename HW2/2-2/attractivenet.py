import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from attractiveembedding import AttractiveEmbedding, CategoryEmbedding


class AttractiveNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embedding = AttractiveEmbedding(vocab_size=config['input_dim'], embedding_size=config['embedding_dim'])
        self.category_embedding = CategoryEmbedding(vocab_size=config['category_dim'], embed_size=config['category_embedding_dim'])

        self.cnn = nn.Conv1d(in_channels=config['embedding_dim'], out_channels=config['hidden_dim'], kernel_size=config['kernel_size'])
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool1d(kernel_size=config['max_size']-1+1)    # - kernel_size + 1

        self.encoder = nn.LSTM(input_size=config['hidden_dim'], hidden_size=config['hidden_dim'], num_layers=config['num_layers'], dropout=config['dropout'], bidirectional=True)

        # self.encoder = nn.LSTM(input_size=config['embedding_dim'], hidden_size=config['hidden_dim'], num_layers=config['num_layers'], dropout=config['dropout'], bidirectional=True)

        self.linear_output = nn.Linear(config['hidden_dim']+config['hidden_dim']+config['category_embedding_dim'], config['output_dim'])

        self.init_weights()

    def init_weights(self):
        for name, param in self.encoder.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.xavier_normal(param)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param)
        # self.hidden = torch.zeros(self.config['num_layers'], self.config['batch_size'], self.config['hidden_dim'])
        # self.cell = torch.zeros(self.config['num_layers'], self.config['batch_size'], self.config['hidden_dim'])

    def forward(self, x, category):
        batch = x.shape[1]
        x = self.embedding(x)
        category_embedding = self.category_embedding(category)

        # CNN
        # (batch_size, seq_length, embedding_size) -> (batch_size, embedding_size, seq_length)
        x = x.permute(1, 2, 0)
        x = self.cnn(x)
        # x = self.relu(x)
        # x = self.maxpool(x)       # look worse
        # (batch_size, hidden_size, seq_length) -> (seq_length, batch_size, hidden_size)
        x = x.permute(2, 0, 1)


        # LSTM: (seq_length, batch_size, embedding_size)

        output, (self.hidden, self.cell) = self.encoder(x)

        # print(self.hidden.shape)

        h_n = self.hidden.view(self.config['num_layers'], 2, batch, self.config['hidden_dim'])[-1]
        x_category = torch.cat((h_n[0], h_n[1], category_embedding), dim=1)

        prediction = self.linear_output(x_category)
        return prediction