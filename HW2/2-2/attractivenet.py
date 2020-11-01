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
        # self.category_embedding = CategoryEmbedding(vocab_size=config['category_dim'], embed_size=config['category_embedding_dim'])

        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=config['embedding_dim'], out_channels=220, kernel_size=config['kernel_size'], padding=1),
            nn.ReLU()
        )

        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=220, out_channels=150, kernel_size=config['kernel_size'], padding=1),
            nn.ReLU()
        )

        self.cnn3 = nn.Sequential(
            nn.Conv1d(in_channels=150, out_channels=100, kernel_size=config['kernel_size'], padding=1),
            nn.ReLU()
        )

        self.encoder = nn.LSTM(input_size=100, hidden_size=config['hidden_dim'], num_layers=config['num_layers'], dropout=config['dropout'], bidirectional=True, batch_first=True)

        # self.encoder = nn.LSTM(input_size=config['embedding_dim'], hidden_size=config['hidden_dim'], num_layers=config['num_layers'], dropout=config['dropout'], bidirectional=True, batch_first=True)

        # self.linear_output = nn.Linear(config['hidden_dim']*4+config['category_embedding_dim'], config['output_dim'])
        # self.linear_output = nn.Linear(config['hidden_dim']*4, config['output_dim'])
        
        self.linear = nn.Sequential(
            nn.Linear(config['hidden_dim']*4, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )
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

    def forward(self, x, category, phase):
        batch = x.shape[0]
        x = self.embedding(x)
        # category_embedding = self.category_embedding(category)

        # CNN
        # (batch_size, seq_length, embedding_size) -> (batch_size, embedding_size, seq_length)
        # print(x.shape, flush=True)
        x = x.transpose(1, 2)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        # (batch_size, hidden_size, seq_length) -> (seq_length, batch_size, hidden_size)
        x = x.transpose(1, 2)
        # print(x.shape, flush=True)
        # 1/0

        # LSTM: (seq_length, batch_size, embedding_size)
        # print(x.shape, flush=True)
        # x = x.transpose(1, 2)

        output, (h, c) = self.encoder(x)

        h, c = h.transpose(0, 1), c.transpose(0, 1)
        h, c = h.reshape(batch, -1), c.reshape(batch, -1)
        x_category = torch.cat((h, c), dim=1)

        prediction = self.linear(x_category)

        if phase == 'train':
            prediction += 3.15
        else:
            prediction += 2.8
        return prediction