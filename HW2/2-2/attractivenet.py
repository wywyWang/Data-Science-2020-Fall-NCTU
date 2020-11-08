import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from attractiveembedding import AttractiveEmbedding, CategoryEmbedding


class AttractiveNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = AttractiveEmbedding(vocab_size=config['input_dim'], embedding_size=config['embedding_dim'])
        # self.category_embedding = CategoryEmbedding(vocab_size=config['category_dim'], embed_size=config['category_embedding_dim'])

        self.bigramcnn = nn.Sequential(
            nn.Conv1d(in_channels=config['embedding_dim'], out_channels=210, kernel_size=config['kernel_size']-1, padding=1),
            nn.ReLU6(),
            nn.Conv1d(in_channels=210, out_channels=100, kernel_size=config['kernel_size']-1, padding=1),
            nn.ReLU6(),
            nn.Dropout(config['dropout'])
        )
        
        self.trigramcnn = nn.Sequential(
            nn.Conv1d(in_channels=config['embedding_dim'], out_channels=210, kernel_size=config['kernel_size'], padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=210, out_channels=100, kernel_size=config['kernel_size'], padding=1),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )

        self.encoder_bigram = nn.LSTM(input_size=100, hidden_size=config['hidden_dim'], num_layers=config['num_layers'], dropout=config['dropout'], bidirectional=True, batch_first=True)
        self.encoder_trigram = nn.LSTM(input_size=100, hidden_size=config['hidden_dim'], num_layers=config['num_layers'], dropout=config['dropout'], bidirectional=True, batch_first=True)

        self.linear = nn.Sequential(
            nn.Linear(config['hidden_dim']*4+2*2, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.encoder_bigram.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

        for name, param in self.encoder_trigram.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, category, phase):
        batch = x.shape[0]
        x = self.embedding(x)
        # category_embedding = self.category_embedding(category)

        # CNN
        # (batch_size, seq_length, embedding_size) -> (batch_size, embedding_size, seq_length)
        x_cnn = x.transpose(1, 2)
        x_tricnn = self.trigramcnn(x_cnn)
        x_bicnn = self.bigramcnn(x_cnn)

        x_tricnn = x_tricnn.transpose(1, 2)
        x_bicnn = x_bicnn.transpose(1, 2)

        # LSTM: (seq_length, batch_size, embedding_size)

        output_tri, (h_tri, c_tri) = self.encoder_trigram(x_tricnn)
        h_tri = h_tri.transpose(0, 1)
        h_tri_avg_pool = torch.mean(h_tri, 2)
        h_tri_max_pool, _ = torch.max(h_tri, 2)
        h_tri = h_tri.reshape(batch, -1)

        output_bi, (h_bi, c_bi) = self.encoder_bigram(x_bicnn)
        h_bi = h_bi.transpose(0, 1)
        h_bi_avg_pool = torch.mean(h_bi, 2)
        h_bi_max_pool, _ = torch.max(h_bi, 2)
        h_bi = h_bi.reshape(batch, -1)

        x_category = torch.cat((h_tri, h_tri_max_pool, 
                                h_bi, h_bi_max_pool), dim=1)

        prediction = self.linear(x_category)

        return prediction