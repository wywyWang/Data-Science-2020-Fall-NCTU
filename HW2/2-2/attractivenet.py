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

        self.bigramcnn = nn.Sequential(
            nn.Conv1d(in_channels=config['embedding_dim'], out_channels=200, kernel_size=config['kernel_size'], padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=200, out_channels=100, kernel_size=config['kernel_size'], padding=1),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )
        
        self.trigramcnn = nn.Sequential(
            nn.Conv1d(in_channels=config['embedding_dim'], out_channels=200, kernel_size=config['kernel_size'], padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=200, out_channels=100, kernel_size=config['kernel_size'], padding=1),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )

        self.encoder_bigram_first = nn.LSTM(input_size=100, hidden_size=config['hidden_dim'], num_layers=config['num_layers'], dropout=config['dropout'], bidirectional=True, batch_first=True)
        self.encoder_trigram_first = nn.LSTM(input_size=100, hidden_size=config['hidden_dim'], num_layers=config['num_layers'], dropout=config['dropout'], bidirectional=True, batch_first=True)
        
        self.linear = nn.Sequential(
            nn.Linear(config['hidden_dim'] * 4 + 2 * 2 + config['category_embedding_dim'], 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.encoder_bigram_first.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

        for name, param in self.encoder_trigram_first.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, category, phase):
        batch = x.shape[0]
        x = self.embedding(x)
        category_embedding = self.category_embedding(category)

        # CNN
        # (batch_size, seq_length, embedding_size) -> (batch_size, embedding_size, seq_length)
        x_cnn = x.transpose(1, 2)
        x_tricnn = self.trigramcnn(x_cnn)
        x_bicnn = self.bigramcnn(x_cnn)

        # (batch_size, hidden_size, seq_length) -> (seq_length, batch_size, hidden_size)
        x_tricnn = x_tricnn.transpose(1, 2)
        x_bicnn = x_bicnn.transpose(1, 2)

        # LSTM: (seq_length, batch_size, embedding_size)

        output_tri_first, (h_tri_first, c_tri_first) = self.encoder_trigram_first(x_tricnn)  
        h_tri_first = h_tri_first.transpose(0, 1)
        h_tri_first = h_tri_first.reshape(batch, -1)
        h_tri_first_avg_pool = torch.mean(h_tri_first, 1)
        h_tri_first_max_pool, _ = torch.max(h_tri_first, 1)
        h_tri_first_avg_pool, h_tri_first_max_pool = h_tri_first_avg_pool.unsqueeze(1), h_tri_first_max_pool.unsqueeze(1)

        output_bi_first, (h_bi_first, c_bi_first) = self.encoder_bigram_first(x_bicnn)
        h_bi_first = h_bi_first.transpose(0, 1)
        h_bi_first = h_bi_first.reshape(batch, -1)
        h_bi_first_avg_pool = torch.mean(h_bi_first, 1)
        h_bi_first_max_pool, _ = torch.max(h_tri_first, 1)
        h_bi_first_avg_pool, h_bi_first_max_pool = h_bi_first_avg_pool.unsqueeze(1), h_bi_first_max_pool.unsqueeze(1)
        
        x_category = torch.cat((h_tri_first, h_tri_first_avg_pool, h_tri_first_max_pool, 
                                h_bi_first, h_bi_first_avg_pool, h_bi_first_max_pool, 
                                category_embedding), dim=1)

        prediction = self.linear(x_category)

        return prediction