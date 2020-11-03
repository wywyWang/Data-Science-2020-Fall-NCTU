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
            nn.ReLU()
        )
        
        self.trigramcnn = nn.Sequential(
            nn.Conv1d(in_channels=config['embedding_dim'], out_channels=200, kernel_size=config['kernel_size'], padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=200, out_channels=100, kernel_size=config['kernel_size'], padding=1),
            nn.ReLU()
        )

        # self.encoder_origin = nn.LSTM(input_size=config['embedding_dim'], hidden_size=config['hidden_dim'], num_layers=config['num_layers'], dropout=config['dropout'], bidirectional=True, batch_first=True)
        self.encoder_bigram = nn.LSTM(input_size=100, hidden_size=config['hidden_dim'], num_layers=config['num_layers'], dropout=config['dropout'], bidirectional=True, batch_first=True)
        self.encoder_trigram = nn.LSTM(input_size=100, hidden_size=config['hidden_dim'], num_layers=config['num_layers'], dropout=config['dropout'], bidirectional=True, batch_first=True)

        self.linear_lstm = nn.Linear(config['hidden_dim'] * 4, 30)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config['dropout'])
        self.linear = nn.Linear(30+config['category_embedding_dim'], 1)
        
        # self.linear = nn.Sequential(
        #     nn.Linear(config['hidden_dim'] * 4, 30),
        #     nn.ReLU(),
        #     nn.Linear(30, 1)
        # )
        self.init_weights()

    def init_weights(self):
        for name, param in self.encoder_bigram.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

        for name, param in self.encoder_trigram.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

        # for m in self.linear.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, mean=0, std=0.02)

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

        output_tri, (h_tri, c_tri) = self.encoder_trigram(x_tricnn)
        h_tri, c_tri = h_tri.transpose(0, 1), c_tri.transpose(0, 1)
        h_tri, c_tri = h_tri.reshape(batch, -1), c_tri.reshape(batch, -1)
        layernorm_tri = nn.LayerNorm(h_tri.size()[1:])
        # h_tri = layernorm_tri(h_tri)

        output_bi, (h_bi, c_bi) = self.encoder_bigram(x_bicnn)
        h_bi, c_bi = h_bi.transpose(0, 1), c_bi.transpose(0, 1)
        h_bi, c_bi = h_bi.reshape(batch, -1), c_bi.reshape(batch, -1)
        layernorm_bi = nn.LayerNorm(h_bi.size()[1:])
        # h_bi = layernorm_bi(h_bi)

        # output_ori, (h_ori, c_ori) = self.encoder_origin(x)
        # h_ori, c_ori = h_ori.transpose(0, 1), c_ori.transpose(0, 1)
        # h_ori, c_ori = h_ori.reshape(batch, -1), c_ori.reshape(batch, -1)

        # x_category = torch.cat((h_tri, h_bi), dim=1)

        x_lstm = torch.cat((h_tri, h_bi), dim=1)
        x_linear = self.relu(self.linear_lstm(x_lstm))
        x_linear = self.dropout(x_linear)
        x_category = torch.cat((x_linear, category_embedding), dim=1)
        prediction = self.linear(x_category)
        
        # prediction = self.linear(x_category)

        return prediction