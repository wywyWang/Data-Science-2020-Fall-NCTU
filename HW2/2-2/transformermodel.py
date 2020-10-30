import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attractiveembedding import AttractiveEmbedding, CategoryEmbedding

class TransformerModel(nn.Module):

    def __init__(self, nhead, input_dim, category_dim, category_embedding_dim, embedding_dim, hidden_dim, output_dim, dropout, num_layers):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.embedding = AttractiveEmbedding(vocab_size=input_dim, embedding_size=embedding_dim)
        self.category_embedding = CategoryEmbedding(vocab_size=category_dim, embed_size=category_embedding_dim)
        encoder_layers = TransformerEncoderLayer(embedding_dim, nhead, hidden_dim, dropout, activation='relu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(embedding_dim+category_embedding_dim, output_dim)

        # self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, category):
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            device = x.device
            mask = self._generate_square_subsequent_mask(len(x)).to(device)
            self.src_mask = mask

        # need to mask padding token
        # src_key_padding_mask(batch_size, sequence)
        src_key_padding_mask = torch.zeros([x.shape[0], x.shape[1]]).bool().to(x.device)
        src_key_padding_mask[x == 1] = 1
        src_key_padding_mask = torch.transpose(src_key_padding_mask, 0, 1)
            
        x = self.embedding(x)
        category_embedding = self.category_embedding(category)

        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        x_category = torch.cat((output[0, :], category_embedding), dim=1)

        prediction = self.linear(x_category)
        return prediction