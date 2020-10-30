import torch
import torch.nn as nn
import math

class AttractiveEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, dropout=0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embedding_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_size = embedding_size

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        # x = self.token(sequence)
        return self.dropout(x)

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=128):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=128):
        super().__init__(vocab_size, embed_size, padding_idx=1)

class CategoryEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=128):
        super().__init__(vocab_size, embed_size, padding_idx=0)