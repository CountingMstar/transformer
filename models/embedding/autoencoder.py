import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, tok_emb, dropout=0.1):
        super(AutoEncoder, self).__init__()

        tok_batch_size, tok_sentence_size, tok_embedding_size = tok_emb.shape
        self.w_1 = nn.Linear(tok_batch_size*2, 64)
        self.w_2 = nn.Linear(64, tok_batch_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x

