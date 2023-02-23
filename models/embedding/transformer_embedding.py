"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.embedding.positional_encoding import PostionalEncoding
from models.embedding.token_embeddings import TokenEmbedding
from models.embedding.autoencoder import AutoEncoder

from conf import device


class SummationEmbedding(nn.Module):
    def __init__(self, token_emb, positional_emb, cat_token_emb, cat_positional_emb):
        super(SummationEmbedding, self).__init__()

        self.token_emb = token_emb
        self.positional_emb = positional_emb
        self.cat_token_emb = cat_token_emb
        self.cat_positional_emb = cat_positional_emb
         

    def summation(self):
        embedding = self.token_emb + self.positional_emb       
        return embedding

    def concatenate(self):
        embedding = torch.cat([self.cat_token_emb, self.cat_positional_emb], 2)
        return embedding

    def autoencoder(self):
        embedding = torch.cat([self.token_emb, self.positional_emb], 2)
        # print('###')
        # print(embedding.shape)
        batch_size, sentence_size, embedding_size = embedding.shape
        embedding = embedding.view(batch_size*sentence_size, -1)
        # print(embedding.shape)
        self.auto_encoder = AutoEncoder(embedding).to(device)
        embedding = self.auto_encoder(embedding)

        # print(embedding.shape)
        embedding = embedding.view(batch_size, sentence_size, int(embedding_size/2))
        # print(embedding.shape)
        return embedding




class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        # "Concatenate" Token, Position 임베딩 크기 조절 파라미터
        k = 12

        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PostionalEncoding(d_model, max_len, device)

        self.cat_tok_emb = TokenEmbedding(vocab_size, d_model-k)
        self.cat_pos_emb = PostionalEncoding(k, max_len, device)

        self.drop_out = nn.Dropout(p=drop_prob)

    def expand(self, x):
        # normal
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        tok_batch_size, tok_sentence_size, tok_embedding_size = tok_emb.shape
        pos_sentence_size, pos_embedding_size = pos_emb.shape
        pos_emb = pos_emb.expand(tok_batch_size, pos_sentence_size, pos_embedding_size)

        # concatenate
        cat_tok_emb = self.cat_tok_emb(x)
        cat_pos_emb = self.cat_pos_emb(x)
        tok_batch_size, tok_sentence_size, tok_embedding_size = cat_tok_emb.shape
        pos_sentence_size, pos_embedding_size = cat_pos_emb.shape
        cat_pos_emb = cat_pos_emb.expand(tok_batch_size, pos_sentence_size, pos_embedding_size)
        return tok_emb, pos_emb, cat_tok_emb, cat_pos_emb

##########################auto encoder를 집어넣자##############################
    def forward(self, x):
        # print('---------------')
        # print(x.shape)
        # print(x)

        tok_emb, pos_emb, cat_tok_emb, cat_pos_emb = self.expand(x)
        
        # print('===============')
        # print(tok_emb.shape)
        # print(pos_emb.shape)
        # print((tok_emb + pos_emb).shape)

        # tok_emb: 여러 문장에 대한 "토큰 임베딩"
        # torch.Size([128, 34, 512])
        # pos_emb: 각 문장에 대한 "포지셔널 임베딩"
        # torch.Size([34, 512])
        # torch.Size([128, 34, 512])


        model = SummationEmbedding(tok_emb, pos_emb, cat_tok_emb, cat_pos_emb)
        # final_emb = model.summation()
        # final_emb = model.concatenate()
        final_emb = model.autoencoder()


        # print('+++++++++++')
        # print(final_emb.shape)
        # print(final_emb)
        # print(d)

        # return self.drop_out(tok_emb + pos_emb)
        return self.drop_out(final_emb)



