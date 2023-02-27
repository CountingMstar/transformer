"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.embedding.positional_encoding import PostionalEncoding
from models.embedding.token_embeddings import TokenEmbedding
from models.embedding.autoencoder import AutoEncoder, LinearLayer

from conf import device


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device, k):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        # "Concatenate" Token, Position 임베딩 크기 조절 파라미터
        # k = 12
        print('#####KKKKKKKKK#####')
        print(k)

        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PostionalEncoding(d_model, max_len, device)

        self.cat_tok_emb = TokenEmbedding(vocab_size, d_model-k)
        self.cat_pos_emb = PostionalEncoding(k, max_len, device)

        self.drop_out = nn.Dropout(p=drop_prob)

    def expander(self, x):
        # normal
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        # print('##########Emb#########')
        # print(x.shape)
        # print(x)
        # print(tok_emb.shape)
        # print(tok_emb)
        # print(pos_emb.shape)
        # print(pos_emb)
        tok_batch_size, tok_sentence_size, tok_embedding_size = tok_emb.shape
        pos_sentence_size, pos_embedding_size = pos_emb.shape
        pos_emb = pos_emb.expand(tok_batch_size, pos_sentence_size, pos_embedding_size)

        # concatenate
        cat_tok_emb = self.cat_tok_emb(x)
        cat_pos_emb = self.cat_pos_emb(x)
        # print('##########Emb#########')
        # print(x.shape)
        # print(x)
        # print(cat_tok_emb.shape)
        # print(tok_emb)
        # print(cat_pos_emb.shape)
        # print(pos_emb)
        tok_batch_size, tok_sentence_size, tok_embedding_size = cat_tok_emb.shape
        pos_sentence_size, pos_embedding_size = cat_pos_emb.shape
        cat_pos_emb = cat_pos_emb.expand(tok_batch_size, pos_sentence_size, pos_embedding_size)
        return tok_emb, pos_emb, cat_tok_emb, cat_pos_emb

##########################auto encoder를 집어넣자##############################
    def forward(self, x):

        tok_emb, pos_emb, cat_tok_emb, cat_pos_emb = self.expander(x)
        # print('########transformer_emb#######')
        # print(tok_emb.shape)
        # print(tok_emb)
        # print(pos_emb.shape)
        # print(pos_emb)

        # tok_emb: 여러 문장에 대한 "토큰 임베딩"
        # torch.Size([128, 34, 512])
        # pos_emb: 각 문장에 대한 "포지셔널 임베딩"
        # torch.Size([34, 512])
        # torch.Size([128, 34, 512])

        model = SummationEmbedding(tok_emb, pos_emb, cat_tok_emb, cat_pos_emb)
        """
        positional encoding type 결정
        """
        # final_emb = model.summation()
        # final_emb = model.concatenate()
        final_emb = model.linear()
        # final_emb = model.autoencoder()
        

        # return self.drop_out(tok_emb + pos_emb)
        # return self.drop_out(final_emb)
        return final_emb


class SummationEmbedding(TransformerEmbedding):
    def __init__(self, token_emb, positional_emb, cat_token_emb, cat_positional_emb):
        super(TransformerEmbedding, self).__init__()

        self.token_emb = token_emb
        self.positional_emb = positional_emb
        self.cat_token_emb = cat_token_emb
        self.cat_positional_emb = cat_positional_emb

        """
        from models.embedding.autoencoder import LinearLayer의
        linear layer 선언
        """
        self.linearlayer = LinearLayer()

    def summation(self):
        embedding = self.token_emb + self.positional_emb       
        return embedding

    def concatenate(self):
        embedding = torch.cat([self.cat_token_emb, self.cat_positional_emb], 2)
        # print('#####CAT#####')
        # print(embedding.shape)
        return embedding

    # def linear(self):
    #     embedding = torch.cat([self.token_emb, self.positional_emb], 2)
    #     batch_size, sentence_size, embedding_size = embedding.shape
    #     embedding = embedding.view(batch_size*sentence_size, -1)
    #     self.linearlayer = LinearLayer(embedding).to(device)
    #     embedding = self.linearlayer(embedding)
    #     embedding = embedding.view(batch_size, sentence_size, int(embedding_size/2))
    #     return embedding

    def linear(self):
        """
        token embedding과 positional embedding을 결합하는 linear layer
        """
        embedding = torch.cat([self.cat_token_emb, self.cat_positional_emb], 2)
        batch_size, sentence_size, embedding_size = embedding.shape
        embedding = embedding.view(batch_size*sentence_size, -1)
        # print('#########')
        # print(embedding.shape)

        # self.linearlayer = LinearLayer(embedding).to(device)
        embedding = self.linearlayer(embedding)

        embedding = embedding.view(batch_size, sentence_size, embedding_size)
        return embedding

    def autoencoder(self):
        embedding = torch.cat([self.token_emb, self.positional_emb], 2)
        batch_size, sentence_size, embedding_size = embedding.shape
        embedding = embedding.view(batch_size*sentence_size, -1)
        self.auto_encoder = AutoEncoder(embedding).to(device)
        encoded, decoded = self.auto_encoder(embedding)
        embedding = embedding.view(batch_size, sentence_size, embedding_size)

        # 텐서를 비교하는 함수
        # print(torch.eq(a, embedding))
        return embedding


