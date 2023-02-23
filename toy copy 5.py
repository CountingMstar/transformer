from torch import nn
import torch


class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


model = TokenEmbedding(6, 7)
print(model)

x = torch.ones(5, 10).int()
print(x)

x = model(x)
print(x.shape)
print(x)