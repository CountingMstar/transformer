import torchtext
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

# define the fields
SRC = Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', lower=True)

# load the dataset
train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

print('0000000000000000')
print(train_data)

for i, example in enumerate(train_data[:10]):
    print(f"Example {i+1}:")
    print("Src: ", example.src)
    print("Trg: ", example.trg)
    print("")

SRC.build_vocab(train_data, min_freq=2)
print(SRC)
print("Source vocabulary size:", len(SRC.vocab))
print("Source vocabulary:", SRC.vocab.itos)