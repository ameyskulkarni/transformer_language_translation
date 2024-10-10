import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # TODO: Why is this multiplication done? The paper states that "They share the same weight matrix between the two embedding layers. 
        # In the embedding layers, we multiply those weights by square root of d_model."
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of length (seq_len, d_model). seq_len is the number of tokens in a sentence and d_model is the number of input embeddings per token.
        self.pe = torch.zeros(seq_len, d_model)

        # create the numerator and denominator of the positional embedding formula. 
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin and cosine to even and odd positions respectively
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        # The following only adds the positional encodings to the state dict and does not update these parameters during training. Other such parameters are usually the batch norm statistics like mean and variance. 
        # These are not model parameters and thier requires_grad is False. These are just needed in the model state dict to load data.
        self.register_buffer('pe', self.pe)

    def forward(self, x):
        x += self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps:float) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Learnable parameter that is multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Learnable parameter that is added

    def forward(x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * ((x-mean)/(std+self.eps)) + self.bias


class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    
    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        # The above happens first through transformation with (W1, B1) and then with (W2, B2)

        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))



