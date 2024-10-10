import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # TODO: Why is this multiplication done? The paper states that "They share the same weight matrix between the two embedding layers. 
        # In the embedding layers, we multiply those weights by square root of d_model."
        return self.embedding(x) * math.sqrt(self.d_model)
    