import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class RNNLM(nn.Module):

    def __init__(self, vocab_size, hidden_size, dropout = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, batch):
        x = self.embeddings(batch.inputs)
        x = self.dropout(x)
        packed = pack(x, batch.lengths, enforce_sorted = False)
        output, _ = self.rnn(packed)
        x, _ = unpack(output)
        x = self.dropout(x)
        x = self.proj(x)
        return x

