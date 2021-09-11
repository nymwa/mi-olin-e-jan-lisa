import torch
from torch.nn.utils.rnn import pad_sequence as pad
from src.batch import Batch

class Dataset(torch.utils.data.Dataset):

    def __init__(self, vocab, inputs, outputs):
        self.vocab = vocab
        self.inputs = [[self.vocab.vocab_dict[token] for token in sent] for sent in inputs]
        self.outputs = [[self.vocab.vocab_dict[token] for token in sent] for sent in outputs]
        self.pad = -100

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]

    def collate(self, batch):
        inputs = pad([torch.tensor(sent) for sent, _ in batch], padding_value = self.vocab.pad)
        outputs = pad([torch.tensor(sent) for _, sent in batch], padding_value = self.pad)
        lengths = torch.tensor([len(sent) for sent, _ in batch])
        return Batch(inputs, outputs, lengths)

