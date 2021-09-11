import numpy as np
import torch
from ilonimi.wannimi import Detokenizer
from ilonimi.joiner import Joiner
from .vocab import load_tp_vocab, load_full_vocab
from .batch import Batch
from .model import RNNLM

def load_full_model():
    vocab = load_full_vocab()
    model = RNNLM(len(vocab), 256)
    checkpoint_path = 'checkpoint/checkpoint.99.pt'
    model.load_state_dict(torch.load(checkpoint_path, map_location = 'cpu')['model'])
    model.eval()
    return vocab, model


def load_model():
    _, full_model = load_full_model()
    vocab = load_tp_vocab()
    model = RNNLM(len(vocab), 256)
    model.rnn = full_model.rnn
    model.embeddings.weight.data = full_model.embeddings.weight.data[:len(vocab), :]
    model.proj.weight.data = full_model.proj.weight.data[:len(vocab), :]
    model.proj.bias.data = full_model.proj.bias.data[:len(vocab)]
    model.eval()
    return vocab, model


def process_logit(vocab, logit):
    for x in [vocab.pad, vocab.tp, vocab.de, vocab.en, vocab.unk]:
        logit[x] = float('-inf')
    return logit


def top_p_sampling(logit, temperature, top_p):
    logit = logit / temperature
    probs = torch.softmax(logit, dim = -1)
    values, indices = torch.sort(probs)
    cumlated = torch.cumsum(values, -1)
    is_removed = cumlated < (1 - top_p)
    logit[indices[is_removed]] = -float('Inf')
    probs = torch.softmax(logit, dim = -1)
    probs = probs.cpu().numpy()
    next_token = np.random.choice(range(len(probs)), p=probs)
    return next_token


class Generator:

    def __init__(self):
        self.vocab, self.model = load_model()
        self.detokenizer = Detokenizer()
        self.joiner = Joiner()

    def next_token(self, sent):
        x = Batch(torch.tensor([sent]).T, lengths = [len(sent)])
        with torch.no_grad():
            logit = self.model(x)[-1][0]
        logit = process_logit(self.vocab, logit)
        y = top_p_sampling(logit, 1.0, 0.8)
        return y

    def postproc(self, x):
        x = ' '.join([self.vocab.vocab_list[t] for t in x[1:]])
        x = self.joiner(x)
        x = self.detokenizer(x)
        return x

    def generate(self):
        sent = [self.vocab.vocab_dict['<tp>']]
        for _ in range(50):
            y = self.next_token(sent)
            if y == 1:
                break
            sent.append(y)
        sent = self.postproc(sent)
        return sent

