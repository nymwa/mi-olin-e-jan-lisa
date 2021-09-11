from logging import getLogger
from collections import defaultdict
import random as rd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.log import init_logging
from src.vocab import load_full_vocab
from src.dataset import Dataset
from src.model import RNNLM
from src.sampler import Sampler
from src.trainer import Trainer
init_logging()
logger = getLogger(__name__)


def load_tp_data():
    data = []
    with open('data/filter/tp.txt') as f:
        for line in f:
            data.append(line.strip().split())
    logger.info('data loaded: {}'.format(len(data)))
    return data


def load_pair_data(file):
    src_list = []
    trg_list = []
    with open(file) as f:
        for line in f:
            src, trg = line.strip().split('\t')
            src_list.append(src.split())
            trg_list.append(trg.split())
    logger.info('data loaded: {} ({})'.format(file, len(src_list)))
    return src_list, trg_list


def make_tp_pair(data):
    inputs = [['<tp>'] + x for x in data]
    outputs = [x + ['<eos>'] for x in data]
    return inputs, outputs


def make_mt_pair(src_tag, trg_tag, src_data, trg_data):
    inputs =  [[src_tag] + src + [trg_tag] + trg for src, trg in zip(src_data, trg_data)]
    inputs += [[trg_tag] + trg + [src_tag] + src for src, trg in zip(src_data, trg_data)]
    outputs =  [src + ['<eos>'] + trg + ['<eos>'] for src, trg in zip(src_data, trg_data)]
    outputs += [trg + ['<eos>'] + src + ['<eos>'] for src, trg in zip(src_data, trg_data)]
    return inputs, outputs


def load_data():
    tp_data = load_tp_data()
    ende_en_data, ende_de_data = load_pair_data('data/filter/ende.tsv')
    tpde_tp_data, tpde_de_data = load_pair_data('data/filter/tpde.tsv')
    tpen_tp_data, tpen_en_data = load_pair_data('data/filter/tpen.tsv')
    tp_inputs, tp_outputs = make_tp_pair(tp_data)
    ende_inputs, ende_outputs = make_mt_pair('<en>', '<de>', ende_en_data, ende_de_data)
    tpde_inputs, tpde_outputs = make_mt_pair('<tp>', '<de>', tpde_tp_data, tpde_de_data)
    tpen_inputs, tpen_outputs = make_mt_pair('<tp>', '<en>', tpen_tp_data, tpen_en_data)
    inputs = tp_inputs + ende_inputs + tpde_inputs + tpen_inputs
    outputs = tp_outputs + ende_outputs + tpde_outputs + tpen_outputs
    return inputs, outputs


def make_model(vocab, hidden_size, dropout):
    model = RNNLM(len(vocab), hidden_size, dropout)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('RNN LM: {}'.format(num_params))
    return model


def make_loader(dataset, max_tokens):
    sampler = Sampler(dataset, max_tokens)
    loader = DataLoader(dataset, batch_sampler = sampler, collate_fn = dataset.collate)
    logger.info('loader: {}'.format(len(loader)))
    return loader


def main():
    vocab = load_full_vocab()
    inputs, outputs = load_data()
    dataset = Dataset(vocab, inputs, outputs)

    hidden_size = 256
    max_tokens = 4000
    max_epoch = 100
    lr = 0.002
    dropout = 0.1
    model = make_model(vocab, hidden_size, dropout = dropout)
    loader = make_loader(dataset, max_tokens)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    trainer = Trainer(max_epoch, model, loader, criterion, optimizer)
    trainer.train()


if __name__ == '__main__':
    main()

