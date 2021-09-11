import torch
from src.accum import Accum
from pathlib import Path
from logging import getLogger
logger = getLogger(__name__)

class Trainer:

    def __init__(self, max_epoch, model, loader, criterion, optimizer):
        self.max_epoch = max_epoch
        self.model = model.cuda()
        self.loader = loader
        self.criterion = criterion
        self.optimizer = optimizer

    def calculate_loss(self, pred, batch):
        pred = pred.view(-1, pred.size(-1))
        loss = self.criterion(pred, batch.outputs.view(-1))
        return loss

    def train_step(self, accum, batch):
        batch.cuda()
        pred = self.model(batch)
        loss = self.calculate_loss(pred, batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train_round(self, epoch):
        self.model.train()
        accum = Accum()
        for i, batch in enumerate(self.loader):
            loss = self.train_step(accum, batch)
            accum(batch, loss)
            logger.info('| epoch {} | train-round {}/{} | loss: {:.4f}'.format(epoch, i + 1, len(self.loader), loss))
        logger.info('| epoch {} | train | loss: {:.4f}'.format(epoch, accum.loss()))


    def save(self, epoch):
        base = Path('checkpoint')
        base.mkdir(exist_ok = True)
        state_dict = self.model.state_dict()
        vocab_size = self.model.vocab_size
        hidden_size = self.model.hidden_size
        dct = {
            'model': state_dict,
            'vocab_size': vocab_size,
            'hidden_size': hidden_size}
        torch.save(dct, base / 'checkpoint.{}.pt'.format(epoch))

    def train(self):
        for epoch in range(self.max_epoch):
            self.train_round(epoch)
            self.save(epoch)

