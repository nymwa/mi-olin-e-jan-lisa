from logging import getLogger
logger = getLogger(__name__)

from pathlib import Path

def load_vocab(vocab):
    vocab = Vocabulary(vocab)
    logger.info('vocab loaded: {}'.format(
        len(vocab)))
    return vocab


def load_tp_vocab():
    file1 = 'vocabulary/special.txt'
    file2 = 'vocabulary/tp.txt'
    with open(file1) as f:
        vocab1 = [x.strip() for x in f]
    with open(file2) as f:
        vocab2 = [x.strip() for x in f]
    return load_vocab(vocab1 + vocab2)


def load_full_vocab():
    file1 = 'vocabulary/special.txt'
    file2 = 'vocabulary/tp.txt'
    file3 = 'vocabulary/ende.txt'
    with open(file1) as f:
        vocab1 = [x.strip() for x in f]
    with open(file2) as f:
        vocab2 = [x.strip() for x in f]
    with open(file3) as f:
        vocab3 = [x.strip() for x in f]
    return load_vocab(vocab1 + vocab2 + vocab3)


class Vocabulary:

    def __init__(self, vocab):
        for index, token in enumerate(vocab):
            if token.startswith('<') and token.endswith('>'):
                setattr(self, token[1:-1], index)

        self.vocab_list = vocab
        self.vocab_dict = {word: i for i, word in enumerate(self.vocab_list)}

    def __len__(self):
        return len(self.vocab_list)

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.vocab_list, f)

