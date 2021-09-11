class Accum:

    def __init__(self):
        self.accum = 0.0
        self.example = 0

    def __call__(self, batch, loss):
        self.accum += loss.item() * len(batch)
        self.example += len(batch)

    def loss(self):
        return self.accum / self.example

