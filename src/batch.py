class Batch:

    def __init__(self, inputs, outputs = None, lengths = None):
        self.inputs = inputs
        self.outputs = outputs
        self.lengths = lengths

    def __len__(self):
        return self.inputs.shape[1]

    def cuda(self):
        self.inputs = self.inputs.cuda()

        if self.outputs is not None:
            self.outputs = self.outputs.cuda()

