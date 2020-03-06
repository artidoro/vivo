import torch
import torch.nn as nn

model_dict = {
    'lstm_attn': LstmAttn,
    'transformer': Transformer
}

loss_dict = {
    'xent': nn.CrossEntropyLoss
}

class LstmAttn(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, **kargs):
        super(LstmAttn, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, **kargs):
        super(LstmAttn, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab