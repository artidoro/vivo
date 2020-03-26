import logging
import spacy
import sys
import torch
import torchtext
from typing import Optional

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'

def torchtext_iterators(device='cpu', batch_size=32,  min_freq=1, max_len=sys.maxsize):
    logger = logging.getLogger('logger')
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    # TODO: Use same preprocessing as OpenNMT.
    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]
    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    # Only target needs BOS/EOS.
    de_field = torchtext.data.Field(tokenize=tokenize_de, lower=True)
    en_field = torchtext.data.Field(tokenize=tokenize_en,
        init_token=BOS_TOKEN, eos_token=EOS_TOKEN, lower=True)

    # TODO: Actually load the WMT 16, sachin data.
    train, val, test = torchtext.datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(de_field, en_field),
        filter_pred=lambda x: len(vars(x)['src']) <= max_len and
                              len(vars(x)['trg']) <= max_len)

    de_field.build_vocab(train.src, min_freq=min_freq)
    en_field.build_vocab(train.trg, min_freq=min_freq)

    train_iter, val_iter = torchtext.data.BucketIterator.splits((train, val),
        batch_size=batch_size, device=torch.device(device), repeat=False,
        sort_key=lambda x: len(x.src))

    # TODO: Load pretrained embeddings.
    # en_field.vocab.load_vectors(vectors=GloVe(name='6B', dim=300))

    logger.info('The size of src vocab is {} and trg vocab is {}.'.format(
        len(de_field.vocab.itos), len(en_field.vocab.itos)))

    return train_iter, val_iter, test, de_field, en_field

def get_nearest_neighbor(
    x: torch.Tensor,
    neighbors: torch.Tensor,
    neighbor_norms: Optional[torch.Tensor] = None,
    return_indexes: bool = False,
) -> torch.Tensor:
    if neighbor_norms is None:
        neighbor_norms = neighbors.norm(dim=-1)
    norms = neighbor_norms.repeat(x.shape[0], 1) * x.norm(dim=-1).unsqueeze(-1)
    dots = (neighbors.unsqueeze(0).repeat(x.shape[0], 1, 1) @ x.unsqueeze(-1)).squeeze()
    if return_indexes:
        return (dots / norms).argmax(-1)
    else:
        return neighbors[(dots / norms).argmax(-1)]
