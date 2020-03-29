import logging
import os
import sys
import torch
import torchtext
from typing import Optional
import numpy as np

BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'

GERMAN_DATA_PATH = 'deen'
FRENCH_DATA_PATH = 'fren'
TRAIN_PREFEX = 'train'
VALID_PREFIX = 'tst201314'
TEST_PREFIX = 'tst201516'
TOKENIZATION_FLAG = 'tok'
LOWERCASE_FLAG = 'true'

def torchtext_iterators(
    root_data_path,
    src_language,
    trg_language,
    device,
    batch_size,
    min_freq,
    max_len,
    fasttext_embeds_path,
    src_vocab_size,
    trg_vocab_size,
    is_vivo
):
    '''
    Validation data TED TEST 2013-2014
    Test data TED TEST 2015-2016

    Note: we are using tokenized lowercased text so split() is sufficient
        for preprocessing.
    '''
    logger = logging.getLogger('vivo_logger')

    if 'fr' in {src_language, trg_language}:
        data_path = FRENCH_DATA_PATH
    else:
        data_path = GERMAN_DATA_PATH

    src_ext = '.'.join(['', TOKENIZATION_FLAG, LOWERCASE_FLAG, src_language])
    trg_ext = '.'.join(['', TOKENIZATION_FLAG, LOWERCASE_FLAG, trg_language])

    # Only target needs BOS/EOS.
    src_field = torchtext.data.Field()
    trg_field = torchtext.data.Field(
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN
    )

    train, val, test = torchtext.datasets.TranslationDataset.splits(
        path=os.path.join(root_data_path, data_path),
        train=TRAIN_PREFEX,
        validation=VALID_PREFIX,
        test=TEST_PREFIX,
        exts=(src_ext, trg_ext),
        fields=(src_field, trg_field),
        filter_pred=lambda x: len(vars(x)['src']) <= max_len and
                              len(vars(x)['trg']) <= max_len
    )

    src_field.build_vocab(train.src, min_freq=min_freq, max_size=src_vocab_size)
    trg_field.build_vocab(train.trg, min_freq=min_freq, max_size=trg_vocab_size)

    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits((train, val, test),
        batch_size=batch_size, device=torch.device(device), repeat=False,
        sort_key=lambda x: len(x.src))

    # Load pretrained embeddings.
    if fasttext_embeds_path is not None:
        embedding_vectors = torchtext.vocab.Vectors(name=fasttext_embeds_path)
        trg_field.vocab.load_vectors(
            vectors=embedding_vectors
        )
        # TODO Remove words without embedding from the output vocabulary
        trg_field.vocab.vectors[trg_field.vocab.stoi[BOS_TOKEN]] = 1/16
        trg_field.vocab.vectors[(trg_field.vocab.vectors == 0).all(-1)] = trg_field.vocab.vectors.mean(0)
        if is_vivo:
            full_vocab = torchtext.data.Field()
            full_vocab.build_vocab(train.trg, min_freq=min_freq)
            excluded_words = [word for word in full_vocab.vocab.stoi if word not in trg_field.vocab.stoi]
            if excluded_words:
                excluded_vectors = torch.stack([embedding_vectors[excluded_word] for excluded_word in excluded_words if excluded_word in embedding_vectors.stoi])
                unk_vector = -torch.mean(excluded_vectors,dim=0)
            else:
                unk_vector = torch.randn(trg_field.vocab.vectors[trg_field.vocab.stoi[UNK_TOKEN]].shape)
            trg_field.vocab.vectors[trg_field.vocab.stoi[UNK_TOKEN]] = unk_vector
            
            zero_idxs = (trg_field.vocab.vectors == 0).all(-1)
            zero_idxs[trg_field.vocab.stoi[BOS_TOKEN]] = False
            zero_idxs[trg_field.vocab.stoi[PAD_TOKEN]] = False
            vector_dim = trg_field.vocab.vectors.shape[-1]
            for i in np.argwhere(zero_idxs).squeeze(0):
                trg_field.vocab.vectors[i] = torch.Tensor(
                np.random.uniform(-1.0, 1.0, vector_dim)
                )

    logger.info('The size of src vocab is {} and trg vocab is {}.'.format(
        len(src_field.vocab.itos), len(trg_field.vocab.itos)))

    return train_iter, val_iter, test_iter, src_field, trg_field

def get_nearest_neighbor(
    x: torch.Tensor,
    neighbors: torch.Tensor,
    neighbor_norms: Optional[torch.Tensor] = None,
    top_k: int = 1,
) -> torch.Tensor:
    if neighbor_norms is None:
        neighbor_norms = neighbors.norm(dim=-1)
    batch_dims = len(x.shape) - 1
    norms = neighbor_norms.repeat(*(1,) * batch_dims, 1) * x.norm(dim=-1).unsqueeze(-1)
    zero_mask = norms == 0.0
    dots = (
        neighbors.unsqueeze(0).repeat(*(1,) * batch_dims, 1, 1) @ x.unsqueeze(-1)
    ).squeeze(-1)
    distances = (dots / norms)
    distances[zero_mask] = 0.0
    if top_k > 1:
        return torch.topk(distances, top_k, sorted=True).indices
    else:
        return distances.argmax(-1)
