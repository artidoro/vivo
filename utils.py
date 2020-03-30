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

def torchtext_iterators(args, src_vocab=None, trg_vocab=None):
    '''
    Validation data TED TEST 2013-2014
    Test data TED TEST 2015-2016

    Note: we are using tokenized lowercased text so split() is sufficient
        for preprocessing.
    '''
    logger = logging.getLogger('vivo_logger')

    if 'fr' in {args['src_language'], args['trg_language']}:
        data_path = FRENCH_DATA_PATH
    else:
        data_path = GERMAN_DATA_PATH

    src_ext = '.'.join(['', TOKENIZATION_FLAG, LOWERCASE_FLAG, args['src_language']])
    trg_ext = '.'.join(['', TOKENIZATION_FLAG, LOWERCASE_FLAG, args['trg_language']])

    # Only target needs BOS/EOS.
    src_field = torchtext.data.Field()
    trg_field = torchtext.data.Field(
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN
    )

    train, val, test = torchtext.datasets.TranslationDataset.splits(
        path=os.path.join(args['data_path'], data_path),
        train=TRAIN_PREFEX,
        validation=VALID_PREFIX,
        test=TEST_PREFIX,
        exts=(src_ext, trg_ext),
        fields=(src_field, trg_field),
        filter_pred=lambda x: len(vars(x)['src']) <= args['max_len'] and
                              len(vars(x)['trg']) <= args['max_len']
    )

    if src_vocab is not None:
        src_field.vocab = src_vocab
    else:
        src_field.build_vocab(train.src, min_freq=args['min_freq'], max_size=args['src_vocab_size'])
    if trg_vocab is not None:
        trg_field.vocab = trg_vocab
    else:
        trg_field.build_vocab(train.trg, min_freq=args['min_freq'], max_size=args['trg_vocab_size'])

    # Create iterators and batch (different batch size for train and eval/test).
    train_iter = torchtext.data.BucketIterator(train,
        batch_size=args['train_batch_size'], device=torch.device(args['device']), repeat=False,
        sort_key=lambda x: len(x.src))
    val_iter = torchtext.data.BucketIterator(val,
        batch_size=args['eval_batch_size'], device=torch.device(args['device']), repeat=False,
        sort_key=lambda x: len(x.src), train=False)
    test_iter = torchtext.data.BucketIterator(test,
        batch_size=args['eval_batch_size'], device=torch.device(args['device']), repeat=False,
        sort_key=lambda x: len(x.src), train=False)

    # Load pretrained embeddings.
    if args['fasttext_embeds_path'] is not None:
        embedding_vectors = torchtext.vocab.Vectors(name=args['fasttext_embeds_path'])
        trg_field.vocab.load_vectors(vectors=embedding_vectors)
        if args['loss_function'] == 'vmf':
            # Intialize UNK to the negative mean of the vectors of words not in vocab.
            excluded_words = [word for word in embedding_vectors.stoi if word not in trg_field.vocab.stoi]
            if excluded_words:
                excluded_vectors = torch.stack(
                    [embedding_vectors[excluded_word] for excluded_word in excluded_words\
                        if excluded_word in embedding_vectors.stoi]
                )
                unk_vector = -torch.mean(excluded_vectors, dim=0)
            else:
                unk_vector = torch.randn(trg_field.vocab.vectors[trg_field.vocab.stoi[UNK_TOKEN]].shape)
            trg_field.vocab.vectors[trg_field.vocab.stoi[UNK_TOKEN]] = unk_vector
            # BOS and PAD are initialized to the zero vector.
            zero_idxs = (trg_field.vocab.vectors == 0).all(-1)
            zero_idxs[trg_field.vocab.stoi[BOS_TOKEN]] = False
            zero_idxs[trg_field.vocab.stoi[PAD_TOKEN]] = False
            if args["eos_vector_replace"]:
                period_vector = trg_field.vocab.vectors[trg_field.vocab.stoi['.']]
                trg_field.vocab.vectors[trg_field.vocab.stoi[EOS_TOKEN]] = -period_vector
            vector_dim = trg_field.vocab.vectors.shape[-1]
            # Other vectors of words not in fasttext are randomly initialized.
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
