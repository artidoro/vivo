import torch
import torchtext


BOS_WORD = '<s>'
EOS_WORD = '</s>'

def torchtext_iterators(device='cpu', batch_size=32,  min_freq=1):
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    de_field = torchtext.data.Field(tokenize=tokenize_de)
    en_field = torchtext.data.Field(tokenize=tokenize_en,
        init_token=BOS_WORD, eos_token=EOS_WORD)  # only target needs BOS/EOS

    # TODO: actually load the WMT 16, sachin data.
    train, val, test = torchtext.datasets.IWSLT.splits(exts=('.de', '.en'),
        fields=(de_field, en_field))

    de_field.build_vocab(train.src, min_freq=min_freq)
    en_field.build_vocab(train.trg, min_freq=min_freq)

    train_iter, val_iter = torchtext.data.BucketIterator.splits((train, val),
        batch_size=batch_size, device=torch.device(device), repeat=False,
        sort_key=lambda x: len(x.src))

    # TODO: Load pretrained embeddings.
    # en_field.vocab.load_vectors(vectors=GloVe(name='6B', dim=300))
    return train_iter, val_iter, test, de_field, en_field