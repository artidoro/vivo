from typing import List, Tuple

import math
import numpy as np
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer,TransformerDecoderLayer
import torch
import torch.nn as nn

import loss
import utils

class AttentionEncoderDecoder(nn.Module):
    def __init__(self, src_vocab, trg_vocab, **kargs):
        super(AttentionEncoderDecoder, self).__init__()
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        self.encoder = EncoderLSTM(src_vocab, **kargs)
        self.decoder = AttentionDecoder(trg_vocab, **kargs)

    def forward(self, src, trg):
        h_encoder = self.encoder(src)
        return self.decoder(trg, h_encoder)

    def decode(self, src: Tensor, max_decoding_len: int) -> List[List[int]]:
        h_encoder = self.encoder(src)
        bos_idx = self.trg_vocab.stoi[utils.BOS_TOKEN]
        eos_idx = self.trg_vocab.stoi[utils.EOS_TOKEN]
        return self.decoder.decode(h_encoder, max_decoding_len, bos_idx, eos_idx)

class Transformer(nn.Module):
    ''' Transformer '''
    class PositionalEncoding(nn.Module):
        ''' Position encoder based on Attention is all you need paper.
            Inputs:
                input_dimension: Dimension of word embeddings
                dropout: Dropout rate [0-1]
                max_len: Fixed length input
            Outputs:
                Embeddings with position information
        '''
        def __init__(self, input_dim, dropout, max_len):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            position_embeddings = torch.zeros(max_len, input_dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, input_dim, 2).float() * (-math.log(10000.0) / input_dim))
            position_embeddings[:, 0::2] = torch.sin(position * div_term)
            position_embeddings[:, 1::2] = torch.cos(position * div_term)
            position_embeddings = position_embeddings.unsqueeze(0).transpose(0, 1)
            self.register_buffer('position_embeddings', position_embeddings)

        def forward(self, x):
            ''' Sums the position embeddings, embeddings and dropsout over them '''
            return self.dropout(x + self.position_embeddings[:x.size(0), :])

    def __init__(self, src_vocab, trg_vocab, **kwargs):
        super(Transformer, self).__init__()
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = kwargs['max_len']
        self.dropout = kwargs['dropout']
        self.input_dim = kwargs['input_dim']
        self.num_layers = kwargs['num_layers']
        self.num_heads = kwargs['num_heads']
        self.hidden_dim = kwargs['hidden_dim']

        # Input embeddings
        self.src_embeddings = nn.Embedding(len(self.src_vocab), self.input_dim)
        self.trg_embeddings = nn.Embedding(len(self.trg_vocab), self.input_dim)
        nn.init.xavier_uniform_(self.src_embeddings.weight)
        nn.init.xavier_uniform_(self.trg_embeddings.weight)

        # Init position encoder, nested class
        self.position_encoder = PositionalEncoding(self.input_dim ,
         self.dropout, self.max_len)

        # Encoder and decoder
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(self.input_dim, self.num_heads,
             self.hidden_dim, self.dropout),
            self.num_layers)

        self.transformer_decoder = TransformerDecoder(
            TransformerDecoderLayer(self.input_dim, self.num_heads,
            self.hidden_dim, self.dropout),
            self.num_layers)

        # Output layer
        self.output_layer = nn.Linear(self.hidden_dim, len(self.trg_vocab))

    # from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    def forward(self, src_sents, trg_sents, mask=None):
        # Mask the decoding to avoid looking forward
        if mask is None or mask.size(0) != len(trg_sents):
            mask = self._generate_square_subsequent_mask(len(trg_sents)).to(trg_sents.device)

        src_embeddings = self.src_embeddings(src_sents) * math.sqrt(self.input_dim)
        src_embeddings = self.pos_encoder(src_embeddings)

        trg_embeddings = self.trg_embeddings(trg_sents) * math.sqrt(self.input_dim)
        trg_embeddings = self.pos_encoder(trg_embeddings)

        encoder_output = self.transformer_encoder(src_embeddings)
        decoder_output = self.transformer_decoder(trg_embeddings, encoder_output,
         trg_mask = mask)

        return self.output_layer(decoder_output)

class EncoderLSTM(nn.Module):
    def __init__(self, vocab, **kwargs):
        super(EncoderLSTM, self).__init__()
        self.embedding = nn.Embedding(len(vocab), kwargs['enc_embed_size'])
        self.lstm = nn.LSTM(kwargs['enc_embed_size'], kwargs['enc_hidden_size'],
            kwargs['enc_num_layers'], dropout=kwargs['dropout'],
            bidirectional=kwargs['enc_bidirectional'])

    def forward(self, src_batch):
        src_embed = self.embedding(src_batch)
        output, _ = self.lstm(src_embed)
        return output

class GlobalAttention(nn.Module):
    def __init__(self, query_size, values_size, out_size):
        super(GlobalAttention, self).__init__()
        self.linear1 = nn.Linear(query_size, values_size)
        self.linear2 = nn.Linear(query_size + values_size, out_size)
        self.tanh = nn.Tanh()
        self.alphas = []

    def forward(self, query, values):
        query_resize = self.linear1(query) # t=1 x b x d

        attn_scores = torch.bmm(query_resize.transpose(0,1), values.permute(1,2,0))
        attn = torch.softmax(attn_scores, dim=-1)
        self.alphas = attn.detach().squeeze(1) # b x num_src_words
        attended_values = torch.bmm(attn, values.transpose(0,1)).transpose(0,1)

        query_feed = torch.cat([attended_values, query], dim=-1)
        output_resize = self.tanh(self.linear2(query_feed))

        return output_resize

class AttentionDecoder(nn.Module):
    def __init__(self, vocab, **kwargs):
        super(AttentionDecoder, self).__init__()
        self.xent = kwargs["loss_function"] == "xent"
        # LSTM input dimension changes with input feeding.
        lstm_input_size = kwargs["dec_embed_size"]
        if kwargs["input_feed"]:
            lstm_input_size *= 2
        self.input_feed = kwargs["input_feed"]

        self.embedding = nn.Embedding(len(vocab), kwargs["dec_embed_size"])
        if kwargs['fasttext_embeds_path']:
            self.embedding.weight = nn.Parameter(vocab.vectors)
        if not self.xent:
            # Freeze embeddings when using VMF.
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            lstm_input_size,
            kwargs["dec_hidden_size"],
            num_layers=kwargs["dec_num_layers"],
            dropout=kwargs["dropout"],
        )
        self.global_attn = GlobalAttention(
            query_size=kwargs["dec_hidden_size"],
            values_size=kwargs["enc_hidden_size"] * (2 ** kwargs["enc_bidirectional"]),
            out_size=kwargs["dec_embed_size"],
        )
        if self.xent:
            self.linear1 = nn.Linear(kwargs["dec_embed_size"], len(vocab))
            # Weight tying.
            if kwargs["tie_embed"]:
                self.linear1.weight = self.embedding.weight

        self.dropout = nn.Dropout(kwargs["dropout"])
        self.attention = []

    def step(
        self,
        token_embedding: Tensor,
        model_output: Tensor,
        hidden: Tensor,
        h_encoder: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        input_combined = token_embedding
        if self.input_feed:
            input_combined = torch.cat([input_combined, model_output], dim=-1)
        h_decoder, hidden = self.lstm(input_combined, hidden)
        attended_output = self.global_attn(h_decoder, h_encoder)
        self.attention.append(self.global_attn.alphas) # num_target_words x b x num_src_words
        return self.dropout(attended_output), hidden

    def forward(self, trg_batch: Tensor, h_encoder: Tensor) -> Tensor:
        trg_embeddings = self.embedding(trg_batch)

        output = torch.zeros([1, trg_batch.shape[1], trg_embeddings.shape[2]]).to(
            self.embedding.weight.device
        )
        hidden = None
        outputs = []
        self.attention = []
        for i in range(trg_embeddings.shape[0]):
            trg_embed = trg_embeddings[i : i + 1, :, :]  # t=1 x b x d
            output, hidden = self.step(trg_embed, output, hidden, h_encoder)
            if self.xent:
                output_projected = self.linear1(output)
                outputs.append(output_projected)
            else:
                outputs.append(output)
        return torch.cat(outputs, 0)

    def _batched_nearest_neighbor(self, x: Tensor, size: int) -> Tensor:
        idxs = [
            utils.get_nearest_neighbor(
                x[:, i : i + size, ...], self.embedding.weight, return_indexes=True
            )
            for i in range(0, x.shape[1], size)
        ]
        return torch.cat(idxs, dim=1)

    def decode(
        self, h_encoder: Tensor, max_decoding_len: int, bos_idx: int, eos_idx: int,
    ) -> List[List[int]]:
        bos_idx_tensor = torch.LongTensor([bos_idx]).to(self.embedding.weight.device)
        batch_size = h_encoder.shape[1]
        model_out = output = torch.zeros(
            [1, batch_size, self.embedding.weight.shape[1]]
        ).to(self.embedding.weight.device)
        hidden = None
        decoded_idxs = [
            torch.LongTensor([bos_idx])
            .repeat(batch_size)
            .unsqueeze(0)
            .to(self.embedding.weight.device)
        ]
        eos_generated = np.zeros((1, batch_size), dtype=np.bool)
        while len(decoded_idxs) < max_decoding_len and (eos_generated == 0).any():
            decoded_embeds = self.embedding(decoded_idxs[-1])
            model_out, hidden = self.step(decoded_embeds, model_out, hidden, h_encoder,)
            if self.xent:
                model_sm = self.linear1(model_out)
                decoded_idxs.append(model_sm.argmax(-1))
            else:
                # TODO Dynamically set eval batch size
                idxs = self._batched_nearest_neighbor(model_out, 128)
                decoded_idxs.append(idxs)
            eos_generated += (decoded_idxs[-1] == eos_idx).cpu().numpy()

        return (
            np.array([x.cpu().numpy() for x in decoded_idxs])
            .squeeze()
            .transpose(1, 0)
            .tolist()
        )


model_dict = {
    "lstm_attn": AttentionEncoderDecoder,
    # "transformer": Transformer
}

loss_dict = {
    "xent": nn.CrossEntropyLoss,
    "vmf": loss.VonMisesFisherLoss,
}
