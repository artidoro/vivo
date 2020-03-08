import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer,TransformerDecoderLayer

class AttentionEncoderDecoder(nn.Module):
    def __init__(self, src_vocab, trg_vocab, **kargs):
        super(AttentionEncoderDecoder, self).__init__()
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        self.encoder = EncoderLSTM(len(src_vocab), **kargs)
        self.decoder = AttentionDecoder(len(trg_vocab), **kargs)

    def forward(self, src, trg):
        h_encoder = self.encoder(src)
        return self.decoder(trg, h_encoder)

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
    def __init__(self, vocab_size, **kwargs):
        super(EncoderLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, kwargs['enc_embed_size'])
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

    def forward(self, query, values):
        query_resize = self.linear1(query)

        attn_scores = torch.bmm(query_resize, values.transpose(1, 2))
        attn = torch.functional.softmax(attn_scores, dim=-1)
        attended_values = torch.bmm(attn, values)

        query_feed = torch.cat([attended_values, query], dim=-1)
        output_resize = self.tanh(self.linear2(query_feed))

        return output_resize

class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, **kwargs):
        super(AttentionDecoder, self).__init__()
        self.xent = kwargs['loss_function'] == 'xent'
        # LSTM input dimension changes with input feeding.
        lstm_input_size = kwargs['dec_embed_size']
        if kwargs['input_feed']:
            lstm_input_size *= 2

        self.embedding = nn.Embedding(vocab_size, kwargs['dec_embed_size'])
        self.output_init = nn.Parameter(torch.zeros(kwargs['batch_size'],
            kwargs['dec_embed_size']))
        self.hidden_init = nn.Parameter(torch.zeros([kwargs['dec_num_layers'],
            kwargs['batch_size'], kwargs['dec_hidden_size']]))
        self.lstm = nn.LSTM(lstm_input_size, kwargs['dec_hidden_size'],
            num_layers=kwargs['dec_num_layers'], dropout=kwargs['dropout'])
        self.global_attn = GlobalAttention(query_size=kwargs['dec_hidden_size'],
            values_size=kwargs['enc_hidden_size'] * (2 ** kwargs['enc_bidirectional']),
            out_size=kwargs['dec_embed_size'])
        if self.xent:
            self.linear1 = nn.Linear(kwargs['dec_embed_size'], vocab_size)
        self.dropout = nn.Dropout(kwargs['dropout'])

        # Weight tying.
        if kwargs['tie_embed']:
            self.linear1.weight = self.embedding.weight

    def forward(self, trg_batch, h_encoder):
        trg_embeddings = self.embedding(trg_batch)

        output = self.output_init
        hidden = self.hidden_init
        outputs = []
        for i in range(trg_embeddings.shape[0]):
            trg_embed = trg_embeddings[i,:,:]
            input_combined = torch.cat([trg_embed, output], dim=1)
            h_decoder, hidden = self.lstm(input_combined, hidden)
            attended_output = self.global_attn(h_decoder, h_encoder)
            output = self.dropout(attended_output)
            if self.xent:
                output = torch.nn(self.linear1(attended_output), dim=2)
            outputs.append(output)
        return torch.stack(outputs, 1)

model_dict = {
    'lstm_attn': AttentionEncoderDecoder,
    # 'transformer': Transformer
}

loss_dict = {
    'xent': nn.CrossEntropyLoss
}