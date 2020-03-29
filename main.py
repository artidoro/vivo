import argparse
import torch
import os
import logging
import pprint
import sys
import time

import evaluation
import logging_utils
from model import model_dict, loss_dict
import train
import utils

def parse_args(args):
    parser = argparse.ArgumentParser(description='Arguments for the text classification model.')
    # Data related.
    parser.add_argument('--min_freq', default=1, type=int)
    parser.add_argument('--max_len', default=100, type=int,
        help='Filters inputs to be at most the specified length.')
    parser.add_argument('--src_language', default='de',
        help='Choose "de", "fr", or "en".')
    parser.add_argument('--trg_language', default='en',
        help='Choose "de", "fr", or,"en".')
    parser.add_argument('--src_vocab_size', default=50000, type=int)
    parser.add_argument('--trg_vocab_size', default=50000, type=int)

    # Modeling.
    parser.add_argument('--device', default='cpu', help='Select cuda for the gpu.')
    parser.add_argument('--model_name', default='lstm_attn')
    parser.add_argument('--loss_function', default='xent')
    # Encoder.
    parser.add_argument('--enc_embed_size', default=512, type=int)
    parser.add_argument('--enc_hidden_size', default=1024, type=int)
    parser.add_argument('--enc_num_layers', default=1, type=int)
    parser.add_argument('--enc_bidirectional', action='store_true')
    # Decoder.
    parser.add_argument('--dec_embed_size', default=300, type=int)
    parser.add_argument('--dec_hidden_size', default=1024, type=int)
    parser.add_argument('--dec_num_layers', default=2, type=int)
    parser.add_argument('--input_feed', action='store_true')
    parser.add_argument('--tie_embed', action='store_true')
    parser.add_argument('--unk_replace', action='store_true')
    parser.add_argument('--fasttext_embeds_path', default=None,
        help='Path to file containing fasttext embeddings.')

    # Training.
    parser.add_argument('--mode', default='train', choices=['eval','train','test'])
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--train_epochs', default=20, type=int)
    parser.add_argument('--eval_epochs', default=None, type=int)
    parser.add_argument('--eval_examples', default=100000, type=int)
    parser.add_argument('--vmf_lambda_1', default=2e-2, type=float)
    parser.add_argument('--vmf_lambda_2', default=1e-1, type=float)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--factor', default=0.1, type=float, help='Scheduler lr reduction factor.')
    parser.add_argument('--patience', default=100, type=int, help='Scheduler patience.')
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--gradient_clipping', default=5, type=float)
    parser.add_argument('--top_k', default=5, type=int)

    # Save-load ops.
    parser.add_argument('--data_path', default='.data', help='Path to IWSLT16.')
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--load_checkpoint_path', default=None)
    parser.add_argument('--use_checkpoint_args', action='store_true')
    parser.add_argument('--load_optimizer', action='store_true')
    parser.add_argument('--load_scheduler', action='store_true')
    parser.add_argument('--write_to_file', action='store_true', help='Write predictions to file.')
    return vars(parser.parse_args(args))

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    # Initialize logging.
    checkpoint_path = os.path.join('log', args['checkpoint_path'])
    logger = logging_utils.setup_logging(logger_name='vivo_logger', path=checkpoint_path)
    logger.info('Starting with args:\n{}'.format(pprint.pformat(args)))

    # Load the data.
    logger.info('Loading data and building iterators.')
    train_iter, val_iter, test_iter, src_field, trg_field = utils.torchtext_iterators(
        root_data_path=args['data_path'],
        src_language=args['src_language'],
        trg_language=args['trg_language'],
        device=args['device'],
        batch_size=args['batch_size'],
        min_freq=args['min_freq'],
        max_len=args['max_len'],
        fasttext_embeds_path=args['fasttext_embeds_path'],
        src_vocab_size=args['src_vocab_size'],
        trg_vocab_size=args['trg_vocab_size']
    )

    # Initialize model and optimizer. This requires loading checkpoint if specified in the arguments.
    if args['load_checkpoint_path'] is None:
        model = model_dict[args['model_name']](src_field.vocab, trg_field.vocab, **args)
        model.to(torch.device(args['device']))
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
            weight_decay=args['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            factor=args['factor'], patience=args['patience'], verbose=True)
    else:
        # Load checkpoint.
        logger.info('Loading checkpoint {}'.format(args['load_checkpoint_path']))
        checkpoint_path = os.path.join('log', args['load_checkpoint_path'])
        assert os.path.exists(checkpoint_path), 'Checkpoint path {} does not exists.'.format(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(args['device']))

        # Use checkpoint arguments if required.
        if args['use_checkpoint_args']:
            mode = args['mode']
            device = args['device']
            args = checkpoint['args']
            args['mode'] = mode
            args['device'] = device

        # Initialize model, optimizer, scheduler.
        model = model_dict[checkpoint['args']['model_name']](src_field.vocab, trg_field.vocab, **args)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
            lr=args['lr'], weight_decay=args['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            factor=args['factor'], patience=args['patience'], verbose=True)
        # Load the parameters.
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(torch.device(args['device']))
        if args['load_optimizer']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if args['load_scheduler']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if args['mode'] == 'train':
        logger.info('Starting training.')
        if args["loss_function"] == "xent":
            loss_function = loss_dict["xent"](
                ignore_index=trg_field.vocab.stoi[trg_field.pad_token]
            )
        elif args["loss_function"] == "vmf":
            loss_function = loss_dict["vmf"](
                args["dec_embed_size"],
                device=args["device"],
                lambda_1=args["vmf_lambda_1"],
                lambda_2=args["vmf_lambda_2"],
                reduction="mean",
            )
        else:
            raise ValueError(f"Unknown loss function: {args['loss_function']}")
        train.train(
            model,
            optimizer,
            scheduler,
            loss_function,
            train_iter,
            val_iter,
            args,
            ignore_index=trg_field.vocab.stoi[trg_field.pad_token],
        )

    else:
        if args['mode'] == 'eval':
            data_iter = val_iter
        elif args['mode'] == 'test':
            data_iter = test_iter
        else:
            raise NotImplementedError
        logger.info('Starting evaluation.')
        evaluation_results = {}
        evaluation_results[args['mode']] = evaluation.decode(
            model,
            data_iter,
            args['max_len'],
            args['unk_replace'],
            args['write_to_file'],
            args['checkpoint_path']
        )
        logger.info('\n' + pprint.pformat(evaluation_results), args)