import argparse
import torch
import os
import logging
import pprint
import sys
import time

import logging_utils
from model import model_dict, loss_dict
import train
import utils

def parse_args(args):
    parser = argparse.ArgumentParser(description='Arguments for the text classification model.')
    # Data related.
    #parser.add_argument('--train_path', default='../topicclass/topicclass_train.txt')
    #parser.add_argument('--valid_path', default='../topicclass/topicclass_valid.txt')
    #parser.add_argument('--test_path', default='../topicclass/topicclass_test.txt')
    parser.add_argument('--min_freq', default=1, type=int)

    # Modeling.
    parser.add_argument('--device', default='cpu', help='Select cuda for the gpu.')
    parser.add_argument('--model_name', default='lstm_attn')
    parser.add_argument('--loss_function', default='xent')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--train_epochs', default=20, type=int)
    parser.add_argument('--eval_epochs', default=1, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--factor', default=0.1, type=float, help='Scheduler lr reduction factor.')
    parser.add_argument('--patience', default=100, type=int, help='Scheduler patience.')


    #parser.add_argument('--alpha', default=1, type=float)
    #parser.add_argument('--batch_size', default=10, type=int)
    #parser.add_argument('--embed_size', default=300, type=int)
    #parser.add_argument('--hidden_size', default=100, type=int)
    #parser.add_argument('--dropout', default=0.5, type=float)
    #parser.add_argument('--num_layers', default=1, type=int)
    #parser.add_argument('--no_pretrained_vectors', action='store_true')
    #parser.add_argument('--num_heads', default=5, type=int)

    # Save-load ops.
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--load_checkpoint_path', default=None)
    parser.add_argument('--use_checkpoint_args', action='store_true')
    parser.add_argument('--load_optimizer', action='store_true')
    parser.add_argument('--load_scheduler', action='store_true')
    return vars(parser.parse_args(args))


if __name__ == '__main__':
    args = parse_args(sys.argv[1])

    # Initialize logging.
    checkpoint_path = os.path.join('log', args['checkpoint_path'])
    logger = logging_utils.setup_logging(logger_name='logger', path=checkpoint_path)

    # Load the data.
    logger.info('Starting to train text classification model with args:\n{}'.format(pprint.pformat(args)))
    train_iter, val_iter, test, en_field, de_field = utils.torchtext_iterators(
        device=args['device'], batch_size=args['batch_size'], min_freq=args['min_freq'])

    # Initialize model and optimizer. This requires loading checkpoint if specified in the arguments.
    if args['load_checkpoint_path'] == None:
        model = model_dict[args['model_name']](en_field.vocab, de_field.vocab, **args)
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
        checkpoint = torch.load(checkpoint_path)

        # Use checkpoint arguments if required.
        if args['use_checkpoint_args']:
            args = checkpoint['args']

        # Initialize model, optimizer, scheduler.
        model = model_dict[checkpoint['args']['model_name']]({}, {})
        model.to(torch.device(args['device']))
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
            weight_decay=args['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            factor=args['factor'], patience=args['patience'], verbose=True)
        # Load the parameters.
        model.load_state_dict(checkpoint['model_state_dict'])
        if args['load_optimizer']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if args['load_scheduler']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if args['mode'] == 'train':
        logger.info('Starting training.')
        train.train(module, optimizer, scheduler,
            loss_dict[args['loss_function']](de_field.vocab.stoi[de_field.pad_token]),
            train_iter, val_iter, args)

    elif args['mode'] == 'eval':
        logger.info('Starting evaluation.')
        evaluation_results = {}
        # evaluation_results['train'] = utils.eval(module, train_iter, args)
        evaluation_results['valid'] = utils.eval(module, val_iter, args)
        logger.info('\n' + pprint.pformat(evaluation_results), args)

    elif args['mode'] == 'test':
        logger.info('Starting testing.')
        utils.predict_write_to_file(module, test, args)
        logger.info('Done writing predictions to file.')