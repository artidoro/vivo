from datetime import datetime
import logging
import os
import pprint
import torch
import tqdm
from typing import Any

import evaluation
from loss import VonMisesFisherLoss

def train(model, optimizer, scheduler, loss_function, train_iter, val_iter, args, ignore_index=-100) -> Any:
    logger = logging.getLogger('vivo_logger')

    for epoch in range(args['train_epochs']):
        logger.info('Starting training for epoch {} of {}.'.format(epoch+1, args['train_epochs']))
        loss_tot = 0
        for batch in tqdm.tqdm(train_iter):
            optimizer.zero_grad()
            scores = model(batch.src, batch.trg)
            if isinstance(loss_function, VonMisesFisherLoss):
                target = model.decoder.embedding(batch.trg[1:,:].view(-1))
                raw_loss = loss_function(scores[:-1,:,:].view(-1, scores.shape[2]), target)
                mask = (batch.trg[1:,:].view(-1) != ignore_index)
                loss = (raw_loss * mask).sum() / mask.sum()
            else:
                target = batch.trg[1:,:].view(-1)
                loss = loss_function(scores[:-1,:,:].view(-1, scores.shape[2]), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['gradient_clipping'])
            optimizer.step()
            loss_tot += loss.item()
            evaluation.greedy_decoding(
    model,
    val_iter,
    args['max_len'],
    args['unk_replace']
)
        loss_avg = loss_tot/len(train_iter)
        logger.info('Train Loss: {:.4f}'.format(loss_avg))

        if (epoch + 1) % args['eval_epochs'] == 0:
            logger.info('Starting evaluation.')
            evaluation_results = {}
            evaluation_results['valid'] = evaluation.eval(model, loss_function, val_iter, args, ignore_index=ignore_index)
            logger.info('\n' + pprint.pformat(evaluation_results))

            # Update the scheduler.
            scheduler.step(evaluation_results['valid']['loss'])

            # Checkpoint
            checkpoint_path = os.path.join('log', args['checkpoint_path'])
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            # datetime object containing current date and time
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
            logger.info('Saving Checkpoint: {}'.format(dt_string))

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': loss_avg,
                'evaluation_results': evaluation_results,
                'args': args
                }, os.path.join(checkpoint_path, dt_string + '.pt'))

