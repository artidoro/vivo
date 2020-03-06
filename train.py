import torch

import eval

def train(model, optimizer, scheduler, loss_function, train_iter, val_iter, args):
    logger = logging.getLogger('logger')

    for epoch in range(args['train_epochs']):
        logger.info('Starting training for epoch {} of {}.'.format(epoch+1, args['train_epochs']))
        loss_tot = 0
        for batch in tqdm(train_iter):
            optimizer.zero_grad()
            scores = model.forward(batch.text)
            loss = loss_function(scores, batch.label)
            loss.backward()
            optimizer.step()
            loss_tot += loss.item()

        loss_avg = loss_tot/len(train_iter)
        logger.info('Train Loss: {:.4f}'.format(loss_avg))

        if (epoch + 1) % args['eval_epochs'] == 0:
            logger.info('Starting evaluation.')
            evaluation_results = {}
            #evaluation_results['train'] = eval(model, train_iter, args)
            evaluation_results['valid'] = eval.eval(model, loss_function, val_iter, args)
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

