from datetime import datetime
import logging
import math
import os
import tqdm

import utils

def write_predictions(predictions, trg_vocab, args):
    # Datetime object containing current date and time.
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    logger = logging.getLogger('logger')
    logger.info('Saving Predictions to file: {}'.format(dt_string))

    # Checkpoint path.
    checkpoint_path = os.path.join('log', args['checkpoint_path'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    # Convert indices to words.
    prediction_strings = idx_to_words(predictions, trg_vocab)
    file_path = os.path.join(checkpoint_path, 'prediction_' + dt_string + '.txt')
    with open(file_path, 'w') as out_file:
        for sentence in prediction_strings:
            out_file.write(sentence + '\n')

def idx_to_words(predictions, trg_vocab):
    """
    Given a list of lists of indices maps to a list of strings.
    Each string is an example. We use the vocab to map between indices
    and words.
    """
    mapped_predictions = []
    for prediction_example in predictions:
        mapped_example = []
        for index in prediction_example:
            word = trg_vocab.itos[index]
            if word is utils.EOS_WORD:
                break
            mapped_example.append(word)
        mapped_predictions.append(' '.join(mapped_example))
    return mapped_predictions

def eval(model, loss_function, test_iter, args):
    mode = model.training
    model.eval()

    correct = 0
    total = 0
    loss_tot = 0
    eval_results = {}
    predictions = []

    for batch in tqdm.tqdm(test_iter):
        scores = model.forward(batch.src, batch.trg)
        loss = loss_function(scores[:-1,:,:].view(-1, scores.shape[2]), batch.trg[1:,:].view(-1))
        loss_tot += loss.item()
        preds = scores[:-1,:,:].argmax(2).squeeze()
        correct += sum((preds.view(-1) == batch.trg[1:,:].view(-1))).item()
        total += len(preds.view(-1))
        if args['write_to_file']:
            predictions += list(preds.transpose(0,1).tolist())

    eval_results['loss'] = loss_tot/len(test_iter)
    eval_results['perplexity'] = math.exp(loss_tot/len(test_iter))
    eval_results['accuracy'] = correct / total

    # Write predictions to file.
    if args['write_to_file']:
        write_predictions(predictions, model.trg_vocab, args)

    model.train(mode)
    return eval_results
