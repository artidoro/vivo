from datetime import datetime
import logging
import math
import os
from typing import Any

import sacrebleu
from tqdm import tqdm
import torch
import numpy as np

import utils
from utils import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN
from loss import VonMisesFisherLoss

def write_predictions(predictions, checkpoint_path):
    # Datetime object containing current date and time.
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    logger = logging.getLogger('vivo_logger')
    logger.info('Saving Predictions to file: {}'.format(dt_string))

    # Checkpoint path.
    checkpoint_path = os.path.join('log', checkpoint_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    file_path = os.path.join(checkpoint_path, 'prediction_' + dt_string + '.txt')
    with open(file_path, 'w') as out_file:
        for sentence in predictions:
            out_file.write(sentence + '\n')

def idx_to_TOKENs(predictions, trg_vocab):
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
            if word is EOS_TOKEN:
                break
            mapped_example.append(word)
        mapped_predictions.append(' '.join(mapped_example))
    return mapped_predictions

def eval(model, loss_function, test_iter, args) -> Any:
    mode = model.training
    model.eval()

    correct = 0
    total = 0
    loss_tot = 0
    eval_results = {}
    predictions = []
    is_vmf_loss = isinstance(loss_function, VonMisesFisherLoss)

    for batch in tqdm(test_iter):
        scores = model.forward(batch.src, batch.trg)
        if is_vmf_loss:
            # Remove first elem of batch.trg which is start token.
            target = model.decoder.embedding(batch.trg[1:,:].view(-1))
        else:
            target = batch.trg[1:,:].view(-1)
        loss = loss_function(scores[:-1,:,:].view(-1, scores.shape[2]), target)
        loss_tot += loss.item()
        if is_vmf_loss:
            # TODO bb: I dont' know why the last elem is removed.
            pred_embeds = scores[:-1,:,:].reshape(-1, scores.shape[-1])
            preds = utils.get_nearest_neighbor(
                pred_embeds,
                model.decoder.embedding.weight,
                return_indexes=True,
            )
            correct += (preds.view(-1) == batch.trg[1:,:].view(-1)).sum().item()
        else:
            preds = scores[:-1,:,:].argmax(2).squeeze()
            correct += (preds.view(-1) == batch.trg[1:,:].view(-1)).sum().item()
        total += len(preds.view(-1))
        if args['write_to_file']:
            predictions += list(preds.transpose(0,1).tolist())

    eval_results['loss'] = loss_tot/len(test_iter)
    eval_results['perplexity'] = math.exp(loss_tot/len(test_iter))
    eval_results['accuracy'] = correct / total

    # Write predictions to file.
    if args['write_to_file']:
        # Convert indices to words.
        prediction_strings = idx_to_TOKENs(predictions, model.trg_vocab)
        write_predictions(prediction_strings, args)

    model.train(mode)
    return eval_results

def _idxs_to_sentences(idxss):
    raw_sentences = [[model.trg_vocab.itos[i] for i in idxs] for idxs in idxss]
    sentences = []
    for rs in raw_sentences:
        s = []
        for tok in rs:
            if tok in (BOS_TOKEN, PAD_TOKEN):
                continue
            elif tok == EOS_TOKEN:
                break
            else:
                s.append(tok)
        sentences.append(" ".join(s))
    return sentences


def greedy_decoding(model, test_iter, max_decoding_len):
    mode = model.training
    model.eval()

    ground_truth = []
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_iter):
            ground_truth += batch.trg.transpose(1, 0).tolist()
            predictions += model.decode(batch.src, max_decoding_len)

    # TODO What is this?
    # model.train(mode)
    # TODO move this function to a better place
    prediction_strings = idxs_to_sentences(predictions)
    gt_strings = idxs_to_sentences(ground_truth)
    assert len(gt_strings) == len(prediction_strings)
    return prediction_strings, gt_strings


def decode(
    model, test_iter, max_decoding_len, write_to_file=False, checkpoint_path=None,
):
    """
    TODO: Think about returning more than just bleu (ex: ppx, loss, ...).
    """
    predictions, ground_truth = greedy_decoding(model, test_iter, max_decoding_len)
    bleu = {"bleu": sacrebleu.corpus_bleu(predictions, [ground_truth])}
    print(bleu["bleu"].score)
    print(ground_truth[0])
    print(predictions[0])
    print(ground_truth[1])
    print(predictions[1])
    if write_to_file:
        assert checkpoint_path is not None
        write_predictions(predictions, checkpoint_path)
    return bleu
