from datetime import datetime
import logging
import math
import os
import sacrebleu
import tqdm

from utils import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN

def write_predictions(predictions, args):
    # Datetime object containing current date and time.
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    logger = logging.getLogger('logger')
    logger.info('Saving Predictions to file: {}'.format(dt_string))

    # Checkpoint path.
    checkpoint_path = os.path.join('log', args['checkpoint_path'])
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
        # Convert indices to words.
        prediction_strings = idx_to_TOKENs(predictions, model.trg_vocab)
        write_predictions(prediction_strings, args)

    model.train(mode)
    return eval_results

def greedy_decoding(model, test_iter, args):
    mode = model.training
    model.eval()

    ground_truth = []
    predictions = []
    EOS_IDX = model.trg_vocab.stoi(EOS_TOKEN)
    PAD_IDX = model.trg_vocab.stoi(PAD_TOKEN)
    vocab_id = torch.eye(len(model.trg_vocab))
    with torch.no_grad():
        for batch in tqdm(test_iter):
            still_decoding = True
            current_time_step = 0
            decoded_sentences = torch.zeros(batch.trg.shape) # t x b x v
            decoded_sentences[:,:,PAD_IDX] = 1 # Transform the entire sentence into pads
            ground_truth += batch.trg.tolist()
            while current_time_step < args['max_decoding_len'] and still_decoding:
                logits = model(batch.src, decoded_sentences)
                decoded_sentences[current_time_step,:,:] = vocab_id[logits.argmax(-1)]
                current_time_step += 1
                still_decoding = torch.all(torch.any(torch.argmax(decoded_sentences, dim = -1) == EOS_IDX, dim = 0))
            predictions += list(logits.argmax(-1).transpose(1,0).tolist())

    model.train(mode)
    prediction_strings = idx_to_TOKENs(predictions, model.trg_vocab)
    return predictions, ground_truth

def decode(model, test_iter, args):
    """
    TODO: Think about returning more than just bleu (ex: ppx, loss, ...).
    """
    predictions, ground_truth = greedy_decoding(model, test_iter, args)
    bleu = {'blue': sacrebleu.corpus_bleu(predictions, [ground_truth])}
    if args['write_to_file']:
        write_predictions(predictions, args)
    return bleu