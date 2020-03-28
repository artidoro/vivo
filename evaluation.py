from datetime import datetime
from typing import Any, List, Tuple, Dict
import numpy as np
import logging
import math
from pathlib import Path
from typing import Any, List, Tuple, Dict, Union
import torch

import sacrebleu
from tqdm import tqdm


import utils
from utils import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN
from loss import VonMisesFisherLoss

PathLike = Union[Path, str]


def write_predictions(predictions: List[str], checkpoint_path: PathLike) -> None:
    log_path = Path("log")
    dir_path = log_path / checkpoint_path
    # Datetime object containing current date and time.
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    logger = logging.getLogger("vivo_logger")
    logger.info(f"Saving Predictions to file: {dt_string}")
    if not dir_path.exists():
        dir_path.mkdir()
    file_path = dir_path / f"prediction_{dt_string}.txt"
    with file_path.open("w") as out_file:
        for sentence in predictions:
            out_file.write(sentence + "\n")


def eval(model, loss_function, test_iter, args, ignore_index=-100) -> Any:
    mode = model.training
    model.eval()

    correct = 0
    total = 0
    loss_tot = 0
    eval_results = {}
    predictions = []
    prediction_strings = []
    is_vmf_loss = isinstance(loss_function, VonMisesFisherLoss)
    for batch in tqdm(test_iter):
        scores = model.forward(batch.src, batch.trg)
        attn_vectors = torch.stack(model.decoder.attention).permute(1,0,2)
        mask = (batch.trg[1:,:].view(-1) != ignore_index)
        if is_vmf_loss:
            # Remove first elem of batch.trg which is start token.
            target = model.decoder.embedding(batch.trg[1:,:].view(-1))
            raw_loss = loss_function(scores[:-1,:,:].view(-1, scores.shape[2]), target)
            loss = (raw_loss * mask).sum() / mask.sum()
        else:
            target = batch.trg[1:,:].view(-1)
            loss = loss_function(scores[:-1,:,:].view(-1, scores.shape[2]), target)
        loss_tot += loss.item()
        if is_vmf_loss:
            pred_embeds = scores[:-1,:,:]
            eval_batch_size = 128 // pred_embeds.shape[0]
            assert eval_batch_size > 0
            def batch_predict(i, x):
                return utils.get_nearest_neighbor(
                    x[:, i:i+eval_batch_size, ...],
                    model.decoder.embedding.weight,
                    return_indexes=True,
                )
            _preds = [
                batch_predict(i, pred_embeds)
                for i in range(0, pred_embeds.shape[1], eval_batch_size)]
            preds = torch.cat(_preds, dim=1)
            correct += (preds.view(-1) == batch.trg[1:,:].view(-1)).sum().item()
        else:
            preds = scores[:-1,:,:].argmax(2).squeeze()
            correct += (preds.view(-1) == batch.trg[1:,:].view(-1)).sum().item()
        total += mask.sum().to(torch.float32)
        if args['write_to_file']:
            predictions = list(preds.transpose(0,1).tolist())
            if args['unk_replace']: 
                prediction_strings += idxs_to_sentences(predictions, model.trg_vocab, src_sents = batch.src.permute(1,0), copy_lut = model.src_vocab, attn = attn_vectors)
            else:
                prediction_strings += idxs_to_sentences(predictions, model.trg_vocab)

    eval_results['loss'] = loss_tot/len(test_iter)
    eval_results['perplexity'] = math.exp(loss_tot/len(test_iter))
    eval_results['accuracy'] = correct / total

    # Write predictions to file.
    if args['write_to_file']:
        # Convert indices to words.
        write_predictions(prediction_strings, args["checkpoint_path"])

    model.train(mode)
    return eval_results

def idxs_to_sentences(predictions, vocab, src_sents = None, copy_lut = None, attn = None) -> List[str]:
    mapped_predictions = []
    for pred_idx, prediction_example in enumerate(predictions):
        mapped_example = []
        for index_idx, index in enumerate(prediction_example):
            word = vocab.itos[index]
            if word is EOS_TOKEN:
                break
            elif word in (BOS_TOKEN, PAD_TOKEN):
                continue
            elif word is UNK_TOKEN and type(src_sents) != type(None) and type(attn) != type(None) and type(copy_lut) != type(None):
                _, max_attn_idx = attn[pred_idx,index_idx].max(-1)
                word = copy_lut.itos[src_sents[pred_idx, max_attn_idx]]
            
            mapped_example.append(word)

        mapped_predictions.append(' '.join(mapped_example))
    return mapped_predictions

def greedy_decoding(
    model,
    test_iter,
    max_decoding_len,
    unk_replace
) -> Tuple[List[str], List[str]]:
    mode = model.training
    model.eval()
    ground_truth = []
    prediction_strings = []
    with torch.no_grad():
        for batch in tqdm(test_iter):
            ground_truth += batch.trg.transpose(1, 0).tolist()
            predictions = model.decode(batch.src, max_decoding_len)
            if unk_replace:
                attn_vectors = torch.stack(model.decoder.attention).permute(1,0,2)
                prediction_strings += idxs_to_sentences(predictions, model.trg_vocab, src_sents = batch.src.permute(1,0), copy_lut = model.src_vocab, attn = attn_vectors)
            else:
                prediction_strings += idxs_to_sentences(predictions, model.trg_vocab)
    model.train(mode)
    gt_strings = idxs_to_sentences(ground_truth, model.trg_vocab)
    assert len(gt_strings) == len(prediction_strings)
    return prediction_strings, gt_strings


def decode(
    model,
    test_iter,
    max_decoding_len,
    unk_replace,
    write_to_file=True,
    checkpoint_path="checkpoint",
) -> Dict:
    # TODO: Think about returning more than just bleu (ex: ppx, loss, ...).
    predictions, ground_truth = greedy_decoding(
        model,
        test_iter,
        max_decoding_len,
        unk_replace
    )
    bleu = {"bleu": sacrebleu.corpus_bleu(predictions, [ground_truth]).score}
    if write_to_file:
        assert checkpoint_path is not None
        write_predictions(predictions, checkpoint_path)
    return bleu
