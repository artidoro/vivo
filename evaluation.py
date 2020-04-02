from datetime import datetime
import numpy as np
import logging
import math
from pathlib import Path
from typing import Any, List, Tuple, Dict, Union
import torch
from sacremoses import MosesDetokenizer
import sacrebleu
from tqdm import tqdm


import utils
from utils import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN
from loss import VonMisesFisherLoss

PathLike = Union[Path, str]

def loadUnkLUT(unk_LUT_path):
    import pickle
    return pickle.load(open(unk_LUT_path,'rb'))

def write_predictions(
    predictions: List[str],
    ground_truth: List[str],
    checkpoint_path: PathLike
) -> None:
    log_path = Path("log")
    dir_path = log_path / checkpoint_path
    # Datetime object containing current date and time.
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    logger = logging.getLogger('vivo_logger')
    logger.info(f"Saving Predictions to file: {dt_string}")
    if not dir_path.exists():
        dir_path.mkdir()
    file_path = dir_path / f"prediction_{dt_string}.txt"
    with file_path.open("w") as out_file:
        for i in range(len(predictions)):
            out_file.write(ground_truth[i] + " ||| " + predictions[i] + "\n")


def eval(model, loss_function, test_iter, args, ignore_index=-100) -> Any:
    mode = model.training
    model.eval()

    top_k_val = args['top_k']
    correct = np.zeros(top_k_val)
    total = 0
    loss_tot = 0
    eval_results = {}
    predictions = []
    ground_truth = []
    prediction_strings = []
    is_vmf_loss = isinstance(loss_function, VonMisesFisherLoss)
    for batch in tqdm(test_iter):
        scores = model.forward(batch.src, batch.trg)
        attn_vectors = torch.stack(model.decoder.attention).permute(1,0,2)
        mask = (batch.trg[1:,:].view(-1) != ignore_index)
        if is_vmf_loss:
            target = model.decoder.embedding(batch.trg[1:,:].view(-1))
        else:
            target = batch.trg[1:,:].view(-1)
        loss = loss_function(scores[:-1,:,:].view(-1, scores.shape[2]), target)
        if args['loss_reduction'] == 'sentence_mean':
                loss = loss.view(-1, scores.shape[1]).sum(0).mean()
        loss_tot += loss.item()
        if is_vmf_loss:
            pred_embeds = scores[:-1,:,:]
            eval_batch_size = 128 // pred_embeds.shape[0]
            assert eval_batch_size > 0
            def batch_predict(i, x):
                return utils.get_nearest_neighbor(
                    x[:, i:i+eval_batch_size, ...],
                    model.decoder.embedding.weight,
                    top_k=top_k_val,
                )
            _preds = [
                batch_predict(i, pred_embeds)
                for i in range(0, pred_embeds.shape[1], eval_batch_size)]
            preds = torch.cat(_preds, dim=1)
        else:
            preds = torch.topk(scores[:-1,:,:], top_k_val, sorted=True).indices
        for k in range(top_k_val):
            correct_tokens = preds[..., :k + 1] == batch.trg[1:, :, np.newaxis]
            correct_tokens_in_top_k = correct_tokens.any(-1)
            total_correct_tokens = correct_tokens_in_top_k.sum().item()
            correct[k] += total_correct_tokens
        # Keep only the top 1
        preds = preds[..., 0]
        total += mask.sum().item()
        if args['write_to_file']:
            ground_truth += idxs_to_sentences(batch.trg.transpose(0, 1).tolist(), model.trg_vocab)
            predictions = preds.transpose(0, 1).tolist()
            if args['unk_replace']:
                copy_lut = model.src_vocab
                if args['unk_lut_path']:
                    copy_lut = loadUnkLUT(args['unk_lut_path']) 
                prediction_strings += idxs_to_sentences(predictions, model.trg_vocab,
                    src_sents=batch.src.permute(1,0), copy_lut=copy_lut, attn=attn_vectors)
            else:
                prediction_strings += idxs_to_sentences(predictions, model.trg_vocab)

    eval_results['loss'] = loss_tot/len(test_iter)
    eval_results['perplexity'] = math.exp(loss_tot/len(test_iter))
    for i in range(top_k_val):
        eval_results[f'accuracy_top_{i + 1}'] = correct[i] / total

    # Write predictions to file.
    if args['write_to_file']:
        # Convert indices to words.
        write_predictions(
            prediction_strings,
            ground_truth,
            args["checkpoint_path"]
        )

    model.train(mode)
    return eval_results

def idxs_to_sentences(
    predictions,
    vocab,
    src_sents = None,
    copy_lut = None,
    attn = None,
    deduplicate = True
) -> List[str]:
    mapped_predictions = []
    prev_word = ""
    md = MosesDetokenizer(lang='en')
    for pred_idx, prediction_example in enumerate(predictions):
        mapped_example = []
        # Iterates through sentence to find first EOS or decodes the entire sentence
        sent_len = next((pos for pos,word_idx in enumerate(prediction_example) if word_idx == vocab.stoi[EOS_TOKEN]),len(prediction_example) -1) 
        for index_idx, index in enumerate(prediction_example[:sent_len]):
            word = vocab.itos[index]
            if word is EOS_TOKEN:
                break
            elif word in (BOS_TOKEN, PAD_TOKEN):
                continue
            elif (
                word is UNK_TOKEN
                and type(src_sents) != type(None)
                and type(attn) != type(None)
                and type(copy_lut) != type(None)
            ):
                _, max_attn_idx = attn[pred_idx,index_idx].max(-1)
                word = copy_lut.itos[src_sents[pred_idx, max_attn_idx]]
            if deduplicate and prev_word == word:
                continue
            prev_word = word
            mapped_example.append(word)

        mapped_predictions.append(md.detokenize(mapped_example))
    return mapped_predictions

def greedy_decoding(
    model,
    test_iter,
    max_decoding_len,
    args
) -> Tuple[List[str], List[str]]:
    mode = model.training
    model.eval()
    ground_truth = []
    prediction_strings = []
    with torch.no_grad():
        for batch in tqdm(test_iter):
            ground_truth += batch.trg.transpose(1, 0).tolist()
            predictions = model.decode(batch.src, max_decoding_len)
            if args['unk_replace']:
                attn_vectors = torch.stack(model.decoder.attention).permute(1,0,2)
                copy_lut = model.src_vocab
                if args['unk_lut_path']:
                    copy_lut = loadUNKLut(args['unk_lut_path']) 

                prediction_strings += idxs_to_sentences(
                    predictions,
                    model.trg_vocab,
                    src_sents=batch.src.permute(1,0),
                    copy_lut=copy_lut,
                    attn=attn_vectors
                )
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
    write_to_file,
    checkpoint_path,
    args
) -> Dict:
    # TODO: Think about returning more than just bleu (ex: ppx, loss, ...).
    predictions, ground_truth = greedy_decoding(
        model,
        test_iter,
        max_decoding_len,
        args
    )
    bleu = {"bleu": sacrebleu.corpus_bleu(predictions, [ground_truth]).score}
    if write_to_file:
        assert checkpoint_path is not None
        write_predictions(predictions, ground_truth, checkpoint_path)
    return bleu
