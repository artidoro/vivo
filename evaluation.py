from utils import BOS_WORD, EOS_WORD, UNK_WORD , PAD_WORD

import math
import sacrebleu

def eval(module, test_iter, loss_function, args):
    mode = module.training
    module.eval()

    correct = 0
    total = 0
    loss_tot = 0
    eval_results = {}
    predictions = []

    for batch in tqdm(test_iter):
        scores = module.forward(batch.src, batch.trg)
        loss = loss_function(scores[:-1,:,:].view(-1, scores.shape[2]), batch.trg[1:,:].view(-1)) # t x b x Vocab
        loss_tot += loss.item()
        preds = scores.argmax(1).squeeze()
        correct += sum((preds == batch.trg[1:,:].view(-1))).item()
        total += batch.src.shape[1]
        if args['write_to_file']:
            predictions += list(preds.cpu().numpy())

    eval_results['loss'] = loss_tot/len(test_iter)
    eval_results['accuracy'] = correct / total
    eval_results['perplexity'] = math.exp(loss_tot)
    
    
    # Write predictions to file.
    # if write_to_file:
    #     write_predictions(predictions, args, eval_results)

    module.train(mode)
    return eval_results

def greedy_decoding(model, test_iter, max_decoding_len):
    predictions = []
    import pdb;pdb.set_trace()
    EOS_IDX = model.vocab.stoi(EOS_WORD)
    PAD_IDX = model.vocab.stoi(PAD_WORD)
    vocab_id = torch.eye(len(model.vocab.tgt))
    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_iter):
            current_time_step = 0
            decoded_sentences = torch.zeros(batch.tgt.shape) # t x b x v
            decoded_sentences[:,:,PAD_IDX] = 1 # Transform the entire sentence into pads
            while current_time_step < max_decoding_len and still_decoding:
                logits = model(batch.src, decoded_sentences)
                decoded_sentences[current_time_step,:,:] = vocab_id[logits.argmax(-1)]
                current_time_step += 1
                still_decoding = torch.all(torch.any(torch.argmax(decoded_sentences, dim = -1) == EOS_IDX, dim = 0))
            predictions += list(preds.tolist())
    
    return predictions
                
