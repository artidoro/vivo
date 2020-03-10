
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
    # TODO: Perplexity.

    # Write predictions to file.
    # if write_to_file:
    #     write_predictions(predictions, args, eval_results)

    module.train(mode)
    return eval_results
