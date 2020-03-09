
def eval(module, test_iter, loss_function, args):
    mode = module.training
    module.eval()

    correct = 0
    total = 0
    loss_tot = 0
    eval_results = {}
    predictions = []

    for batch in tqdm(test_iter):
        scores = module.forward(batch.text)
        loss = model.loss(scores, batch.label)
        loss_tot += loss.item()
        preds = scores.argmax(1).squeeze()
        correct += sum((preds == batch.label)).item()
        total += batch.text.shape[0]
        # if write_to_file:
        # if write_to_file:
        #     predictions += list(preds.cpu().numpy())

    eval_results['loss'] = loss_tot/len(test_iter)
    eval_results['accuracy'] = correct / total
    # TODO: Perplexity.

    # Write predictions to file.
    # if write_to_file:
    #     write_predictions(predictions, args, eval_results)

    module.train(mode)
    return eval_results
