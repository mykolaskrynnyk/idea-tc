from tqdm import tqdm
import evaluate

def train(model, epochs, optimiser, criterion, dataloader_train, dataloader_valid = None, patience: int = 3, progress_bar: bool = False) -> int:
    metric = evaluate.load('f1')
    with tqdm(range(1, epochs + 1), disable=not progress_bar) as t:
        f_score_train, f_score_valid_best, epoch_valid_best = 0., 0., 0
        for epoch in t:
            model.train()
            for idx, (labels, *inputs) in enumerate(dataloader_train):
                optimiser.zero_grad()
                predictions = model(*inputs)
                loss = criterion(predictions, labels)
                loss.backward()
                optimiser.step()
                metric.add_batch(predictions=predictions.argmax(1), references=labels)
                t.set_description(f'{idx}/{len(dataloader_train)} batches | F1-score {f_score_train:.2f}')
            f_score_train = metric.compute(average='weighted')['f1']

            # validation and early stopping
            if dataloader_valid is not None:
                model.eval()
                for labels, *inputs in dataloader_valid:
                    predictions = model(*inputs)
                    metric.add_batch(predictions=predictions.argmax(1), references=labels)
                f_score_valid = metric.compute(average='weighted')['f1']
                if f_score_valid_best <= f_score_valid and f_score_valid_best < 1.:
                    f_score_valid_best = f_score_valid
                    epoch_valid_best = epoch
                else:
                    if epoch - epoch_valid_best > patience:
                        print(f'Early stopping epoch {epoch} | Best F1-score {f_score_valid_best:.2f}')
                        break
            else:
                epoch_valid_best = epoch
    return epoch_valid_best
