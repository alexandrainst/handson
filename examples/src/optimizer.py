import torch
from tqdm.auto import tqdm


def optimize_model(model, dataloaders, optimizer, loss_fnc, epochs, **kwargs):
    """
    Simple tracking of model optimization.
    """
    dataset_types = ['train', 'val']
    assert set(dataset_types).issubset(dataloaders), f'dataloaders are expected to contain {dataset_types}'
    losses = {dataset_type: [] for dataset_type in dataset_types}
    accuracies = {dataset_type: [] for dataset_type in dataset_types}

    for epoch in (pbar := tqdm(range(epochs),desc=f"Training...")):
        for dataset_type in dataset_types:
            loss, accuracy = train_or_eval_model(model, dataloaders[dataset_type], optimizer, loss_fnc, is_training=dataset_type=='train')
            losses[dataset_type].append(loss)
            accuracies[dataset_type].append(accuracy)
        pbar.set_postfix(loss_train=losses['train'][-1], loss_val=losses['val'][-1], accuracy_train = accuracies['train'][-1], accuracy_val=accuracies['val'][-1])
    return losses, accuracies

def train_or_eval_model(model, dataloader, optimizer, loss_fnc, is_training):
    """
    Runs a dataloader through a model.
    Supports both training and evaluation mode.
    """
    model.train(is_training)
    torch.set_grad_enabled(is_training)
    loss_agg = 0.0
    true_predictions = 0
    predictions = 0
    assert len(dataloader) > 0, f"The dataloader should contain at least one batch of samples."
    for i, (x,target) in enumerate(dataloader):
        prediction_prob = model(x)
        loss = loss_fnc(prediction_prob,target)
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_agg += loss.item()
        prediction = torch.argmax(prediction_prob,dim=1)
        true_predictions += (prediction == target).sum().item()
        predictions += target.shape[0]

    accuracy_average = true_predictions / predictions
    loss_average = loss_agg / predictions
    return loss_average, accuracy_average



