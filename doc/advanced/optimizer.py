"""


"""
import torch
from tqdm.auto import tqdm


def optimize_model(model, dataloaders, optimizer, loss_fnc, epochs, **kwargs):
    """
    Minimal working example of optimization.
    """
    dataset_types = ['train', 'val']
    for epoch in range(epochs):
        for dataset_type in dataset_types:
            train_or_eval_model(model, dataloaders[dataset_type], optimizer, loss_fnc, is_training=dataset_type=='train')
    return

def train_or_eval_model(model, dataloader, optimizer, loss_fnc, is_training):
    """
    Minimal working example of optimization.
    """
    model.train(is_training)
    torch.set_grad_enabled(is_training)
    for i, (x,target) in enumerate(dataloader):
        prediction_prob = model(x)
        loss = loss_fnc(prediction_prob,target)
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return


def optimize_model_with_validation(model, dataloaders, optimizer, loss_fnc, epochs, **kwargs):
    """
    An example of how validation might be used to track performance and keep the best performing model.
    Note that save_model and load_model are not defined and needs to actually be made for this to work.

    """
    dataset_types = ['train', 'val']
    assert set(dataset_types).issubset(dataloaders), f'dataloaders are expected to contain {dataset_types}'
    losses = {dataset_type: [] for dataset_type in dataset_types}
    accuracies = {dataset_type: [] for dataset_type in dataset_types}
    best_loss_val = 1e9
    for epoch in (pbar := tqdm(range(epochs),desc=f"Training...")):
        for dataset_type in dataset_types:
            loss, accuracy = train_or_eval_model(model, dataloaders[dataset_type], optimizer, loss_fnc, is_training=dataset_type=='train')
            losses[dataset_type].append(loss)
            accuracies[dataset_type].append(accuracy)
        if losses['val'][-1] < best_loss_val:
            save_model(model,model_checkpoint_path)
            best_loss_val = losses['val'][-1]
        pbar.set_postfix(loss_train=losses['train'][-1], loss_val=losses['val'][-1], accuracy_train = accuracies['train'][-1], accuracy_val=accuracies['val'][-1])
    load_model(model,model_checkpoint_path)
    return losses, accuracies
