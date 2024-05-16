import torch

def select_loss_fnc(loss_type,**kwargs):
    """
    Wrapper for various loss functions.

    The two main types of loss functions are mean square error loss functions, which are used for regression problems
    and cross entropy loss functions which are used for catagorical problems.

    The important thing to remember about loss functions is they need to lead to stable training.
    L1 loss is for instance generally not a good loss function since the gradient of such a loss function does not go towards zero as we approach our desired solution.
    
    """

    if loss_type == 'mse':
        loss_fnc = torch.nn.MSELoss()
    elif loss_type == 'bce_logitloss':
        loss_fnc = torch.nn.BCEWithLogitsLoss()
    elif loss_type == 'cross_entropy':
        loss_fnc = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"loss type: {loss_type} has not been implemented.")
    return loss_fnc