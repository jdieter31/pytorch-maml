import torch

from collections import OrderedDict
from torchmeta.modules import MetaModule
from .model import BatchLinear
from .transformer_metric import TransformerMetric

from .expm import torch_expm as expm

def compute_accuracy(logits, targets):
    """Compute the accuracy"""
    with torch.no_grad():
        if logits.dim() == 2:
            _, predictions = torch.max(logits, dim=1)
            accuracy = torch.mean(predictions.eq(targets).float())
        else:
            _, predictions = torch.max(logits, dim=2)
            accuracy = torch.mean(predictions.eq(targets).float(), dim=-1)
    return accuracy.detach().cpu().numpy()

def tensors_to_device(tensors, device=torch.device('cpu')):
    """Place a collection of tensors in a specific device"""
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(tensors_to_device(tensor, device=device)
            for tensor in tensors)
    elif isinstance(tensors, (dict, OrderedDict)):
        return type(tensors)([(name, tensors_to_device(tensor, device=device))
            for (name, tensor) in tensors.items()])
    else:
        raise NotImplementedError()

class ToTensor1D(object):
    """Convert a `numpy.ndarray` to tensor. Unlike `ToTensor` from torchvision,
    this converts numpy arrays regardless of the number of dimensions.

    Converts automatically the array to `float32`.
    """
    def __call__(self, array):
        return torch.from_numpy(array.astype('float32'))

    def __repr__(self):
        return self.__class__.__name__ + '()'

def make_warp_model(model):
    metric_params = []
    for layer in model.modules():
        if isinstance(layer, BatchLinear):
            metric_params.append(layer.weight)

    return TransformerMetric(metric_params)

def kronecker_warp(grad, kronecker_matrices) -> torch.Tensor:
    """
    Function for doing Kronecker based warping of gradient batches of an
        m x n matrix parameter

    Params:
        grad (torch.Tensor): gradient batch of shape [meta_batch_size, batch_size m, n]
        kronecker_matrices (Tuple[torch.Tensor, torch.Tensor]): kronecker
            matrices to do the warping. First element of tuple is of shape
            [meta_batch_size, batch_size n, n] second is of shape
            [meta_batch_size, batch_size, m, m]
    """

    matrix_a = kronecker_matrices[0]
    matrix_a = matrix_a.view(-1, matrix_a.size(-2), matrix_a.size(-1))
    matrix_b = kronecker_matrices[1]
    matrix_b = matrix_b.view(-1, matrix_b.size(-2), matrix_b.size(-1))
    if len(kronecker_matrices) > 2:
        matrix_c = kronecker_matrices[2]
        matrix_c = matrix_c.view(-1, matrix_c.size(-2), matrix_c.size(-1))

        grad_size = grad.size()
        grad = grad.view(-1, matrix_a.size(-1), matrix_c.size(-1))

        matrix_a = matrix_a.unsqueeze(1).expand(matrix_a.size(0), grad_size[-2], *matrix_a.size()[1:]).reshape(-1, matrix_a.size(-2), matrix_a.size(-1))
        matrix_c = matrix_c.unsqueeze(1).expand(matrix_c.size(0), grad_size[-2], *matrix_c.size()[1:]).reshape(-1, matrix_c.size(-2), matrix_c.size(-1))
        prod1 = torch.bmm(matrix_a.transpose(1, 2), grad)
        prod2 = torch.bmm(prod1, matrix_c)

        temp = prod2.view(-1, matrix_b.size(-1), grad_size[-1])
        prod3 = torch.bmm(matrix_b, temp)
        return prod3.view(grad_size)
    else:
        grad_shape = grad.size()
        grad = grad.view(-1, grad.size(-2), grad.size(-1))

        prod1 = torch.bmm(matrix_b, grad)
        prod2 = torch.bmm(prod1, torch.transpose(matrix_a, 1, 2))

        return prod2.view(grad_shape)

def gradient_update_parameters_warp(model,
                               loss,
                               params=None,
                               warp_model=None,
                               step_size=0.5,
                               first_order=False,
                               state=None):
    """Update of the meta-parameters with one step of gradient descent on the
    loss function.
    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.
    loss : `torch.Tensor` instance
        The value of the inner-loss. This is the result of the training dataset
        through the loss function.
    params : `collections.OrderedDict` instance, optional
        Dictionary containing the meta-parameters of the model. If `None`, then
        the values stored in `model.meta_named_parameters()` are used. This is
        useful for running multiple steps of gradient descent as the inner-loop.
    step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
        The step size in the gradient update. If an `OrderedDict`, then the
        keys must match the keys in `params`.
    first_order : bool (default: `False`)
        If `True`, then the first order approximation of MAML is used.
    Returns
    -------
    updated_params : `collections.OrderedDict` instance
        Dictionary containing the updated meta-parameters of the model, with one
        gradient update wrt. the inner-loss.
    """
    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.'
                         'MetaModule`, got `{0}`'.format(type(model)))

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    param_jacobs_lst = [[] for _ in range(len(params))]
    for i in range(loss.size(0)):
        grads = torch.autograd.grad(loss[i], params.values(), retain_graph=True, create_graph=not first_order)
        for j, grad in enumerate(grads):
            param_jacobs_lst[j].append(grad)

    param_jacobs = [torch.stack(param_jacob, dim=1) for param_jacob in param_jacobs_lst]

    if warp_model is not None:
        warp_model_input = []
        for param in warp_model.warp_parameters:
            warp_model_input.append([param.input_data, param.grad_data])

        kronecker_matrix_logs, state = warp_model(warp_model_input, state=state)
        kronecker_matrices = []

        #for matrix_a, matrix_b in kronecker_matrix_logs:
        for kronecker_matrix_list in kronecker_matrix_logs:
            matrix_a = kronecker_matrix_list[0]
            matrix_b = kronecker_matrix_list[1]
            if len(kronecker_matrix_list) > 2:
                matrix_c = kronecker_matrix_list[2]

            # Negative because we want metric inverse
            exp_matrix_a = torch.matrix_exp(-matrix_a.reshape((-1, matrix_a.size(-2),
                                                   matrix_a.size(-1))))
            exp_matrix_a = exp_matrix_a.reshape(matrix_a.size())

            exp_matrix_b = torch.matrix_exp(-matrix_b.reshape((-1, matrix_b.size(-2),
                                                   matrix_b.size(-1))))
            exp_matrix_b = exp_matrix_b.reshape(matrix_b.size())

            if len(kronecker_matrix_list) > 2:
                exp_matrix_c = torch.matrix_exp(-matrix_c.reshape((-1, matrix_c.size(-2),
                                                    matrix_c.size(-1))))
                exp_matrix_c = exp_matrix_c.reshape(matrix_c.size())

                kronecker_matrices.append([exp_matrix_a, exp_matrix_b, exp_matrix_c])
            else:
                kronecker_matrices.append([exp_matrix_a, exp_matrix_b])


    updated_params = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for i, ((name, param), grad) in enumerate(zip(params.items(), param_jacobs)):
            if warp_model is not None:
                grad = kronecker_warp(grad, kronecker_matrices[i])
            updated_params[name] = param - step_size[name] * grad.mean(dim=-3)

    else:
        for i, ((name, param), grad) in enumerate(zip(params.items(), param_jacobs)):
            if warp_model is not None:
                grad = kronecker_warp(grad, kronecker_matrices[i])
            updated_params[name] = param - step_size * grad.mean(dim=-3)

    return updated_params, state
