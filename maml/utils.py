import torch

from collections import OrderedDict
from torchmeta.modules import MetaModule
from .model import BatchParameter
from .transformer_metric import TransformerMetric
from .constant_metric import ConstantMetric

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

def make_warp_model(model, constant=False):
    metric_params = []
    for parameter in model.parameters():
        if isinstance(parameter, BatchParameter):
            metric_params.append(parameter)
    """
    for layer in model.modules():
        if isinstance(layer, BatchLinear):
            metric_params.append(layer.weight)
    """

    if constant:
        return ConstantMetric(metric_params)
    else:
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

    input_matrices = kronecker_matrices[0]
    output_matrices = kronecker_matrices[1]
    all_matrices = input_matrices + output_matrices
    grad = grad.sum(dim=-3)
    grad_size = grad.size()

    first_matrix = all_matrices[0]
    first_matrix = first_matrix.view(-1, first_matrix.size(-2), first_matrix.size(-1))
    temp = grad.view(-1, all_matrices[1].size(-1), first_matrix.size(-1))
    first_matrix = first_matrix.unsqueeze(1).expand(
            first_matrix.size(0), temp.size(0) // first_matrix.size(0), *first_matrix.size()[1:]
            ).reshape(-1, first_matrix.size(-2), first_matrix.size(-1))
    temp = torch.bmm(temp, first_matrix)
    right_size = first_matrix.size(-1)

    for i, matrix in enumerate(all_matrices[1:]):
        matrix = matrix.view(-1, matrix.size(-2), matrix.size(-1))
        matrix = matrix.unsqueeze(1).expand(
                matrix.size(0), temp.size(0) // matrix.size(0), *matrix.size()[1:]
                ).reshape(-1, matrix.size(-2), matrix.size(-1))
        temp = torch.bmm(matrix, temp)

        if i < len(all_matrices) - 2:
            right_size *= matrix.size(-1)
            temp = temp.view(-1, all_matrices[i + 2].size(-1), right_size)

    return temp.view(grad_size)

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
            if param.collect_input:
                warp_model_input.append([param.input_data, param.grad_data])

        kronecker_matrix_logs = warp_model(warp_model_input)
        kronecker_matrices = []

        for kronecker_matrix_list in kronecker_matrix_logs:
            input_matrices = kronecker_matrix_list[0]
            output_matrices = kronecker_matrix_list[1]

            exp_input_matrices = []
            for matrix in input_matrices:
                #exp_matrix = torch.matrix_exp(matrix.reshape((-1, matrix.size(-2), matrix.size(-1))))
                #exp_matrix = exp_matrix.reshape(matrix.size())
                #exp_matrix = matrix.reshape((-1, matrix.size(-2), matrix.size(-1)))
                #exp_matrix = torch.bmm(exp_matrix, exp_matrix)
                #exp_matrix = exp_matrix.reshape(matrix.size())
                exp_input_matrices.append(matrix)

            exp_output_matrices = []
            for matrix in output_matrices:
                #exp_matrix = torch.matrix_exp(matrix.reshape((-1, matrix.size(-2), matrix.size(-1))))
                #exp_matrix = exp_matrix.reshape(matrix.size())
                #exp_matrix = matrix.reshape((-1, matrix.size(-2), matrix.size(-1)))
                #exp_matrix = torch.bmm(exp_matrix, exp_matrix)
                #exp_matrix = exp_matrix.reshape(matrix.size())
                exp_output_matrices.append(matrix)

            kronecker_matrices.append([exp_input_matrices, exp_output_matrices])

    updated_params = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for i, ((name, param), grad) in enumerate(zip(params.items(), param_jacobs)):
            if warp_model is not None:
                grad = kronecker_warp(grad, kronecker_matrices[i])
            updated_params[name] = param - step_size[name] * grad

    else:
        for i, ((name, param), grad) in enumerate(zip(params.items(), param_jacobs)):
            if warp_model is not None:
                grad = kronecker_warp(grad, kronecker_matrices[i])
            updated_params[name] = param - step_size * grad

    return updated_params
