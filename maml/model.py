import torch.nn as nn
import torch
from torch.nn import init
from torch.nn.functional import has_torch_function, handle_torch_function, assert_int_or_pair, _pair
import math

from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)

class ExpandingParameter(nn.Parameter):
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.Tensor()

        instance = torch.Tensor._make_subclass(cls, data, requires_grad)
        instance.input_data = None
        instance.grad_data = None
        instance.listening = False
        return instance

    def expanded(self, meta_batch_size):
        return self.expand([meta_batch_size] + list(self.size())[1:])

class BatchLinear(MetaModule):

    def __init__(self, input_size, output_size, bias=True):
        super(BatchLinear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.param_update = None
        if bias:
            input_size += 1
        self.weight = ExpandingParameter(torch.Tensor(1, self.output_size,
                                                 input_size))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight[:, :, :-1], a=math.sqrt(5))
        if self.bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(
                self.weight[:, :, :-1])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.weight[:, :, -1], -bound, bound)

    def forward(self, input_data, params=None):
        weight = self.weight.expanded(input_data.size(0)) if params is None else params["weight"]

        if self.bias:
            input_data = torch.cat([input_data,
                torch.ones(*input_data.size()[:-1], 1, device=input_data.device)], -1)

        if input_data.ndim > 3:
            input_data_resized = input_data.view(input_data.size(0), -1, input_data.size(-1))
            out = torch.transpose(torch.bmm(weight, torch.transpose(input_data_resized, 1,
                                                                        2)), 1, 2)
            out = out.reshape(*(input_data.size()[:-1] + out.size()[-1:]))

        else:
            out = torch.transpose(torch.bmm(weight, torch.transpose(input_data, 1,
                                                                        2)), 1, 2)


        if self.weight.listening:
            # Save input data for future reference
            if input_data.ndim > 3:
                self.weight.input_data = input_data.mean(dim=-2)
                self.weight.grad_data = torch.zeros_like(out.mean(dim=-2))
            else:
                self.weight.input_data = input_data
                self.weight.grad_data = torch.zeros_like(out)


            def count(grad):
                # Hacky way of recording the gradient data during the batch
                # jacobian computation
                if grad.ndim > 3:
                    self.weight.grad_data += grad.mean(dim=-2)
                else:
                    self.weight.grad_data += grad

            out.register_hook(count)
        return out

class Im2Col(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size, dilation, padding, stride):
        ctx.shape, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride = (x.shape[2:], kernel_size, dilation, padding, stride)
        return nn.functional.unfold(x, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.enable_grad():
            shape, ks, dilation, padding, stride = ctx.shape, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride
            return (
                nn.functional.fold(grad_output, shape, kernel_size=ks, dilation=dilation, padding=padding, stride=stride),
                None,
                None,
                None,
                None
            )


def im2col(x, kernel_size, dilation=1, padding=0, stride=1):
    return Im2Col.apply(x, kernel_size, dilation, padding, stride)

def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    # type: (Tensor, BroadcastingList2[int], BroadcastingList2[int], BroadcastingList2[int], BroadcastingList2[int]) -> Tensor  # noqa
    r"""Extracts sliding local blocks from an batched input tensor.

    .. warning::
        Currently, only 4-D input tensors (batched image-like tensors) are
        supported.

    .. warning::

        More than one element of the unfolded tensor may refer to a single
        memory location. As a result, in-place operations (especially ones that
        are vectorized) may result in incorrect behavior. If you need to write
        to the tensor, please clone it first.


    See :class:`torch.nn.Unfold` for details
    """

    if not torch.jit.is_scripting():
        if type(input) is not torch.Tensor and has_torch_function((input,)):
            return handle_torch_function(
                unfold, (input,), input, kernel_size, dilation=dilation,
                padding=padding, stride=stride)
    if input.dim() == 4:
        msg = '{} must be int or 2-tuple for 4D input'
        assert_int_or_pair(kernel_size, 'kernel_size', msg)
        assert_int_or_pair(dilation, 'dilation', msg)
        assert_int_or_pair(padding, 'padding', msg)
        assert_int_or_pair(stride, 'stride', msg)

        return im2col(input, _pair(kernel_size),
                                   _pair(dilation), _pair(padding), _pair(stride))
    else:
        raise NotImplementedError("Input Error: Only 4D input Tensors are supported (got {}D)".format(input.dim()))

class BatchConv2d(MetaModule):

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding = 0, dilation = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
        super(BatchConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

        self.linear = BatchLinear(self.in_channels * self.kernel_size[0] * self.kernel_size[1], out_channels, bias)

    def forward(self, inputs, params = None):
        meta_batch_size, batch_size, in_channels, in_h, in_w = inputs.size()
        # out_channels, in_channels, kh, kw =  weight.shape

        inp_unf = unfold(inputs.view(-1, in_channels, in_h, in_w),
                kernel_size=self.kernel_size, dilation=self.dilation,
                padding=self.padding, stride=self.stride).transpose(-1,-2)
        inp_unf = inp_unf.reshape(meta_batch_size, batch_size, *inp_unf.size()[-2:])

        out_unf = self.linear(inp_unf, params=self.get_subdict(params, 'linear')).transpose(-1, -2)

        out_h = math.floor((in_h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)/self.stride[0] + 1)
        out_w = math.floor((in_w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)/self.stride[1] + 1)

        out = out_unf.view(meta_batch_size, batch_size, self.out_channels, out_h, out_w)
        return out

class SquishMetaBatch(nn.Module):

    def __init__(self):
        super(SquishMetaBatch, self).__init__()
        self.meta_batch_size = 0

    def forward(self, inputs):
        self.meta_batch_size = inputs.size(0)
        return inputs.reshape(-1, *inputs.size()[2:])

class ExpandMetaBatch(nn.Module):

    def __init__(self, squish_meta_batch):
        super(ExpandMetaBatch, self).__init__()
        self.squish_meta_batch = squish_meta_batch

    def forward(self, inputs):
        return inputs.reshape(self.squish_meta_batch.meta_batch_size, -1, *inputs.size()[1:])

def conv_block(in_channels, out_channels, **kwargs):
    squish = SquishMetaBatch()
    expand = ExpandMetaBatch(squish)
    return MetaSequential(OrderedDict([
        ('conv', BatchConv2d(in_channels, out_channels, **kwargs)),
        ('squish', squish),
        ('norm', nn.BatchNorm2d(out_channels, momentum=1.,
            track_running_stats=False)),
        ('relu', nn.ReLU()),
        ('pool', nn.MaxPool2d(2)),
        ('expand', expand)
    ]))

class MetaConvModel(MetaModule):
    """4-layer Convolutional Neural Network architecture from [1].

    Parameters
    ----------
    in_channels : int
        Number of channels for the input images.

    out_features : int
        Number of classes (output of the model).

    hidden_size : int (default: 64)
        Number of channels in the intermediate representations.

    feature_size : int (default: 64)
        Number of features returned by the convolutional head.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, in_channels, out_features, hidden_size=64, feature_size=64):
        super(MetaConvModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.feature_size = feature_size

        self.features = MetaSequential(OrderedDict([
            ('layer1', conv_block(in_channels, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer3', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer4', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True))
        ]))
        self.classifier = BatchLinear(feature_size, out_features, bias=True)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), features.size(1), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits

class MetaMLPModel(MetaModule):
    """Multi-layer Perceptron architecture from [1].

    Parameters
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of classes (output of the model).

    hidden_sizes : list of int
        Size of the intermediate representations. The length of this list
        corresponds to the number of hidden layers.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, in_features, out_features, hidden_sizes):
        super(MetaMLPModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes

        layer_sizes = [in_features] + hidden_sizes

        self.features = MetaSequential(OrderedDict([('layer{0}'.format(i + 1),
            MetaSequential(OrderedDict([
                ('linear', BatchLinear(hidden_size, layer_sizes[i + 1], bias=True)),
                ('relu', nn.ReLU())
            ]))) for (i, hidden_size) in enumerate(layer_sizes[:-1])]))
        self.classifier = BatchLinear(hidden_sizes[-1], out_features, bias=True)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits

def ModelConvOmniglot(out_features, hidden_size=64):
    return MetaConvModel(1, out_features, hidden_size=hidden_size,
                         feature_size=hidden_size)

def ModelConvMiniImagenet(out_features, hidden_size=64):
    return MetaConvModel(3, out_features, hidden_size=hidden_size,
                         feature_size=5 * 5 * hidden_size)

def ModelMLPSinusoid(hidden_sizes=[40, 40]):
    return MetaMLPModel(1, 1, hidden_sizes)

if __name__ == '__main__':
    model = ModelMLPSinusoid()
