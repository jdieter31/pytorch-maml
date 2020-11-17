import torch.nn as nn
import torch
from torch.nn import init
from torch.nn.functional import has_torch_function, handle_torch_function, assert_int_or_pair, _pair
import math

from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)

__mode__ = "Expanding"

class BatchParameter(nn.Parameter):
    def __new__(cls, data=None, convolutional=False, in_size=0, in_channels=0, requires_grad=True, expanding=False, expanding_factor=32, in_division=None, out_division=None, collect_input=True):
        if data is None:
            data = torch.Tensor()

        instance = torch.Tensor._make_subclass(cls, data, requires_grad)
        instance.expanding = expanding
        instance.input_data = None
        instance.grad_data = None
        instance.convolutional = convolutional
        instance.listening = False
        instance.in_channels = in_channels
        instance.in_size = in_size
        instance.expanding_factor = expanding_factor
        instance.in_division = in_division
        instance.out_division = out_division
        instance.collect_input = collect_input
        return instance

    def expanded(self, meta_batch_size):
        return self.repeat(*([self.expanding_factor] + [1 for _ in range(self.dim() - 1)]))

class BatchLinear(MetaModule):

    def __init__(self, input_size, output_size, meta_batch_size=1, bias=True, expanding=True, expanding_factor=32):
        super(BatchLinear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias

        self.expanding = expanding
        if expanding:
            self.weight = BatchParameter(torch.Tensor(meta_batch_size // expanding_factor, self.output_size,
                                                    input_size), expanding=True, expanding_factor=expanding_factor)
            if bias:
                self.bias_value = BatchParameter(torch.Tensor(meta_batch_size // expanding_factor, self.output_size, 1), collect_input=False, expanding=True)
        else:
            self.weight = BatchParameter(torch.Tensor(meta_batch_size, self.output_size,
                                                    input_size))
            if bias:
                self.bias_value = BatchParameter(torch.Tensor(meta_batch_size, self.output_size, 1), collect_input=False)
        self.reset_parameters()

    def reset_parameters(self):

        init.kaiming_uniform_(self.weight[:, :, :], a=math.sqrt(5))
        if self.bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(
                self.weight[:, :, :])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_value[:, :, 0], -bound, bound)

    def forward(self, input_data, params=None, save_warp_data=True):
        if self.expanding:
            weight = self.weight if params is None else params["weight"]
        else:
            weight = self.weight.expanded(input_data.size(0)) if params is None else params["weight"]

        if self.bias:
            if self.expanding:
                bias_value = self.bias_value if params is None else params["bias_value"]
            else:
                bias_value = self.bias_value.expanded(input_data.size(0)) if params is None else params["bias_value"]

            weight = torch.cat([weight, bias_value], dim=-1)

        if self.bias:
            input_data_exp = torch.cat([input_data,
                torch.ones(*input_data.size()[:-1], 1, device=input_data.device)], -1)
        else:
            input_data_exp = input_data

        if input_data.ndim > 3:
            input_data_resized = input_data_exp.view(input_data_exp.size(0), -1, input_data_exp.size(-1))
            out = torch.transpose(torch.bmm(weight, torch.transpose(input_data_resized, 1,
                                                                        2)), 1, 2)
            out = out.reshape(*(input_data.size()[:-1] + out.size()[-1:]))

        else:
            out = torch.transpose(torch.bmm(weight, torch.transpose(input_data_exp, 1,
                                                                        2)), 1, 2)
        if save_warp_data:
            self.enable_warp_data(input_data, out)

        return out

    def enable_warp_data(self, input_data, out):
        if self.weight.listening:
            # Save input data for future reference
            self.weight.input_data = input_data
            self.weight.grad_data = torch.zeros_like(out)

            def count(grad):
                # Hacky way of recording the gradient data during the batch
                # jacobian computation
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

class BatchConv2d(BatchLinear):

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding = 0, dilation = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', in_size=28, meta_batch_size=1, expanding=True, expanding_factor=32, out_division=None):
        # Call grandparent
        super(BatchLinear, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

        # self.linear = BatchLinear(self.in_channels * self.kernel_size[0] * self.kernel_size[1], out_channels, bias)

        # Init BatchLinear
        total_kernel_size = self.kernel_size[0] * self.kernel_size[1]

        self.input_size = self.in_channels * total_kernel_size
        self.output_size = out_channels
        self.expanding = expanding
        in_division = [self.in_channels, self.kernel_size[0], self.kernel_size[1]]

        if expanding:
            self.weight = BatchParameter(torch.Tensor(meta_batch_size // expanding_factor, self.output_size,
                                                    self.input_size), convolutional=True, in_size=in_size,
                                                    in_channels=self.in_channels, expanding=True, expanding_factor=expanding_factor, in_division=in_division, out_division=out_division)
            if bias:
                self.bias_value = BatchParameter(torch.Tensor(meta_batch_size // expanding_factor, self.output_size, 1), collect_input=False, expanding=True)
        else:
            self.weight = BatchParameter(torch.Tensor(meta_batch_size, self.output_size,
                                                    self.input_size), convolutional=True, in_size=in_size, in_channels=self.in_channels, in_division=in_division, out_division=out_division)
            if bias:
                self.bias_value = BatchParameter(torch.Tensor(meta_batch_size, self.output_size, 1), collect_input=False)
        self.reset_parameters()

    def forward(self, inputs, params = None):
        meta_batch_size, batch_size, in_channels, in_h, in_w = inputs.size()
        # out_channels, in_channels, kh, kw =  weight.shape

        inp_unf = unfold(inputs.view(-1, in_channels, in_h, in_w),
                kernel_size=self.kernel_size, dilation=self.dilation,
                padding=self.padding, stride=self.stride).transpose(-1,-2)
        inp_unf = inp_unf.reshape(meta_batch_size, batch_size, *inp_unf.size()[-2:])

        # out_unf = self.linear(inp_unf, params=self.get_subdict(params, 'linear')).transpose(-1, -2)

        """
        if self.conv_bias:
            inp_unf = inp_unf.view(meta_batch_size, batch_size, -1, in_channels, inp_unf.size(-1) // in_channels)
            inp_unf = torch.cat([inp_unf,
                torch.ones(*inp_unf.size()[:-1], 1, device=inp_unf.device)], -1)
            inp_unf = inp_unf.view(meta_batch_size, batch_size, -1, in_channels * self.weight.kernel_size)
        """

        out_unf = super().forward(inp_unf, params=params, save_warp_data=False).transpose(-1, -2)

        self.out_h = math.floor((in_h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)/self.stride[0] + 1)
        self.out_w = math.floor((in_w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)/self.stride[1] + 1)

        out = out_unf.view(meta_batch_size, batch_size, self.out_channels, self.out_h, self.out_w)

        self.enable_warp_data(inputs, out)

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

def conv_block(in_channels, out_channels, in_size, **kwargs):
    squish = SquishMetaBatch()
    expand = ExpandMetaBatch(squish)
    return MetaSequential(OrderedDict([
        ('conv', BatchConv2d(in_channels, out_channels, in_size=in_size, **kwargs)),
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
    def __init__(self, in_channels, out_features, hidden_size=64, feature_size=64, in_size=28, meta_batch_size=1, hidden_division=None):
        super(MetaConvModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.feature_size = feature_size

        in_sizes = [in_size]
        for i in range(3):
            in_sizes.append(math.floor(in_sizes[-1]/2))

        self.features = MetaSequential(OrderedDict([
            ('layer1', conv_block(in_channels, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True, in_size=in_sizes[0]**2, meta_batch_size=meta_batch_size, out_division=hidden_division)),
            ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True, in_size=in_sizes[1]**2, meta_batch_size=meta_batch_size, out_division=hidden_division)),
            ('layer3', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True, in_size=in_sizes[2]**2, meta_batch_size=meta_batch_size, out_division=hidden_division)),
            ('layer4', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True, in_size=in_sizes[3]**2, meta_batch_size=meta_batch_size, out_division=hidden_division))
        ]))
        self.classifier = BatchLinear(feature_size, out_features, bias=True, meta_batch_size=meta_batch_size)

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
    def __init__(self, in_features, out_features, hidden_sizes, meta_batch_size=1):
        super(MetaMLPModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes

        layer_sizes = [in_features] + hidden_sizes

        self.features = MetaSequential(OrderedDict([('layer{0}'.format(i + 1),
            MetaSequential(OrderedDict([
                ('linear', BatchLinear(hidden_size, layer_sizes[i + 1], meta_batch_size=meta_batch_size, bias=True)),
                ('relu', nn.ReLU())
            ]))) for (i, hidden_size) in enumerate(layer_sizes[:-1])]))
        self.classifier = BatchLinear(hidden_sizes[-1], out_features, meta_batch_size=meta_batch_size, bias=True)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits

def ModelConvOmniglot(out_features, hidden_size=64, meta_batch_size=1):
    return MetaConvModel(1, out_features, hidden_size=hidden_size,
                         feature_size=hidden_size, meta_batch_size=meta_batch_size)

def ModelConvMiniImagenet(out_features, hidden_size=64, meta_batch_size=1):
    return MetaConvModel(3, out_features, hidden_size=hidden_size,
                         feature_size=5 * 5 * hidden_size, in_size=84, meta_batch_size=meta_batch_size)

def ModelMLPSinusoid(hidden_sizes=[40, 40], meta_batch_size=1):
    return MetaMLPModel(1, 1, hidden_sizes, meta_batch_size=meta_batch_size)

if __name__ == '__main__':
    model = ModelMLPSinusoid()
