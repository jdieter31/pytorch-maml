import torch
from torch import nn
from typing import List
from math import sqrt, floor, ceil
from torch.nn.init import orthogonal_
from torch.nn.modules.normalization import LayerNorm
import numpy as np

def primes(n):
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
       primfac.append(n)
    return primfac

class MiniConvBlock(nn.Module):

    def __init__(self, in_size, in_channels, out_channels, pool=True, pool_size=2, **kwargs):
        super(MiniConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels, momentum=1,
                track_running_stats=False)
        self.relu = nn.ReLU()
        self.pool = None
        if pool:
            self.pool = nn.MaxPool2d(pool_size)

    def forward(self, input_data):
        batch_sizes = input_data.size()[:2]
        out = input_data.reshape(-1, *input_data.size()[2:])
        out = self.conv(out)
        out = self.norm(out)
        out = self.relu(out)
        if self.pool is not None:
            out = self.pool(out)
        out = out.reshape(*batch_sizes, -1)
        return out


class TransformerMetric(nn.Module):

    def __init__(self, warp_parameters, hidden_size: int = 64, transformer_layers: int = 3, transformer_heads: int = 8, transformer_feedforward_dim: int = 512, transformer_d_model: int = 64, conv_hidden_dim: int = 2, reduction_factor=4):

        super(TransformerMetric, self).__init__()

        self.input_modules = []
        self.warp_parameters = warp_parameters

        self.hidden_size = hidden_size
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_feedforward_dim = transformer_feedforward_dim
        self.transformer_d_model = transformer_d_model
        self.conv_hidden_dim = conv_hidden_dim
        total_input_dim = 0

        for param in warp_parameters:
            if not param.collect_input:
                continue
            if param.convolutional:
                dim = sqrt(param.in_size)
                if dim >= 32:
                    pool = True
                    pool_size = int(dim // 16)
                elif dim >= 16:
                    pool = True
                    pool_size = int(dim // 4)
                else:
                    pool = False
                    pool_size = 0

                output_size = param.in_size
                if pool:
                    output_size  = floor(dim / pool_size) ** 2
                output_size *= conv_hidden_dim
                total_input_dim += 2 * ceil(output_size / reduction_factor)

                mini_conv_block_1 = MiniConvBlock(param.in_size, param.in_channels, conv_hidden_dim, kernel_size=3, stride=1, padding=1, bias=True, pool=pool, pool_size=pool_size)
                mini_conv_block_2 = MiniConvBlock(param.in_size, param.size(-2), conv_hidden_dim, kernel_size=3, stride=1, padding=1, bias=True, pool=pool, pool_size=pool_size)
                linear_1 = nn.Linear(output_size, ceil(output_size / reduction_factor), bias = True)
                linear_2 = nn.Linear(output_size, ceil(output_size / reduction_factor), bias = True)
                in_module_1 = nn.Sequential(mini_conv_block_1, linear_1, nn.ReLU())
                in_module_2 = nn.Sequential(mini_conv_block_2, linear_2, nn.ReLU())
                self.input_modules.append(nn.ModuleList([in_module_1, in_module_2]))
            else:
                total_input_dim += ceil(param.size(2) / reduction_factor) + ceil(param.size(1) / reduction_factor)
                linear_1 = nn.Linear(param.size(2), ceil(param.size(2) / reduction_factor), bias = True)
                linear_2 = nn.Linear(param.size(1), ceil(param.size(1) / reduction_factor), bias = True)
                self.input_modules.append(nn.ModuleList([linear_1, linear_2]))

        self.input_modules = nn.ModuleList(self.input_modules)
        self.input_linear = nn.Linear(total_input_dim, self.transformer_d_model, bias=True)
        self.relu = nn.ReLU()
        self.layer_norms = nn.ModuleList([LayerNorm(self.transformer_d_model) for _ in range(self.transformer_layers)])

        self.transformer_encoder_layers = []
        for _ in range(self.transformer_layers):
            self.transformer_encoder_layers.append(
                nn.TransformerEncoderLayer(self.transformer_d_model,
                                            self.transformer_heads,
                                            self.transformer_feedforward_dim,
                                            dropout=0)
            )

        self.transformer_encoder_layers = nn.ModuleList(self.transformer_encoder_layers)

        self.output_shapes = []
        self.input_shapes = []

        output_head_dim = 0
        for param in warp_parameters:
            total_input_shape = []
            if param.in_division is not None:
                for dimension in param.in_division:
                    total_input_shape.append(dimension)
            else:
                total_input_shape.append(param.size(2))
            self.input_shapes.append(total_input_shape)

            total_output_shape = []
            if param.out_division is not None:
                for dimension in param.out_division:
                    total_output_shape.append(dimension)
            else:
                total_output_shape.append(param.size(1))
            self.output_shapes.append(total_output_shape)

            for dim in total_input_shape:
                output_head_dim += (dim * (dim + 1)) // 2

            for dim in total_output_shape:
                output_head_dim += (dim * (dim + 1)) // 2

        self.output_head = nn.Linear(self.transformer_d_model, output_head_dim, bias=True)
        with torch.no_grad():
            self.output_head.weight /= 10000
            self.output_head.bias /= 10000

        import ipdb; ipdb.set_trace()

    def forward(self, warp_inputs: List[List[torch.Tensor]], state: List[List[torch.Tensor]] = None):
        embeddings = []

        # Get all inputs to be same dimension so we can pass through the
        # transformer cascade
        for i, warp_input in enumerate(warp_inputs):
            embeddings_1 = self.input_modules[i][0](warp_input[0])
            embeddings_2 = self.input_modules[i][1](warp_input[1])
            embeddings += [embeddings_1, embeddings_2]

        embeddings = torch.cat(embeddings, dim=-1)
        embeddings = self.input_linear(embeddings)
        embeddings = self.relu(embeddings)
        embeddings = embeddings.transpose(0,1)

        for i in range(self.transformer_layers):
            embeddings = self.transformer_encoder_layers[i](embeddings)
            embeddings = self.layer_norms[i](embeddings)

        embeddings = embeddings.transpose(0,1)

        output_raw = self.output_head(embeddings)
        output_raw_size = output_raw.size()
        output_raw = output_raw.reshape([-1, output_raw.size(-1)])
        range_start = 0
        kronecker_matrices = []
        for i, param in enumerate(self.warp_parameters):
            output_matrices = []
            for dim in self.output_shapes[i]:
                length = (dim * (dim + 1)) // 2
                matrix_gen = output_raw[:, range_start : range_start + length]
                range_start += length
                matrix = torch.zeros(matrix_gen.size(0), dim, dim, device=matrix_gen.device)
                triu_indices = torch.triu_indices(dim, dim, 1, device=matrix.device)
                if triu_indices.size(-1) > 0:
                    matrix[:, triu_indices[0], triu_indices[1]] = matrix_gen[:,:-dim]
                    matrix[:, triu_indices[1], triu_indices[0]] = matrix_gen[:,:-dim]
                eye_indices = torch.arange(0, dim, device=matrix.device, dtype=torch.long)
                matrix[:, eye_indices, eye_indices] = matrix_gen[:,-dim:]
                matrix = matrix.reshape(list(output_raw_size)[:-1] + 2 * [dim])
                output_matrices.append(matrix)

            input_matrices = []
            for dim in self.input_shapes[i]:
                length = (dim * (dim + 1)) // 2
                matrix_gen = output_raw[:, range_start : range_start + length]
                range_start += length
                matrix = torch.zeros(matrix_gen.size(0), dim, dim, device=matrix_gen.device)
                triu_indices = torch.triu_indices(dim, dim, 1, device=matrix.device)
                if triu_indices.size(-1) > 0:
                    matrix[:, triu_indices[0], triu_indices[1]] = matrix_gen[:,:-dim]
                    matrix[:, triu_indices[1], triu_indices[0]] = matrix_gen[:,:-dim]
                eye_indices = torch.arange(0, dim, device=matrix.device, dtype=torch.long)
                matrix[:, eye_indices, eye_indices] = matrix_gen[:,-dim:]
                matrix = matrix.reshape(list(output_raw_size)[:-1] + 2 * [dim])
                input_matrices.append(matrix)

            kronecker_matrices.append([input_matrices, output_matrices])
        return kronecker_matrices

    def set_listening(self, listening: bool):
        for param in self.warp_parameters:
            param.listening = listening
