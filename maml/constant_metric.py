import torch
from torch import nn
from typing import List
from math import sqrt, floor, ceil
from torch.nn.init import orthogonal_
from torch.nn.modules.normalization import LayerNorm
import numpy as np

class ConstantMetric(nn.Module):

    def __init__(self, warp_parameters):

        super(ConstantMetric, self).__init__()

        self.warp_parameters = warp_parameters

        self.input_matrices = []
        self.output_matrices = []
        for param in warp_parameters:
            param_input_matrices = []
            if param.in_division is not None:
                for dimension in param.in_division:
                    matrix = nn.Parameter(torch.eye(dimension, device=param.device))
                    param_input_matrices.append(matrix)
            else:
                matrix = nn.Parameter(torch.eye(param.size(2), device=param.device))
                param_input_matrices.append(matrix)
            self.input_matrices.append(nn.ParameterList(param_input_matrices))

            param_output_matrices = []
            if param.out_division is not None:
                for dimension in param.out_division:
                    matrix = nn.Parameter(torch.eye(dimension, device=param.device))
                    param_output_matrices.append(matrix)
            else:
                matrix = nn.Parameter(torch.eye(param.size(1), device=param.device))
                param_output_matrices.append(matrix)
            self.output_matrices.append(nn.ParameterList(param_output_matrices))

        self.input_matrices = nn.ModuleList(self.input_matrices)
        self.output_matrices = nn.ModuleList(self.output_matrices)

    def forward(self, warp_inputs: List[List[torch.Tensor]]):
        meta_batch_size = warp_inputs[0][0].size(0)
        kronecker_matrices = []
        for i, param in enumerate(self.warp_parameters):
            output_matrices = []
            for matrix in self.output_matrices[i]:
                matrix = matrix.unsqueeze(0).expand(meta_batch_size, *list(matrix.size()))
                output_matrices.append(matrix)

            input_matrices = []
            for matrix in self.input_matrices[i]:
                matrix = matrix.unsqueeze(0).expand(meta_batch_size, *list(matrix.size()))
                input_matrices.append(matrix)

            kronecker_matrices.append([input_matrices, output_matrices])

        return kronecker_matrices

    def set_listening(self, listening: bool):
        for param in self.warp_parameters:
            param.listening = listening
