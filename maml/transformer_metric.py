import torch
from torch import nn
from typing import List
from math import sqrt, ceil
from .model import ExpandingParameter
from torch.nn.init import orthogonal_
from torch.nn.modules.normalization import LayerNorm

class MiniConvBlock(nn.Module):

    def __init__(self, in_size, in_channels, out_channels, out_flat, **kwargs):
        super(MiniConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels, momentum=1,
                track_running_stats=False)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_size * out_channels, out_flat, bias=True)

    def forward(self, input_data):
        batch_sizes = input_data.size()[:2]
        out = input_data.reshape(-1, *input_data.size()[2:])
        out = self.conv(out)
        out = self.norm(out)
        out = self.relu(out)
        out = out.reshape(*batch_sizes, -1)
        out = self.linear(out)
        out = self.relu(out)
        return out


class TransformerMetric(nn.Module):

    def __init__(self, warp_parameters, hidden_size: int = 512, transformer_layers: int = 4, transformer_heads: int = 8, transformer_feedforward_dim: int = 2048, transformer_d_model: int = 512, transformer_cascade_direction: str = "bidirectional", num_outer_products = 32, conv_hidden_dim: int = 2048):

        super(TransformerMetric, self).__init__()

        self.input_modules = []
        self.warp_parameters = warp_parameters

        self.hidden_size = hidden_size
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_feedforward_dim = transformer_feedforward_dim
        self.transformer_d_model = transformer_d_model
        self.transformer_cascade_direction = transformer_cascade_direction
        self.num_outer_products = num_outer_products
        self.conv_hidden_dim = conv_hidden_dim

        for param in warp_parameters:
            if param.convolutional:
                hidden_dim = ceil(conv_hidden_dim / param.in_size)

                mini_conv_block_1 = MiniConvBlock(param.in_size, param.in_channels, hidden_dim, self.transformer_d_model // 2, kernel_size=3, stride=1, padding=1, bias=True)
                mini_conv_block_2 = MiniConvBlock(param.in_size, param.size(-2), hidden_dim, self.transformer_d_model // 2, kernel_size=3, stride=1, padding=1, bias=True)
                self.input_modules.append(nn.ModuleList([mini_conv_block_1, mini_conv_block_2]))
            else:
                input_dim = param.size(2) + param.size(1)
                self.input_modules.append(nn.Linear(input_dim,
                                                self.transformer_d_model))
        self.input_modules = nn.ModuleList(self.input_modules)
        direction = self.transformer_cascade_direction


        self.layer_norms = []
        for i in range(self.transformer_layers):
            self.layer_norms.append([])
            for j in range(len(warp_parameters)):
                self.layer_norms[-1].append(LayerNorm(self.transformer_d_model))

        self.layer_norms = nn.ModuleList([nn.ModuleList(self.layer_norms[i]) for i in range(self.transformer_layers)])


        if direction == "forward" or direction == "bidirectional":
            self.forward_transformers = [[] for _ in range(self.transformer_layers)]
            for i in range(self.transformer_layers):
                for j in range(len(warp_parameters)):
                    """
                    if j == 0:
                        self.forward_transformers[i].append(
                            nn.TransformerEncoderLayer(self.transformer_d_model,
                                                      self.transformer_heads,
                                                      self.transformer_feedforward_dim,
                                                      dropout=0)
                        )
                    else:
                    """
                    self.forward_transformers[i].append(
                        nn.TransformerDecoderLayer(self.transformer_d_model,
                                                    self.transformer_heads,
                                                    self.transformer_feedforward_dim,
                                                    dropout=0)
                    )

            self.forward_transformers = nn.ModuleList([nn.ModuleList(transformers) for transformers in self.forward_transformers])

        if direction == "backward" or direction =="bidirectional":
            self.backward_transformers = [[] for _ in
                                         range(self.transformer_layers)]
            for i in range(self.transformer_layers):
                for j in range(len(warp_parameters)):
                    """
                    if j == len(warp_parameters) - 1:
                        self.backward_transformers[i].append(
                            nn.TransformerEncoderLayer(self.transformer_d_model,
                                                      self.transformer_heads,
                                                      self.transformer_feedforward_dim,
                                                      dropout=0)
                        )
                    else:
                    """
                    self.backward_transformers[i].append(
                        nn.TransformerDecoderLayer(self.transformer_d_model,
                                                    self.transformer_heads,
                                                    self.transformer_feedforward_dim,
                                                    dropout=0)
                    )
            self.backward_transformers = nn.ModuleList([nn.ModuleList(transformers) for transformers in self.backward_transformers])

        if direction == "bidirectional":
            # If we have a bidirectional flow we need a layer to take in the
            # two d_model tensors from the outputs of the forward and backward
            # transformers and half their dimension to be able to be fed into
            # the next layer
            self.dimension_halfs = nn.ModuleList([
                nn.Linear(self.transformer_d_model * 2,
                       self.transformer_d_model) \
                for _ in range(self.transformer_layers)])

        self.output_heads = []
        for param in warp_parameters:
            self.output_heads.append(
                [
                    nn.Linear(self.transformer_d_model,
                            self.num_outer_products *
                            param.size(1)),
                    nn.Linear(self.transformer_d_model,
                              self.num_outer_products),
                    nn.Linear(self.transformer_d_model,
                            self.num_outer_products *
                            param.size(2)),
                    nn.Linear(self.transformer_d_model,
                              self.num_outer_products),
                    nn.Linear(self.transformer_d_model,
                              1),
                    nn.Linear(self.transformer_d_model,
                              1)
                ]
            )

            # Initialize to be sufficiently small to be suitable for matrix
            # exponential
            with torch.no_grad():
                self.output_heads[-1][0].weight /= param.size(1) * param.size(2)
                self.output_heads[-1][2].weight /= param.size(1) * param.size(2)
                orthogonal_(self.output_heads[-1][0].bias.view(
                    self.num_outer_products, param.size(1)))
                orthogonal_(self.output_heads[-1][2].bias.view(
                    self.num_outer_products, param.size(2)))

                # Don't make scalars too big
                self.output_heads[-1][1].weight /= self.num_outer_products
                self.output_heads[-1][1].bias /= self.num_outer_products
                self.output_heads[-1][3].weight /= self.num_outer_products
                self.output_heads[-1][3].bias /= self.num_outer_products
        self.output_heads = nn.ModuleList([nn.ModuleList(output_head) for output_head in self.output_heads])

    def forward(self, warp_inputs: List[List[torch.Tensor]], state: List[List[torch.Tensor]] = None):
        embeddings = []
        direction = self.transformer_cascade_direction

        # Get all inputs to be same dimension so we can pass through the
        # transformer cascade
        for i, warp_input in enumerate(warp_inputs):
            if self.warp_parameters[i].convolutional:
                embeddings_1 = self.input_modules[i][0](warp_input[0])
                embeddings_2 = self.input_modules[i][1](warp_input[1])
                embeddings.append(torch.cat([embeddings_1, embeddings_2], dim=-1))
            else:
                embeddings.append(self.input_modules[i](torch.cat(warp_input, dim=-1)))

        if state is None:
            state = [None, None]
            if direction == "forward" or direction == "bidirectional":
                state[0] = []
                for i in range(self.transformer_layers):
                    state_tensor = torch.zeros_like(embeddings[0].transpose(0,1))
                    state[0].append(state_tensor)

            if direction == "backward" or direction == "bidirectional":
                state[1] = []
                for i in range(self.transformer_layers):
                    state_tensor = torch.zeros_like(embeddings[0].transpose(0,1))
                    state[1].append(state_tensor)

        for i in range(self.transformer_layers):
            if direction == "forward" or direction == "bidirectional":
                forward_transformer_out = []
                memory = None
                for j, embedding in enumerate(embeddings):
                    if j == 0:
                        memory = self.forward_transformers[i][j](
                            embedding.transpose(0,1), state[0][i])
                    else:
                        memory = self.forward_transformers[i][j](
                            embedding.transpose(0,1), memory)

                    if j == len(embeddings) - 1:
                        state[0][i] = memory

                    forward_transformer_out.append(memory)
                forward_transformer_out = torch.stack(forward_transformer_out)

            if direction == "backward" or direction == "bidirectional":
                backward_transformer_out = []
                memory = None
                for j, embedding in reversed(list(enumerate(embeddings))):
                    if j == len(embeddings) - 1:
                        memory = self.backward_transformers[i][j](
                            embedding.transpose(0,1), state[1][i])
                    else:
                        memory = self.backward_transformers[i][j](
                            embedding.transpose(0,1), memory)

                    if j == 0:
                        state[1][i] = memory

                    backward_transformer_out.insert(0, memory)
                backward_transformer_out = torch.stack(backward_transformer_out)

            if direction == "bidirectional":
                embeddings = torch.cat((forward_transformer_out,
                                        backward_transformer_out), dim=-1)
                embeddings = self.dimension_halfs[i](embeddings)
                embeddings = embeddings.transpose(1,2)

            if direction == "forward":
                embeddings = forward_transformer_out.transpose(1,2)

            if direction == "backward":
                embeddings = backward_transformer_out.transpose(1,2)

            for j in range(len(embeddings)):
                embeddings[i][j] = self.layer_norms[i][j](embeddings[i][j])

        output_matrices = []
        for i, (embedding, param) in enumerate(zip(embeddings,
                                                   self.warp_parameters)):
            matrix_b_gen = self.output_heads[i][0](embedding)
            matrix_b_scalar = self.output_heads[i][1](embedding)
            matrix_a_gen = self.output_heads[i][2](embedding)
            matrix_a_scalar = self.output_heads[i][3](embedding)

            matrix_b_eye = self.output_heads[i][4](embedding)
            matrix_a_eye = self.output_heads[i][5](embedding)

            original_shape_b = matrix_b_gen.size()
            new_shape_b = [-1, param.size(1)]
            matrix_b_gen = matrix_b_gen.reshape(new_shape_b)
            matrix_b_scalar = matrix_b_scalar.reshape([-1,1])

            # Get scalars times outer products
            matrix_b = torch.bmm(matrix_b_scalar.unsqueeze(-1) * \
                                 matrix_b_gen.unsqueeze(-1),
                                 matrix_b_gen.unsqueeze(-2))
            matrix_b = matrix_b.reshape(list(original_shape_b)[:-1] +
                             [self.num_outer_products] + 2 * [param.size(1)])
            matrix_b = matrix_b.sum(dim=-3)

            matrix_b += matrix_b_eye.unsqueeze(-1) * \
                    torch.eye(matrix_b.size(-1), device=matrix_b.device
                            ).unsqueeze(0).unsqueeze(0).expand_as(matrix_b)

            original_shape_a = matrix_a_gen.size()
            new_shape_a = [-1, param.size(2)]
            matrix_a_gen = matrix_a_gen.reshape(new_shape_a)
            matrix_a_scalar = matrix_a_scalar.reshape([-1,1])
            matrix_a = torch.bmm(matrix_a_scalar.unsqueeze(-1) * \
                                 matrix_a_gen.unsqueeze(-1),
                                 matrix_a_gen.unsqueeze(-2))
            matrix_a = matrix_a.reshape(list(original_shape_a)[:-1] +
                             [self.num_outer_products] + 2 * [param.size(2)])
            matrix_a = matrix_a.sum(dim=-3)

            matrix_a += matrix_a_eye.unsqueeze(-1) * \
                    torch.eye(matrix_a.size(-1), device=matrix_a.device
                            ).unsqueeze(0).unsqueeze(0).expand_as(matrix_a)

            output_matrices.append([matrix_a, matrix_b])

        return output_matrices, state

    def set_listening(self, listening: bool):
        for param in self.warp_parameters:
            param.listening = listening
