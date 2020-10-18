import torch
from torch import nn
from typing import List
from math import sqrt
from .model import ExpandingParameter
from torch.nn.init import orthogonal_

class TransformerMetric(nn.Module):

    def __init__(self, warp_parameters, hidden_size: int = 16, transformer_layers: int = 2, transformer_heads: int = 4, transformer_feedforward_dim: int = 32, transformer_d_model: int = 16, transformer_cascade_direction: str = "bidirectional", num_outer_products = 5):

        super(TransformerMetric, self).__init__()

        self.linear_input = []
        self.warp_parameters = warp_parameters

        self.hidden_size = hidden_size
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_feedforward_dim = transformer_feedforward_dim
        self.transformer_d_model = transformer_d_model
        self.transformer_cascade_direction = transformer_cascade_direction
        self.num_outer_products = num_outer_products

        for param in warp_parameters:
            input_dim = param.size(2) + param.size(1)
            self.linear_input.append(nn.Linear(input_dim,
                                               self.transformer_d_model))
        self.linear_input = nn.ModuleList(self.linear_input)
        direction = self.transformer_cascade_direction

        if direction == "forward" or direction == "bidirectional":
            self.forward_transformers = [[] for _ in range(self.transformer_layers)]
            for i in range(self.transformer_layers):
                for j in range(len(warp_parameters)):
                    if j == 0:
                        self.forward_transformers[i].append(
                            nn.TransformerEncoderLayer(self.transformer_d_model,
                                                      self.transformer_heads,
                                                      self.transformer_feedforward_dim,
                                                      dropout=0)
                        )
                    else:
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
                    if j == len(warp_parameters) - 1:
                        self.backward_transformers[i].append(
                            nn.TransformerEncoderLayer(self.transformer_d_model,
                                                      self.transformer_heads,
                                                      self.transformer_feedforward_dim,
                                                      dropout=0)
                        )
                    else:
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

    def forward(self, warp_inputs: List[List[torch.Tensor]]):
        embeddings = []
        direction = self.transformer_cascade_direction

        # Get all inputs to be same dimension so we can pass through the
        # transformer cascade
        for i, warp_input in enumerate(warp_inputs):
            embeddings.append(self.linear_input[i](torch.cat(warp_input, dim=-1)))

        for i in range(self.transformer_layers):
            if direction == "forward" or direction == "bidirectional":
                forward_transformer_out = []
                memory = None
                for j, embedding in enumerate(embeddings):
                    if j == 0:
                        memory = \
                                self.forward_transformers[i][j](embedding.transpose(0,1))
                    else:
                        memory = self.forward_transformers[i][j](
                            embedding.transpose(0,1), memory)

                    forward_transformer_out.append(memory)
                forward_transformer_out = torch.stack(forward_transformer_out)

            if direction == "backward" or direction == "bidirectional":
                backward_transformer_out = []
                memory = None
                for j, embedding in reversed(list(enumerate(embeddings))):
                    if j == len(embeddings) - 1:
                        memory = \
                                self.backward_transformers[i][j](embedding.transpose(0,1))
                    else:
                        memory = self.backward_transformers[i][j](
                            embedding.transpose(0,1), memory)

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

        output_matrices = []
        for i, (embedding, param) in enumerate(zip(embeddings,
                                                   self.warp_parameters)):
            matrix_b_gen = self.output_heads[i][0](embedding)
            matrix_b_scalar = self.output_heads[i][1](embedding)
            matrix_a_gen = self.output_heads[i][2](embedding)
            matrix_a_scalar = self.output_heads[i][3](embedding)

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

            output_matrices.append([matrix_a, matrix_b])

        return output_matrices

    def set_listening(self, listening: bool):
        for param in self.warp_parameters:
            param.listening = listening
