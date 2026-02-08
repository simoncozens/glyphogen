#!/usr/bin/env python
import random
import torch
from torch import nn
import torch.nn.functional as F

from glyphogen.hyperparameters import PROJ_SIZE

from glyphogen.representations.model import MODEL_REPRESENTATION


class LSTMDecoder(nn.Module):
    def __init__(self, d_model, latent_dim, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.proj_size = PROJ_SIZE
        self.rate = rate

        self.command_embedding = nn.Linear(MODEL_REPRESENTATION.command_width, d_model)
        self.coord_embedding = nn.Linear(MODEL_REPRESENTATION.coordinate_width, d_model)
        self.heading_embedding = nn.Linear(2, d_model)  # New heading embedding
        self.lstm = nn.LSTM(
            d_model + latent_dim, d_model, batch_first=True, proj_size=self.proj_size
        )
        self.layer_norm = nn.LayerNorm(self.proj_size)
        self.dropout = nn.Dropout(rate)
        self.output_command = nn.Linear(
            self.proj_size, MODEL_REPRESENTATION.command_width
        )
        self.output_command_activation = nn.ReLU()
        self.output_coords = nn.Linear(
            self.proj_size + MODEL_REPRESENTATION.command_width,
            MODEL_REPRESENTATION.coordinate_width,
        )
        self.tanh = nn.Tanh()
        nn.init.zeros_(self.output_coords.bias)

    def _forward_step(self, input_token, context, hidden_state=None):
        """
        Performs a single decoding step.
        Args:
            input_token (Tensor): Shape (batch_size, 1, MODEL_REPRESENTATION.command_width + MODEL_REPRESENTATION.coordinate_width + 2)
            context (Tensor): Shape (batch_size, 1, latent_dim)
            hidden_state (tuple, optional): Previous hidden state of the LSTM.
        Returns:
            command_logits (Tensor): Shape (batch_size, 1, MODEL_REPRESENTATION.command_width)
            coord_output (Tensor): Shape (batch_size, 1, MODEL_REPRESENTATION.coordinate_width)
            hidden_state (tuple): New hidden state of the LSTM.
        """
        command_input = input_token[:, :, : MODEL_REPRESENTATION.command_width].float()
        coord_input = input_token[
            :,
            :,
            MODEL_REPRESENTATION.command_width : MODEL_REPRESENTATION.command_width
            + MODEL_REPRESENTATION.coordinate_width,
        ].float()
        heading_input = input_token[
            :,
            :,
            MODEL_REPRESENTATION.command_width
            + MODEL_REPRESENTATION.coordinate_width :,
        ].float()

        command_emb = self.command_embedding(command_input)
        coord_emb = self.coord_embedding(coord_input)
        heading_emb = self.heading_embedding(heading_input)

        x = command_emb + coord_emb + heading_emb
        x = self.dropout(x)

        if context is not None:
            x = torch.cat([x, context], dim=-1)

        x, hidden_state = self.lstm(x, hidden_state)
        x = self.layer_norm(x)

        command_logits = self.output_command(x)
        coord_head_input = torch.cat([x, command_logits], dim=-1)
        coord_output = self.output_coords(coord_head_input)

        return command_logits, coord_output, hidden_state

    def forward(self, x_std, context=None, teacher_forcing_ratio=1.0):
        """
        Training forward pass with scheduled sampling.
        Operates entirely in STANDARDIZED coordinate space.
        `x_std` is the ground truth sequence, already standardized.
        The returned `coord_output_std` is also in STANDARDIZED space.
        """
        batch_size, seq_len, _ = x_std.shape
        current_input_std = x_std[:, 0:1, :]
        hidden_state = None
        all_command_logits = []
        all_coord_outputs_std = []

        for i in range(seq_len):
            command_logits, coord_output_std, hidden_state = self._forward_step(
                current_input_std, context, hidden_state
            )
            all_command_logits.append(command_logits)
            all_coord_outputs_std.append(coord_output_std)

            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if i + 1 < seq_len:
                if use_teacher_forcing:
                    current_input_std = x_std[:, i + 1 : i + 2, :]
                else:
                    command_probs = F.softmax(command_logits.squeeze(1), dim=-1)
                    predicted_command_idx = torch.argmax(
                        command_probs, dim=1, keepdim=True
                    )
                    next_command_onehot = F.one_hot(
                        predicted_command_idx,
                        num_classes=MODEL_REPRESENTATION.command_width,
                    ).float()

                    # Calculate Heading for next step
                    next_means, next_stds = MODEL_REPRESENTATION.get_stats_for_sequence(
                        predicted_command_idx
                    )
                    coord_output_norm = MODEL_REPRESENTATION.de_standardize(
                        coord_output_std, next_means, next_stds
                    )
                    temp_token_norm = torch.cat(
                        [next_command_onehot, coord_output_norm], dim=-1
                    )
                    deltas = MODEL_REPRESENTATION.compute_deltas(temp_token_norm)
                    deltas_norm = torch.norm(deltas, p=2, dim=-1, keepdim=True)
                    heading = deltas / (deltas_norm + 1e-8)

                    current_input_std = torch.cat(
                        [next_command_onehot, coord_output_std, heading], dim=-1
                    )

        command_output = torch.cat(all_command_logits, dim=1)
        coord_output_std = torch.cat(all_coord_outputs_std, dim=1)
        return command_output, coord_output_std
