from typing import List

from jaxtyping import Float
import torch
import torch.nn.functional as F
from torch import nn

from glyphogen.representations.model import MODEL_REPRESENTATION
from glyphogen.typing import CollatedGlyphData, ModelResults
from glyphogen.hyperparameters import GEN_IMAGE_SIZE
from glyphogen.latent_encoder import MaskLatentEncoder
from glyphogen.lstm import LSTMDecoder


class ContourVectorizer(nn.Module):
    """Vectorizer that operates only on normalized contour masks."""

    def __init__(self, d_model: int, latent_dim: int = 32, rate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.rate = rate
        self.img_height, self.img_width = GEN_IMAGE_SIZE
        self.img_size = self.img_height

        self.mask_encoder = MaskLatentEncoder(
            image_size=GEN_IMAGE_SIZE,
            latent_dim=latent_dim,
            rate=rate,
        )

        self.decoder_core = LSTMDecoder(
            d_model=d_model, latent_dim=latent_dim, rate=rate
        )
        self.decoder = torch.compile(self.decoder_core)
        self.arg_counts: torch.Tensor = torch.tensor(
            list(MODEL_REPRESENTATION.grammar.values()), dtype=torch.long
        )

        self.use_raster = False

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return latent vectors for normalized contour masks [N, 1, H, W]."""
        return self.mask_encoder(inputs)

    def forward(
        self, collated_batch: CollatedGlyphData, teacher_forcing_ratio: float = 1.0
    ) -> ModelResults:
        normalized_masks_batch = collated_batch["normalized_masks"]
        target_sequences = collated_batch["target_sequences"]
        labels = collated_batch["labels"]

        z_batch = self.encode(normalized_masks_batch).unsqueeze(1)

        if self.training:
            return self.teacher_forcing(
                target_sequences, labels, z_batch, teacher_forcing_ratio
            )
        return self.teacher_forcing(
            target_sequences,
            labels,
            z_batch,
            teacher_forcing_ratio=1.0,
        )

    def generate_from_normalized(
        self,
        normalized_masks_batch: torch.Tensor,
        pred_categories: List[int],
    ) -> ModelResults:
        """Autoregress from normalized contour masks."""
        if normalized_masks_batch.numel() == 0:
            return ModelResults.empty()

        z_batch = self.encode(normalized_masks_batch).unsqueeze(1)
        return self.autoregression(z_batch, pred_categories)

    def vectorize_contour(self, normalized_mask: torch.Tensor) -> torch.Tensor:
        """Vectorize one normalized contour mask and return a full mask-space sequence."""
        if normalized_mask.dim() == 2:
            normalized_mask = normalized_mask.unsqueeze(0).unsqueeze(0)
        elif normalized_mask.dim() == 3:
            normalized_mask = normalized_mask.unsqueeze(0)

        normalized_mask = normalized_mask.to(torch.float32)
        device = normalized_mask.device

        outputs = self.generate_from_normalized(
            normalized_mask,
            [1],
        )
        if not outputs.pred_commands:
            return torch.empty(
                0,
                MODEL_REPRESENTATION.command_width
                + MODEL_REPRESENTATION.coordinate_width,
                device=device,
            )

        pred_commands = outputs.pred_commands[0]
        pred_coords_norm = outputs.pred_coords_norm[0]

        sos_idx = MODEL_REPRESENTATION.encode_command("SOS")
        sos_cmd = F.one_hot(
            torch.tensor(sos_idx, device=device),
            num_classes=MODEL_REPRESENTATION.command_width,
        ).float()
        sos_coords = torch.zeros(MODEL_REPRESENTATION.coordinate_width, device=device)
        sos_token = torch.cat([sos_cmd, sos_coords]).unsqueeze(0)

        return torch.cat(
            [sos_token, torch.cat([pred_commands, pred_coords_norm], dim=-1)], dim=0
        )

    def teacher_forcing(
        self, target_sequences, labels, z_batch, teacher_forcing_ratio
    ) -> ModelResults:
        glyph_pred_commands = []
        glyph_pred_coords_norm = []
        glyph_pred_coords_std = []
        glyph_gt_coords_std = []
        glyph_lstm_outputs = []

        decoder_inputs_norm = [seq for seq in target_sequences]

        padded_decoder_inputs = torch.nn.utils.rnn.pad_sequence(
            decoder_inputs_norm, batch_first=True, padding_value=0.0
        )
        decoder_input_batch = padded_decoder_inputs[:, :-1, :]

        commands_norm, coords_norm = MODEL_REPRESENTATION.split_tensor(
            decoder_input_batch
        )
        command_indices = torch.argmax(commands_norm, dim=-1)
        means, stds = MODEL_REPRESENTATION.get_stats_for_sequence(command_indices)
        coords_std = MODEL_REPRESENTATION.standardize(coords_norm, means, stds)

        deltas = MODEL_REPRESENTATION.compute_deltas(decoder_input_batch)
        deltas_norm = torch.norm(deltas, p=2, dim=-1, keepdim=True)
        heading = deltas / (deltas_norm + 1e-8)

        decoder_input_std = torch.cat([commands_norm, coords_std, heading], dim=-1)

        pred_commands_batch, pred_coords_std_batch, lstm_outputs_batch = self.decoder(
            decoder_input_std,
            context=z_batch,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        pred_command_indices_batch = torch.argmax(pred_commands_batch, dim=-1)
        pred_means_batch, pred_stds_batch = MODEL_REPRESENTATION.get_stats_for_sequence(
            pred_command_indices_batch
        )
        pred_coords_norm_batch = MODEL_REPRESENTATION.de_standardize(
            pred_coords_std_batch, pred_means_batch, pred_stds_batch
        )

        for i in range(len(target_sequences)):
            seq_len = decoder_inputs_norm[i].shape[0] - 1

            gt_coords_for_loss_std = coords_std[i, :seq_len, :]
            glyph_gt_coords_std.append(gt_coords_for_loss_std)

            pred_commands = pred_commands_batch[i, :seq_len, :]
            pred_coords_std = pred_coords_std_batch[i, :seq_len, :]
            pred_coords_norm = pred_coords_norm_batch[i, :seq_len, :]
            lstm_outputs = lstm_outputs_batch[i, :seq_len, :]
            glyph_pred_commands.append(pred_commands)
            glyph_pred_coords_std.append(pred_coords_std)
            glyph_pred_coords_norm.append(pred_coords_norm)
            glyph_lstm_outputs.append(lstm_outputs)

        return ModelResults(
            pred_commands=glyph_pred_commands,
            pred_coords_std=glyph_pred_coords_std,
            gt_coords_std=glyph_gt_coords_std,
            pred_coords_norm=glyph_pred_coords_norm,
            used_teacher_forcing=True,
            pred_categories=(
                labels.tolist() if isinstance(labels, torch.Tensor) else labels
            ),
            lstm_outputs=glyph_lstm_outputs,
        )

    def autoregression(
        self,
        z_batch: torch.Tensor,
        pred_categories: List[int],
    ) -> ModelResults:
        glyph_pred_commands = []
        glyph_pred_coords_norm = []
        glyph_pred_coords_std = []

        batch_size = z_batch.shape[0]
        device = z_batch.device
        sos_index = MODEL_REPRESENTATION.encode_command("SOS")

        command_part = torch.zeros(
            batch_size, 1, MODEL_REPRESENTATION.command_width, device=device
        )
        command_part[:, 0, sos_index] = 1.0
        coords_part_norm = torch.zeros(
            batch_size, 1, MODEL_REPRESENTATION.coordinate_width, device=device
        )
        command_indices = torch.argmax(command_part, dim=-1)
        means, stds = MODEL_REPRESENTATION.get_stats_for_sequence(command_indices)
        coords_part_std = MODEL_REPRESENTATION.standardize(
            coords_part_norm, means, stds
        )
        heading_part = torch.zeros(batch_size, 1, 2, device=device)
        current_input_std = torch.cat(
            [command_part, coords_part_std, heading_part], dim=-1
        )

        hidden_state = None
        batch_contour_commands = [[] for _ in range(batch_size)]
        batch_contour_coords_std = [[] for _ in range(batch_size)]
        active_indices = list(range(batch_size))

        for _ in range(50):
            if not active_indices:
                break

            active_z = z_batch[active_indices]

            command_logits, coord_output_std, hidden_state, _ = self.decoder._forward_step(  # type: ignore
                current_input_std, active_z, hidden_state
            )

            for i, original_idx in enumerate(active_indices):
                batch_contour_commands[original_idx].append(command_logits[i : i + 1])
                batch_contour_coords_std[original_idx].append(
                    coord_output_std[i : i + 1]
                )

            command_probs = F.softmax(command_logits.squeeze(1), dim=-1)
            predicted_command_idx = torch.argmax(command_probs, dim=1, keepdim=True)
            eos_mask = predicted_command_idx.squeeze(
                1
            ) == MODEL_REPRESENTATION.encode_command("EOS")

            if any(eos_mask):
                active_indices_mask = ~eos_mask
                active_indices = [
                    idx
                    for i, idx in enumerate(active_indices)
                    if active_indices_mask[i]
                ]
                if not active_indices:
                    break
                if hidden_state is not None:
                    h, c = hidden_state
                    hidden_state = (
                        h[:, active_indices_mask, :],
                        c[:, active_indices_mask, :],
                    )
                predicted_command_idx = predicted_command_idx[active_indices_mask]
                coord_output_std = coord_output_std[active_indices_mask]

            next_command_onehot = F.one_hot(
                predicted_command_idx,
                num_classes=MODEL_REPRESENTATION.command_width,
            ).float()

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

        for i in range(batch_size):
            if not batch_contour_commands[i]:
                pred_commands = torch.empty(
                    0, MODEL_REPRESENTATION.command_width, device=device
                )
                pred_coords_std = torch.empty(
                    0, MODEL_REPRESENTATION.coordinate_width, device=device
                )
            else:
                pred_commands = torch.cat(batch_contour_commands[i], dim=1).squeeze(0)
                pred_coords_std = torch.cat(batch_contour_coords_std[i], dim=1).squeeze(
                    0
                )

            glyph_pred_commands.append(pred_commands)
            glyph_pred_coords_std.append(pred_coords_std)

            pred_command_indices = torch.argmax(pred_commands, dim=-1)
            pred_means, pred_stds = MODEL_REPRESENTATION.get_stats_for_sequence(
                pred_command_indices
            )

            pred_coords_norm = MODEL_REPRESENTATION.de_standardize(
                pred_coords_std, pred_means, pred_stds
            )
            glyph_pred_coords_norm.append(pred_coords_norm)

        return ModelResults(
            pred_commands=glyph_pred_commands,
            pred_coords_std=glyph_pred_coords_std,
            gt_coords_std=[],
            pred_coords_norm=glyph_pred_coords_norm,
            used_teacher_forcing=False,
            pred_categories=pred_categories,
            lstm_outputs=[],
        )
