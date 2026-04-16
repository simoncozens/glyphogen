#!/usr/bin/env python
from collections import defaultdict
from typing import Optional, Tuple, List
from jaxtyping import Float

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN

from glyphogen.representations.model import MODEL_REPRESENTATION
from glyphogen.typing import (
    CollatedGlyphData,
    GroundTruthContour,
    LossDictionary,
    ModelResults,
    SegmenterOutput,
    Target,
)
from glyphogen.hyperparameters import GEN_IMAGE_SIZE
from glyphogen.losses import (
    dump_debug_sequences,
    losses,
)
from glyphogen.lstm import LSTMDecoder

DEBUG = True


class Mask:
    bounds: Tuple[float, float, float, float]
    mask_tensor: torch.Tensor

    def __init__(self, bounds, mask_tensor):
        self.bounds = bounds
        self.mask_tensor = mask_tensor


def get_model_instance_segmentation(num_classes, load_pretrained=True) -> MaskRCNN:
    if load_pretrained:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights="MaskRCNN_ResNet50_FPN_Weights.DEFAULT"
        )
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels  # type: ignore
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


class VectorizationGenerator(nn.Module):
    segmenter: MaskRCNN

    def __init__(
        self, segmenter_state, d_model: int, latent_dim: int = 32, rate: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.rate = rate
        self.img_height, self.img_width = GEN_IMAGE_SIZE
        self.img_size = self.img_height

        # Build the convolutional pyramid from image resolution.
        target_feature_size = 8
        channel_schedule = [16, 32, 64, 128, 256]
        kernel_schedule = [7, 5, 5, 3, 3]
        in_channels = 1
        current_h, current_w = self.img_height, self.img_width
        self.num_pyramid_layers = 0

        for i, (out_channels, kernel_size) in enumerate(
            zip(channel_schedule, kernel_schedule), start=1
        ):
            if (
                i > 1
                and current_h <= target_feature_size
                and current_w <= target_feature_size
            ):
                break

            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                stride=2,
            )
            current_h = (current_h + 1) // 2
            current_w = (current_w + 1) // 2
            norm = nn.LayerNorm([out_channels, current_h, current_w])
            relu = nn.ReLU()

            setattr(self, f"conv{i}", conv)
            setattr(self, f"norm{i}", norm)
            setattr(self, f"relu{i}", relu)

            in_channels = out_channels
            self.num_pyramid_layers += 1

        self.dropout = nn.Dropout(rate)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_channels * current_h * current_w, latent_dim)
        self.norm_dense = nn.LayerNorm(latent_dim)
        self.sigmoid = nn.Sigmoid()
        self.output_dense = nn.Linear(latent_dim, latent_dim)
        self.contour_head = nn.Linear(latent_dim, 1)
        torch.nn.init.ones_(self.contour_head.bias)

        # Load and freeze the segmentation model
        self.segmenter: MaskRCNN = get_model_instance_segmentation(
            num_classes=3, load_pretrained=False
        )
        self.segmenter.load_state_dict(segmenter_state)
        self.segmenter.eval()
        for param in self.segmenter.parameters():
            param.requires_grad = False

        # The new context will be the latent vector of the normalized mask
        self.decoder = torch.compile(
            LSTMDecoder(d_model=d_model, latent_dim=latent_dim, rate=rate)
        )
        self.arg_counts: torch.Tensor = torch.tensor(
            list(MODEL_REPRESENTATION.grammar.values()), dtype=torch.long
        )

        self.use_raster = False

    @torch.compile
    def encode(self, inputs):
        """Return a latent vector encoding of the input mask images."""
        x = inputs
        for i in range(1, self.num_pyramid_layers + 1):
            x = getattr(self, f"conv{i}")(x)
            x = getattr(self, f"norm{i}")(x)
            x = getattr(self, f"relu{i}")(x)
            x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.norm_dense(x)
        x = self.sigmoid(x)
        z = self.output_dense(x)
        return z

    def forward(
        self, collated_batch: "CollatedGlyphData", teacher_forcing_ratio=1.0
    ) -> ModelResults:
        # This model implements the canonical mask normalization strategy.
        # It processes a batch of contours collated from multiple images.

        # Data is pre-processed by the collate_fn. We can use it directly.
        normalized_masks_batch = collated_batch["normalized_masks"]
        target_sequences = collated_batch["target_sequences"]
        valid_boxes = collated_batch["contour_boxes"]
        labels = collated_batch["labels"]

        # Send the normalized masks through the latent vector encoder
        z_batch = self.encode(normalized_masks_batch).unsqueeze(1)

        # and then decode a sequence for each contour
        if self.training:
            return self.teacher_forcing(
                target_sequences, valid_boxes, labels, z_batch, teacher_forcing_ratio
            )
        else:  # Validation
            # Always use teacher forcing during validation for a stable loss metric and speed.
            return self.teacher_forcing(
                target_sequences,
                valid_boxes,
                labels,
                z_batch,
                teacher_forcing_ratio=1.0,
            )

    def generate(self, raster_image: torch.Tensor) -> ModelResults:
        """
        This method is for pure inference from a raw image.
        """
        # Step 1: Run the segmenter to get predicted boxes and masks
        with torch.no_grad():
            # The segmenter expects a batch, so unsqueeze the image
            segmenter_output: SegmenterOutput = self.segmenter(
                raster_image.unsqueeze(0)
            )[0]

        if not segmenter_output["masks"].numel():
            return ModelResults.empty()

        # Sort by mask area descending
        areas = (segmenter_output["boxes"][:, 2] - segmenter_output["boxes"][:, 0]) * (
            segmenter_output["boxes"][:, 3] - segmenter_output["boxes"][:, 1]
        )
        sorted_indices = torch.argsort(areas, descending=True)
        contour_boxes = segmenter_output["boxes"][sorted_indices]
        contour_masks = segmenter_output["masks"][sorted_indices].squeeze(1)
        pred_labels = segmenter_output["labels"][sorted_indices]

        # Step 2: Normalize the predicted masks
        normalized_masks = []
        valid_boxes = []
        pred_categories = []
        for i in range(len(contour_boxes)):
            box = contour_boxes[i].clamp(min=0, max=raster_image.shape[-1] - 1)
            mask = contour_masks[i]
            label = pred_labels[i].item()
            x1, y1, x2, y2 = box.long()
            if x1 >= x2 or y1 >= y2:
                continue

            valid_boxes.append(box)
            pred_categories.append(label)
            cropped_mask = mask[y1:y2, x1:x2].unsqueeze(0).unsqueeze(0)
            normalized_mask = F.interpolate(
                cropped_mask.to(torch.float32),
                size=GEN_IMAGE_SIZE,
                mode="bilinear",
                align_corners=False,
            )
            normalized_masks.append(normalized_mask)

        if not normalized_masks:
            return ModelResults.empty()

        # Step 3: Encode the masks and decode using autoregression
        normalized_masks_batch = torch.cat(normalized_masks, dim=0)
        z_batch = self.encode(normalized_masks_batch).unsqueeze(1)

        return self.autoregression(z_batch, valid_boxes, pred_categories)

    def teacher_forcing(
        self, target_sequences, valid_boxes, labels, z_batch, teacher_forcing_ratio
    ) -> ModelResults:
        glyph_pred_commands = []
        glyph_pred_coords_norm = []
        glyph_pred_coords_img_space = []
        glyph_pred_coords_std = []
        glyph_gt_coords_std = []
        glyph_pred_means = []
        glyph_pred_stds = []
        glyph_lstm_outputs = []

        # Prepare batch for decoder
        decoder_inputs_norm = [
            MODEL_REPRESENTATION.image_space_to_mask_space(seq, box)
            for seq, box in zip(target_sequences, valid_boxes)
        ]

        # Pad sequences
        padded_decoder_inputs = torch.nn.utils.rnn.pad_sequence(
            decoder_inputs_norm, batch_first=True, padding_value=0.0
        )

        # Prepare decoder input (all but last token)
        decoder_input_batch = padded_decoder_inputs[:, :-1, :]

        # --- NEW STANDARDIZATION LOGIC ---
        commands_norm, coords_norm = MODEL_REPRESENTATION.split_tensor(
            decoder_input_batch
        )
        command_indices = torch.argmax(commands_norm, dim=-1)
        means, stds = MODEL_REPRESENTATION.get_stats_for_sequence(command_indices)
        coords_std = MODEL_REPRESENTATION.standardize(coords_norm, means, stds)

        # Calculate Heading
        deltas = MODEL_REPRESENTATION.compute_deltas(decoder_input_batch)
        deltas_norm = torch.norm(deltas, p=2, dim=-1, keepdim=True)
        heading = deltas / (deltas_norm + 1e-8)

        decoder_input_std = torch.cat([commands_norm, coords_std, heading], dim=-1)
        # --- END NEW LOGIC ---

        pred_commands_batch, pred_coords_std_batch, lstm_outputs_batch = self.decoder(
            decoder_input_std,  # Pass standardized input
            context=z_batch,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        # Unpad and handle results
        for i in range(len(valid_boxes)):
            box = valid_boxes[i]
            seq_len = decoder_inputs_norm[i].shape[0] - 1  # Original length

            # --- Ground Truth ---
            # Get the GT coords for the loss (the part that was predicted)
            gt_coords_for_loss_std = coords_std[i, :seq_len, :]
            glyph_gt_coords_std.append(gt_coords_for_loss_std)

            # --- Predictions ---
            pred_commands = pred_commands_batch[i, :seq_len, :]
            pred_coords_std = pred_coords_std_batch[i, :seq_len, :]
            lstm_outputs = lstm_outputs_batch[i, :seq_len, :]
            glyph_pred_commands.append(pred_commands)
            glyph_pred_coords_std.append(pred_coords_std)
            glyph_lstm_outputs.append(lstm_outputs)

            # --- De-standardize for metrics and logging ---
            pred_command_indices = torch.argmax(pred_commands, dim=-1)
            pred_means, pred_stds = MODEL_REPRESENTATION.get_stats_for_sequence(
                pred_command_indices
            )
            glyph_pred_means.append(pred_means)
            glyph_pred_stds.append(pred_stds)

            pred_coords_norm = MODEL_REPRESENTATION.de_standardize(
                pred_coords_std, pred_means, pred_stds
            )
            glyph_pred_coords_norm.append(pred_coords_norm)

            # --- De-normalize to image space for logging/decoding ---
            pred_sequence_norm = torch.cat([pred_commands, pred_coords_norm], dim=-1)
            sos_token = decoder_inputs_norm[i][0:1, :]
            full_pred_sequence_norm = torch.cat([sos_token, pred_sequence_norm], dim=0)

            pred_sequence_img_space = MODEL_REPRESENTATION.mask_space_to_image_space(
                full_pred_sequence_norm, box
            )
            glyph_pred_coords_img_space.append(
                pred_sequence_img_space[1:, MODEL_REPRESENTATION.command_width :]
            )

        return ModelResults(
            pred_commands=glyph_pred_commands,
            pred_coords_std=glyph_pred_coords_std,
            gt_coords_std=glyph_gt_coords_std,
            pred_means=glyph_pred_means,
            pred_stds=glyph_pred_stds,
            pred_coords_norm=glyph_pred_coords_norm,
            pred_coords_img_space=glyph_pred_coords_img_space,
            used_teacher_forcing=True,
            contour_boxes=valid_boxes,
            pred_categories=labels,
            lstm_outputs=glyph_lstm_outputs,
        )

    def autoregression(
        self,
        z_batch,
        valid_boxes: List[Float[torch.Tensor, "4"]],
        pred_categories: List[int],
    ) -> ModelResults:
        glyph_pred_commands = []
        glyph_pred_coords_norm = []
        glyph_pred_coords_img_space = []
        glyph_pred_coords_std = []
        glyph_pred_means = []
        glyph_pred_stds = []
        glyph_lstm_outputs = []

        batch_size = len(valid_boxes)
        device = z_batch.device
        sos_index = MODEL_REPRESENTATION.encode_command("SOS")
        # --- Standardize the initial SOS input ---
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
        # ---

        hidden_state = None
        batch_contour_commands = [[] for _ in range(batch_size)]
        batch_contour_coords_std = [[] for _ in range(batch_size)]
        active_indices = list(range(batch_size))

        for _ in range(50):  # max_length
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
            glyph_pred_means.append(pred_means)
            glyph_pred_stds.append(pred_stds)

            pred_coords_norm = MODEL_REPRESENTATION.de_standardize(
                pred_coords_std, pred_means, pred_stds
            )
            glyph_pred_coords_norm.append(pred_coords_norm)

            pred_sequence_norm = torch.cat([pred_commands, pred_coords_norm], dim=-1)
            sos_cmd = (
                F.one_hot(
                    torch.tensor(sos_index),
                    num_classes=MODEL_REPRESENTATION.command_width,
                )
                .float()
                .to(device)
            )
            sos_coords = torch.zeros(
                MODEL_REPRESENTATION.coordinate_width, device=device
            )
            sos_token = torch.cat([sos_cmd, sos_coords]).unsqueeze(0)
            full_pred_sequence_norm = torch.cat([sos_token, pred_sequence_norm], dim=0)

            pred_sequence_img_space = MODEL_REPRESENTATION.mask_space_to_image_space(
                full_pred_sequence_norm, valid_boxes[i]
            )
            glyph_pred_coords_img_space.append(
                pred_sequence_img_space[1:, MODEL_REPRESENTATION.command_width :]
            )
        return ModelResults(
            pred_commands=glyph_pred_commands,
            pred_coords_std=glyph_pred_coords_std,
            gt_coords_std=[],  # No GT in autoregression
            pred_means=glyph_pred_means,
            pred_stds=glyph_pred_stds,
            pred_coords_norm=glyph_pred_coords_norm,
            pred_coords_img_space=glyph_pred_coords_img_space,
            used_teacher_forcing=False,
            contour_boxes=valid_boxes,
            pred_categories=pred_categories,
            lstm_outputs=[],  # No lstm_outputs in autoregression
        )


def step(
    model,
    batch,
    writer,
    global_step,
    teacher_forcing_ratio=1.0,
) -> Tuple[LossDictionary, List[ModelResults]]:
    device = next(model.parameters()).device
    collated_batch = batch

    if collated_batch is None:
        # Batch was empty or invalid, skip.
        final_losses: LossDictionary = {
            "total_loss": torch.tensor(0.0, device=device, requires_grad=True),
            "command_loss": torch.tensor(0.0, device=device),
            "coord_loss": torch.tensor(0.0, device=device),
            "signed_area_loss": torch.tensor(0.0, device=device),
            "command_accuracy_metric": torch.tensor(0.0, device=device),
            "coordinate_mae_metric": torch.tensor(0.0, device=device),
            "alignment_loss": torch.tensor(0.0, device=device),
            "contrastive_loss": torch.tensor(0.0, device=device),
        }
        return final_losses, []

    # Move all tensors in the collated batch to the correct device
    for key, value in collated_batch.items():
        if key == "gt_targets":  # This is handled by the losses function
            continue
        if isinstance(value, torch.Tensor):
            collated_batch[key] = value.to(device)
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            collated_batch[key] = [v.to(device) for v in value]

    # The model now processes the entire batch of contours at once.
    if model.training:
        outputs = model(
            collated_batch,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
    else:  # Validation/Inference
        outputs = model(collated_batch, teacher_forcing_ratio=1.0)

    # Loss is also calculated over the entire batch.
    loss_values = losses(
        collated_batch, outputs, model, device, validation=not model.training
    )

    if DEBUG and writer is not None and global_step % 100 == 0:
        # Debugging needs to be adapted for the new batch structure
        # For now, let's debug the first image in the batch
        gt_targets = collated_batch["gt_targets"]
        if gt_targets:
            contour_image_idx = collated_batch["contour_image_idx"]
            for i in range(len(gt_targets)):
                image_contour_mask = contour_image_idx == i
                # Create a sliced version of outputs for the image
                debug_outputs = ModelResults(
                    **{
                        field: [
                            getattr(outputs, field)[i]
                            for i, mask in enumerate(image_contour_mask)
                            if mask
                        ]
                        for field in ModelResults._fields
                        if field != "used_teacher_forcing"
                    },
                    used_teacher_forcing=outputs.used_teacher_forcing,
                )

                try:
                    dump_debug_sequences(
                        writer,
                        global_step,
                        i,
                        gt_targets[i]["gt_contours"],
                        debug_outputs,
                        loss_values,
                    )
                except Exception as e:
                    print("Error dumping sequences", e)

    return loss_values, [outputs]  # Return list with one item for consistency
