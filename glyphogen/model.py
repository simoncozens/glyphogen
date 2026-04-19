#!/usr/bin/env python
from typing import List, Tuple

import torch
from torch import nn

from glyphogen.losses import dump_debug_sequences, losses
from glyphogen.segmenter import (
    SegmenterOutput as SegmentedMask,
    get_model_instance_segmentation,
    segment_single_image,
)
from glyphogen.typing import (
    CollatedGlyphData,
    ContourVectorization,
    LossDictionary,
    ModelResults,
)
from glyphogen.vectorizer import ContourVectorizer

DEBUG = True


class VectorizationGenerator(nn.Module):
    """Wrapper that composes a segmenter and a contour-only vectorizer."""

    def __init__(
        self,
        segmenter_state,
        d_model: int,
        latent_dim: int = 32,
        rate: float = 0.1,
        segmenter_score_threshold: float = 0.0,
    ):
        super().__init__()
        self.vectorizer = ContourVectorizer(
            d_model=d_model,
            latent_dim=latent_dim,
            rate=rate,
        )

        self.segmenter = get_model_instance_segmentation(
            num_classes=3, load_pretrained=False
        )
        self.segmenter.load_state_dict(segmenter_state)
        self.segmenter.eval()
        for param in self.segmenter.parameters():
            param.requires_grad = False

        self.segmenter_score_threshold = segmenter_score_threshold

        # Preserve existing attribute access used by training code.
        self.decoder = self.vectorizer.decoder
        self.mask_encoder = self.vectorizer.mask_encoder
        self.arg_counts = self.vectorizer.arg_counts
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.rate = rate
        self.use_raster = False

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.vectorizer.encode(inputs)

    def forward(
        self, collated_batch: CollatedGlyphData, teacher_forcing_ratio: float = 1.0
    ) -> ModelResults:
        return self.vectorizer(collated_batch, teacher_forcing_ratio)

    def teacher_forcing(self, *args, **kwargs) -> ModelResults:
        return self.vectorizer.teacher_forcing(*args, **kwargs)

    def autoregression(self, *args, **kwargs) -> ModelResults:
        return self.vectorizer.autoregression(*args, **kwargs)

    def segment_contours(self, raster_image: torch.Tensor) -> List[SegmentedMask]:
        return segment_single_image(
            self.segmenter,
            raster_image,
            score_threshold=self.segmenter_score_threshold,
        )

    def vectorize_contour(self, normalized_mask: torch.Tensor) -> torch.Tensor:
        return self.vectorizer.vectorize_contour(normalized_mask)

    def vectorize_contours(
        self, raster_image: torch.Tensor
    ) -> List[ContourVectorization]:
        segmented_contours = self.segment_contours(raster_image)
        if not segmented_contours:
            return []

        normalized_masks_batch = torch.stack(
            [contour.normalized_mask() for contour in segmented_contours], dim=0
        ).to(raster_image.device)
        pred_categories = [contour.label for contour in segmented_contours]

        results = self.vectorizer.generate_from_normalized(
            normalized_masks_batch,
            pred_categories,
        )

        contour_results: List[ContourVectorization] = []
        for i, contour in enumerate(segmented_contours):
            contour_results.append(
                {
                    "box": contour.box,
                    "label": contour.label,
                    "raw_mask": contour.raw_mask,
                    "normalized_mask": contour.normalized_mask(),
                    "pred_commands": results.pred_commands[i],
                    "pred_coords_std": results.pred_coords_std[i],
                    "pred_coords_norm": results.pred_coords_norm[i],
                }
            )

        return contour_results

    def generate(self, raster_image: torch.Tensor) -> ModelResults:
        contour_results = self.vectorize_contours(raster_image)
        if not contour_results:
            return ModelResults.empty()

        return ModelResults(
            pred_commands=[contour["pred_commands"] for contour in contour_results],
            pred_coords_std=[contour["pred_coords_std"] for contour in contour_results],
            gt_coords_std=[],
            pred_coords_norm=[
                contour["pred_coords_norm"] for contour in contour_results
            ],
            used_teacher_forcing=False,
            pred_categories=[contour["label"] for contour in contour_results],
            lstm_outputs=[],
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

    for key, value in collated_batch.items():
        if key == "gt_targets":
            continue
        if isinstance(value, torch.Tensor):
            collated_batch[key] = value.to(device)
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            collated_batch[key] = [v.to(device) for v in value]

    if model.training:
        outputs = model(
            collated_batch,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
    else:
        outputs = model(collated_batch, teacher_forcing_ratio=1.0)

    loss_values = losses(
        collated_batch, outputs, model, device, validation=not model.training
    )

    if DEBUG and writer is not None and global_step % 5000 == 0:
        gt_targets = collated_batch["gt_targets"]
        if gt_targets:
            contour_image_idx = collated_batch["contour_image_idx"]
            for i in range(len(gt_targets)):
                image_contour_mask = contour_image_idx == i
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
                    # Extract original boxes for contours in this image
                    original_boxes_all = collated_batch["original_boxes"]
                    original_boxes_for_image = original_boxes_all[image_contour_mask]

                    dump_debug_sequences(
                        writer,
                        global_step,
                        i,
                        gt_targets[i]["gt_contours"],
                        debug_outputs,
                        loss_values,
                        original_boxes_for_image,
                    )
                except Exception as e:
                    print("Error dumping sequences", e)

    return loss_values, [outputs]
