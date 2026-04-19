from typing import NamedTuple, List, TypedDict
import torch
from jaxtyping import Float, UInt2


class ModelResults(NamedTuple):
    """NamedTuple for model results."""

    pred_commands: List[Float[torch.Tensor, "seq_len command_dim"]]
    pred_coords_norm: List[Float[torch.Tensor, "contour point_dim"]]
    pred_coords_std: List[Float[torch.Tensor, "contour point_dim"]]
    gt_coords_std: List[Float[torch.Tensor, "contour point_dim"]]
    used_teacher_forcing: bool
    pred_categories: List[int]
    lstm_outputs: List[torch.Tensor]

    @classmethod
    def empty(cls) -> "ModelResults":
        """Create an empty ModelResults instance."""
        return cls(
            pred_commands=[],
            pred_coords_norm=[],
            pred_coords_std=[],
            gt_coords_std=[],
            used_teacher_forcing=False,
            pred_categories=[],
            lstm_outputs=[],
        )


class SegmenterOutput(TypedDict):
    """TypedDict for segmenter outputs."""

    boxes: Float[torch.Tensor, "num_boxes 4"]
    masks: Float[torch.Tensor, "num_boxes height width"]
    labels: UInt2[torch.Tensor, "num_boxes"]


class SegmentedContour(TypedDict):
    box: Float[torch.Tensor, "4"]
    label: int
    raw_mask: Float[torch.Tensor, "height width"]
    normalized_mask: Float[torch.Tensor, "1 512 512"]


class ContourVectorization(TypedDict):
    box: Float[torch.Tensor, "4"]
    label: int
    raw_mask: Float[torch.Tensor, "height width"]
    normalized_mask: Float[torch.Tensor, "1 512 512"]
    pred_commands: Float[torch.Tensor, "seq_len command_dim"]
    pred_coords_norm: Float[torch.Tensor, "contour point_dim"]
    pred_coords_std: Float[torch.Tensor, "contour point_dim"]


class LossDictionary(TypedDict):
    """TypedDict for loss values."""

    total_loss: Float[torch.Tensor, ""]
    command_loss: Float[torch.Tensor, ""]
    coord_loss: Float[torch.Tensor, ""]
    signed_area_loss: Float[torch.Tensor, ""]
    alignment_loss: Float[torch.Tensor, ""]
    contrastive_loss: Float[torch.Tensor, ""]
    command_accuracy_metric: Float[torch.Tensor, ""]
    coordinate_mae_metric: Float[torch.Tensor, ""]


class GroundTruthContour(TypedDict):
    box: Float[torch.Tensor, "4"]
    label: UInt2[torch.Tensor, ""]
    mask: UInt2[torch.Tensor, "height width"]
    sequence: Float[
        torch.Tensor, "seq_len 17"
    ]  # Assuming 17 command dimensions, relative NodeCommand encoding
    x_aligned_point_indices: List[List[int]]
    y_aligned_point_indices: List[List[int]]
    filename: str
    character: str
    contour_number: int


class Target(TypedDict):
    image_id: int
    filename: str
    character: str
    gt_contours: List[GroundTruthContour]


class CollatedGlyphData(TypedDict):
    """
    Data structure for a batch of glyph data, collated by the custom collate_fn.
    Contours from all images in the batch are flattened into single tensors.
    """

    images: Float[torch.Tensor, "batch channels height width"]
    gt_targets: List[Target]  # Original targets for loss calculation
    normalized_masks: Float[torch.Tensor, "total_contours 1 512 512"]
    original_boxes: Float[
        torch.Tensor, "total_contours 4"
    ]  # Original image-space bounding boxes for denormalization
    target_sequences: List[
        Float[torch.Tensor, "seq_len 17"]
    ]  # Normalized-space contour sequences (variable length)
    contour_image_idx: torch.Tensor  # Maps each contour to its image index in the batch
    contour_filenames: List[str]
    contour_characters: List[str]
    contour_numbers: torch.Tensor
    x_aligned_point_indices: List[List[List[int]]]
    y_aligned_point_indices: List[List[List[int]]]
    labels: Float[torch.Tensor, "total_contours"]
