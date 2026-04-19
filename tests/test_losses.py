import torch
import numpy as np
import pytest

from glyphogen.losses import losses
from glyphogen.representations.nodecommand import NodeCommand
from glyphogen.nodeglyph import NodeGlyph, NodeContour, Node
from glyphogen.typing import ModelResults, CollatedGlyphData


def _create_square_glyph():
    """Helper to create a simple 100x100 square NodeGlyph."""
    contour = NodeContour([])
    contour.push(np.array([0, 0]), None, None)
    contour.push(np.array([100, 0]), None, None)
    contour.push(np.array([100, 100]), None, None)
    contour.push(np.array([0, 100]), None, None)
    return NodeGlyph([contour])


def _get_glyph_bounds(glyph: NodeGlyph):
    """Helper to get the bounding box of a glyph."""
    all_coords = [
        node.coordinates for contour in glyph.contours for node in contour.nodes
    ]
    if not all_coords:
        return torch.tensor([0, 0, 0, 0], dtype=torch.float32)
    coords_arr = np.array(all_coords)
    min_x, min_y = coords_arr.min(axis=0)
    max_x, max_y = coords_arr.max(axis=0)
    return torch.tensor([min_x, min_y, max_x, max_y], dtype=torch.float32)


def _run_loss_test(gt_glyph: NodeGlyph, pred_glyph: NodeGlyph) -> dict:
    """
    A helper function to perform all the boilerplate for running the main
    losses function with a ground truth and predicted glyph.
    """
    device = torch.device("cpu")

    # 1. Encode GT and Pred glyphs to get their command sequences.
    # The command sequence must match so only geometry differs.
    gt_encoded = gt_glyph.encode(NodeCommand)
    if not gt_encoded:
        raise ValueError("Could not encode GT glyph")
    gt_sequence_img = torch.from_numpy(gt_encoded[0]).to(torch.float32)
    gt_commands_tensor, _ = NodeCommand.split_tensor(gt_sequence_img)

    # For the prediction, we generate a sequence with the same commands but different geometry.
    pred_encoded = pred_glyph.encode(NodeCommand)
    if not pred_encoded:
        raise ValueError("Could not encode Pred glyph")
    pred_coords_unnorm = torch.from_numpy(
        pred_encoded[0][:, NodeCommand.command_width :]
    ).to(torch.float32)

    # 2. Manually create the `collated_batch` and `ModelResults` objects.
    gt_box = _get_glyph_bounds(gt_glyph)

    # Build normalized GT and Pred sequences in the same space used by collate_fn.
    gt_sequence_norm = NodeCommand.image_space_to_mask_space(gt_sequence_img, gt_box)

    pred_sequence_img = torch.cat([gt_commands_tensor, pred_coords_unnorm], dim=1)
    pred_sequence_norm = NodeCommand.image_space_to_mask_space(
        pred_sequence_img, gt_box
    )

    gt_commands_norm, gt_coords_norm = NodeCommand.split_tensor(gt_sequence_norm)
    _, pred_coords_norm = NodeCommand.split_tensor(pred_sequence_norm)

    # The alignment indices are based on the original NodeGlyph node order.
    x_alignments = [[[0, 3], [1, 2]]]  # For the single contour
    y_alignments = [[[0, 1], [3, 2]]]

    collated_batch: CollatedGlyphData = {
        "target_sequences": [gt_sequence_norm],
        "original_boxes": torch.stack([gt_box], dim=0),
        "x_aligned_point_indices": x_alignments,
        "y_aligned_point_indices": y_alignments,
        "images": torch.empty(1, 1, 1, 1),
        "gt_targets": [],
        "normalized_masks": torch.empty(1, 1, 1, 1),
        "contour_image_idx": torch.zeros(1, dtype=torch.long),
        "contour_filenames": ["synthetic"],
        "contour_characters": ["#"],
        "contour_numbers": torch.tensor([0], dtype=torch.long),
        "labels": torch.tensor([1], dtype=torch.int64),
    }

    # The ModelResults should contain the model's *output*, which does not
    # include the SOS token. We slice it off here.
    # Build standardized coordinate tensors exactly like training does.
    pred_commands = gt_commands_norm[1:]
    pred_command_indices = torch.argmax(pred_commands, dim=-1)
    pred_means, pred_stds = NodeCommand.get_stats_for_sequence(pred_command_indices)
    pred_coords_std = NodeCommand.standardize(
        pred_coords_norm[1:], pred_means, pred_stds
    )
    gt_coords_std = NodeCommand.standardize(gt_coords_norm[1:], pred_means, pred_stds)

    model_results = ModelResults(
        pred_commands=[pred_commands],
        pred_coords_norm=[pred_coords_norm[1:]],
        pred_coords_std=[pred_coords_std],
        gt_coords_std=[gt_coords_std],
        used_teacher_forcing=True,
        pred_categories=[0],
        lstm_outputs=[torch.zeros(pred_commands.shape[0], 128)],  # Dummy LSTM output
    )

    # 3. Run the main losses function
    loss_dict = losses(collated_batch, model_results, None, device)
    return {k: v.item() for k, v in loss_dict.items()}


def test_scaling_losses():
    """
    Tests that coordinate and area loss increase with scale, while alignment
    loss remains constant.
    """
    gt_glyph = _create_square_glyph()

    # Create a 1.1x scaled version
    pred_glyph_1_1x = _create_square_glyph()
    for node in pred_glyph_1_1x.contours[0].nodes:
        node.coordinates = (node.coordinates - 50) * 1.1 + 50

    # Create a 1.2x scaled version
    pred_glyph_1_2x = _create_square_glyph()
    for node in pred_glyph_1_2x.contours[0].nodes:
        node.coordinates = (node.coordinates - 50) * 1.2 + 50

    losses1 = _run_loss_test(gt_glyph, pred_glyph_1_1x)
    losses2 = _run_loss_test(gt_glyph, pred_glyph_1_2x)

    assert losses2["coord_loss"] > losses1["coord_loss"]
    assert losses2["signed_area_loss"] > losses1["signed_area_loss"]
    assert np.isclose(
        losses1["alignment_loss"], losses2["alignment_loss"]
    ), "Alignment loss should not change for uniform scaling"
    assert np.isclose(
        losses1["alignment_loss"], 0.0
    ), "Alignment loss should be zero for a scaled square"


def test_alignment_loss_shift():
    """
    Tests that alignment loss increases when a point is shifted out of
    alignment, while other losses behave as expected.
    """
    gt_glyph = _create_square_glyph()

    # Create a version with one point shifted
    pred_glyph_misaligned = _create_square_glyph()
    # This breaks the x-alignment for group [0,3] and y-alignment for [2,3]
    pred_glyph_misaligned.contours[0].nodes[3].coordinates += np.array([10, -10])

    # Get losses for a perfect prediction vs. the misaligned one
    losses_perfect = _run_loss_test(gt_glyph, gt_glyph)
    losses_misaligned = _run_loss_test(gt_glyph, pred_glyph_misaligned)

    # Assert that alignment loss is zero for the perfect case
    assert np.isclose(
        losses_perfect["alignment_loss"], 0.0
    ), "Alignment loss for identical glyphs should be zero"

    # Assert that alignment loss increased for the misaligned case
    assert (
        losses_misaligned["alignment_loss"] > losses_perfect["alignment_loss"]
    ), "Alignment loss should increase when a point is misaligned"

    # Sanity check: coordinate loss should also increase
    assert (
        losses_misaligned["coord_loss"] > losses_perfect["coord_loss"]
    ), "Coordinate loss should increase for a misaligned point"
