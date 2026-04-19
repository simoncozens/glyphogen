import torch
import numpy as np
import pytest
from torch.utils.data import DataLoader

from glyphogen.nodeglyph import NodeGlyph
from glyphogen.representations.nodecommand import NodeCommand
from glyphogen.dataset import get_hierarchical_data, collate_fn, font_files
from glyphogen.hyperparameters import ALPHABET
from glyphogen.svgglyph import SVGGlyph

# Get the index for the commands from the grammar
SOS_INDEX = NodeCommand.encode_command("SOS")
EOS_INDEX = NodeCommand.encode_command("EOS")
N_INDEX = NodeCommand.encode_command("N")

# Set our own batch size
BATCH_SIZE = 16


@pytest.fixture(scope="module")
def dataset():
    # It's important to load the dataset once and reuse it for all tests
    test_dataset, train_dataset = get_hierarchical_data()
    return DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)


def test_svg_generation(dataset):
    """
    Tests that a batch from the dataset can be successfully decoded and
    converted into an SVG.
    """
    # Get one batch from the dataset
    batch = next(iter(dataset))
    if batch is None:
        pytest.skip("Could not get a batch from the dataset.")

    # Get the ground truth data for the first glyph in the batch
    first_glyph_targets = batch["gt_targets"][0]
    gt_contours = first_glyph_targets["gt_contours"]

    # We need the raw command sequences from each contour
    contour_sequences = [contour["sequence"] for contour in gt_contours]

    # Decode the sequences into a NodeGlyph object
    decoded_glyph = NodeGlyph.decode(contour_sequences, NodeCommand)

    # Generate an SVG from the NodeGlyph
    svg_string = SVGGlyph.from_node_glyph(decoded_glyph).to_svg_string()

    # The test just asserts that an SVG was produced without errors
    assert isinstance(svg_string, str)
    assert len(svg_string) > 0


def test_collate_fn_output_ranges(dataset):
    """
    Tests that the collate_fn produces outputs with expected value ranges.
    - Normalized masks should be in [0, 1]
    - Original boxes should be finite and positive
    - Target sequences should be properly formatted (commands one-hot, normalized coords)
    """
    batch = next(iter(dataset))
    if batch is None:
        pytest.skip("Could not get a batch from the dataset.")

    # Check normalized masks are in valid range
    normalized_masks = batch["normalized_masks"]
    assert normalized_masks.min() >= -0.1, "Masks have unexpectedly low values"
    assert normalized_masks.max() <= 1.1, "Masks have unexpectedly high values"

    # Check original boxes are finite and positive
    original_boxes = batch["original_boxes"]
    assert torch.isfinite(
        original_boxes
    ).all(), "Original boxes contain non-finite values"
    assert (original_boxes[:, 0] >= 0).all(), "Box x1 is negative"
    assert (original_boxes[:, 1] >= 0).all(), "Box y1 is negative"
    assert (original_boxes[:, 2] >= original_boxes[:, 0]).all(), "Box x2 < x1"
    assert (original_boxes[:, 3] >= original_boxes[:, 1]).all(), "Box y2 < y1"

    assert (original_boxes[:, 2] >= 3).all(), "Box x2 is tiny"
    assert (original_boxes[:, 3] >= 3).all(), "Box y2 is tiny"

    # Check target sequences have proper structure
    # Commands should be one-hot (max value 1.0, single max per sequence element)
    offenders = []
    for seq_idx, seq in enumerate(batch["target_sequences"]):
        # Each sequence should be [seq_len, command_dim + coord_dim + possibly_heading]
        # The dimension should be at least 10 (command) + 2 (coord) = 12
        assert (
            seq.shape[1] >= 12
        ), f"Sequence {seq_idx} has unexpected shape: {seq.shape}"

        # First token should be SOS
        commands, coords = NodeCommand.split_tensor(seq)
        first_cmd_idx = torch.argmax(commands[0])
        assert first_cmd_idx == SOS_INDEX, f"First token is not SOS: {first_cmd_idx}"

        # Commands should have max value close to 1.0 (one-hot encoded)
        max_command_vals = torch.amax(commands, dim=-1)
        assert (
            max_command_vals > 0.9
        ).all(), f"Sequence {seq_idx} has non-one-hot commands"

        # Raw mask-space relative deltas can legitimately reach +/-2 when a step spans
        # one full bbox width/height, so min/max on mixed coordinate semantics is noisy.
        # Instead verify the robust invariant:
        #   x == image_space_to_mask_space(mask_space_to_image_space(x))
        # and that absolute on-curve positions stay inside the contour bbox.
        box = batch["original_boxes"][seq_idx]
        seq_img = NodeCommand.mask_space_to_image_space(seq, box)
        seq_cycle = NodeCommand.image_space_to_mask_space(seq_img, box)
        cycle_err = torch.abs(seq_cycle - seq)

        # Numerical noise should be tiny.
        cycle_linf = float(cycle_err.max().item())
        cycle_l1 = float(cycle_err.mean().item())

        # Convert to absolute positions and ensure they remain inside bbox.
        seq_abs = NodeCommand.unroll_relative_coordinates(seq_img)
        abs_commands, abs_coords = NodeCommand.split_tensor(seq_abs)
        abs_command_idx = torch.argmax(abs_commands, dim=-1)

        pos_cmd_names = ["M", "L", "LH", "LV", "N", "NS", "NH", "NV", "NCI", "NCO"]
        pos_cmd_indices = torch.tensor(
            [NodeCommand.encode_command(name) for name in pos_cmd_names],
            dtype=abs_command_idx.dtype,
            device=abs_command_idx.device,
        )
        pos_mask = (abs_command_idx[:, None] == pos_cmd_indices[None, :]).any(dim=1)

        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        tol = 1e-3
        pos_xy = abs_coords[pos_mask, 0:2]
        if pos_xy.numel() > 0:
            x_ok = (pos_xy[:, 0] >= x1 - tol) & (pos_xy[:, 0] <= x2 + tol)
            y_ok = (pos_xy[:, 1] >= y1 - tol) & (pos_xy[:, 1] <= y2 + tol)
            pos_ok = bool((x_ok & y_ok).all().item())
        else:
            pos_ok = True

        if cycle_linf > 1e-5 or not pos_ok:
            filename = batch["contour_filenames"][seq_idx]
            character = batch["contour_characters"][seq_idx]
            contour_number = int(batch["contour_numbers"][seq_idx].item())
            coord_min = float(coords.min().item())
            coord_max = float(coords.max().item())
            offenders.append(
                (
                    f"idx={seq_idx} file={filename} char={character!r} "
                    f"contour={contour_number} coord_min={coord_min:.4f} "
                    f"coord_max={coord_max:.4f} cycle_linf={cycle_linf:.3e} "
                    f"cycle_l1={cycle_l1:.3e} pos_ok={pos_ok} box={[x1, y1, x2, y2]}"
                )
            )

    if offenders:
        pytest.fail(
            "Found normalization inconsistencies (cycle error / bbox position violation).\n"
            + "\n".join(offenders)
        )
