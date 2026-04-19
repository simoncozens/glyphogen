from glyphogen.nodeglyph import NodeGlyph
import torch
import numpy as np
from glyphogen.glyph import SVGGlyph
from glyphogen.representations.nodecommand import (
    NodeCommand,
)
import uharfbuzz as hb


def test_coordinate_transforms():
    """
    Tests that the coordinate space transformations are perfect inverses.
    """
    box = torch.tensor([100.0, 100.0, 300.0, 400.0])  # x1, y1, x2, y2

    # Create a sample sequence tensor
    # [M, L, N]
    commands = torch.zeros(3, NodeCommand.command_width)
    commands[0, NodeCommand.encode_command("M")] = 1.0
    commands[1, NodeCommand.encode_command("L")] = 1.0
    commands[2, NodeCommand.encode_command("N")] = 1.0

    coords = torch.tensor(
        [
            [150, 250, 0, 0, 0, 0],  # M (absolute)
            [10, 20, 0, 0, 0, 0],  # L (relative)
            [-5, 15, 5, 5, -5, -5],  # N (relative)
        ],
        dtype=torch.float32,
    )

    # Pad to full sequence width
    padding = torch.zeros(3, NodeCommand.coordinate_width - coords.shape[1])
    coords = torch.cat([coords, padding], dim=1)

    sequence = torch.cat([commands, coords], dim=1)

    # Round trip
    sequence_mask_space = NodeCommand.image_space_to_mask_space(sequence, box)
    assert (
        sequence_mask_space[0, NodeCommand.command_width] == -0.5
    )  # One-quarter of the way across box, which is now -1 to +1
    sequence_roundtrip = NodeCommand.mask_space_to_image_space(sequence_mask_space, box)

    assert torch.allclose(
        sequence[:, NodeCommand.command_width :],
        sequence_roundtrip[:, NodeCommand.command_width :],
        atol=1e-6,
    ), f"Coordinate space round-trip failed:\n{sequence[:, NodeCommand.command_width :]}\n{sequence_roundtrip[:, NodeCommand.command_width :]}"


import pytest
from pathlib import Path
from glyphogen.hyperparameters import ALPHABET
from glyphogen.glyph import Glyph

# ... (existing test_coordinate_transforms and test_nodeglyph_encoding_decoding) ...


@pytest.mark.parametrize("char_to_test", list(ALPHABET))
def test_real_glyph_roundtrip(char_to_test):
    """
    Tests the full round-trip process on a real glyph from a font file.
    """
    font_path = Path("NotoSans[wdth,wght].ttf")
    if not font_path.exists():
        pytest.skip("NotoSans[wdth,wght].ttf not found, skipping real glyph test.")
    if char_to_test == "l":
        pytest.skip("Skipping 'l' as it has a bad contour construction.")

    # 1. Load original glyph and convert to NodeGlyph
    hb_face = hb.Face(hb.Blob.from_file_path(font_path))
    hb_font = hb.Font(hb_face)
    gid = hb_font.get_nominal_glyph(ord(char_to_test))
    glyph = Glyph(font_path, gid, hb_face, {})
    svg_glyph_orig = None
    try:
        svg_glyph_orig = glyph.vectorize()
    except NotImplementedError:
        pytest.skip(
            f"Skipping glyph '{char_to_test}' due to fontTools NotImplementedError."
        )
    if not svg_glyph_orig:
        return  # Skip empty glyphs

    # Get the original SVG string for comparison
    svg_orig_str = svg_glyph_orig.to_svg_string()

    # Now convert to NodeGlyph
    nodeglyph_orig = svg_glyph_orig.to_node_glyph()
    # print("Debug node glyph: ", nodeglyph_orig.to_debug_string())
    # and back again
    svg_glyph_roundtrip = SVGGlyph.from_node_glyph(nodeglyph_orig)
    svg_roundtrip_str = svg_glyph_roundtrip.to_svg_string()

    # Compare original and round-tripped SVG strings
    if svg_orig_str != svg_roundtrip_str:
        print(f"Original SVG:\n{svg_orig_str}")
        print(f"Roundtrip SVG:\n{svg_roundtrip_str}")
    assert svg_orig_str == svg_roundtrip_str

    # 2a. NodeGlyph -> List[List[NodeCommand]]
    contours_commands = nodeglyph_orig.command_lists(NodeCommand)

    # 2b. List[List[NodeCommand]] -> NodeGlyph
    nodeglyph_reconstructed = NodeGlyph(
        [
            NodeCommand.contour_from_commands(contour, tolerant=False)
            for contour in contours_commands
        ],
        nodeglyph_orig.origin,
    )

    # Assert that the on-curve points are preserved.
    # We don't assert full handle equality, because the NS encoding "perfects"
    # the handles of almost-smooth nodes, which is a desirable cleanup.
    assert len(nodeglyph_orig.contours) == len(nodeglyph_reconstructed.contours)
    for i, orig_contour in enumerate(nodeglyph_orig.contours):
        reconstructed_contour = nodeglyph_reconstructed.contours[i]
        assert len(orig_contour.nodes) == len(reconstructed_contour.nodes)
        for j, orig_node in enumerate(orig_contour.nodes):
            reconstructed_node = reconstructed_contour.nodes[j]
            assert np.allclose(
                orig_node.coordinates, reconstructed_node.coordinates, atol=1e-5
            ), f"Node {j} in contour {i} of glyph '{char_to_test}' has mismatched coordinates"
            # Optional: check that smoothness is preserved
            if orig_node.is_smooth:
                assert (
                    reconstructed_node.is_smooth
                ), f"Node {j} in contour {i} of glyph '{char_to_test}' lost its smoothness"

    # 3a. NodeGlyph -> encoded sequence
    encoded_sequences = nodeglyph_orig.encode(NodeCommand)

    # This can happen for blank glyphs like 'space'
    if encoded_sequences is None:
        assert svg_orig_str == ""
        return

    # 3b. Encoded sequence -> NodeGlyph
    nodeglyph_roundtrip = NodeGlyph.decode(encoded_sequences, NodeCommand)

    # 4. Compare original and round-tripped NodeGlyph objects
    # We assert on-curve coordinate preservation, not perfect handle equality.
    assert len(nodeglyph_orig.contours) == len(nodeglyph_roundtrip.contours)
    for i, orig_contour in enumerate(nodeglyph_orig.contours):
        reconstructed_contour = nodeglyph_roundtrip.contours[i]
        assert len(orig_contour.nodes) == len(reconstructed_contour.nodes)
        for j, orig_node in enumerate(orig_contour.nodes):
            reconstructed_node = reconstructed_contour.nodes[j]
            assert np.allclose(
                orig_node.coordinates, reconstructed_node.coordinates, atol=1e-5
            ), f"Node {j} in contour {i} of glyph '{char_to_test}' has mismatched coordinates after encode/decode"


def test_nodeglyph_decoding():
    # This test ensures that the decoding process (tensor -> NodeGlyph) works correctly.
    # The command sequence is defined semantically, and tensors are generated from it,
    # making the test robust to changes in the command vocabulary.
    command_strs = [
        "M",
        "L",
        "N",
        "NCO",
        "NCI",
        "NV",
        "NCO",
        "EOS",
        "NCO",
        "NCI",
        "NCI",
        "NCO",
        "EOS",
    ]
    commands_tensor = torch.stack(
        [NodeCommand.encode_command_one_hot(s) for s in command_strs]
    )

    coords_list = [
        [66, 182, 49, 48, 50, 48],
        [9, 8, 39, 42, 46, 46],
        [8, 8, 37, 37, 42, 43],
        [8, 10, 39, 39, 47, 45],
        [8, 10, 37, 38, 44, 47],
        [9, 9, 37, 36, 42, 46],
        [8, 10, 38, 38, 46, 45],
        [9, 8, 37, 34, 40, 42],  # Coords for EOS are ignored
        [8, 10, 37, 37, 45, 44],
        [8, 10, 35, 37, 42, 45],
        [8, 9, 34, 34, 40, 46],
        [8, 10, 35, 36, 45, 45],
        [10, 9, 34, 33, 41, 42],  # Coords for EOS are ignored
    ]
    # Ensure coordinates are padded to the correct width
    padded_coords_list = [
        row + [0.0] * (NodeCommand.coordinate_width - len(row)) for row in coords_list
    ]
    coords_tensor = torch.tensor(padded_coords_list, dtype=torch.float32)

    glyph = NodeGlyph.decode(
        [torch.cat([commands_tensor, coords_tensor], dim=-1)],
        NodeCommand,
    )

    # The first command is M (absolute), establishing the initial point.
    # The second command is L (relative), which creates the first node in the contour.
    # The assertion checks that this first node's absolute coordinates are correct.
    assert np.allclose(
        glyph.contours[0].nodes[0].coordinates, np.array([66 + 9, 182 + 8])
    )
