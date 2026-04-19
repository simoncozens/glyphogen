import torch
import numpy as np
from glyphogen.representations.nodecommand import NodeCommand
from glyphogen.svgglyph import SVGGlyph
from glyphogen.nodeglyph import NodeGlyph
import pytest

# This test needs updating to our new command definitions


@pytest.mark.skip(reason="Needs updating to new command definitions")
def test_unroll_relative_coordinates():
    letter_h = "M 643 714 L 553 714 L 553 412 L 187 412 L 187 714 L 97 714 L 97 0 L 187 0 L 187 333 L 553 333 L 553 0 L 643 0 L 643 714 Z"
    contour = SVGGlyph.from_svg_string(letter_h)
    nodeglyph_orig = contour.to_node_glyph()
    real_coords = [x.coordinates for x in nodeglyph_orig.contours[0].nodes]
    # Encode it in relative form
    encoded = torch.from_numpy(nodeglyph_orig.encode(NodeCommand)[0])
    commands, coords = NodeCommand.split_tensor(encoded)
    decoded_commands = [NodeCommand.decode_command_one_hot(cmd) for cmd in commands]
    assert " ".join(decoded_commands) == "SOS M L L L L L L L L L L L L EOS"
    x_y = coords[:, 0:2]
    expected = [
        [0.0, 0.0],
        [643.0, 714.0],
        [0.0, 0.0],  # First command after M has no movement
        [-90.0, 0.0],
        [0.0, -302.0],
        [-366.0, 0.0],
        [0.0, 302.0],
        [-90.0, 0.0],
        [0.0, -714.0],
        [90.0, 0.0],
        [0, 333.0],
        [366.0, 0.0],
        [0.0, -333.0],
        [90.0, 0.0],
        [0.0, 0.0],
    ]
    for i in range(len(expected)):
        assert np.allclose(
            x_y[i].numpy(), expected[i]
        ), f"Mismatch at index {i}: got {x_y[i].numpy()}, expected {expected[i]}"
    # Now unroll relative coordinates to absolute
    unrolled = NodeCommand.unroll_relative_coordinates(encoded)
    _, unrolled_coords = NodeCommand.split_tensor(unrolled)
    unrolled_x_y = unrolled_coords[:, 0:2]
    # Add a 0,0 to the real_coords for the SOS and a 643,714 for the M
    real_coords = [[0.0, 0.0], [643.0, 714.0]] + real_coords
    for i in range(len(real_coords)):
        assert np.allclose(
            unrolled_x_y[i].numpy(), real_coords[i]
        ), f"Unrolled mismatch at index {i}: got {unrolled_x_y[i].numpy()}, expected {real_coords[i]}"

    # OK, try the decode
    nodeglyph_roundtrip = NodeGlyph.decode([encoded], NodeCommand)
    assert len(nodeglyph_orig.contours) == len(nodeglyph_roundtrip.contours)
    for i, orig_contour in enumerate(nodeglyph_orig.contours):
        reconstructed_contour = nodeglyph_roundtrip.contours[i]
        assert len(orig_contour.nodes) == len(reconstructed_contour.nodes)
        for j, orig_node in enumerate(orig_contour.nodes):
            reconstructed_node = reconstructed_contour.nodes[j]
            assert np.allclose(
                orig_node.coordinates, reconstructed_node.coordinates, atol=1e-5
            ), f"Node {j} in contour {i} of glyph '{char_to_test}' has mismatched coordinates after encode/decode"
