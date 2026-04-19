import math
from pathlib import Path
import numpy as np
import pytest
import torch
from glyphogen.representations.relativepolar import RelativePolarCommand
from glyphogen.glyph import SVGGlyph, Glyph
from glyphogen.nodeglyph import Node, NodeContour, NodeGlyph
from glyphogen.hyperparameters import ALPHABET
import uharfbuzz as hb


def test_relative_polar_line_roundtrip():
    """
    Tests that a simple square contour can be encoded into TangentNormalCommands
    and decoded back to the original contour, using L_LEFT.
    """
    # 1. Create a simple square NodeContour
    nodes = [
        Node(np.array([0, 0]), contour=None),
        Node(np.array([100, 0]), contour=None),
        Node(np.array([100, 100]), contour=None),
        Node(np.array([0, 100]), contour=None),
    ]
    original_contour = NodeContour(nodes)
    for node in nodes:
        node._contour = original_contour

    # 2. Encode it
    commands = RelativePolarCommand.emit(original_contour.nodes)

    # 3. Check for correct commands
    assert len(commands) == 3 + len(nodes)  # SOS, M, nodes, EOS
    assert commands[0].command == "SOS"
    assert commands[1].command == "M"
    assert np.allclose(commands[1].coordinates, [0, 0])

    # Node 0: (0,0) - relative to itself
    assert commands[2].command == "L_POLAR"
    r_node0, cos_phi0, sin_phi0 = commands[2].coordinates
    assert np.isclose(r_node0, 0.0)
    assert np.isclose(cos_phi0, 1.0)
    assert np.isclose(sin_phi0, 0.0)

    # Node 1: (100,0) - straight line from (0,0)
    assert commands[3].command == "L_POLAR"
    r_node1, cos_phi1, sin_phi1 = commands[3].coordinates
    assert np.isclose(r_node1, 100)
    assert np.isclose(cos_phi1, 1.0)
    assert np.isclose(sin_phi1, 0.0)

    # Node 2: (100,100) - left turn from (100,0)
    assert commands[4].command == "L_LEFT_90"
    r_node2 = commands[4].coordinates[0]
    assert np.isclose(r_node2, 100)

    # Node 3: (0,100) - left turn from (100,100)
    assert commands[5].command == "L_LEFT_90"
    r_node3 = commands[5].coordinates[0]
    assert np.isclose(r_node3, 100)

    assert commands[6].command == "EOS"

    # 4. Decode it
    decoded_contour = RelativePolarCommand.contour_from_commands(commands)

    # 5. Compare
    assert len(original_contour.nodes) == len(
        decoded_contour.nodes
    ), f"Expected {len(original_contour.nodes)} nodes, got {len(decoded_contour.nodes)}"

    for i, original_node in enumerate(original_contour.nodes):
        decoded_node = decoded_contour.nodes[i]
        assert np.allclose(
            original_node.coordinates, decoded_node.coordinates, atol=1e-4
        ), f"Node {i} mismatch: Original {original_node.coordinates}, Decoded {decoded_node.coordinates}"

    print("\n--- RelativePolarCommand line roundtrip test passed! ---")


def test_relative_polar_z_shape_roundtrip():
    """
    Tests that a Z-shaped contour can be encoded and decoded correctly,
    including diagonal lines with non-zero phi.
    """
    nodes = [
        Node(np.array([0, 100]), contour=None),
        Node(np.array([100, 100]), contour=None),
        Node(np.array([0, 0]), contour=None),
        Node(np.array([100, 0]), contour=None),
    ]
    original_contour = NodeContour(nodes)
    for node in nodes:
        node._contour = original_contour

    commands = RelativePolarCommand.emit(original_contour.nodes)
    assert len(commands) == 3 + len(nodes)  # SOS, M, nodes, EOS

    # Expected commands: SOS, M, then 4 node commands, then EOS
    assert commands[0].command == "SOS"
    assert commands[1].command == "M"
    assert np.allclose(commands[1].coordinates, [0, 100])

    # Node 0: (0,100) - relative to itself
    assert commands[2].command == "L_POLAR"
    r, cos_phi, sin_phi = commands[2].coordinates
    assert np.isclose(r, 0.0)
    assert np.isclose(cos_phi, 1.0)
    assert np.isclose(sin_phi, 0.0)

    # Node 1: (100,100) - straight right from (0,100)
    assert commands[3].command == "L_POLAR"
    r, cos_phi, sin_phi = commands[3].coordinates
    assert np.isclose(r, 100)
    assert np.isclose(cos_phi, 1.0)
    assert np.isclose(sin_phi, 0.0)

    # Node 2: (0,0) - diagonal down-left from (100,100)
    assert commands[4].command == "L_POLAR"
    r, cos_phi, sin_phi = commands[4].coordinates
    phi = -np.pi * 3 / 4
    assert np.isclose(r, np.sqrt(100**2 + 100**2))
    assert np.isclose(cos_phi, np.cos(phi))
    assert np.isclose(sin_phi, np.sin(phi))

    # Node 3: (100,0) - straight right from (0,0)
    assert commands[5].command == "L_POLAR"
    r, cos_phi, sin_phi = commands[5].coordinates
    phi = np.pi * 3 / 4
    assert np.isclose(r, 100)
    assert np.isclose(cos_phi, np.cos(phi))
    assert np.isclose(sin_phi, np.sin(phi))

    assert commands[6].command == "EOS"

    # Decode and compare
    decoded_contour = RelativePolarCommand.contour_from_commands(commands)
    assert len(original_contour.nodes) == len(decoded_contour.nodes)
    for i, original_node in enumerate(original_contour.nodes):
        decoded_node = decoded_contour.nodes[i]
        assert np.allclose(
            original_node.coordinates, decoded_node.coordinates, atol=1e-4
        ), f"Node {i} mismatch: Original {original_node.coordinates}, Decoded {decoded_node.coordinates}"

    # Unroll and verify absolute coordinates
    command_tensors = []
    coord_tensors = []
    max_coords = RelativePolarCommand.coordinate_width
    for cmd in commands:
        command_tensors.append(RelativePolarCommand.encode_command_one_hot(cmd.command))
        padded_coords = cmd.coordinates + [0] * (max_coords - len(cmd.coordinates))
        coord_tensors.append(torch.tensor(padded_coords, dtype=torch.float32))

    sequence_tensor = torch.cat(
        [torch.stack(command_tensors), torch.stack(coord_tensors)], dim=1
    )
    unrolled_sequence = RelativePolarCommand.unroll_relative_coordinates(
        sequence_tensor
    )
    _, abs_coords = RelativePolarCommand.split_tensor(unrolled_sequence)

    assert np.allclose(abs_coords[1, 0:2], [0, 100])  # M
    assert np.allclose(abs_coords[2, 0:2], [0, 100])  # Node 0
    assert np.allclose(abs_coords[3, 0:2], [100, 100])  # Node 1
    assert np.allclose(abs_coords[4, 0:2], [0, 0])  # Node 2
    assert np.allclose(abs_coords[5, 0:2], [100, 0])  # Node 3
    # For a closed contour, the last node's coordinates should match the first node's coordinates
    assert np.allclose(abs_coords[5, 0:2], original_contour.nodes[3].coordinates)

    print("\n--- RelativePolarCommand Z-shape roundtrip test passed! ---")


def test_relative_polar_real_z_glyph_roundtrip():
    """
    Tests the full round-trip process for a real Z-shaped glyph using
    RelativePolarCommand.
    """
    svg_string = "M 34 0 L 584 0 L 584 96 L 192 96 L 572 638 L 572 690 L 49 690 L 49 594 L 414 594 L 34 52 L 34 0 Z"
    svg_glyph = SVGGlyph.from_svg_string(svg_string)
    nodeglyph_orig = svg_glyph.to_node_glyph()

    # 2a. NodeGlyph -> List[List[RelativePolarCommand]]
    contours_commands = nodeglyph_orig.command_lists(RelativePolarCommand)
    assert len(contours_commands) == 1
    commands = contours_commands[0]
    assert len(commands) == 3 + len(
        nodeglyph_orig.contours[0].nodes
    )  # SOS, M, nodes, EOS
    assert len(commands) == 3 + 10  # Hand counted

    # Manually trace expected commands and coordinates
    # Initial f_hat = [1,0]
    # M 34 0
    assert commands[0].command == "SOS"
    assert commands[1].command == "M"
    assert np.allclose(commands[1].coordinates, [34, 0])

    # Node 0: (34,0) - relative to itself
    assert commands[2].command == "L_POLAR"
    assert np.isclose(commands[2].coordinates[0], 0)
    assert np.isclose(commands[2].coordinates[1], 1.0)
    assert np.isclose(commands[2].coordinates[2], 0.0)

    # Node 1: (584,0) (L 550 0) - across the baseline
    # delta_pos = [550, 0], r = 550, phi = 0 (relative to f_hat=[1,0])
    assert commands[3].command == "L_POLAR"
    assert np.isclose(commands[3].coordinates[0], 550)
    assert np.isclose(commands[3].coordinates[1], 1.0)
    assert np.isclose(commands[3].coordinates[2], 0.0)

    # Node 2: (584,96) (L 0 96) - turn left and head upwards
    # f_hat is still [1,0]
    # delta_pos = [0, 96], r = 96, phi = pi/2 (relative to f_hat=[1,0])
    assert commands[4].command == "L_LEFT_90"
    assert np.isclose(commands[4].coordinates[0], 96)

    # Node 3: (192,96) (L -392 0) - turn left and head left
    # f_hat is now [0,1]
    # delta_pos = [-392, 0], r = 392, phi = pi/2 (relative to f_hat=[0,1])
    assert commands[5].command == "L_LEFT_90"
    assert np.isclose(commands[5].coordinates[0], 392)

    # Node 4: (572,638) (L 380 542) - up the outside diagonal
    # f_hat is now [-1,0]
    # delta_pos = [380, 542], r = sqrt(380^2 + 542^2) = 661.09
    # r_hat = [0,-1]
    # phi = arctan2(dot([380,542], [0,-1]), dot([380,542], [-1,0])) = arctan2(-542, -380) = -2.19 rad
    assert commands[6].command == "L_POLAR"
    phi = np.arctan2(-542, -380)
    assert np.isclose(commands[6].coordinates[0], np.sqrt(380**2 + 542**2))
    assert np.isclose(commands[6].coordinates[1], np.cos(phi))
    assert np.isclose(commands[6].coordinates[2], np.sin(phi))

    # Node 5: (572,690) (L 0 52) - straight up to the top
    assert commands[7].command == "L_POLAR"

    # Node 6: (49,690) (L -523 0) - straight left along the top
    assert commands[8].command == "L_LEFT_90"

    # Node 7: (49,594) (L 0 -96) - down top left corner
    assert commands[9].command == "L_LEFT_90"

    # Node 8: (414,594) (L 365 0) - straight right towards bend
    assert commands[10].command == "L_LEFT_90"

    # Node 9: (34,52) (L -380 -542) - down the inside diagonal
    assert commands[11].command == "L_POLAR"

    # We're done, automatically close the path
    assert commands[12].command == "EOS"

    # 3a. NodeGlyph -> encoded sequence
    encoded_sequences = nodeglyph_orig.encode(RelativePolarCommand)

    # This can happen for blank glyphs like 'space'
    if encoded_sequences is None:
        assert svg_glyph.to_svg_string() == ""
        return

    # 3b. Encoded sequence -> NodeGlyph
    nodeglyph_roundtrip = NodeGlyph.decode(encoded_sequences, RelativePolarCommand)

    # 4. Compare original and round-tripped NodeGlyph objects
    assert len(nodeglyph_orig.contours) == len(nodeglyph_roundtrip.contours)
    for i, orig_contour in enumerate(nodeglyph_orig.contours):
        reconstructed_contour = nodeglyph_roundtrip.contours[i]
        assert len(orig_contour.nodes) == len(reconstructed_contour.nodes)
        for j, orig_node in enumerate(orig_contour.nodes):
            reconstructed_node = reconstructed_contour.nodes[j]
            assert np.allclose(
                orig_node.coordinates, reconstructed_node.coordinates, atol=1e-5
            ), f"Node {j} in contour {i} of glyph has mismatched coordinates"
            # For now, we don't have handles, so we don't check them.

    print("\n--- RelativePolarCommand real Z-glyph roundtrip test passed! ---")

    # 3a. NodeGlyph -> encoded sequence
    encoded_sequences = nodeglyph_orig.encode(RelativePolarCommand)

    # This can happen for blank glyphs like 'space'
    if encoded_sequences is None:
        assert svg_glyph.to_svg_string() == ""
        return

    # 3b. Encoded sequence -> NodeGlyph
    nodeglyph_roundtrip = NodeGlyph.decode(encoded_sequences, RelativePolarCommand)

    # 4. Compare original and round-tripped NodeGlyph objects
    assert len(nodeglyph_orig.contours) == len(nodeglyph_roundtrip.contours)
    for i, orig_contour in enumerate(nodeglyph_orig.contours):
        reconstructed_contour = nodeglyph_roundtrip.contours[i]
        assert len(orig_contour.nodes) == len(reconstructed_contour.nodes)
        for j, orig_node in enumerate(orig_contour.nodes):
            reconstructed_node = reconstructed_contour.nodes[j]
            assert np.allclose(
                orig_node.coordinates, reconstructed_node.coordinates, atol=1e-5
            ), f"Node {j} in contour {i} of glyph has mismatched coordinates"
            # For now, we don't have handles, so we don't check them.

    print("\n--- RelativePolarCommand real Z-glyph roundtrip test passed! ---")


def test_relative_polar_curve_roundtrip():
    """
    Tests that a 'D'-shaped contour with a curve can be encoded and decoded.
    """
    # 1. Create a 'D' shape. It has two nodes: top-left and bottom-left.
    # The segment from node 0 to 1 is a curve.
    # The segment from node 1 to 0 is a straight line.
    h_len = 100 * 2 / 3  # Approximation for a nice curve

    nodes = [
        Node(
            np.array([0, 100]),
            out_handle=np.array([h_len, 100]),
            in_handle=None,
            contour=None,
        ),
        Node(
            np.array([0, 0]),
            out_handle=None,
            in_handle=np.array([h_len, 0]),
            contour=None,
        ),
    ]
    original_contour = NodeContour(nodes)
    for node in nodes:
        node._contour = original_contour

    # 2. Encode it
    commands = RelativePolarCommand.emit(original_contour.nodes)
    assert len(commands) == 3 + len(original_contour.nodes)  # SOS, M, nodes, EOS

    # 3. Check for correct commands
    assert commands[0].command == "SOS"
    assert commands[1].command == "M"
    assert np.allclose(commands[1].coordinates, [0, 100])

    # Node 0: (0,100) - relative to itself
    assert commands[2].command == "N_POLAR_OUT"
    (
        r_node0,
        cos_phi0,
        sin_phi0,
        out_len0,
        out_cos_phi0,
        out_sin_phi0,
    ) = commands[2].coordinates
    assert np.isclose(r_node0, 0.0)
    assert np.isclose(cos_phi0, 1.0)
    assert np.isclose(sin_phi0, 0.0)
    assert np.isclose(out_len0, h_len)
    assert np.isclose(out_cos_phi0, 1.0)
    assert np.isclose(out_sin_phi0, 0.0)

    # Node 1: (0,0) - curve from (0,100)
    assert commands[3].command == "N_POLAR_IN"
    r_node1, cos_phi1, sin_phi1, in_len1, in_cos_phi1, in_sin_phi1 = commands[
        3
    ].coordinates
    phi1 = -np.pi / 2
    assert np.isclose(r_node1, 100)  # distance from (0,100) to (0,0)
    assert np.isclose(cos_phi1, np.cos(phi1))
    assert np.isclose(sin_phi1, np.sin(phi1))
    assert np.isclose(in_len1, h_len)
    # in_handle is [h_len, 0] - [0,0] = [h_len, 0]. f_hat is [0,-1]. r_hat is [1,0].
    # in_phi is arctan2(dot([h,0],[1,0]), dot([h,0],[0,-1])) = arctan2(h,0) = 0
    assert np.isclose(in_cos_phi1, 1.0)
    assert np.isclose(in_sin_phi1, 0.0)

    assert commands[4].command == "EOS"

    # 4. Decode it
    decoded_contour = RelativePolarCommand.contour_from_commands(commands)

    # 5. Compare
    assert len(original_contour.nodes) == len(
        decoded_contour.nodes
    ), f"Expected {len(original_contour.nodes)} nodes, got {len(decoded_contour.nodes)}"

    for i, original_node in enumerate(original_contour.nodes):
        decoded_node = decoded_contour.nodes[i]
        assert np.allclose(
            original_node.coordinates, decoded_node.coordinates, atol=1e-4
        ), f"Node {i} coordinate mismatch: Original {original_node.coordinates}, Decoded {decoded_node.coordinates}"

        if original_node.out_handle is not None:
            assert (
                decoded_node.out_handle is not None
            ), f"Node {i} decoded out-handle is None"
            assert np.allclose(
                original_node.out_handle, decoded_node.out_handle, atol=1e-4
            ), f"Node {i} out-handle mismatch: Original {original_node.out_handle}, Decoded {decoded_node.out_handle}"

        if original_node.in_handle is not None:
            assert (
                decoded_node.in_handle is not None
            ), f"Node {i} decoded in-handle is None"
            assert np.allclose(
                original_node.in_handle, decoded_node.in_handle, atol=1e-4
            ), f"Node {i} in-handle mismatch: Original {original_node.in_handle}, Decoded {decoded_node.in_handle}"

    print("\n--- RelativePolarCommand curve roundtrip test passed! ---")


#     nodes = [
#         Node(
#             np.array([0, 100]),
#             out_handle=np.array([h_len, 100]),
#             in_handle=None,
#             contour=None,
#         ),
#         Node(
#             np.array([0, 0]),
#             out_handle=None,
#             in_handle=np.array([h_len, 0]),
#             contour=None,
#         ),
#     ]
#     original_contour = NodeContour(nodes)
#     for node in nodes:
#         node._contour = original_contour

#     # 2. Encode it
#     commands = RelativePolarCommand.emit(original_contour.nodes)

#     # 3. Decode it
#     decoded_contour = RelativePolarCommand.contour_from_commands(commands)

#     # 4. Compare
#     # The decoded contour should have the same number of nodes
#     assert len(original_contour.nodes) == len(
#         decoded_contour.nodes
#     ), f"Expected {len(original_contour.nodes)} nodes, got {len(decoded_contour.nodes)}"

#     for i, original_node in enumerate(original_contour.nodes):
#         decoded_node = decoded_contour.nodes[i]
#         assert np.allclose(
#             original_node.coordinates, decoded_node.coordinates, atol=1e-4
#         ), f"Node {i} coordinate mismatch: Original {original_node.coordinates}, Decoded {decoded_node.coordinates}"

#         if original_node.out_handle is not None:
#             assert (
#                 decoded_node.out_handle is not None
#             ), f"Node {i} decoded out-handle is None"
#             assert np.allclose(
#                 original_node.out_handle, decoded_node.out_handle, atol=1e-4
#             ), f"Node {i} out-handle mismatch: Original {original_node.out_handle}, Decoded {decoded_node.out_handle}"

#         if original_node.in_handle is not None:
#             assert (
#                 decoded_node.in_handle is not None
#             ), f"Node {i} decoded in-handle is None"
#             assert np.allclose(
#                 original_node.in_handle, decoded_node.in_handle, atol=1e-4
#             ), f"Node {i} in-handle mismatch: Original {original_node.in_handle}, Decoded {decoded_node.in_handle}"

#     print("\n--- RelativePolarCommand curve roundtrip test passed! ---")


def test_relative_polar_smooth_roundtrip():
    """
    Tests that a contour with a smooth connection is correctly
    encoded and decoded with N_SMOOTH.
    """
    nodes = [
        Node(
            np.array([0, 0]),
            out_handle=np.array([50, 0]),
            contour=None,
        ),
        Node(
            np.array([100, 0]),
            in_handle=np.array([50, 0]),
            out_handle=np.array([150, 0]),
            contour=None,
        ),
        Node(
            np.array([200, 0]),
            in_handle=np.array([150, 0]),
            contour=None,
        ),
    ]
    original_contour = NodeContour(nodes)
    for node in nodes:
        node._contour = original_contour

    commands = RelativePolarCommand.emit(original_contour.nodes)

    # 3. Check for correct commands
    assert commands[0].command == "SOS"
    assert commands[1].command == "M"
    assert np.allclose(commands[1].coordinates, [0, 0])

    # Node 0: (0,0) - relative to itself
    assert commands[2].command == "N_POLAR_OUT"
    (
        r_node0,
        cos_phi0,
        sin_phi0,
        out_len0,
        out_cos_phi0,
        out_sin_phi0,
    ) = commands[2].coordinates
    assert np.isclose(r_node0, 0.0)
    assert np.isclose(cos_phi0, 1.0)
    assert np.isclose(sin_phi0, 0.0)
    assert np.isclose(out_len0, 50)
    assert np.isclose(out_cos_phi0, 1.0)
    assert np.isclose(out_sin_phi0, 0.0)

    # Node 1: (100,0) - smooth curve from (0,0)
    assert commands[3].command == "N_SMOOTH"
    (
        r_node1,
        cos_phi1,
        sin_phi1,
        out_cos_phi1,
        out_sin_phi1,
        in_len1,
        out_len1,
    ) = commands[3].coordinates
    assert np.isclose(r_node1, 100)
    assert np.isclose(cos_phi1, 1.0)
    assert np.isclose(sin_phi1, 0.0)
    assert np.isclose(out_cos_phi1, 1.0)
    assert np.isclose(out_sin_phi1, 0.0)
    assert np.isclose(in_len1, 50)
    assert np.isclose(out_len1, 50)

    # Node 2: (200,0) - smooth curve from (100,0)
    assert commands[4].command == "N_POLAR_IN"
    r_node2, cos_phi2, sin_phi2, in_len2, in_cos_phi2, in_sin_phi2 = commands[
        4
    ].coordinates
    assert np.isclose(r_node2, 100)
    assert np.isclose(cos_phi2, 1.0)
    assert np.isclose(sin_phi2, 0.0)
    assert np.isclose(in_len2, 50)
    # f_hat is [1,0], r_hat is [0,1]. in_handle is [150,0]-[200,0] = [-50,0]
    # in_phi = arctan2(dot([-50,0],[0,1]), dot([-50,0],[1,0])) = arctan2(0,-50) = pi
    assert np.isclose(in_cos_phi2, -1.0)
    assert np.isclose(in_sin_phi2, 0.0)

    assert commands[5].command == "EOS"

    # Decode and compare
    decoded_contour = RelativePolarCommand.contour_from_commands(commands)
    assert len(original_contour.nodes) == len(decoded_contour.nodes)
    for i, original_node in enumerate(original_contour.nodes):
        decoded_node = decoded_contour.nodes[i]
        assert np.allclose(
            original_node.coordinates, decoded_node.coordinates, atol=1e-4
        )
        if original_node.out_handle is not None:
            assert (
                decoded_node.out_handle is not None
            ), f"Node {i} decoded out-handle is None"
            assert np.allclose(
                original_node.out_handle, decoded_node.out_handle, atol=1e-4
            ), f"Node {i} out-handle mismatch: Original {original_node.out_handle}, Decoded {decoded_node.out_handle}"
        if original_node.in_handle is not None:
            assert (
                decoded_node.in_handle is not None
            ), f"Node {i} decoded in-handle is None"
            assert np.allclose(
                original_node.in_handle, decoded_node.in_handle, atol=1e-4
            ), f"Node {i} in-handle mismatch: Original {original_node.in_handle}, Decoded {decoded_node.in_handle}"
    print("\n--- RelativePolarCommand smooth roundtrip test passed! ---")


def test_relative_polar_space_conversion_roundtrip():
    """
    Tests that the coordinate space conversion methods can roundtrip from
    image space to mask space and back.
    """
    # 1. Create a sequence of commands in image space
    commands = [
        RelativePolarCommand("SOS", []),
        RelativePolarCommand("M", [100, 200]),
        RelativePolarCommand(
            "N_POLAR",
            [
                50,
                np.cos(0.2),
                np.sin(0.2),
                10,
                np.cos(-0.3),
                np.sin(-0.3),
                12,
                np.cos(0.4),
                np.sin(0.4),
            ],
        ),
        RelativePolarCommand("L_POLAR", [30, np.cos(0.1), np.sin(0.1)]),
        RelativePolarCommand("EOS", []),
    ]

    # 2. Convert to a tensor
    command_tensors = []
    coord_tensors = []
    max_coords = RelativePolarCommand.coordinate_width
    for cmd in commands:
        command_tensors.append(RelativePolarCommand.encode_command_one_hot(cmd.command))

        # Pad coordinates to max_coords
        padded_coords = cmd.coordinates + [0] * (max_coords - len(cmd.coordinates))
        coord_tensors.append(torch.tensor(padded_coords, dtype=torch.float32))

    command_tensor = torch.stack(command_tensors)
    coord_tensor = torch.stack(coord_tensors)
    sequence_tensor = torch.cat([command_tensor, coord_tensor], dim=1)

    # 3. Define a bounding box
    box = torch.tensor([50, 50, 250, 350], dtype=torch.float32)  # x1, y1, x2, y2

    # 4. Normalize to mask space
    normalized_sequence = RelativePolarCommand.image_space_to_mask_space(
        sequence_tensor, box
    )

    # 5. Denormalize back to image space
    denormalized_sequence = RelativePolarCommand.mask_space_to_image_space(
        normalized_sequence, box
    )

    # 6. Compare
    assert torch.allclose(
        sequence_tensor, denormalized_sequence, atol=1e-4
    ), "Roundtrip failed: tensors do not match"

    print("\n--- RelativePolarCommand space conversion roundtrip test passed! ---")


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

    nodeglyph_orig.normalize()
    normalized_svg = SVGGlyph.from_node_glyph(nodeglyph_orig).to_svg_string()

    # 2a. NodeGlyph -> List[List[NodeCommand]]
    contours_commands = nodeglyph_orig.command_lists(RelativePolarCommand)

    # 2b. List[List[NodeCommand]] -> NodeGlyph
    nodeglyph_reconstructed = NodeGlyph(
        [
            RelativePolarCommand.contour_from_commands(contour, tolerant=False)
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
    encoded_sequences = nodeglyph_orig.encode(RelativePolarCommand)

    # This can happen for blank glyphs like 'space'
    if encoded_sequences is None:
        assert svg_orig_str == ""
        return

    # 3b. Encoded sequence -> NodeGlyph
    nodeglyph_roundtrip = NodeGlyph.decode(encoded_sequences, RelativePolarCommand)

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


def test_check_left90s():
    svg_string = "M 51 101 L 51 121 L 51 156 Z"
    svg_glyph = SVGGlyph.from_svg_string(svg_string)
    nodeglyph_orig = svg_glyph.to_node_glyph()
    contours_commands = nodeglyph_orig.command_lists(RelativePolarCommand)
    commands = [x.command for x in contours_commands[0]]
    assert commands == ["SOS", "M", "L_POLAR", "L_LEFT_90", "L_POLAR", "EOS"]

    # Forward, left turn, left turn, curve
    svg_string = "M 275 245 L 275 101 L 201 101 L 201 228 C 201 263 196 281 275 245"
    svg_glyph = SVGGlyph.from_svg_string(svg_string)
    nodeglyph_orig = svg_glyph.to_node_glyph()
    contours_commands = nodeglyph_orig.command_lists(RelativePolarCommand)
    commands = [x.command for x in contours_commands[0]]
    assert commands == [
        "SOS",
        "M",
        "N_POLAR_IN",  # Because of the curve at the end
        "L_RIGHT_90",
        "L_RIGHT_90",
        "N_POLAR_OUT",
        "EOS",
    ]
