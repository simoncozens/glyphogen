from pathlib import Path
import pytest
from glyphogen.glyph import (
    Glyph,
    SVGGlyph,
    SVGCommand,
)
from glyphogen.nodeglyph import (
    NodeGlyph,
    NodeContour,
    Node,
)
from glyphogen.representations.nodecommand import (
    NodeCommand,
)
from glyphogen.hyperparameters import GEN_IMAGE_SIZE
import numpy as np
import torch
import uharfbuzz as hb


def test_glyph_extraction():
    """Test extracting and encoding a real glyph from a font."""
    font_path = Path("tests/data/NotoSans[wdth,wght].ttf")
    location = {"wght": 400.0}
    codepoint = ord("a")

    hb_face = hb.Face(hb.Blob.from_file_path(font_path))
    hb_font = hb.Font(hb_face)
    gid = hb_font.get_nominal_glyph(codepoint)
    glyph = Glyph(font_path, gid, hb_face, {})

    # Rasterize the glyph
    rasterized_glyph = glyph.rasterize(GEN_IMAGE_SIZE[0])

    # Check the size of the rasterized numpy array
    assert rasterized_glyph.shape == (*GEN_IMAGE_SIZE, 1)

    # Vectorize the glyph
    vectorized_glyph = glyph.vectorize()

    assert isinstance(vectorized_glyph, SVGGlyph)
    assert len(vectorized_glyph.commands) > 0
    assert vectorized_glyph.commands[0].command == "M"
    node_glyph = vectorized_glyph.to_node_glyph()
    assert isinstance(node_glyph, NodeGlyph)

    # Encode the glyph - should now return a list of sequences
    contour_sequences = node_glyph.encode(NodeCommand)
    if contour_sequences is None:
        pytest.skip("Glyph 'a' in NotoSans is too complex, skipping remainder of test.")

    assert isinstance(contour_sequences, list), "encode() should return a list"
    assert len(contour_sequences) > 0, "Should have at least one contour"

    # Each contour sequence should have proper shape
    for seq in contour_sequences:
        assert seq.shape[1] == NodeCommand.command_width + NodeCommand.coordinate_width
        # First should be SOS
        assert np.argmax(
            seq[0, : NodeCommand.command_width]
        ) == NodeCommand.encode_command("SOS")
        # Last should be EOS
        assert np.argmax(
            seq[-1, : NodeCommand.command_width]
        ) == NodeCommand.encode_command("EOS")

    # Test round-trip encoding/decoding
    decoded_glyph = NodeGlyph.decode(
        [torch.from_numpy(s) for s in contour_sequences], NodeCommand
    )

    assert len(decoded_glyph.contours) == len(node_glyph.contours)

    # Test that we can convert to SVG without errors
    svg_glyph = SVGGlyph.from_node_glyph(decoded_glyph)
    assert isinstance(svg_glyph, SVGGlyph)
    svg_string = svg_glyph.to_svg_string()
    assert len(svg_string) > 0


def test_glyph_simplify():
    # Test one with cross-contour overlaps
    font_path = Path("tests/data/Roboto[wdth,wght].ttf")
    location = {"wght": 400.0}
    codepoint = ord("A")

    hb_face = hb.Face(hb.Blob.from_file_path(font_path))
    hb_font = hb.Font(hb_face)
    gid = hb_font.get_nominal_glyph(codepoint)
    glyph = Glyph(font_path, gid, hb_face, {})

    vectorized_glyph = glyph.vectorize(remove_overlaps=False)
    assert len(vectorized_glyph.to_node_glyph().contours) == 3

    vectorized_glyph = glyph.vectorize(remove_overlaps=True)
    # Should now have two contours
    assert len(vectorized_glyph.to_node_glyph().contours) == 2


def test_node_glyph_normalization():
    # Contour 1: A square with starting point (20, 80)
    # Lowest Y/X point is (10, 70)
    contour1 = NodeContour([])
    nodes1_data = [
        (20, 80),
        (10, 80),
        (10, 70),
        (20, 70),
    ]
    nodes1 = [Node(coords, contour1) for coords in nodes1_data]
    contour1.nodes = nodes1

    # Contour 2: A square with starting point (60, 40)
    # Lowest Y/X point is (50, 30)
    contour2 = NodeContour([])
    nodes2_data = [
        (60, 40),
        (50, 40),
        (50, 30),
        (60, 30),
    ]
    nodes2 = [Node(coords, contour2) for coords in nodes2_data]
    contour2.nodes = nodes2

    # Create glyph with contours out of order
    # Contour1 starts at y=80, Contour2 starts at y=40
    node_glyph = NodeGlyph([contour1, contour2])

    # Normalize the glyph
    node_glyph.normalize()

    # After normalization:
    # 1. Contours should be sorted by their starting node's lowest (y, x)
    # 2. Each contour's node list should be rolled to start at its lowest (y, x) node

    # Contour2's lowest point is (50, 30) (y=30)
    # Contour1's lowest point is (10, 70) (y=70)
    # So, contour2 should now be first.

    assert len(node_glyph.contours) == 2

    # Check that the first contour is the original contour2, now normalized
    first_contour = node_glyph.contours[0]
    np.testing.assert_array_equal(first_contour.nodes[0].coordinates, [50, 30])
    # Check if the original nodes are present
    original_contour2_coords = {tuple(n.coordinates) for n in contour2.nodes}
    new_first_contour_coords = {tuple(n.coordinates) for n in first_contour.nodes}
    assert original_contour2_coords == new_first_contour_coords

    # Check that the second contour is the original contour1, now normalized
    second_contour = node_glyph.contours[1]
    np.testing.assert_array_equal(second_contour.nodes[0].coordinates, [10, 70])
    original_contour1_coords = {tuple(n.coordinates) for n in contour1.nodes}
    new_second_contour_coords = {tuple(n.coordinates) for n in second_contour.nodes}
    assert original_contour1_coords == new_second_contour_coords


def test_get_segmentation_data():
    # Case 1: Simple single contour glyph (a square)
    simple_glyph = SVGGlyph(
        [
            SVGCommand("M", [10, 10]),
            SVGCommand("L", [100, 10]),
            SVGCommand("L", [100, 100]),
            SVGCommand("L", [10, 100]),
            SVGCommand("Z", []),
        ]
    )
    seg_data_simple = simple_glyph.get_segmentation_data()
    assert len(seg_data_simple) == 1
    assert seg_data_simple[0]["label"] == 0  # Outer contour
    # Bounds will be in image space.
    expected_bounds = torch.tensor([[10, 10, 100, 100]]).numpy()
    np.testing.assert_allclose([seg_data_simple[0]["bbox"]], expected_bounds)

    # Case 2: Glyph with a hole (like an 'O')
    hole_glyph = SVGGlyph(
        [
            # Outer contour
            SVGCommand("M", [0, 0]),
            SVGCommand("L", [200, 0]),
            SVGCommand("L", [200, 200]),
            SVGCommand("L", [0, 200]),
            SVGCommand("Z", []),
            # Inner contour (hole)
            SVGCommand("M", [50, 50]),
            SVGCommand("L", [150, 50]),
            SVGCommand("L", [150, 150]),
            SVGCommand("L", [50, 150]),
            SVGCommand("Z", []),
        ]
    )
    seg_data_hole = hole_glyph.get_segmentation_data()
    assert len(seg_data_hole) == 2
    # Sort by bbox area (descending) to have a stable order for checking
    seg_data_hole.sort(
        key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
        reverse=True,
    )

    # Check outer contour
    assert seg_data_hole[0]["label"] == 0
    np.testing.assert_allclose(seg_data_hole[0]["bbox"], [0, 0, 200, 200])

    # Check inner contour (hole)
    assert seg_data_hole[1]["label"] == 1
    np.testing.assert_allclose(seg_data_hole[1]["bbox"], [50, 50, 150, 150])

    # Case 3: Glyph with two separate contours (like an 'i')
    multi_contour_glyph = SVGGlyph(
        [
            # Top contour (dot)
            SVGCommand("M", [40, 0]),
            SVGCommand("L", [60, 0]),
            SVGCommand("L", [60, 10]),
            SVGCommand("L", [40, 10]),
            SVGCommand("Z", []),
            # Bottom contour (stem)
            SVGCommand("M", [40, 20]),
            SVGCommand("L", [60, 20]),
            SVGCommand("L", [60, 100]),
            SVGCommand("L", [40, 100]),
            SVGCommand("Z", []),
        ]
    )
    seg_data_multi = multi_contour_glyph.get_segmentation_data()
    assert len(seg_data_multi) == 2
    # Sort by y-coordinate to have a stable order
    seg_data_multi.sort(key=lambda d: d["bbox"][1])

    # Check top contour
    assert seg_data_multi[0]["label"] == 0
    np.testing.assert_allclose(seg_data_multi[0]["bbox"], [40, 0, 60, 10])

    # Check bottom contour
    assert seg_data_multi[1]["label"] == 0
    np.testing.assert_allclose(seg_data_multi[1]["bbox"], [40, 20, 60, 100])

    # Case 4: Empty glyph
    empty_glyph = SVGGlyph([])
    seg_data_empty = empty_glyph.get_segmentation_data()
    assert len(seg_data_empty) == 0
