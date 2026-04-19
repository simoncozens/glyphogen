from pathlib import Path
import numpy as np
import pytest
from glyphogen.representations.nodecommand import NodeCommand
from glyphogen.glyph import Glyph
from glyphogen.nodeglyph import NodeContour
import uharfbuzz as hb


def test_smooth_node_roundtrip():
    """
    Tests that a NodeContour with a smooth node can be converted to a command
    sequence (emitting an NS command) and back to an identical NodeContour.
    """
    # 1. Create a contour with a smooth node
    # A simple "S" curve with three points. P1 is the smooth node.
    # The handles for P1 are collinear.
    original_contour = NodeContour([])
    p0 = original_contour.push(
        coordinates=np.array([0, 0]),
        in_handle=np.array([-25, 25]),  # Not used for first segment
        out_handle=np.array([25, 0]),
    )
    p1 = original_contour.push(
        coordinates=np.array([50, 50]),
        in_handle=np.array([25, 25]),
        out_handle=np.array([75, 75]),
    )
    p2 = original_contour.push(
        coordinates=np.array([0, 100]),
        in_handle=np.array([25, 100]),
        out_handle=np.array([-25, 75]),  # Not used for last segment
    )

    # 2. Emit the command sequence
    commands = original_contour.commands(NodeCommand)

    # 3. Check that an NS command was emitted for the smooth node
    # The command for a node describes the segment *leading to it*.
    # The `is_smooth` property is on `p1`, so the command at index 2
    # (after SOS and M) should correspond to the segment ending at p1.
    # However, our `emit` logic checks the node type and emits a command for it.
    # The sequence is SOS, M, N (for p0), NS (for p1), N (for p2), EOS.
    # So the NS command should be at index 3.
    command_names = [c.command for c in commands]
    assert "NS" in command_names
    assert command_names[3] == "NS"

    # 4. Convert back to a NodeContour
    reconstructed_contour = NodeCommand.contour_from_commands(commands)

    # 5. Assert equality
    # The contour_from_commands will create a closed contour, so we need to
    # adjust the original to match if it's not already closed.
    # In this case, the logic implicitly closes it. Let's check the logic.
    # The `emit` does not create a closed path, it just adds EOS.
    # The `contour_from_commands` does not close it either.
    # Let's compare the nodes directly.
    assert len(original_contour.nodes) == len(reconstructed_contour.nodes)
    assert reconstructed_contour == original_contour


def test_we_generate_some():
    font_path = Path("NotoSans[wdth,wght].ttf")
    if not font_path.exists():
        pytest.skip("NotoSans[wdth,wght].ttf not found, skipping real glyph test.")
    char_to_test = "s"  # Should be some smooth nodes in 's'

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
    # Now convert to NodeGlyph
    nodeglyph_orig = svg_glyph_orig.to_node_glyph()
    # Now encode to commands
    contours_commands = nodeglyph_orig.command_lists(NodeCommand)
    # Check that at least one NS command is present
    ns_found = False
    for contour_commands in contours_commands:
        for command in contour_commands:
            if command.command == "NS":
                ns_found = True
                break
        if ns_found:
            break
    assert ns_found, f"No NS command found in glyph '{char_to_test}'"
