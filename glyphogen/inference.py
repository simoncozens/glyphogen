from typing import cast

from glyphogen.nodeglyph import NodeGlyph
import torch
from glyphogen.representations.nodecommand import NodeCommand


# Main entry point to model
def vectorize(pipeline_model, raster_image: torch.Tensor) -> NodeGlyph:
    contour_results = pipeline_model.vectorize_contours(raster_image)
    if not contour_results:
        return NodeGlyph([])

    contour_sequences = []
    for contour in contour_results:
        if contour["pred_commands"].numel() == 0:
            continue

        contour_sequences.append(
            torch.cat(
                [contour["pred_commands"], contour["pred_coords_norm"]],
                dim=-1,
            )
            .detach()
            .cpu()
        )

    if not contour_sequences:
        return NodeGlyph([])

    decoded = NodeGlyph.decode(contour_sequences, NodeCommand)
    return cast(NodeGlyph, decoded)
