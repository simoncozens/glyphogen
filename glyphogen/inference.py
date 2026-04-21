from typing import cast

from glyphogen.nodeglyph import NodeGlyph
import torch
from glyphogen.representations.nodecommand import NodeCommand
import torch.nn.functional as F


# Main entry point to model
def vectorize(pipeline_model, raster_image: torch.Tensor) -> NodeGlyph:
    contour_results = pipeline_model.vectorize_contours(raster_image)
    if not contour_results:
        return NodeGlyph([])

    contour_sequences = []
    sos_idx = NodeCommand.encode_command("SOS")
    for contour in contour_results:
        if contour["pred_commands"].numel() == 0:
            continue

        pred_commands = contour["pred_commands"]
        pred_coords_norm = contour["pred_coords_norm"]

        sos_cmd = F.one_hot(
            torch.tensor(sos_idx, device=pred_commands.device),
            num_classes=NodeCommand.command_width,
        ).float()
        sos_coords = torch.zeros(
            NodeCommand.coordinate_width,
            device=pred_coords_norm.device,
            dtype=pred_coords_norm.dtype,
        )
        sos_token = torch.cat([sos_cmd, sos_coords]).unsqueeze(0)

        pred_sequence_norm = torch.cat([pred_commands, pred_coords_norm], dim=-1)
        full_pred_sequence_norm = torch.cat([sos_token, pred_sequence_norm], dim=0)
        full_pred_sequence_img = NodeCommand.mask_space_to_image_space(
            full_pred_sequence_norm, contour["box"]
        )

        contour_sequences.append(full_pred_sequence_img[1:].detach().cpu())

    if not contour_sequences:
        return NodeGlyph([])

    decoded = NodeGlyph.decode(contour_sequences, NodeCommand)
    return cast(NodeGlyph, decoded)
