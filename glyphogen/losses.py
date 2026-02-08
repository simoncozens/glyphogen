from typing import List, TYPE_CHECKING
from glyphogen.nodeglyph import NodeGlyph
from glyphogen.svgglyph import SVGGlyph
import torch
import torch.nn.functional as F
import sys

from glyphogen.representations.model import MODEL_REPRESENTATION
from glyphogen.hyperparameters import (
    ALIGNMENT_LOSS_WEIGHT,
    HUBER_DELTA,
    SIGNED_AREA_WEIGHT,
    VECTOR_LOSS_WEIGHT_COMMAND,
    VECTOR_LOSS_WEIGHT_COORD,
    VECTOR_LOSS_WEIGHT_COORD_ABSOLUTE,
)
from glyphogen.typing import (
    CollatedGlyphData,
    GroundTruthContour,
    LossDictionary,
    ModelResults,
)

if TYPE_CHECKING:
    from glyphogen.model import VectorizationGenerator


@torch.compile
def losses(
    collated_batch: CollatedGlyphData,
    outputs: ModelResults,
    model: "VectorizationGenerator",
    device,
    validation=False,
) -> LossDictionary:
    """
    Calculates losses for the hierarchical vectorization model.
    This function now iterates through all contours in a collated batch.
    """
    # Standardized outputs from the model
    pred_commands_list = outputs.pred_commands
    pred_coords_std_list = outputs.pred_coords_std
    gt_coords_std_list = outputs.gt_coords_std
    pred_means_list = outputs.pred_means
    pred_stds_list = outputs.pred_stds

    # Data from the collated batch
    gt_target_sequences = collated_batch["target_sequences"]
    contour_boxes = collated_batch["contour_boxes"]
    x_aligned_point_indices = collated_batch["x_aligned_point_indices"]
    y_aligned_point_indices = collated_batch["y_aligned_point_indices"]

    num_contours_to_compare = len(gt_target_sequences)
    total_command_loss = torch.tensor(0.0, device=device)
    total_coord_loss = torch.tensor(0.0, device=device)
    total_signed_area_loss = torch.tensor(0.0, device=device)
    total_alignment_loss = torch.tensor(0.0, device=device)
    total_coord_mae_metric = torch.tensor(0.0, device=device)
    total_correct_cmds = 0
    total_cmds = 0

    if num_contours_to_compare == 0:
        return {
            "total_loss": torch.tensor(0.0, device=device, requires_grad=True),
            "command_loss": torch.tensor(0.0, device=device),
            "coord_loss": torch.tensor(0.0, device=device),
            "signed_area_loss": torch.tensor(0.0, device=device),
            "alignment_loss": torch.tensor(0.0, device=device),
            "command_accuracy_metric": torch.tensor(0.0, device=device),
            "coordinate_mae_metric": torch.tensor(0.0, device=device),
        }

    for i in range(num_contours_to_compare):
        pred_command = pred_commands_list[i]
        pred_coords_std = pred_coords_std_list[i]
        gt_coords_std = gt_coords_std_list[i]
        pred_means = pred_means_list[i]
        pred_stds = pred_stds_list[i]
        box = contour_boxes[i]

        gt_sequence_img_space = gt_target_sequences[i].to(device)
        gt_sequence_norm = MODEL_REPRESENTATION.image_space_to_mask_space(
            gt_sequence_img_space, box
        )
        gt_commands_norm, _ = MODEL_REPRESENTATION.split_tensor(gt_sequence_norm)
        gt_command_for_loss = gt_commands_norm[1 : pred_command.shape[0] + 1]

        # Slice gt_coords_std to remove the SOS token and match the length of pred_coords_std
        # gt_coords_std comes from outputs.gt_coords_std, which includes the SOS token.
        gt_coords_std_sliced = gt_coords_std[1 : pred_command.shape[0] + 1]
        # But now the pred coordinates have a different length, so pad the gt coords to match
        if pred_coords_std.shape[0] > gt_coords_std_sliced.shape[0]:
            padding_size = pred_coords_std.shape[0] - gt_coords_std_sliced.shape[0]
            padding = torch.zeros(
                (padding_size, gt_coords_std_sliced.shape[1]),
                device=device,
                dtype=gt_coords_std_sliced.dtype,
            )
            gt_coords_std_sliced = torch.cat([gt_coords_std_sliced, padding], dim=0)

        # 1. Command Loss (Cross-Entropy)
        command_loss = F.cross_entropy(
            pred_command.unsqueeze(0).permute(0, 2, 1),
            gt_command_for_loss.unsqueeze(0).argmax(dim=-1),
            label_smoothing=0.1,
        )

        # 2. Coordinate Loss (Hybrid)
        # 2a. Stable loss on standardized, relative coordinates
        coord_loss_relative = masked_relative_std_coordinate_loss(
            device, gt_command_for_loss, gt_coords_std_sliced, pred_coords_std
        )

        # --- De-standardize and Unroll for other losses and metrics ---
        pred_coords_norm = MODEL_REPRESENTATION.de_standardize(  # type: ignore
            pred_coords_std, pred_means, pred_stds
        )
        (
            abs_gt_command,
            abs_gt_coords,
            abs_pred_command,
            abs_pred_coords,
        ) = align_sequences(
            device,
            gt_sequence_norm,
            pred_command,
            pred_coords_norm,
        )

        if abs_pred_command.shape[0] == 0:
            continue

        # 2b. Global loss on absolute, unrolled coordinates
        coord_loss_absolute = masked_absolute_coordinate_loss(
            device, abs_gt_command, abs_gt_coords, abs_pred_coords
        )
        coord_loss = (1 - VECTOR_LOSS_WEIGHT_COORD_ABSOLUTE) * coord_loss_relative + (
            VECTOR_LOSS_WEIGHT_COORD_ABSOLUTE * coord_loss_absolute
        )

        # 3. Signed Area Loss
        eos_idx = MODEL_REPRESENTATION.encode_command("EOS")
        sos_idx = MODEL_REPRESENTATION.encode_command("SOS")
        gt_vertex_mask = (abs_gt_command.argmax(dim=-1) != eos_idx) & (
            abs_gt_command.argmax(dim=-1) != sos_idx
        )
        pred_vertex_mask = (abs_pred_command.argmax(dim=-1) != eos_idx) & (
            abs_pred_command.argmax(dim=-1) != sos_idx
        )
        gt_on_curve_points = abs_gt_coords[gt_vertex_mask, 0:2]
        pred_on_curve_points = abs_pred_coords[pred_vertex_mask, 0:2]
        signed_area_loss = abs_signed_area_loss(
            gt_on_curve_points, pred_on_curve_points
        )

        # 4. Alignment Loss
        align_loss = alignment_loss(
            abs_pred_coords,
            x_aligned_point_indices[i],
            y_aligned_point_indices[i],
            device,
            abs_gt_command,
        )

        total_command_loss += command_loss
        total_coord_loss += coord_loss
        total_signed_area_loss += signed_area_loss
        total_alignment_loss += align_loss

        # Accumulate accuracy stats
        pred_indices = torch.argmax(abs_pred_command, dim=-1)
        gt_indices = torch.argmax(abs_gt_command, dim=-1)
        correct_commands = (pred_indices == gt_indices).sum().item()
        total_correct_cmds += correct_commands
        total_cmds += len(pred_indices)

        # And metric
        coord_mae_metric = masked_coordinate_mae_metric(
            device, abs_gt_command, abs_gt_coords, abs_pred_coords
        )
        total_coord_mae_metric += coord_mae_metric

    def average_a_loss(loss):
        return (
            loss / num_contours_to_compare
            if num_contours_to_compare > 0
            else torch.tensor(0.0)
        )

    avg_command_loss = average_a_loss(total_command_loss)
    avg_coord_loss = average_a_loss(total_coord_loss)
    avg_signed_area_loss = average_a_loss(total_signed_area_loss)
    avg_alignment_loss = average_a_loss(total_alignment_loss)
    avg_coord_mae_metric = average_a_loss(total_coord_mae_metric)
    command_accuracy = (
        torch.tensor(total_correct_cmds / total_cmds)
        if total_cmds > 0
        else torch.tensor(0.0)
    )

    total_loss = (
        VECTOR_LOSS_WEIGHT_COMMAND * avg_command_loss
        + VECTOR_LOSS_WEIGHT_COORD * avg_coord_loss
        + SIGNED_AREA_WEIGHT * avg_signed_area_loss
        + ALIGNMENT_LOSS_WEIGHT * avg_alignment_loss
    )

    return {
        "total_loss": total_loss,
        "command_loss": avg_command_loss.detach(),
        "coord_loss": avg_coord_loss.detach(),
        "signed_area_loss": avg_signed_area_loss.detach(),
        "alignment_loss": avg_alignment_loss.detach(),
        "command_accuracy_metric": command_accuracy,
        "coordinate_mae_metric": avg_coord_mae_metric.detach(),
    }


def abs_signed_area_loss(true_points, pred_points):
    """
    Calculates the signed area of a polygon using the Shoelace formula.
    The input should be a tensor of shape (N, 2) where N is the number of vertices.
    """

    def compute_signed_area(points):
        x = points[:, 0]
        y = points[:, 1]
        signed_area = 0.5 * (torch.sum(x[:-1] * y[1:]) - torch.sum(y[:-1] * x[1:]))
        return signed_area

    true_signed_area = compute_signed_area(true_points)
    pred_signed_area = compute_signed_area(pred_points)

    return torch.abs(true_signed_area - pred_signed_area)


def alignment_loss(
    pred_coords, x_alignment_sets, y_alignment_sets, device, gt_command_for_loss
):
    """
    Calculates a loss based on the variance of coordinates that should be aligned.
    """
    total_x_variance = torch.tensor(0.0, device=device)
    total_y_variance = torch.tensor(0.0, device=device)

    on_curve_points = pred_coords[:, 0:2]
    valid_nodes_mask = torch.argmax(
        gt_command_for_loss, dim=-1
    ) != MODEL_REPRESENTATION.encode_command("EOS")
    num_valid_nodes = valid_nodes_mask.sum().item()

    for alignment_set in x_alignment_sets:
        # This +1 is suspicious, but it *is* correct. Remember that in the alignment sets,
        # the first node is index 0, which corresponds to the first on-curve point,
        # but in our representation this is split over two nodes (the M and the first command).
        row_indices = [idx + 1 for idx in alignment_set]
        valid_set = [idx for idx in row_indices if idx < num_valid_nodes]
        if len(valid_set) > 1:
            aligned_x_coords = on_curve_points[valid_set, 0]
            total_x_variance += torch.var(aligned_x_coords)

    for alignment_set in y_alignment_sets:
        row_indices = [idx + 1 for idx in alignment_set]
        valid_set = [idx for idx in row_indices if idx < num_valid_nodes]
        if len(valid_set) > 1:
            aligned_y_coords = on_curve_points[valid_set, 1]
            total_y_variance += torch.var(aligned_y_coords)

    return total_x_variance + total_y_variance


def coordinate_width_mask(commands: torch.Tensor, coords: torch.Tensor):
    device = commands.device
    command_indices = torch.argmax(commands, dim=-1)
    arg_counts_list = [
        MODEL_REPRESENTATION.grammar[cmd] for cmd in MODEL_REPRESENTATION.grammar
    ]
    arg_counts = torch.tensor(arg_counts_list, device=device)
    num_relevant_coords = arg_counts[command_indices]
    coord_mask = torch.arange(
        coords.shape[1], device=device
    ) < num_relevant_coords.unsqueeze(1)
    return coord_mask


def masked_relative_std_coordinate_loss(
    device, gt_command_for_loss, gt_coords_std, pred_coords_std
):
    """
    Calculates MAE loss on standardized, relative coordinates.
    """
    coord_mask = coordinate_width_mask(gt_command_for_loss, gt_coords_std)
    elementwise_coord_loss = F.l1_loss(
        pred_coords_std,
        gt_coords_std,
        #delta = 0.1,
        reduction="none",
    )
    masked_coord_loss = elementwise_coord_loss * coord_mask
    num_coords_in_loss = coord_mask.sum()
    if num_coords_in_loss > 0:
        coord_loss = masked_coord_loss.sum() / num_coords_in_loss
    else:
        coord_loss = torch.tensor(0.0, device=device)
    return coord_loss


def masked_absolute_coordinate_loss(
    device, gt_command_for_loss, abs_gt_coords, abs_pred_coords
):
    """
    Calculates Huber loss on absolute, bbox-normalized coordinates.
    """
    coord_mask = coordinate_width_mask(gt_command_for_loss, abs_gt_coords)
    elementwise_coord_loss = F.huber_loss(
        abs_pred_coords,
        abs_gt_coords,
        reduction="none",
        delta=HUBER_DELTA,
    )
    masked_coord_loss = elementwise_coord_loss * coord_mask
    num_coords_in_loss = coord_mask.sum()
    if num_coords_in_loss > 0:
        coord_loss = masked_coord_loss.sum() / num_coords_in_loss
    else:
        coord_loss = torch.tensor(0.0, device=device)
    return coord_loss


def masked_coordinate_mae_metric(
    device, gt_command_for_loss, abs_gt_coords, abs_pred_coords
):
    """
    Calculate Mean Absolute Error (MAE) for coordinates, masked by command type.
    This version expects absolute, bbox-normalized coordinates.
    """
    coord_mask = coordinate_width_mask(gt_command_for_loss, abs_gt_coords)
    elementwise_coord_error = torch.abs(abs_pred_coords * 256.0 - abs_gt_coords * 256.0)
    masked_coord_error = elementwise_coord_error * coord_mask
    num_coords_in_metric = coord_mask.sum()
    if num_coords_in_metric > 0:
        coord_mae = masked_coord_error.sum() / num_coords_in_metric
    else:
        coord_mae = torch.tensor(0.0, device=device)
    return coord_mae


def align_sequences(
    device,
    gt_sequence_norm,
    pred_command,
    pred_coords_norm,
):
    gt_command_all, _ = MODEL_REPRESENTATION.split_tensor(gt_sequence_norm)
    gt_command_for_loss = gt_command_all[1 : pred_command.shape[0] + 1]
    pred_command_for_loss = pred_command

    abs_gt_sequence = MODEL_REPRESENTATION.unroll_relative_coordinates(gt_sequence_norm)
    _, abs_gt_coords_all = MODEL_REPRESENTATION.split_tensor(abs_gt_sequence)
    gt_coords_for_loss = abs_gt_coords_all[1 : pred_command.shape[0] + 1]

    sos_token = gt_sequence_norm[0:1, :]
    predicted_relative_sequence = torch.cat([pred_command, pred_coords_norm], dim=-1)
    full_predicted_relative_sequence = torch.cat(
        [sos_token, predicted_relative_sequence], dim=0
    )
    abs_pred_sequence = MODEL_REPRESENTATION.unroll_relative_coordinates(
        full_predicted_relative_sequence
    )
    _, abs_pred_coords_all = MODEL_REPRESENTATION.split_tensor(abs_pred_sequence)
    pred_coords_for_loss = abs_pred_coords_all[1:]

    return (
        gt_command_for_loss,
        gt_coords_for_loss,
        pred_command_for_loss,
        pred_coords_for_loss,
    )


def predictions_to_image_space(
    outputs: ModelResults, gt_contours: List[GroundTruthContour]
):
    pred_commands_list = outputs.pred_commands
    pred_coords_norm_list = outputs.pred_coords_norm
    contour_boxes = outputs.contour_boxes

    pred_commands_and_coords_img_space = []
    for i in range(len(pred_commands_list)):
        pred_cmd = pred_commands_list[i].detach().cpu()
        pred_coords_norm = pred_coords_norm_list[i].detach().cpu()
        box = contour_boxes[i].detach().cpu()

        pred_sequence_norm = torch.cat([pred_cmd, pred_coords_norm], dim=-1)
        sos_token = MODEL_REPRESENTATION.image_space_to_mask_space(
            gt_contours[i]["sequence"].cpu(), box
        )[0:1, :]
        full_pred_sequence_norm = torch.cat([sos_token, pred_sequence_norm], dim=0)
        pred_sequence_img_space = MODEL_REPRESENTATION.mask_space_to_image_space(
            full_pred_sequence_norm, box
        )
        pred_commands_and_coords_img_space.append(pred_sequence_img_space[1:])
    return pred_commands_and_coords_img_space


def dump_debug_sequences(
    writer, global_step, batch_idx, gt_contours, outputs: ModelResults, loss_values
):
    """For debugging, dump ground truth and predicted sequences"""
    pred_commands_and_coords_img_space = predictions_to_image_space(
        outputs, gt_contours
    )
    pred_command_lists = NodeGlyph.decode(
        pred_commands_and_coords_img_space,
        MODEL_REPRESENTATION,
        return_raw_command_lists=True,
    )
    pred_glyph = NodeGlyph.from_command_lists(pred_command_lists)
    pred_debug_command_lists = [
        " ".join([cmd.debug_string() for cmd in cmd_list])
        for cmd_list in pred_command_lists
    ]
    pred_nodes = ", ".join(pred_debug_command_lists)

    debug_string = SVGGlyph.from_node_glyph(pred_glyph).to_svg_string()
    gt_command_lists = NodeGlyph.decode(
        [x["sequence"].cpu() for x in gt_contours],
        MODEL_REPRESENTATION,
        return_raw_command_lists=True,
    )
    gt_debug_command_lists = [
        " ".join([cmd.debug_string() for cmd in cmd_list])
        for cmd_list in gt_command_lists
    ]
    gt_glyph = NodeGlyph.from_command_lists(gt_command_lists)
    gt_nodes = ", ".join(gt_debug_command_lists)
    gt_debug_string = SVGGlyph.from_node_glyph(gt_glyph).to_svg_string()

    writer.add_text(
        f"SVG/Debug_{batch_idx}",
        f"""
GT: {gt_debug_string}
Pred: {debug_string}
GT Commands: {gt_nodes}
Pred Commands: {pred_nodes}
Command loss: {loss_values['command_loss'].item():.4f}
Coord loss: {loss_values['coord_loss'].item():.4f}
""",
        global_step,
    )
    writer.flush()
