from glyphogen.representations.svgcommand import SVGCommand
from glyphogen.typing import CollatedGlyphData, ModelResults
import torch

from glyphogen.losses import align_sequences
from glyphogen.rasterizer import rasterize_batch
from glyphogen.inference import vectorize

from .representations.model import MODEL_REPRESENTATION
from .nodeglyph import NodeGlyph
from .svgglyph import SVGGlyph


def log_vectorizer_outputs(
    model, pipeline_model, data_loader, writer, epoch, num_images=4, log_svgs=True
):
    """
    Unified logging function for vectorizer outputs.

    Runs inference on images using autoregressive generation and logs:
    - Raster visualizations (overlay of predicted vs ground truth)
    - SVG outputs (generated vector graphics and debug info)
    """
    # Only log SVGs every 5 epochs to save storage
    skip_svgs = log_svgs and (epoch % 5 != 0)

    device = next(model.parameters()).device
    model.eval()
    try:
        collated_batch = next(iter(data_loader))
    except StopIteration:
        print("Warning: Could not get a batch from data_loader for logging.")
        return

    if collated_batch is None:
        print("Warning: Collated batch is None, skipping logging.")
        return

    images = collated_batch["images"].to(device)

    with torch.no_grad():
        for i in range(min(num_images, images.shape[0])):
            img = images[i]

            # Run full-glyph inference via segmenter+vectorizer pipeline.
            predicted_nodeglyph = vectorize(pipeline_model, img)

            # --- Log Raster Visualization ---
            predicted_raster = torch.ones_like(img[0:1])
            if predicted_nodeglyph.contours:
                svg_contours = predicted_nodeglyph.encode(SVGCommand) or []
                # Filter out dead
                svg_contours = [
                    torch.from_numpy(c) for c in svg_contours if c.shape[0] > 0
                ]

                if svg_contours:
                    predicted_raster = rasterize_batch(
                        [
                            [
                                SVGCommand.split_tensor(svg_contour)
                                for svg_contour in svg_contours
                            ]
                        ],
                        SVGCommand,
                        device=device,
                    )[0]

            # Create a 3-channel overlay for visualization
            # We want predicted in red, ground truth in green, blue channel empty
            predicted_inv = 1.0 - predicted_raster
            true_inv = 1.0 - img[0:1, :, :]  # Take first channel and keep dims
            zeros = torch.zeros_like(predicted_inv)

            overlay_image = torch.cat([predicted_inv, true_inv, zeros], dim=0)
            writer.add_image(f"Vectorizer_Images/Overlay_{i}", overlay_image, epoch)

            # --- Log SVG Outputs (if enabled and epoch matches) ---
            if log_svgs and not skip_svgs and predicted_nodeglyph.contours:
                try:
                    svg_string = SVGGlyph.from_node_glyph(
                        predicted_nodeglyph
                    ).to_svg_string()
                    debug_string = predicted_nodeglyph.to_debug_string()
                except Exception as e:
                    svg_string = f"Couldn't generate SVG: {e}"
                    debug_string = "Error in decoding glyph."

                writer.add_text(f"SVG/Generated_{i}", svg_string, epoch)
                writer.add_text(f"SVG/Node_{i}", debug_string, epoch)

    writer.flush()


def init_confusion_matrix_state():
    """Initializes a state dictionary for collecting confusion matrix data."""

    return {"all_true_indices": [], "all_pred_indices": []}


def collect_confusion_matrix_data(
    state, outputs: ModelResults, collated_batch: CollatedGlyphData
):
    """
    Collects prediction and ground truth data from a validation batch.
    This version works with a single ModelResults and CollatedGlyphData
    object that contains data for the entire batch.
    """
    # Data is already batched; we iterate over all contours in the batch.
    pred_commands_list = outputs.pred_commands
    pred_coords_norm_list = outputs.pred_coords_norm

    gt_target_sequences = collated_batch["target_sequences"]
    device = pred_commands_list[0].device  # Get device from a tensor

    num_contours_to_compare = len(gt_target_sequences)

    for j in range(num_contours_to_compare):
        pred_command = pred_commands_list[j]
        pred_coords_norm = pred_coords_norm_list[j]
        gt_sequence_norm = gt_target_sequences[j].to(device)

        (
            gt_command_for_loss,
            _,
            pred_command_for_loss,
            _,
        ) = align_sequences(
            device,
            gt_sequence_norm,
            pred_command,
            pred_coords_norm,
        )

        if gt_command_for_loss.shape[0] > 0:
            true_indices = torch.argmax(gt_command_for_loss, dim=-1)
            pred_indices = torch.argmax(pred_command_for_loss, dim=-1)
            state["all_true_indices"].append(true_indices.detach().cpu())
            state["all_pred_indices"].append(pred_indices.detach().cpu())


def log_confusion_matrix(state, writer, epoch):
    """Computes and logs the confusion matrix at the end of an epoch."""
    if not state["all_true_indices"]:
        return

    true_indices = torch.cat(state["all_true_indices"])
    pred_indices = torch.cat(state["all_pred_indices"])
    num_classes = len(MODEL_REPRESENTATION.grammar)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int32)
    for i in range(true_indices.shape[0]):
        true_label = true_indices[i]
        pred_label = pred_indices[i]
        if true_label < num_classes and pred_label < num_classes:
            matrix[true_label, pred_label] += 1

    # Format as Markdown table
    command_names = list(MODEL_REPRESENTATION.grammar.keys())
    header = "| True \\ Pred | " + " | ".join(command_names) + " |\n"
    separator = "|--- " * (num_classes + 1) + "|\n"
    body = ""
    for i, name in enumerate(command_names):
        row = f"| **{name}** | "
        for j in range(num_classes):
            row += f"{matrix[i, j].item()} | "
        body += row + "\n"

    markdown_string = header + separator + body

    writer.add_text("Diagnostics/Confusion Matrix", markdown_string, epoch)
    writer.flush()

    # State is reset in the training loop
