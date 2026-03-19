from glyphogen.representations.svgcommand import SVGCommand
from glyphogen.typing import CollatedGlyphData, ModelResults
import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes

from glyphogen.losses import align_sequences
from glyphogen.rasterizer import rasterize_batch

from .representations.model import MODEL_REPRESENTATION
from .nodeglyph import NodeGlyph
from .svgglyph import SVGGlyph


def log_vectorizer_outputs(
    model, data_loader, writer, epoch, num_images=4, log_svgs=True
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

            # Run inference using the new generate method
            outputs: ModelResults = model.generate(img)

            # --- Log Raster Visualization ---
            predicted_raster = torch.ones_like(img[0:1])
            if outputs.pred_commands:
                # The rasterizer expects a list of contours for a single glyph,
                # and each contour is a tuple of (commands, coords)
                contour_sequences = [
                    (
                        outputs.pred_commands[idx],
                        outputs.pred_coords_img_space[idx],
                    )
                    for idx in range(len(outputs.pred_commands))
                ]
                # De-encode the glyph from our model representation
                ng = NodeGlyph.decode(
                    [
                        torch.cat([cmds, coords], dim=-1).cpu()
                        for cmds, coords in contour_sequences
                    ],
                    MODEL_REPRESENTATION,
                )
                # Now re-encode as SVG
                svg_contours = ng.encode(SVGCommand)
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
            if log_svgs and not skip_svgs and outputs.pred_commands:
                # Prepare contour sequences for decoding
                contour_sequences = [
                    torch.cat(
                        [
                            outputs.pred_commands[idx].cpu(),
                            outputs.pred_coords_img_space[idx].cpu(),
                        ],
                        dim=-1,
                    )
                    for idx in range(len(outputs.pred_commands))
                ]

                try:
                    decoded_glyph = NodeGlyph.decode(
                        contour_sequences, MODEL_REPRESENTATION
                    )
                    svg_string = SVGGlyph.from_node_glyph(decoded_glyph).to_svg_string()
                    debug_string = decoded_glyph.to_debug_string()
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
    contour_boxes = collated_batch["contour_boxes"]
    device = pred_commands_list[0].device  # Get device from a tensor

    num_contours_to_compare = len(gt_target_sequences)

    for j in range(num_contours_to_compare):
        pred_command = pred_commands_list[j]
        pred_coords_norm = pred_coords_norm_list[j]
        box = contour_boxes[j]

        # Convert GT sequence from image space to normalized mask space
        gt_sequence_img_space = gt_target_sequences[j]
        gt_sequence_norm = MODEL_REPRESENTATION.image_space_to_mask_space(
            gt_sequence_img_space.to(device), box
        )

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


def log_bounding_boxes(model, data_loader, writer, epoch, num_images=4):
    """Logs images with ground truth and predicted bounding boxes."""
    device = next(model.parameters()).device
    model.eval()

    # Get a batch of data
    try:
        collated_batch = next(iter(data_loader))
    except StopIteration:
        print("Warning: Could not get a batch from data_loader for logging.")
        return

    if collated_batch is None:
        print("Warning: Collated batch is None, skipping logging.")
        return

    # Unpack the collated batch
    images = collated_batch["images"]
    targets = collated_batch["gt_targets"]

    # Take only num_images from the batch
    images = images[:num_images]
    targets = targets[:num_images]

    # images is already a stacked tensor from the collate_fn.
    images_for_segmenter = images.to(device)

    # Get predictions
    with torch.no_grad():
        # The segmenter is part of the main model
        predictions = model.segmenter(images_for_segmenter)

    for i in range(len(images)):
        img_tensor = images[i]
        gt_target = targets[i]
        pred_target = predictions[i]

        # Prepare image for drawing (convert to uint8, 3 channels)
        img_to_draw = (img_tensor.cpu() * 255).to(torch.uint8)
        if img_to_draw.shape[0] == 1:
            img_to_draw = img_to_draw.repeat(3, 1, 1)

        # Get GT boxes and labels
        if gt_target["gt_contours"]:
            gt_boxes = torch.stack([c["box"] for c in gt_target["gt_contours"]])
            gt_labels = [
                f"GT: {'hole' if c['label']==2 else 'outer'}"
                for c in gt_target["gt_contours"]
            ]
        else:
            gt_boxes = torch.empty((0, 4))
            gt_labels = []

        # Get predicted boxes and labels
        pred_boxes = pred_target["boxes"].cpu()
        pred_labels = [
            f"Pred: {'hole' if label==2 else 'outer'} ({(s*100):.0f}%)"
            for label, s in zip(pred_target["labels"], pred_target["scores"])
        ]

        # Draw boxes on the image
        img_with_boxes = img_to_draw
        if pred_boxes.shape[0] > 0:
            img_with_boxes = draw_bounding_boxes(
                img_with_boxes, boxes=pred_boxes, labels=pred_labels, colors="red"
            )
        if gt_boxes.shape[0] > 0:
            img_with_boxes = draw_bounding_boxes(
                img_with_boxes, boxes=gt_boxes, labels=gt_labels, colors="green"
            )

        writer.add_image(f"Bounding_Boxes/Image_{i}", img_with_boxes, epoch)

    writer.flush()
