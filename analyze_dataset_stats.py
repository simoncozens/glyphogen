import torch
from torch.utils.data import DataLoader
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm

from glyphogen.dataset import get_hierarchical_data, collate_fn
from glyphogen.representations.model import MODEL_REPRESENTATION

# Define the coordinate groups
STAT_GROUPS = MODEL_REPRESENTATION.get_initial_stats_dict()


def analyze_dataset_stats():
    """
    Analyzes the training dataset to compute mean and standard deviation for
    different types of coordinates.
    """
    print("Loading dataset...")
    train_dataset, _ = get_hierarchical_data()
    # Use a simple dataloader; batch size 1 is fine.
    # collate_fn is needed to handle the custom batch structure.
    dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )
    command_histogram = Counter()

    # Get command indices for faster lookup
    cmd_indices = {
        cmd: MODEL_REPRESENTATION.encode_command(cmd)
        for cmd in MODEL_REPRESENTATION.grammar.keys()
    }

    print(f"Analyzing {len(train_dataset)} glyphs...")
    for batch in tqdm(dataloader):
        if batch is None:
            continue

        contour_boxes = batch["contour_boxes"]
        target_sequences = batch["target_sequences"]
        contour_image_idx = batch["contour_image_idx"]

        for i in range(len(target_sequences)):
            sequence_img_space = target_sequences[i]
            box = contour_boxes[i]

            # Convert to the model's normalized coordinate space
            sequence_norm = MODEL_REPRESENTATION.image_space_to_mask_space(
                sequence_img_space, box
            )
            commands, coords = MODEL_REPRESENTATION.split_tensor(sequence_norm)
            command_idxs = torch.argmax(commands, dim=-1)

            for j in range(len(command_idxs)):
                cmd_idx = int(command_idxs[j].item())
                command = MODEL_REPRESENTATION.decode_command(cmd_idx)
                command_histogram[command] += 1
                coord_vec = coords[j]
                MODEL_REPRESENTATION.update_stats_dict_with_command(
                    STAT_GROUPS, command, coord_vec
                )

    # --- Calculate and save stats ---
    print("\n--- Coordinate Statistics (Mask Space) ---")
    final_stats = {}
    for name, values in STAT_GROUPS.items():
        if not values:
            print(f"{name}: No values found.")
            continue
        tensor_values = torch.tensor(values, dtype=torch.float32)
        mean = torch.mean(tensor_values)
        std = torch.std(tensor_values)
        # Add a small epsilon to std to avoid division by zero if a value is constant
        if std < 1e-6:
            std = torch.tensor(1.0)
            print(f"Warning: Std dev for {name} is near zero. Setting to 1.0.")

        final_stats[name] = {"mean": mean.item(), "std": std.item()}
        print(
            f"{name}: \tMean: {mean.item():.4f}, \tStd: {std.item():.4f}, \tSamples: {len(values)}"
        )

    print("\n--- Command Histogram ---")
    for cmd, count in sorted(command_histogram.items(), key=lambda x: x[0]):
        print(f"{cmd}: {count}")

    output_file = "data/coord_stats.pt"
    torch.save(final_stats, output_file)
    print(f"\nStatistics saved to {output_file}")


if __name__ == "__main__":
    analyze_dataset_stats()
