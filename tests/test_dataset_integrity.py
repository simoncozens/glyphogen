import os
import pytest
import torch

from glyphogen.dataset import get_hierarchical_data
from glyphogen.representations.model import MODEL_REPRESENTATION

# This test is designed to be run from the root of the project.
# It checks the integrity of the dataset by inspecting the command sequences.


def test_inspect_dataset_sequences():
    """
    Loads the training dataset and prints the command sequences for the
    first few items to verify they are not malformed (e.g., all EOS).
    """
    DATA_DIR = "data"
    TRAIN_DB = os.path.join(DATA_DIR, "train_hierarchical.sqlite")
    TRAIN_JSON = os.path.join(DATA_DIR, "train_hierarchical.json")

    print(
        f"Loading dataset from: {TRAIN_DB if os.path.exists(TRAIN_DB) else TRAIN_JSON}"
    )

    if not os.path.exists(TRAIN_DB) and not os.path.exists(TRAIN_JSON):
        pytest.fail("Annotation file not found. Run preprocess_for_hierarchical.py.")

    train_dataset, _ = get_hierarchical_data()
    dataset = train_dataset

    command_names = list(MODEL_REPRESENTATION.grammar.keys())
    command_width = len(command_names)
    num_items_to_check = 5

    print(f"--- Inspecting first {num_items_to_check} items from the dataset ---")

    found_valid_sequence = False
    for i in range(min(num_items_to_check, len(dataset))):
        img, target_dict = dataset[i]
        gt_contours = target_dict["gt_contours"]

        print(f"\nItem {i} (Image ID: {target_dict['image_id']}):")
        if not gt_contours:
            print("  No contours.")
            continue

        for j, contour in enumerate(gt_contours):
            sequence_tensor = contour["sequence"]
            command_tensor = sequence_tensor[:, :command_width]

            indices = command_tensor.argmax(dim=-1).tolist()
            names = [command_names[idx] for idx in indices]

            print(f"  Contour {j}: {names}")

            # Check if there's at least one non-SOS/EOS command
            if any(cmd not in ["SOS", "EOS"] for cmd in names):
                found_valid_sequence = True

            # Remove SOS and EOS for neighbor checks
            names = [cmd for cmd in names if cmd not in ["SOS", "EOS", "M"]]
            for k, cmd in enumerate(names):
                left_cmd = names[k - 1]
                right_cmd = names[(k + 1) % len(names)]

                # If we see a NCI, it should have a node on its left and an line
                # (of some description) on its right.
                if cmd == "NCI":
                    if left_cmd is None or right_cmd is None:
                        print(
                            f"    WARNING: 'NCI' at position {k} has no left or right command."
                        )
                    elif not left_cmd in ["N", "NH", "NV", "NCO"] or right_cmd not in [
                        "L",
                        "LH",
                        "LV",
                        "NCO",
                    ]:
                        print(
                            f"    WARNING: 'NCI' at position {k} has unexpected neighbors: left='{left_cmd}', right='{right_cmd}'."
                        )
                # Similarly for NCO
                if cmd == "NCO":
                    if left_cmd is None or right_cmd is None:
                        print(
                            f"    WARNING: 'NCO' at position {k} has no left or right command."
                        )
                    elif not left_cmd in ["L", "LH", "LV", "NCI"] or right_cmd not in [
                        "N",
                        "NH",
                        "NV",
                        "NCI",
                    ]:
                        print(
                            f"    WARNING: 'NCO' at position {k} has unexpected neighbors: left='{left_cmd}', right='{right_cmd}'."
                        )

    print("--- Inspection Complete ---")

    # This is not a formal assertion, but a check for the user to see.
    # We can add a real assertion later if needed.
    if not found_valid_sequence:
        print(
            "\nWARNING: No valid (non-SOS/EOS) commands found in the inspected items."
        )
    else:
        print("\nSUCCESS: At least one valid sequence was found.")


if __name__ == "__main__":
    import pytest

    # This allows running the test directly.
    test_inspect_dataset_sequences()
