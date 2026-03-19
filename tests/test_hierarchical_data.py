import torch
from pathlib import Path
import pytest

# This import will fail if the main project is not in the path.
# Pytest handles this automatically.
from glyphogen.dataset import get_hierarchical_data


@pytest.fixture
def hierarchical_test_dataset():
    """Provides a fixture for the hierarchical test dataset."""
    DATA_DIR = Path("data")
    TEST_DB = DATA_DIR / "test_hierarchical.sqlite"
    TEST_JSON = DATA_DIR / "test_hierarchical.json"

    if not TEST_DB.exists() and not TEST_JSON.exists():
        pytest.skip(
            "Hierarchical test dataset not found. Run preprocess_for_hierarchical.py."
        )

    _, test_dataset = get_hierarchical_data()
    return test_dataset


def test_dataset_loading(hierarchical_test_dataset):
    """Tests that the dataset can be loaded and has items."""
    assert len(hierarchical_test_dataset) > 0, "Dataset should not be empty."


def test_mask_properties(hierarchical_test_dataset):
    """
    Tests the properties of the masks loaded from the dataset to ensure
    they are binary torch.uint8 tensors.
    """
    # Get a sample item from the dataset
    # Let's take an item from the middle to get a typical glyph
    img, target = hierarchical_test_dataset[len(hierarchical_test_dataset) // 2]

    assert "gt_contours" in target, "Target dictionary should have 'gt_contours' key."

    gt_contours = target["gt_contours"]
    assert isinstance(gt_contours, list), "gt_contours should be a list."
    assert len(gt_contours) > 0, "Sample should have at least one contour."

    # Get the mask from the first contour
    first_contour = gt_contours[0]
    if "normalized_mask" in first_contour:
        mask = first_contour["normalized_mask"]
        assert isinstance(
            mask, torch.Tensor
        ), f"Mask should be a torch.Tensor, but got {type(mask)}."
        assert (
            mask.dtype == torch.float32
        ), f"Normalized mask dtype should be torch.float32, but got {mask.dtype}."
        assert (
            mask.ndim == 4
        ), f"Normalized mask should be 4D [1,1,H,W], but has {mask.ndim} dimensions."
        assert (
            mask.shape[2] > 0 and mask.shape[3] > 0
        ), f"Mask dimensions should be greater than 0, but got shape {mask.shape}"
    else:
        assert "mask" in first_contour, "Contour dictionary should have a 'mask' key."
        mask = first_contour["mask"]

        assert isinstance(
            mask, torch.Tensor
        ), f"Mask should be a torch.Tensor, but got {type(mask)}."
        assert (
            mask.dtype == torch.uint8
        ), f"Mask dtype should be torch.uint8, but got {mask.dtype}."
        unique_values = torch.unique(mask)
        is_binary = all(v in [0, 1] for v in unique_values)
        assert (
            is_binary
        ), f"Mask should be binary (contain only 0s and 1s), but found values: {unique_values.tolist()}"
        assert (
            mask.ndim == 2
        ), f"Mask should be a 2D tensor [H, W], but has {mask.ndim} dimensions."
        assert (
            mask.shape[0] > 0 and mask.shape[1] > 0
        ), f"Mask dimensions should be greater than 0, but got shape {mask.shape}"
