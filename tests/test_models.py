"""
Tests for model outputs and ranges.

Tests that:
1. ContourVectorizer model output ranges are correct
2. VectorizationGenerator (segmenter + vectorizer) works end-to-end
"""

import torch
import pytest
from pathlib import Path
from torch.utils.data import DataLoader

from glyphogen.vectorizer import ContourVectorizer
from glyphogen.model import VectorizationGenerator
from glyphogen.dataset import get_hierarchical_data, collate_fn
from glyphogen.representations.nodecommand import NodeCommand
from glyphogen.hyperparameters import D_MODEL, LATENT_DIM, RATE


SOS_INDEX = NodeCommand.encode_command("SOS")
EOS_INDEX = NodeCommand.encode_command("EOS")
BATCH_SIZE = 4


@pytest.fixture(scope="module")
def dataset():
    """Load test dataset once for all tests in this module."""
    test_dataset, _ = get_hierarchical_data()
    return DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def vectorizer_model(device):
    """Create and load a ContourVectorizer model."""
    model = ContourVectorizer(
        d_model=D_MODEL,
        latent_dim=LATENT_DIM,
        rate=RATE,
    )
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def generator_model(device):
    """Create and load a VectorizationGenerator model (segmenter + vectorizer)."""
    segmenter_checkpoint = Path("glyphogen.segmenter.pt")
    if not segmenter_checkpoint.exists():
        pytest.skip("Missing segmenter checkpoint: glyphogen.segmenter.pt")

    segmenter_state = torch.load(segmenter_checkpoint, map_location="cpu")
    model = VectorizationGenerator(
        segmenter_state=segmenter_state,
        d_model=D_MODEL,
        latent_dim=LATENT_DIM,
        rate=RATE,
    )
    model = model.to(device)
    model.eval()
    return model


def test_contour_vectorizer_output_ranges(vectorizer_model, dataset, device):
    """Test that ContourVectorizer produces outputs with expected ranges."""
    batch = next(iter(dataset))
    if batch is None:
        pytest.skip("Could not get a batch from the dataset.")

    # Move batch to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
        elif (
            isinstance(batch[key], list)
            and batch[key]
            and isinstance(batch[key][0], torch.Tensor)
        ):
            batch[key] = [t.to(device) for t in batch[key]]

    # Forward pass
    with torch.no_grad():
        outputs = vectorizer_model(batch, teacher_forcing_ratio=1.0)

    # Test pred_commands: should be logits (arbitrary range) or one-hot
    for i, cmd in enumerate(outputs.pred_commands):
        assert cmd is not None, f"pred_commands[{i}] is None"
        assert torch.isfinite(
            cmd
        ).all(), f"pred_commands[{i}] contains non-finite values"
        # Commands can be logits, so we just check they're finite

    # Test pred_coords_norm: should be in normalized space [-1, 2]
    for i, coords in enumerate(outputs.pred_coords_norm):
        assert coords is not None, f"pred_coords_norm[{i}] is None"
        assert torch.isfinite(
            coords
        ).all(), f"pred_coords_norm[{i}] contains non-finite values"
        assert (
            coords.min() >= -2.0
        ), f"pred_coords_norm[{i}] has too-low values: {coords.min()}"
        assert (
            coords.max() <= 2.0
        ), f"pred_coords_norm[{i}] has too-high values: {coords.max()}"

    # Test pred_coords_std: should be standardized (mean~0, std~1)
    for i, coords_std in enumerate(outputs.pred_coords_std):
        assert coords_std is not None, f"pred_coords_std[{i}] is None"
        assert torch.isfinite(
            coords_std
        ).all(), f"pred_coords_std[{i}] contains non-finite values"
        # Standardized coords can have wider range but should be roughly centered
        assert (
            coords_std.abs().mean() < 10.0
        ), f"pred_coords_std[{i}] has unexpectedly large values"

    # Test lstm_outputs
    for i, lstm_out in enumerate(outputs.lstm_outputs):
        if lstm_out is not None:
            assert torch.isfinite(
                lstm_out
            ).all(), f"lstm_outputs[{i}] contains non-finite values"

    # Test used_teacher_forcing
    assert isinstance(
        outputs.used_teacher_forcing, bool
    ), "used_teacher_forcing should be bool"


def test_contour_vectorizer_batch_consistency(vectorizer_model, dataset, device):
    """Test that vectorizer produces consistent number of outputs for each batch."""
    batch = next(iter(dataset))
    if batch is None:
        pytest.skip("Could not get a batch from the dataset.")

    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
        elif (
            isinstance(batch[key], list)
            and batch[key]
            and isinstance(batch[key][0], torch.Tensor)
        ):
            batch[key] = [t.to(device) for t in batch[key]]

    with torch.no_grad():
        outputs = vectorizer_model(batch, teacher_forcing_ratio=1.0)

    num_contours = len(batch["target_sequences"])
    assert (
        len(outputs.pred_commands) == num_contours
    ), f"Expected {num_contours} pred_commands, got {len(outputs.pred_commands)}"
    assert (
        len(outputs.pred_coords_norm) == num_contours
    ), f"Expected {num_contours} pred_coords_norm, got {len(outputs.pred_coords_norm)}"
    assert (
        len(outputs.pred_coords_std) == num_contours
    ), f"Expected {num_contours} pred_coords_std, got {len(outputs.pred_coords_std)}"


def test_vectorization_generator_end_to_end(generator_model, device):
    """
    Test end-to-end VectorizationGenerator: takes a raster image and produces
    a full vectorization.
    """
    # Create a simple test image: 256x256 with a square in the center
    test_image = torch.ones(1, 256, 256, device=device)
    test_image[0, 100:200, 100:200] = 0  # Black square

    # Generate vectorization
    with torch.no_grad():
        outputs = generator_model.generate(test_image)

    # Check that we got some output
    assert outputs is not None, "VectorizationGenerator.generate() returned None"

    # If the model found contours, check their properties
    if len(outputs.pred_commands) > 0:
        # All lists should have same length
        assert len(outputs.pred_coords_norm) == len(
            outputs.pred_commands
        ), "Mismatch between pred_commands and pred_coords_norm"

        # Check value ranges
        for i, coords in enumerate(outputs.pred_coords_norm):
            assert torch.isfinite(
                coords
            ).all(), f"Contour {i}: normalized coords contain non-finite values"
            assert (
                coords.min() >= -2.0 and coords.max() <= 2.0
            ), f"Contour {i}: normalized coords out of expected range"

    # Check that used_teacher_forcing is False during generation (inference mode)
    assert (
        outputs.used_teacher_forcing == False
    ), "used_teacher_forcing should be False during inference"


def test_vectorization_generator_with_real_glyph(generator_model, dataset, device):
    """
    Test VectorizationGenerator on real glyph image from dataset.
    """
    batch = next(iter(dataset))
    if batch is None:
        pytest.skip("Could not get a batch from the dataset.")

    # Get first image from batch
    test_image = batch["images"][0].to(device)

    # Generate vectorization
    with torch.no_grad():
        outputs = generator_model.generate(test_image)

    # Check basic properties
    assert outputs is not None, "VectorizationGenerator.generate() returned None"

    # The image contains real glyphs, so we should find some contours
    if len(outputs.pred_commands) > 0:
        assert len(outputs.pred_coords_norm) == len(
            outputs.pred_commands
        ), "Output lists have inconsistent lengths"

        # Check each contour
        for i, (cmd, coords_norm) in enumerate(
            zip(
                outputs.pred_commands,
                outputs.pred_coords_norm,
            )
        ):
            # Should have commands
            assert cmd.shape[0] > 0, f"Contour {i} has no commands"

            # Commands and coordinates should match in sequence length
            assert (
                cmd.shape[0] == coords_norm.shape[0]
            ), f"Contour {i}: command and coord sequence lengths mismatch"

            # Check value ranges
            assert torch.isfinite(cmd).all(), f"Contour {i} commands contain NaN/Inf"
            assert torch.isfinite(
                coords_norm
            ).all(), f"Contour {i} normalized coords contain NaN/Inf"
