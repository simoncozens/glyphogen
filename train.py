#!/usr/bin/env python
from collections import defaultdict
import datetime
import os

import pkbar
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random
import sys
from pathlib import Path

from glyphogen.scheduler import get_cosine_annealing_with_warmup, WarmupLR

# Add the torchvision references to the path
sys.path.append("vision/references/detection/")
torch.multiprocessing.set_sharing_strategy("file_system")

from glyphogen.callbacks import (
    log_vectorizer_outputs,
    init_confusion_matrix_state,
    collect_confusion_matrix_data,
    log_confusion_matrix,
    log_bounding_boxes,
)
from glyphogen.dataset import collate_fn, get_hierarchical_data
from glyphogen.hyperparameters import (
    BATCH_SIZE,
    D_MODEL,
    EPOCHS,
    LATENT_DIM,
    LEARNING_RATE,
    RATE,
    FINAL_LEARNING_RATE,
    SCHEDULED_SAMPLING_START_EPOCH,
    SCHEDULED_SAMPLING_END_EPOCH,
    SCHEDULED_SAMPLING_MIN_RATIO,
    WARMUP_STEPS,
)
from glyphogen.model import VectorizationGenerator, step

do_validation = True


def dump_accumulators(accumulators, writer, epoch, batch_idx, step=None):
    for key, value in accumulators.items():
        if step is not None:
            avg_value = value
            prefix = "Step"
            epoch = step
        else:
            avg_value = value / (batch_idx + 1)
            prefix = ""
        if key.endswith("_loss"):
            scalar_key = key.replace("_loss", "")
            writer.add_scalar(f"{prefix}Loss/{scalar_key}", avg_value, epoch)
        else:
            scalar_key = key.replace("_metric", "")
            writer.add_scalar(f"{prefix}Metric/{scalar_key}", avg_value, epoch)
    writer.flush()


def write_gradient_norms(model, losses, writer, step):
    for key in losses.keys():
        if key == "total_loss":
            continue
        if losses[key].grad_fn is None:
            continue
        coord_grads = torch.autograd.grad(
            losses[key],
            model.parameters(),
            retain_graph=True,
            allow_unused=True,
        )
        norm = torch.norm(
            torch.stack([torch.norm(g, 2.0) for g in coord_grads if g is not None]),
            2.0,
        )
        writer.add_scalar(f"GradNorm/{key}", norm, step)


def main(
    model_name="glyphogen.vectorizer.pt",
    epochs=EPOCHS,
    canary=None,
    debug_grads=False,
    load_model=False,
    segmentation_model="glyphogen.segmenter.pt",
):
    random.seed(1234)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision("high")

    torch.manual_seed(1234)

    # Model
    segmenter_state = torch.load(segmentation_model, map_location=device)
    model = VectorizationGenerator(
        segmenter_state=segmenter_state,
        d_model=D_MODEL,
        latent_dim=LATENT_DIM,
        rate=RATE,
    ).to(device)

    if load_model and os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name, map_location=device))
        print(f"Loaded model from {model_name}")

    # Data
    train_dataset, test_dataset = get_hierarchical_data()

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        drop_last=True,
        worker_init_fn=seed_worker,
        num_workers=0,
    )

    if canary is not None:
        print("Reducing dataset for canary testing")
        if canary == 1:
            # Find the first non-null batch and repeat it 32 times
            for batch in train_loader:
                if batch is not None:
                    canary_batch = batch
                    break
            train_loader = [canary_batch] * 32
            for batch in test_loader:
                if batch is not None:
                    canary_batch = batch
                    break
            test_loader = [canary_batch] * 16
            canary = 32

    train_batch_count = (
        len(train_loader) if canary is None else min(len(train_loader), canary)
    ) - 1  # Last batch is dropped

    test_batch_count = (
        len(test_loader) if canary is None else min(len(test_loader), canary // 3)
    ) - 1  # Last batch is dropped, simulate a 1/3 size validation set

    # Optimizer and Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=torch.tensor(LEARNING_RATE), 
        weight_decay=0.005
    )

    @torch.compile(fullgraph=False)
    def compiled_opt_step():
        optimizer.step()

    # Work out gamma from number of steps and start/end learning rate
    # final_learning_rate = LEARNING_RATE * (gamma ** steps)
    gamma = (FINAL_LEARNING_RATE / LEARNING_RATE) ** (1 / (train_batch_count * epochs))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    scheduler = WarmupLR(scheduler, 0, num_warmup = WARMUP_STEPS)

    # Note we are stepping the scheduler each batch, not each epoch

    # alpha_f = FINAL_LEARNING_RATE / LEARNING_RATE
    # scheduler = get_cosine_annealing_with_warmup(
    #    optimizer, WARMUP_STEPS, train_batch_count * epochs, alpha_f
    # )
    # Training Loop
    train_writer = SummaryWriter(
        f"logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}/train"
    )
    val_writer = SummaryWriter(
        f"logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}/val"
    )
    train_writer.add_text(
        "Hyperparameters", open("glyphogen/hyperparameters.py").read(), 0
    )
    best_val_metric = 0
    best_val_loss = 12345
    global_step = 0
    torch._dynamo.config.capture_scalar_outputs = True
    if debug_grads:
        torch._functorch.config.donated_buffer = False
    for epoch in range(epochs):
        # Calculate teacher forcing ratio for this epoch
        if epoch < SCHEDULED_SAMPLING_START_EPOCH:
            teacher_forcing_ratio = 1.0
        elif epoch > SCHEDULED_SAMPLING_END_EPOCH:
            teacher_forcing_ratio = SCHEDULED_SAMPLING_MIN_RATIO
        else:
            # Linear decay
            progress = (epoch - SCHEDULED_SAMPLING_START_EPOCH) / (
                SCHEDULED_SAMPLING_END_EPOCH - SCHEDULED_SAMPLING_START_EPOCH
            )
            teacher_forcing_ratio = 1.0 - progress * (
                1.0 - SCHEDULED_SAMPLING_MIN_RATIO
            )
        train_writer.add_scalar("Teacher Forcing Ratio", teacher_forcing_ratio, epoch)
        print(f"\nTeacher forcing ratio for this epoch: {teacher_forcing_ratio:.2f}")

        print()
        model.train()
        loss_accumulators = defaultdict(lambda: 0.0)
        i = 0
        kbar = pkbar.Kbar(
            target=train_batch_count,
            epoch=epoch,
            num_epochs=epochs,
            width=8,
            always_stateful=False,
        )
        for i, batch in enumerate(train_loader):
            if canary is not None and i >= canary:
                break
            optimizer.zero_grad()
            losses, _ = step(
                model,
                batch,
                teacher_forcing_ratio=teacher_forcing_ratio,
                writer=train_writer,
                global_step=global_step,
            )

            if debug_grads:
                write_gradient_norms(model, losses, train_writer, global_step)

            losses["total_loss"].backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            compiled_opt_step()
            for loss_key, loss_value in losses.items():
                loss_accumulators[loss_key] += loss_value.item()
            dump_accumulators(losses, train_writer, epoch, i, step=global_step)

            kbar.update(
                i,
                values=[
                    (label.replace("_loss", ""), losses[label].item())
                    for label in losses.keys()
                ],
            )

            global_step += 1
            scheduler.step()  # Step the scheduler each batch
            train_writer.add_scalar(
                "Learning Rate", scheduler.get_last_lr()[0], global_step
            )

        dump_accumulators(loss_accumulators, train_writer, epoch, i)
        train_writer.flush()

        ## VALIDATION ##

        if do_validation:
            model.eval()
            total_val_loss = 0
            loss_accumulators = defaultdict(lambda: 0.0)
            i = 0
            cm_state = init_confusion_matrix_state()

            kbar = pkbar.Kbar(
                target=test_batch_count,
                epoch=epoch,
                num_epochs=epochs,
                width=8,
                always_stateful=False,
            )
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    if canary is not None and i >= canary // 3:
                        break
                    losses, outputs = step(
                        model,
                        batch,
                        teacher_forcing_ratio=1.0,  # Always use teacher forcing for validation for now
                        writer=val_writer,
                        global_step=global_step,
                    )
                    for loss_key, loss_value in losses.items():
                        loss_accumulators[loss_key] += loss_value.item()
                    total_val_loss += losses["total_loss"].item()
                    if outputs:
                        collect_confusion_matrix_data(cm_state, outputs[0], batch)
                    kbar.update(
                        i,
                        values=[
                            (label.replace("_loss", ""), losses[label])
                            for label in losses.keys()
                        ],
                    )

            avg_val_loss = total_val_loss / (0.1 + i)
            avg_val_metric = loss_accumulators["raster_metric"] / (0.1 + i)
            dump_accumulators(loss_accumulators, val_writer, epoch, i)
            print(
                f"Epoch {epoch}, Validation Loss: {avg_val_loss}; Metric: {avg_val_metric}"
            )

            # Checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), model_name)
                print(f"Saved best model to {model_name}")

            # Callbacks
            log_bounding_boxes(model, test_loader, val_writer, epoch)
            log_vectorizer_outputs(model, test_loader, val_writer, epoch)
            log_confusion_matrix(cm_state, val_writer, epoch)

        # Log the learning rate
        val_writer.flush()

    train_writer.close()
    val_writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the VectorizationGenerator model in PyTorch."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="glyphogen.vectorizer.pt",
        help="Name of the model to save.",
    )
    parser.add_argument(
        "--segmentation-model",
        type=str,
        default="glyphogen.segmenter.pt",
        help="Name of the segmentation model to load.",
    )
    parser.add_argument(
        "--load_model",
        action="store_true",
        help="Whether to load a pre-existing model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of epochs to train the model.",
    )
    parser.add_argument(
        "--canary",
        type=int,
        help="Take a slice of the dataset for canary testing.",
    )
    parser.add_argument(
        "--debug-grads",
        action="store_true",
        help="Whether to log gradient norms for debugging.",
    )
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        epochs=args.epochs,
        debug_grads=args.debug_grads,
        canary=args.canary,
        load_model=args.load_model,
        segmentation_model=args.segmentation_model,
    )
