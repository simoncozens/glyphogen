import sys
from pathlib import Path
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# This allows us to import engine, utils, etc.
sys.path.append("vision/references/detection/")

import engine  # torchvision.vision.references.detection.engine
import utils  # etc.
import torchvision.transforms.v2 as T
from glyphogen.dataset import GlyphSqliteDataset, get_hierarchical_data


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights="MaskRCNN_ResNet50_FPN_Weights.DEFAULT"
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def get_transform(train):
    transforms = []
    # if train:
    # A simple training transform
    # transforms.append(T.RandomHorizontalFlip(0.5))

    # Use newer v2 transforms consistent with dataset.py
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    return T.Compose(transforms)


def collate_fn(batch):
    """
    Custom collate_fn to adapt GlyphSqliteDataset/GlyphCocoDataset structure
    to what MaskRCNN expects.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return tuple(), tuple()

    images = []
    targets = []
    for img, target in batch:
        images.append(img)

        gt_contours = target["gt_contours"]

        if len(gt_contours) == 0:
            # MaskRCNN expects tensors even if empty
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            # Get H, W from image
            _, h, w = img.shape
            masks = torch.empty((0, h, w), dtype=torch.uint8)
        else:
            boxes = torch.stack([t["box"] for t in gt_contours])
            labels = torch.stack([t["label"] for t in gt_contours])
            masks = torch.stack([t["mask"] for t in gt_contours])

        new_target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([target["image_id"]]),
        }
        targets.append(new_target)

    return tuple(images), tuple(targets)


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    num_classes = 3  # 0: background, 1: outer, 2: hole

    # Pass our transform to get_hierarchical_data
    dataset, dataset_test = get_hierarchical_data(
        train_transform=get_transform(train=True),
        test_transform=get_transform(train=False),
    )

    # DataLoaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        engine.train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq=10
        )
        lr_scheduler.step()
        engine.evaluate(model, data_loader_test, device=device)

    print("That's it! Saving model.")
    torch.save(model.state_dict(), "glyphogen.segmenter.pt")


if __name__ == "__main__":
    main()
