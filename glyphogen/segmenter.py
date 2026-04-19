from dataclasses import dataclass
from typing import List

from jaxtyping import Float
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN

from glyphogen.hyperparameters import GEN_IMAGE_SIZE
from glyphogen.typing import SegmenterOutput as MaskRCNNOutput


@dataclass
class SegmenterOutput:
    box: Float[torch.Tensor, "4"]
    label: int
    raw_mask: Float[torch.Tensor, "height width"]

    def cropped_mask(self) -> torch.Tensor:
        """Crop the raw mask to the bounding box."""
        x_min, y_min, x_max, y_max = self.box.long()
        return self.raw_mask[y_min:y_max, x_min:x_max]

    def normalized_mask(self) -> torch.Tensor:
        """Normalize the cropped mask to GEN_IMAGE_SIZE as [1, H, W]."""
        return F.interpolate(
            self.cropped_mask().unsqueeze(0).unsqueeze(0),
            size=GEN_IMAGE_SIZE,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)


def get_model_instance_segmentation(
    num_classes: int, load_pretrained: bool = True
) -> MaskRCNN:
    if load_pretrained:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights="MaskRCNN_ResNet50_FPN_Weights.DEFAULT"
        )
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels  # type: ignore
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def segment_single_image(
    segmenter: MaskRCNN, raster_image: torch.Tensor, score_threshold: float = 0.0
) -> List[SegmenterOutput]:
    """Return sorted contour masks for a single raster image tensor [C, H, W]."""
    with torch.no_grad():
        raw_output: MaskRCNNOutput = segmenter(raster_image.unsqueeze(0))[0]  # type: ignore

    if not raw_output["masks"].numel():
        return []

    indices = torch.arange(raw_output["boxes"].shape[0], device=raster_image.device)
    if "scores" in raw_output:
        indices = indices[raw_output["scores"] >= score_threshold]

    if indices.numel() == 0:
        return []

    boxes = raw_output["boxes"][indices]
    masks = raw_output["masks"][indices].squeeze(1)
    labels = raw_output["labels"][indices]

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_indices = torch.argsort(areas, descending=True)

    image_h, image_w = raster_image.shape[-2:]
    contour_outputs: List[SegmenterOutput] = []
    for idx in sorted_indices:
        box = boxes[idx].clone()
        box[0::2] = box[0::2].clamp(min=0, max=image_w - 1)
        box[1::2] = box[1::2].clamp(min=0, max=image_h - 1)

        x_min, y_min, x_max, y_max = box.long()
        if x_min >= x_max or y_min >= y_max:
            continue

        contour_outputs.append(
            SegmenterOutput(
                box=box,
                label=int(labels[idx].item()),
                raw_mask=masks[idx].to(torch.float32),
            )
        )

    return contour_outputs
