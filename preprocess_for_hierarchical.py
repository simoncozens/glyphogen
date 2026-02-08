from collections import defaultdict
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from pycocotools import mask as mask_util
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from fontTools.ttLib import TTFont

from glyphogen.representations.model import MODEL_REPRESENTATION
from glyphogen.dataset import font_files
from glyphogen.glyph import Glyph
from glyphogen.hyperparameters import ALPHABET, GEN_IMAGE_SIZE
from glyphogen.svgglyph import SVGGlyph
from glyphogen.nodeglyph import NodeGlyph
import uharfbuzz as hb

LIMIT = 500000


def get_alignments(node_glyph: "NodeGlyph") -> List[Dict]:
    """
    Analyzes the NodeGlyph to find aligned points within each contour.
    The idea is similar to font hinting: if multiple points share the same
    x or y coordinate (within a small quantum), they are considered aligned,
    and we want to encourage the model to maintain this alignment.
    Points are considered aligned if their coordinates round to the same
    value within a certain quantum.
    """

    def round_coord(coord, quantum=3):
        return round(coord) // quantum * quantum

    rv = []

    for contour in node_glyph.contours:
        x_coords = defaultdict(list)
        y_coords = defaultdict(list)
        for ix, node in enumerate(contour.nodes):
            x_coords[round_coord(node.coordinates[0])].append(ix)
            y_coords[round_coord(node.coordinates[1])].append(ix)
        x_alignments = []
        y_alignments = []

        for indices in x_coords.values():
            if len(indices) > 1:
                x_alignments.append(indices)
        for indices in y_coords.values():
            if len(indices) > 1:
                y_alignments.append(indices)
        rv.append(
            {
                "x_aligned_point_indices": x_alignments,
                "y_aligned_point_indices": y_alignments,
            }
        )
    return rv


def process_glyph_data(glyph_list, image_dir, start_img_id=0, start_ann_id=0):
    images_json = []
    annotations_json = []

    img_id = start_img_id
    ann_id = start_ann_id

    for font_path, gid, face in tqdm(glyph_list, desc="Processing glyphs"):
        if LIMIT > 0 and len(images_json) >= LIMIT:
            break
        try:
            pth = Path(font_path)
            glyph = Glyph(pth, gid, face, location={})
            glyph_name = glyph.name
            # print(f"Processing glyph '{gid}' from {font_path}")

            # Generate vector data first to ensure it's valid
            node_glyph = glyph.vectorize().to_node_glyph()
            node_glyph.normalize()  # IMPORTANT: ensure canonical order

            # Have to do this on the node glyph *after normalization*
            # as we're going to be using the point indices later
            # for alignment supervision
            alignment_points = get_alignments(node_glyph)

            contour_sequences = node_glyph.encode(MODEL_REPRESENTATION)
            if contour_sequences is None:
                print(f"  Skipping glyph {glyph} due to encoding failure.")
                continue

            # Now get segmentation data, which should be in the same order
            svg_glyph = SVGGlyph.from_node_glyph(node_glyph)
            segmentation_data = svg_glyph.get_segmentation_data()

            if not segmentation_data or len(segmentation_data) != len(
                contour_sequences
            ):
                # Mismatch between number of contours in segmentation and vectorization
                print("  Skipping glyph due to segmentation/vectorization mismatch.")
                continue

            img_filename = f"{pth.stem}-{glyph_name}.png"
            img_path = image_dir / img_filename
            if not img_path.parent.exists():
                img_path.parent.mkdir(parents=True, exist_ok=True)
            if not img_path.exists():
                # Generate and save raster image
                raster_img = glyph.rasterize(GEN_IMAGE_SIZE[0])
                if np.sum(raster_img) < 0.01:  # Skip blank images
                    print(f"Skipping blank image for {glyph_name} in {font_path}")
                    continue

                from PIL import Image

                pil_img = Image.fromarray(
                    (raster_img.squeeze(-1) * 255).astype(np.uint8), mode="L"
                )
                pil_img.save(img_path)

            images_json.append(
                {
                    "id": img_id,
                    "width": GEN_IMAGE_SIZE[1],
                    "height": GEN_IMAGE_SIZE[0],
                    "file_name": img_filename,
                    "font_path": str(font_path),
                    "character": glyph_name,
                }
            )

            # Create annotations for this image
            for i, seg_item in enumerate(segmentation_data):
                bbox = seg_item["bbox"]
                x, y, x2, y2 = bbox
                w = x2 - x
                h = y2 - y
                rle = mask_util.encode(np.asfortranarray(seg_item["mask"]))
                rle["counts"] = rle["counts"].decode("utf-8")  # type: ignore

                annotations_json.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": seg_item["label"] + 1,
                        "segmentation": rle,
                        "area": float(w * h),
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "iscrowd": 0,
                        "sequence": contour_sequences[i].tolist(),
                        "x_aligned_point_indices": alignment_points[i][
                            "x_aligned_point_indices"
                        ],
                        "y_aligned_point_indices": alignment_points[i][
                            "y_aligned_point_indices"
                        ],
                    }
                )
                ann_id += 1

            img_id += 1

        except Exception as e:
            print(f"Could not process from {font_path}: {e}")

    return images_json, annotations_json


def main():
    DATA_DIR = Path("data")
    TRAIN_IMG_DIR = DATA_DIR / "images_hierarchical" / "train"
    TEST_IMG_DIR = DATA_DIR / "images_hierarchical" / "test"

    TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
    TEST_IMG_DIR.mkdir(parents=True, exist_ok=True)

    all_glyphs = []
    for font in font_files:
        blob = hb.Blob.from_file_path(font)  # type: ignore
        face = hb.Face(blob)  # type: ignore
        gids = face.glyph_count
        for gid in range(1, gids + 1):
            all_glyphs.append((font, gid, face))

    random.seed(42)
    random.shuffle(all_glyphs)

    train_glyphs, test_glyphs = train_test_split(
        all_glyphs, test_size=0.2, random_state=42
    )

    print(
        f"Processing {len(train_glyphs)} training glyphs and {len(test_glyphs)} test glyphs."
    )

    categories_json = [
        {"id": 1, "name": "outer", "supercategory": "contour"},
        {"id": 2, "name": "hole", "supercategory": "contour"},
    ]

    train_images, train_annotations = process_glyph_data(
        train_glyphs, TRAIN_IMG_DIR, start_img_id=0, start_ann_id=0
    )
    train_coco_json = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories_json,
    }

    test_images, test_annotations = process_glyph_data(
        test_glyphs,
        TEST_IMG_DIR,
        start_img_id=116472,
        start_ann_id=len(train_annotations),
    )
    test_coco_json = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": categories_json,
    }

    print("\nSaving COCO JSON files...")
    with open(DATA_DIR / "train_hierarchical.json", "w") as f:
        json.dump(train_coco_json, f)

    with open(DATA_DIR / "test_hierarchical.json", "w") as f:
        json.dump(test_coco_json, f)

    print("\nDone.")


if __name__ == "__main__":
    main()
