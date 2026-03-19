from collections import Counter, defaultdict
import io
import json
import random
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import zlib

import numpy as np
from pycocotools import mask as mask_util
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from fontTools.ttLib import TTFont
import torch
import torch.nn.functional as F

from glyphogen.representations.model import MODEL_REPRESENTATION
from glyphogen.dataset import font_files
from glyphogen.glyph import Glyph
from glyphogen.hyperparameters import ALPHABET, GEN_IMAGE_SIZE
from glyphogen.svgglyph import SVGGlyph
from glyphogen.nodeglyph import NodeGlyph
import uharfbuzz as hb

LIMIT = 300_000
NORMALIZED_MASK_SIZE = 512


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


def _encode_array(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return zlib.compress(buffer.getvalue(), level=3)


def _init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            font_path TEXT NOT NULL,
            character TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS masks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mask TEXT NOT NULL UNIQUE
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY,
            image_id INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            area REAL NOT NULL,
            bbox_x REAL NOT NULL,
            bbox_y REAL NOT NULL,
            bbox_w REAL NOT NULL,
            bbox_h REAL NOT NULL,
            iscrowd INTEGER NOT NULL,
            sequence BLOB NOT NULL,
            mask_id INTEGER NOT NULL,
            x_aligned_point_indices TEXT NOT NULL,
            y_aligned_point_indices TEXT NOT NULL,
            FOREIGN KEY(image_id) REFERENCES images(id),
            FOREIGN KEY(mask_id) REFERENCES masks(id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_annotations_image_id ON annotations(image_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_annotations_mask_id ON annotations(mask_id)"
    )
    conn.commit()
    return conn


def process_glyph_data(
    glyph_list,
    image_dir: Path,
    db_conn: sqlite3.Connection,
    stats_groups: Optional[Dict[str, List[float]]] = None,
    command_histogram: Optional[Counter] = None,
    start_img_id: int = 0,
    start_ann_id: int = 0,
    train: bool = True,
):
    img_id = start_img_id
    ann_id = start_ann_id
    cursor = db_conn.cursor()
    insert_count = 0
    processed_count = 0
    limit = LIMIT
    if not train and limit > 0:
        limit = 1 / 4 * limit

    for font_path, gid, face in tqdm(glyph_list, desc="Processing glyphs"):
        if limit > 0 and processed_count >= limit:
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
                #print(f"  Skipping glyph {glyph} due to encoding failure.")
                continue

            # Now get segmentation data, which should be in the same order
            svg_glyph = SVGGlyph.from_node_glyph(node_glyph)
            segmentation_data = svg_glyph.get_segmentation_data()

            if not segmentation_data or len(segmentation_data) != len(
                contour_sequences
            ):
                # Mismatch between number of contours in segmentation and vectorization
                #print("  Skipping glyph due to segmentation/vectorization mismatch.")
                continue

            img_filename = f"{pth.stem}/{glyph_name}.png"
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

            cursor.execute(
                """
                INSERT INTO images (id, width, height, file_name, font_path, character)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    img_id,
                    GEN_IMAGE_SIZE[1],
                    GEN_IMAGE_SIZE[0],
                    img_filename,
                    str(font_path),
                    glyph_name,
                ),
            )

            # Create annotations for this image
            for i, seg_item in enumerate(segmentation_data):
                bbox = seg_item["bbox"]
                x, y, x2, y2 = bbox
                w = x2 - x
                h = y2 - y

                mask = seg_item["mask"]
                rle = mask_util.encode(np.asfortranarray(mask))
                rle_json = json.dumps(
                    {"size": rle["size"], "counts": rle["counts"].decode("utf-8")}
                )

                sequence_tensor = contour_sequences[i]
                if not torch.is_tensor(sequence_tensor):
                    sequence_tensor = torch.tensor(sequence_tensor, dtype=torch.float32)

                sequence_blob = _encode_array(
                    sequence_tensor.cpu().numpy().astype(np.float32)
                )
                x_alignments_json = json.dumps(
                    alignment_points[i]["x_aligned_point_indices"]
                )
                y_alignments_json = json.dumps(
                    alignment_points[i]["y_aligned_point_indices"]
                )

                # Insert or get existing mask ID
                mask_id = cursor.execute(
                    """
                    INSERT INTO masks (mask) VALUES (?)
                    ON CONFLICT (mask) DO UPDATE SET mask = mask
                    RETURNING id
                    """,
                    (rle_json,),
                ).fetchone()[0]

                cursor.execute(
                    """
                    INSERT INTO annotations (
                        id, image_id, category_id, area, bbox_x, bbox_y, bbox_w, bbox_h,
                        iscrowd, sequence, mask_id, x_aligned_point_indices,
                        y_aligned_point_indices
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ann_id,
                        img_id,
                        seg_item["label"] + 1,
                        float(w * h),
                        float(x),
                        float(y),
                        float(w),
                        float(h),
                        0,
                        sqlite3.Binary(sequence_blob),
                        mask_id,
                        x_alignments_json,
                        y_alignments_json,
                    ),
                )

                if stats_groups is not None:
                    box_tensor = torch.tensor([x, y, x2, y2], dtype=torch.float32)
                    sequence_norm = MODEL_REPRESENTATION.image_space_to_mask_space(
                        sequence_tensor, box_tensor
                    )
                    commands, coords = MODEL_REPRESENTATION.split_tensor(sequence_norm)
                    command_idxs = torch.argmax(commands, dim=-1)

                    for j in range(len(command_idxs)):
                        cmd_idx = int(command_idxs[j].item())
                        command = MODEL_REPRESENTATION.decode_command(cmd_idx)
                        if command_histogram is not None:
                            command_histogram[command] += 1
                        MODEL_REPRESENTATION.update_stats_dict_with_command(
                            stats_groups, command, coords[j]
                        )

                ann_id += 1
                insert_count += 1
                if insert_count % 1000 == 0:
                    db_conn.commit()

            img_id += 1
            processed_count += 1

        except Exception as e:
            pass
            #print(f"Could not process from {font_path}: {e}")

    db_conn.commit()
    return img_id, ann_id


def main():
    DATA_DIR = Path("data")
    TRAIN_IMG_DIR = DATA_DIR / "images_hierarchical" / "train"
    TEST_IMG_DIR = DATA_DIR / "images_hierarchical" / "test"
    TRAIN_DB = DATA_DIR / "train_hierarchical.sqlite"
    TEST_DB = DATA_DIR / "test_hierarchical.sqlite"

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

    stats_groups = MODEL_REPRESENTATION.get_initial_stats_dict()
    command_histogram = Counter()

    train_conn = _init_db(TRAIN_DB)
    train_img_id, train_ann_id = process_glyph_data(
        train_glyphs,
        TRAIN_IMG_DIR,
        train_conn,
        stats_groups=stats_groups,
        command_histogram=command_histogram,
        start_img_id=0,
        start_ann_id=0,
        train=True,
    )
    train_conn.close()

    test_conn = _init_db(TEST_DB)
    process_glyph_data(
        test_glyphs,
        TEST_IMG_DIR,
        test_conn,
        start_img_id=train_img_id,
        start_ann_id=train_ann_id,
    )
    test_conn.close()

    print("\n--- Coordinate Statistics (Mask Space) ---")
    final_stats = {}
    for name, values in stats_groups.items():
        if not values:
            print(f"{name}: No values found.")
            continue
        tensor_values = torch.tensor(values, dtype=torch.float32)
        mean = torch.mean(tensor_values)
        std = torch.std(tensor_values)
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

    output_file = DATA_DIR / "coord_stats.pt"
    torch.save(final_stats, output_file)
    print(f"\nStatistics saved to {output_file}")
    histogram_file = DATA_DIR / "command_stats.pt"
    torch.save(command_histogram, histogram_file)

    print("\nDone.")


if __name__ == "__main__":
    main()
