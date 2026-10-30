"""
Prepare the YOLO training sample from a Roboflow PKLot export (COCO format).

Pipeline: gather annotations from every raw folder -> draw a random sample -> bundle frames
by capture date -> hand whole dates to train / val / test -> write YOLO labels and data.yaml.

Run from the project root, where the downloaded train/ valid/ test/ folders live:
    python prepare_data.py
"""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

RAW_SPLITS = ("train", "valid", "test")
CATEGORY_REMAP = {1: 0, 2: 1}                  # COCO id 1 -> bay-free, 2 -> bay-taken
CLASSES = ["bay-free", "bay-taken"]
SPLIT_RATIO = {"train": 0.75, "val": 0.15, "test": 0.10}
COCO_JSON_NAMES = ("_annotations.coco.json", "_annotations_coco.json")


def load_coco_folder(folder: Path) -> list:
    """Return a list of (image_path, width, height, [(class_id, [x, y, w, h]), ...])."""
    json_path = next((folder / n for n in COCO_JSON_NAMES if (folder / n).is_file()), None)
    if json_path is None:
        print(f"[warn] {folder} has no annotation file, skipping")
        return []
    data = json.loads(json_path.read_text())
    boxes_by_image = defaultdict(list)
    for a in data["annotations"]:
        if a["category_id"] in CATEGORY_REMAP:
            boxes_by_image[a["image_id"]].append((CATEGORY_REMAP[a["category_id"]], a["bbox"]))
    records = []
    for meta in data["images"]:
        file = folder / meta["file_name"]
        if file.is_file() and boxes_by_image.get(meta["id"]):
            records.append((file, meta["width"], meta["height"], boxes_by_image[meta["id"]]))
    return records


def assign_by_date(records: list, rng: random.Random) -> dict:
    """Place every frame of a capture date in one split so near-duplicates never straddle splits."""
    by_date = defaultdict(list)
    for rec in records:
        by_date[rec[0].name[:10]].append(rec)            # file names begin with YYYY-MM-DD
    dates = sorted(by_date)
    rng.shuffle(dates)
    splits = {key: [] for key in SPLIT_RATIO}
    for date in dates:
        gap = {k: SPLIT_RATIO[k] * len(records) - len(splits[k]) for k in splits}
        splits[max(gap, key=gap.get)].extend(by_date[date])
    return splits


def yolo_row(class_id: int, box: list, img_w: int, img_h: int) -> str:
    x, y, w, h = box
    cx, cy = (x + w / 2) / img_w, (y + h / 2) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def export_split(split: str, records: list, dest: Path) -> list:
    images_out, labels_out = dest / "images" / split, dest / "labels" / split
    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)
    class_counts = [0] * len(CLASSES)
    for file, img_w, img_h, boxes in records:
        shutil.copy(file, images_out / file.name)
        rows = [yolo_row(cid, box, img_w, img_h) for cid, box in boxes]
        (labels_out / f"{file.stem}.txt").write_text("\n".join(rows))
        for cid, _ in boxes:
            class_counts[cid] += 1
    return class_counts


def main() -> None:
    project_root = Path(__file__).resolve().parents[0]
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--raw-dir", type=Path, default=project_root / "data", help="folder holding train/ valid/ test/")
    cli.add_argument("--out-dir", type=Path, default=project_root / "bay_data")
    cli.add_argument("--sample-size", type=int, default=1500, help="number of images to keep")
    cli.add_argument("--seed", type=int, default=7)
    opts = cli.parse_args()

    rng = random.Random(opts.seed)
    everything = [r for name in RAW_SPLITS for r in load_coco_folder(opts.raw_dir / name)]
    print(f"Annotated images available: {len(everything)}")
    rng.shuffle(everything)
    chosen = everything[: opts.sample_size]

    splits = assign_by_date(chosen, rng)
    if opts.out_dir.exists():
        shutil.rmtree(opts.out_dir)
    print(f"\n{'split':<7}{'images':>8}{'free':>9}{'taken':>9}")
    for split, recs in splits.items():
        free, taken = export_split(split, recs, opts.out_dir)
        print(f"{split:<7}{len(recs):>8}{free:>9}{taken:>9}")

    (opts.out_dir / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CLASSES)}\nnames: {CLASSES}\n"
    )
    bundle = shutil.make_archive(str(opts.out_dir), "zip", opts.out_dir)
    print(f"\nDataset folder: {opts.out_dir}\nZip for Colab:  {bundle}")


if __name__ == "__main__":
    main()
