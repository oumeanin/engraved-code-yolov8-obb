"""Copy a dataset and repeat Train image/label pairs by fixed class multipliers."""

import argparse
from pathlib import Path
import shutil

from cli_support import DEFAULT_CLASSES, classes, directory, fresh_path, run, write_json
from settings import CLASS_MULTIPLIERS

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def replicate(source, output, names):
    source = directory(source)
    output = fresh_path(output, (source,))
    images = directory(source / "images" / "train")
    labels = directory(source / "labels" / "train")
    plans = []
    existing_stems = {path.stem for path in images.iterdir() if path.is_file()}
    for label in sorted(labels.glob("*.txt")):
        matches = [path for path in images.glob(label.stem + ".*") if path.suffix.lower() in IMAGE_EXTENSIONS]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one image for {label.name}; found {len(matches)}")
        image = matches[0]
        class_ids = set()
        for line in label.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                class_id = int(float(line.split()[0]))
                if class_id not in names:
                    raise ValueError(f"Unknown class ID {class_id} in {label}")
                class_ids.add(class_id)
        multiplier = 1 if "_aug" in label.name else max(
            (CLASS_MULTIPLIERS.get(names[index], 1) for index in class_ids), default=1
        )
        for index in range(1, multiplier):
            if f"{image.stem}_aug{index}" in existing_stems:
                raise ValueError(f"Generated filename would collide for {image.name}")
        plans.append((image, label, multiplier))
    if not plans:
        raise ValueError("No Train label files found.")
    image_stems = {path.stem for path in images.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    if image_stems != {label.stem for _, label, _ in plans}:
        raise ValueError("Train image and label stems do not match.")

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output)
    for image, label, multiplier in plans:
        for index in range(1, multiplier):
            stem = f"{image.stem}_aug{index}"
            shutil.copy2(image, output / "images" / "train" / (stem + image.suffix))
            shutil.copy2(label, output / "labels" / "train" / (stem + ".txt"))
    summary = {
        "source": str(source),
        "original_train_images": len(plans),
        "output_train_images": sum(multiplier for _, _, multiplier in plans),
        "original_included_in_multiplier": True,
        "image_transforms_applied": False,
        "class_multipliers": CLASS_MULTIPLIERS,
        "images": [{"file": image.name, "multiplier": multiplier} for image, _, multiplier in plans],
    }
    write_json(output / "oversampling_manifest.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Original dataset with images/ and labels/")
    parser.add_argument("--output", required=True, help="New dataset directory outside source")
    parser.add_argument("--classes", default=str(DEFAULT_CLASSES))
    args = parser.parse_args(argv)
    summary = replicate(args.source, args.output, classes(args.classes))
    print(f"Train images: {summary['original_train_images']} -> {summary['output_train_images']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
