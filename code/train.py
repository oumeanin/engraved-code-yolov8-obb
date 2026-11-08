"""Train one final-paper condition without changing the source dataset."""

import argparse
from pathlib import Path

from cli_support import DEFAULT_CLASSES, classes, directory, existing_file, fresh_path, run
from settings import AUG_OFF, AUG_ON, COMMON, EXPERIMENTS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--data", required=True, help="YOLO dataset YAML")
    parser.add_argument("--weights", required=True, help="Existing DOTA-pretrained local .pt file")
    parser.add_argument("--output", required=True, help="New run directory")
    args = parser.parse_args(argv)

    data_file = existing_file(args.data)
    weights = existing_file(args.weights)
    import yaml

    dataset = yaml.safe_load(data_file.read_text(encoding="utf-8-sig"))
    if not isinstance(dataset, dict):
        raise ValueError("Dataset YAML must contain a mapping.")
    root = Path(dataset.get("path", ".")).expanduser()
    if not root.is_absolute():
        root = data_file.parent / root
    root = directory(root)
    names = list(classes(DEFAULT_CLASSES).values())
    supplied_names = dataset.get("names", names)
    if isinstance(supplied_names, dict):
        supplied_names = [supplied_names[index] for index in range(len(supplied_names))]
    if supplied_names != names or dataset.get("nc", len(names)) != len(names):
        raise ValueError("Dataset class order differs from configs/classes.json.")
    split_directories = []
    for part in ("train", "val", "test"):
        if not isinstance(dataset.get(part), str):
            raise ValueError(f"Dataset must specify a string path for {part}.")
        split_directories.append(directory(root / dataset[part]))
    output = fresh_path(args.output, (root, data_file, weights, *split_directories))

    from ultralytics import YOLO

    output.mkdir(parents=True, exist_ok=False)
    resolved = dict(dataset, path=str(root), names=names, nc=len(names))
    resolved_file = output / "dataset.resolved.yaml"
    with resolved_file.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(resolved, stream, allow_unicode=True, sort_keys=False)
    condition = EXPERIMENTS[args.experiment]
    options = dict(COMMON)
    options.update(AUG_ON if condition["augmentation"] else AUG_OFF)
    YOLO(str(weights)).train(
        data=str(resolved_file),
        project=str(output.parent),
        name=output.name,
        exist_ok=True,
        **options,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
