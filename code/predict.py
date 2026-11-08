"""Save base class-aware detections with the paper's inference settings."""

import argparse

from cli_support import directory, existing_file, fresh_path, run
from settings import INFERENCE


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Existing local best.pt file")
    parser.add_argument("--images", required=True, help="Validation or Test image directory")
    parser.add_argument("--output", required=True, help="New prediction directory")
    args = parser.parse_args(argv)
    weights = existing_file(args.weights)
    images = directory(args.images)
    output = fresh_path(args.output, (images, weights))

    from ultralytics import YOLO

    output.mkdir(parents=True, exist_ok=False)
    YOLO(str(weights)).predict(
        source=str(images),
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(output.parent),
        name=output.name,
        exist_ok=True,
        verbose=False,
        agnostic_nms=False,
        **INFERENCE,
    )
    (output / "labels").mkdir(exist_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
