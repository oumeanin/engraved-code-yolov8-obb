"""Path and input checks shared by the command-line adapters."""

import json
from pathlib import Path
import sys

DEFAULT_CLASSES = Path(__file__).resolve().parents[1] / "configs" / "classes.json"


def directory(value):
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Directory not found: {path}")
    return path


def existing_file(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"File not found: {path}")
    return path


def fresh_path(value, protected=()):
    path = Path(value).expanduser().resolve()
    if path.exists():
        raise ValueError(f"Output already exists: {path}")
    for item in protected:
        source = Path(item).resolve()
        if path == source or path in source.parents or source in path.parents:
            raise ValueError(f"Output overlaps an input: {path} and {source}")
    return path


def classes(value=DEFAULT_CLASSES):
    path = existing_file(value)
    names = json.loads(path.read_text(encoding="utf-8-sig"))
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(names) != len(set(names))):
        raise ValueError("Classes must be a nonempty JSON list of unique strings.")
    return {index: name for index, name in enumerate(names)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run(main):
    try:
        return main()
    except (OSError, ValueError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
