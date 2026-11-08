"""Restore one preserved split from an explicitly supplied original dataset."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from cli_support import directory, existing_file, fresh_path, run, write_json
from oversample import IMAGE_EXTENSIONS


def restore(manifest_path, source_pool, output):
    manifest_path = existing_file(manifest_path)
    source_pool = directory(source_pool)
    output = fresh_path(output, (source_pool, manifest_path))
    spec = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if set(spec.get("splits", {})) != {"train", "val", "test"}:
        raise ValueError("Manifest must contain train, val and test splits.")
    images, labels = {}, {}
    for path in sorted(source_pool.glob("images/*/*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in images:
            raise ValueError(f"Ambiguous image stem in source pool: {path.stem}")
        images[path.stem] = path
    for path in sorted(source_pool.glob("labels/*/*.txt")):
        if path.stem in labels:
            raise ValueError(f"Ambiguous label stem in source pool: {path.stem}")
        labels[path.stem] = path

    seen = set()
    for part, info in spec["splits"].items():
        names = info["files"]
        if len(names) != info["count"]:
            raise ValueError(f"Recorded count does not match {part} file list.")
        for name in names:
            if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name or name in {".", ".."}:
                raise ValueError("Manifest file identifiers must be plain basenames.")
            if name in seen:
                raise ValueError(f"Repeated file across split lists: {name}")
            seen.add(name)
            if name not in images or name not in labels:
                raise ValueError(f"Source image/label pair missing: {name}")
        digest = hashlib.sha256("|".join(sorted(names)).encode()).hexdigest()[:16]
        if digest != spec["split_hash"][part]:
            raise ValueError(f"Manifest filename hash mismatch: {part}")

    output.mkdir(parents=True, exist_ok=False)
    counts = {}
    for part, info in spec["splits"].items():
        image_dir, label_dir = output / "images" / part, output / "labels" / part
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for name in info["files"]:
            shutil.copy2(images[name], image_dir / images[name].name)
            shutil.copy2(labels[name], label_dir / labels[name].name)
        actual = sorted(path.stem for path in label_dir.glob("*.txt"))
        digest = hashlib.sha256("|".join(actual).encode()).hexdigest()[:16]
        if digest != spec["split_hash"][part]:
            raise ValueError(f"Restored filename hash mismatch: {part}")
        counts[part] = len(actual)
    write_json(output / "restoration_manifest.json", {
        "manifest": str(manifest_path),
        "source_pool": str(source_pool),
        "counts": counts,
        "hash_type": "SHA-256 of sorted filename stems, truncated to 16 hex digits",
        "content_hash_verified": False,
    })
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-pool", required=True, help="Dataset with images/<split>/ and labels/<split>/")
    parser.add_argument("--output", required=True, help="New dataset directory outside source pool")
    args = parser.parse_args(argv)
    print(json.dumps(restore(args.manifest, args.source_pool, args.output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
