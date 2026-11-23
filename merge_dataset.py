
from pathlib import Path
import shutil
import yaml

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "dataset"

DATASETS = {
    "person": ROOT / "person",
    "apple": ROOT / "apple",
    "vehicle": ROOT / "vehicle",
}

TARGET_NAMES = ["person", "apple", "vehicle"]

ALIASES = {
    "person": "person",
    "apple": "apple",
    "vehicle": "vehicle",
    "motorcycle": "vehicle"
}

SPLITS = {
    "train": ["train"],
    "val": ["valid", "val"],
    "test": ["test"],
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_class_map(dataset_dir):
    with open(dataset_dir / "data.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names = config["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    else:
        names = {int(k): v for k, v in names.items()}

    mapping = {}
    for old_id, name in names.items():
        normalized = str(name).strip().lower()
        target = ALIASES.get(normalized)

        if target in TARGET_NAMES:
            mapping[old_id] = TARGET_NAMES.index(target)

    return mapping, names

def merge():
    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} 폴더가 이미 존재합니다. "
            "기존 결과를 백업하거나 삭제한 후 실행하세요."
        )

    counts = {s: 0 for s in SPLITS}
    skipped = 0

    for source_name, dataset_dir in DATASETS.items():
        mapping, names = load_class_map(dataset_dir)
        print(f"\n[{source_name}] 원본 클래스: {names}")
        print(f"변환 번호: {mapping}")

        if not mapping:
            raise ValueError(
                f"{source_name}: 일치하는 클래스가 없습니다."
            )

        for split, candidates in SPLITS.items():
            source_split = next(
                (dataset_dir / c for c in candidates
                 if (dataset_dir / c / "images").is_dir()),
                None
            )

            if source_split is None:
                print(f"  {split}: 폴더 없음, 건너뜀")
                continue

            image_dir = source_split / "images"
            label_dir = source_split / "labels"

            dest_images = OUTPUT / "images" / split
            dest_labels = OUTPUT / "labels" / split
            dest_images.mkdir(parents=True, exist_ok=True)
            dest_labels.mkdir(parents=True, exist_ok=True)

            for image in sorted(image_dir.rglob("*")):
                if not image.is_file() or image.suffix.lower() not in IMAGE_EXTS:
                    continue

                relative = image.relative_to(image_dir)
                label = label_dir / relative.with_suffix(".txt")

                if not label.exists():
                    skipped += 1
                    continue

                new_lines = []

                
                
                for line in label.read_text(
                    encoding="utf-8-sig"
                ).splitlines():

                    parts = line.strip().split()

                    if not parts:
                        continue

                    old_id = int(parts[0])

                    if old_id not in mapping:
                        continue

                    new_id = mapping[old_id]

                    if len(parts) == 5:
                        # Object Detection
                        values = [float(v) for v in parts[1:]]

                    elif len(parts) >= 7 and len(parts) % 2 == 1:
                        # Segmentation -> Bounding Box
                        coords = [float(v) for v in parts[1:]]

                        xs = coords[0::2]
                        ys = coords[1::2]

                        xmin = min(xs)
                        xmax = max(xs)
                        ymin = min(ys)
                        ymax = max(ys)

                        values = [
                            (xmin + xmax) / 2,
                            (ymin + ymax) / 2,
                            xmax - xmin,
                            ymax - ymin
                        ]

                    else:
                        raise ValueError(
                            f"지원하지 않는 라벨 형식: {label}"
                        )

                    if (
                        any(not 0 <= v <= 1 for v in values)
                        or values[2] <= 0
                        or values[3] <= 0
                    ):
                        raise ValueError(
                            f"잘못된 좌표: {label}"
                        )

                    new_lines.append(
                        f"{new_id} " +
                        " ".join(f"{v:.6f}" for v in values)
                    )



                if not new_lines:
                        skipped += 1
                        continue

                safe_name = "__".join(relative.parts)
                prefix = f"{source_name}__{split}__"
                dest_image = dest_images / (prefix + safe_name)
                dest_label = dest_labels / (
                    dest_image.stem + ".txt"
                    )

                if dest_image.exists():
                        raise FileExistsError(dest_image)

                shutil.copy2(image, dest_image)
                dest_label.write_text(
                    "\n".join(new_lines) + "\n",
                    encoding="utf-8"
                    )
                counts[split] += 1

        config = {
                "path": "/content/dataset",
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": 3,
                "names": {
                    0: "person",
                    1: "apple",
                    2: "vehicle",
                },
            }

        with open(OUTPUT / "data.yaml", "w", encoding="utf-8") as f:
                    yaml.safe_dump(
                        config, f, allow_unicode=True, sort_keys=False
                    )

        print("\n===== 병합 완료 =====")
        for split, count in counts.items():
            print(f"{split}: {count}장")
            print(f"제외된 이미지: {skipped}장")
            print(f"저장 위치: {OUTPUT}")

if __name__ == "__main__":
    merge()
