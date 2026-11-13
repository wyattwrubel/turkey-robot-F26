"""
ena24_build.py: turn ENA24 (LILA BC, CDLA Permissive 1.0) parquet files into YOLO data
with W26's classes: 0 turkey (Wild Turkey), 1 person (Human), 2 vehicle (Vehicle).

  python -I ena24_build.py RAW_ENA24_DIR OUT_DIR [--neg-ratio 1.5]

- Images with a Wild Turkey, Human or Vehicle box are positives (all their boxes of
  those three classes are written).
- Images with only other mammals (deer, squirrels, raccoons, ...) are negatives
  (empty label files): the "nothing to chase" examples. Capped at neg-ratio x turkeys.
- Images with other birds (Bird, American Crow, Chicken) and no turkey are skipped
  for v1: whether a crow is a target is the size-gate question for Jared.
- Split by blocks of 10 consecutive image_ids (ids run by species, so 100-id blocks put
  almost every turkey in one split; neighbouring blocks may share a burst, a small leak):
  block % 10 == 0 -> test, == 1 -> val, else train. Never split by frame.
- Saved as JPEG, long side 1280 (originals 1920x1080), labels normalized.
"""
import argparse, glob, io, json, os, random
import pyarrow.parquet as pq
from PIL import Image

MAP = {"Wild Turkey": 0, "Human": 1, "Vehicle": 2}
OTHER_BIRDS = {"Bird", "American Crow", "Chicken"}


BLOCK = 10   # ids are grouped by species, not time; 10-id blocks spread every class across splits


def split_of(image_id):
    b = (image_id // BLOCK) % 10
    return "test" if b == 0 else "val" if b == 1 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw"); ap.add_argument("out")
    ap.add_argument("--neg-ratio", type=float, default=1.5)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.raw, "*.parquet")))
    names = json.loads(pq.ParquetFile(files[0]).schema_arrow.metadata[b"huggingface"]) \
        ["info"]["features"]["objects"]["category"]["feature"]["names"]
    cid = {n: i for i, n in enumerate(names)}
    keep_map = {cid[n]: c for n, c in MAP.items()}
    birds = {cid[n] for n in OTHER_BIRDS}

    pos, neg = [], []          # (file, row index, image_id)
    for f in files:
        t = pq.read_table(f, columns=["image_id", "objects"]).to_pylist()
        for r, row in enumerate(t):
            cats = set(row["objects"]["category"])
            if cats & set(keep_map):
                pos.append((f, r, row["image_id"]))
            elif not cats & birds:
                neg.append((f, r, row["image_id"]))
    n_turkey = sum(1 for _ in pos)
    random.Random(0).shuffle(neg)
    neg = neg[: int(a.neg_ratio * n_turkey)]
    want = {(f, r): "pos" for f, r, _ in pos} | {(f, r): "neg" for f, r, _ in neg}
    print(f"positives (turkey/person/vehicle) {len(pos)}, negatives kept {len(neg)}", flush=True)

    counts = {}
    for f in files:
        pf = pq.ParquetFile(f)
        rows = pf.read().to_pylist()
        for r, row in enumerate(rows):
            if (f, r) not in want:
                continue
            sp = split_of(row["image_id"])
            img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            W, H = img.size
            s = 1280 / max(W, H)
            if s < 1:
                img = img.resize((round(W * s), round(H * s)))
            stem = f"ena24_{row['image_id']}"
            idir = os.path.join(a.out, sp, "images"); ldir = os.path.join(a.out, sp, "labels")
            os.makedirs(idir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
            img.save(os.path.join(idir, stem + ".jpg"), quality=92)
            lines = []
            for (x, y, w, h), c in zip(row["objects"]["bbox"], row["objects"]["category"]):
                if c in keep_map:
                    lines.append(f"{keep_map[c]} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
            open(os.path.join(ldir, stem + ".txt"), "w").write("\n".join(lines) + ("\n" if lines else ""))
            key = (sp, want[(f, r)])
            counts[key] = counts.get(key, 0) + 1
    print("written:", dict(sorted(counts.items())), flush=True)


if __name__ == "__main__":
    main()
