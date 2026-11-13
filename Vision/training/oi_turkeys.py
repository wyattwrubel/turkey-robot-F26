"""
oi_turkeys.py: pick real-bird "Turkey" images from Open Images and write YOLO labels.

  python -I oi_turkeys.py RAW_OI_DIR OUT_DIR [--max 3000]

Reads the box CSVs in RAW_OI_DIR (train from v6, validation/test from v5), keeps
images with a Turkey box (/m/0jly1) that are photos (IsDepiction=0) and drops images
that also contain food or table things (roast turkeys), downloads each image from the
public Open Images bucket, and writes YOLO labels (class 0 = turkey).

Split: Open Images train -> our train/val pool; Open Images validation + test -> held-out test.
"""
import argparse, csv, os, sys, concurrent.futures as cf, urllib.request
from collections import defaultdict

TURKEY = "/m/0jly1"
FOOD_NAMES = {  # images that also contain any of these are dropped (roast / cooked / table scenes)
    "Food", "Fast food", "Dish", "Tableware", "Plate", "Bowl", "Table", "Kitchen & dining room table",
    "Coffee table", "Bread", "Sandwich", "Cake", "Fruit", "Vegetable", "Salad", "Fork", "Knife", "Spoon",
    "Chopsticks", "Wine glass", "Bottle", "Coffee cup", "Mug", "Platter", "Cutting board", "Oven",
}
FOODISH = set()  # filled from class-descriptions-boxable.csv by name, so no MID is guessed
# W26's class ids: 0 turkey, 1 person, 2 vehicle. Person/vehicle boxes in kept images are written
# too, so the fine-tune doesn't learn "people are background".
PERSON_NAMES = {"Person", "Man", "Woman", "Boy", "Girl"}
VEHICLE_NAMES = {"Car", "Truck", "Van", "Bus", "Motorcycle", "Land vehicle", "Vehicle", "Taxi"}
EXTRA = {}  # MID -> class id, filled by name
SRC = {"train": "oidv6-train-annotations-bbox.csv",
       "validation": "validation-annotations-bbox.csv",
       "test": "test-annotations-bbox.csv"}


def scan(path):
    boxes, foodish, depiction, extra = defaultdict(list), set(), set(), defaultdict(list)
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            lab, iid = row["LabelName"], row["ImageID"]
            if lab in FOODISH:
                foodish.add(iid)
            elif lab in EXTRA:
                extra[iid].append((EXTRA[lab], float(row["XMin"]), float(row["XMax"]),
                                   float(row["YMin"]), float(row["YMax"])))
            elif lab == TURKEY:
                if row.get("IsDepiction") == "1":
                    depiction.add(iid)
                boxes[iid].append((float(row["XMin"]), float(row["XMax"]),
                                   float(row["YMin"]), float(row["YMax"])))
    keep = {i: b for i, b in boxes.items() if i not in foodish and i not in depiction}
    return keep, len(boxes), len(foodish & set(boxes)), len(depiction), extra


def fetch(split, iid, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    url = f"https://open-images-dataset.s3.amazonaws.com/{split}/{iid}.jpg"
    try:
        with urllib.request.urlopen(url, timeout=30) as r, open(dest, "wb") as f:
            f.write(r.read())
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw"); ap.add_argument("out")
    ap.add_argument("--max", type=int, default=3000, help="cap on train images")
    a = ap.parse_args()
    with open(os.path.join(a.raw, "class-descriptions-boxable.csv"), newline="") as f:
        names = {row[1]: row[0] for row in csv.reader(f)}
    missing = sorted(n for n in FOOD_NAMES if n not in names)
    FOODISH.update(names[n] for n in FOOD_NAMES if n in names)
    EXTRA.update({names[n]: 1 for n in PERSON_NAMES if n in names})
    EXTRA.update({names[n]: 2 for n in VEHICLE_NAMES if n in names})
    assert names.get("Turkey") == TURKEY, names.get("Turkey")
    print(f"food/table classes: {len(FOODISH)} found, not in Open Images: {missing}", flush=True)
    for split, fname in SRC.items():
        keep, n_all, n_food, n_dep, extra = scan(os.path.join(a.raw, fname))
        dst = "test" if split in ("validation", "test") else "pool"
        img_dir = os.path.join(a.out, dst, "images"); lab_dir = os.path.join(a.out, dst, "labels")
        os.makedirs(img_dir, exist_ok=True); os.makedirs(lab_dir, exist_ok=True)
        ids = sorted(keep)[: a.max] if split == "train" else sorted(keep)
        print(f"{split}: {n_all} images with Turkey boxes, dropped {n_food} food/table, "
              f"{n_dep} depictions -> {len(keep)} kept, downloading {len(ids)}", flush=True)
        ok = 0
        with cf.ThreadPoolExecutor(16) as ex:
            futs = {ex.submit(fetch, split, i, os.path.join(img_dir, f"oi_{i}.jpg")): i for i in ids}
            for fu in cf.as_completed(futs):
                i = futs[fu]
                if not fu.result():
                    continue
                ok += 1
                with open(os.path.join(lab_dir, f"oi_{i}.txt"), "w") as f:
                    rows = [(0, *b) for b in keep[i]] + extra.get(i, [])
                    for c, x1, x2, y1, y2 in rows:
                        f.write(f"{c} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}\n")
        print(f"{split}: {ok} images written to {dst}", flush=True)


if __name__ == "__main__":
    main()
