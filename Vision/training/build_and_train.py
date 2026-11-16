"""
build_and_train.py: merge the sources into one YOLO dataset, score W26's models on the
held-out test set ("before"), fine-tune from W26's weights, and score the result ("after").

  python -I build_and_train.py CV_DIR assemble          # data/build/* -> data/turkey_v1/
  python -I build_and_train.py CV_DIR baseline          # W26 turkey_1280@640 and turkey_640 on test
  python -I build_and_train.py CV_DIR train [--epochs 60] [--from turkey_1280.pt] [--fraction 1.0]
  python -I build_and_train.py CV_DIR eval RUN_DIR      # a trained run on the same test set

Classes stay W26's (0 turkey, 1 person, 2 vehicle) so the Hailo pipeline and the planner
need no change. Test set = Open Images validation+test turkeys + ENA24 test blocks; it is
never used for training or threshold choice. Our own robot-height footage and Jared's
trail-camera images get added as new sources later (see MODEL-PLAN.md).
"""
import argparse, glob, os, random, shutil, sys

NAMES = {0: "turkey", 1: "person", 2: "vehicle"}


def link(src_img, dst_dir, split):
    stem = os.path.splitext(os.path.basename(src_img))[0]
    lbl = os.path.join(os.path.dirname(os.path.dirname(src_img)), "labels", stem + ".txt")
    for kind, s in (("images", src_img), ("labels", lbl)):
        d = os.path.join(dst_dir, kind, split)
        os.makedirs(d, exist_ok=True)
        t = os.path.join(d, os.path.basename(s))
        if os.path.exists(s) and not os.path.exists(t):
            os.symlink(s, t)


def assemble(cv):
    b, out = os.path.join(cv, "data", "build"), os.path.join(cv, "data", "turkey_v1")
    if os.path.exists(out):
        shutil.rmtree(out)
    rnd = random.Random(0)
    oi = sorted(glob.glob(os.path.join(b, "openimages", "pool", "images", "*.jpg")))
    rnd.shuffle(oi)
    n_val = len(oi) // 10
    for i, f in enumerate(oi):
        link(f, out, "val" if i < n_val else "train")
    for f in glob.glob(os.path.join(b, "openimages", "test", "images", "*.jpg")):
        link(f, out, "test")
    for sp in ("train", "val", "test"):
        for f in glob.glob(os.path.join(b, "ena24_b10", sp, "images", "*.jpg")):
            link(f, out, sp)
    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write(f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n")
        for k, v in NAMES.items():
            f.write(f"  {k}: {v}\n")
    for sp in ("train", "val", "test"):
        imgs = glob.glob(os.path.join(out, "images", sp, "*"))
        cls = {}
        for l in glob.glob(os.path.join(out, "labels", sp, "*.txt")):
            for line in open(l):
                if line.strip():
                    c = line.split()[0]; cls[c] = cls.get(c, 0) + 1
        empty = sum(1 for l in glob.glob(os.path.join(out, "labels", sp, "*.txt")) if os.path.getsize(l) == 0)
        print(f"{sp:5}: {len(imgs):5} images ({empty} background), boxes by class {dict(sorted(cls.items()))}")
    return os.path.join(out, "data.yaml")


def score(cv, weights, name, imgsz=640):
    from ultralytics import YOLO
    m = YOLO(weights)
    r = m.val(data=os.path.join(cv, "data", "turkey_v1", "data.yaml"), split="test", imgsz=imgsz,
              batch=16, device="mps", conf=0.001, iou=0.6, plots=False, verbose=False,
              project=os.path.join(cv, "runs"), name=f"test_{name}", exist_ok=True)
    # per-class: index 0 = turkey
    p, rc, ap50, ap = r.box.class_result(0)
    print(f"{name:28} turkey  P {p:.3f}  R {rc:.3f}  mAP50 {ap50:.3f}  mAP50-95 {ap:.3f}")
    return dict(precision=p, recall=rc, map50=ap50, map=ap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cv"); ap.add_argument("cmd", choices=["assemble", "baseline", "train", "eval"])
    ap.add_argument("run", nargs="?")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--from", dest="init", default="turkey_1280.pt")
    ap.add_argument("--fraction", type=float, default=1.0, help="train on a fraction (smoke runs)")
    a = ap.parse_args()
    cv = os.path.abspath(a.cv)
    if a.cmd == "assemble":
        assemble(cv)
    elif a.cmd == "baseline":
        score(cv, os.path.join(cv, "weights", "turkey_1280.pt"), "W26 turkey_1280 @640 (baseline)")
        score(cv, os.path.join(cv, "weights", "turkey_640.pt"), "W26 turkey_640 (on the Hailo)")
    elif a.cmd == "train":
        from ultralytics import YOLO
        m = YOLO(os.path.join(cv, "weights", a.init))
        m.train(data=os.path.join(cv, "data", "turkey_v1", "data.yaml"), imgsz=640, epochs=a.epochs,
                batch=16, device="mps", workers=4, fraction=a.fraction, patience=15, seed=0,
                # fine-tuning pre-trained weights: low LR so W26's features aren't wiped early
                optimizer="SGD", lr0=0.002, lrf=0.1, cos_lr=True, warmup_epochs=1,
                # small, distant birds matter most: scale down hard, keep mosaic
                scale=0.6, mosaic=1.0, degrees=0.0, fliplr=0.5, hsv_v=0.5,
                project=os.path.join(cv, "runs"), name=f"turkey_v1_{a.init.split('.')[0]}", exist_ok=False)
    elif a.cmd == "eval":
        score(cv, os.path.join(a.run, "weights", "best.pt"), os.path.basename(a.run))


if __name__ == "__main__":
    main()
