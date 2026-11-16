"""
hailo_kit.py: build everything the Colab notebook (hailo_compile_split.ipynb) needs to
compile a fine-tuned YOLO26n for the robot's Hailo-8L, and check the split before Colab.

  python -I hailo_kit.py CV_DIR WEIGHTS.pt OUT_DIR [--name turkey_v1] [--calib 300] [--eval 24]

Writes to OUT_DIR:
  NAME.onnx                 opset 11, 640, end2end off (the robot decodes the raw heads)
  calib_NAME.zip            training images, letterboxed exactly as Vision/Detector.py does
  eval_NAME.zip             held-out test images for the notebook's quantization gate
  yolo_decode.py            the robot's decoder, used by the gate
  *.part-aa, ...            the ONNX and calibration zip split under 7 MB (the Drive
                            connector's upload limit); the notebook rejoins and hash-checks them
  SHA256.txt                hashes of the whole files

Before writing the zips it cuts the ONNX at the six head convs the HEF will end at
(box + class per stride, W26's working turkey_640_v4 layout), decodes them with the
robot's decode_split, and compares to the .pt on held-out images. It stops if they
disagree, so a wrong cut never reaches Colab. (11-16: turkey_v1, same counts, IoU 0.979.)

The notebook's step 5 needs the ONNX node names ENDS below; keep the two in sync.
"""
import argparse, glob, hashlib, os, random, shutil, subprocess, sys, zipfile

ENDS = [f"/model.23/cv{k}.{i}/cv{k}.{i}.2/Conv" for i in range(3) for k in (2, 3)]
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cv"); ap.add_argument("weights"); ap.add_argument("out")
    ap.add_argument("--name", default="turkey_v1")
    ap.add_argument("--calib", type=int, default=300)
    ap.add_argument("--eval", type=int, default=24)
    a = ap.parse_args()
    import cv2, numpy as np, onnx, onnxruntime as ort
    from ultralytics import YOLO
    from Vision.yolo_decode import letterbox, unletterbox, decode_split

    os.makedirs(a.out, exist_ok=True)
    pt_path = os.path.join(a.out, a.name + ".pt")
    shutil.copy(a.weights, pt_path)
    pt = YOLO(pt_path)
    pt.export(format="onnx", imgsz=640, opset=11, simplify=True, dynamic=False, end2end=False)
    onnx_path = os.path.join(a.out, a.name + ".onnx")

    # 1. the split check: cut at ENDS, decode like the robot, compare with the .pt
    split_path = os.path.join(a.out, a.name + "_split_check.onnx")
    onnx.utils.extract_model(onnx_path, split_path, ["images"], [e + "_output_0" for e in ENDS])
    sess = ort.InferenceSession(split_path)
    test = sorted(glob.glob(f"{a.cv}/data/turkey_v1/images/test/*.jpg"))
    random.seed(0)
    counts, ious = [], []
    for p in random.sample(test, min(8, len(test))):
        bgr = cv2.imread(p); lb, r, pad = letterbox(bgr, (640, 640))
        x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255
        outs = [o[0].transpose(1, 2, 0) for o in sess.run(None, {"images": x})]
        b, _, _ = decode_split(outs, (640, 640), 0.25)
        b = unletterbox(b, r, pad, (bgr.shape[1], bgr.shape[0]))
        pb = pt.predict(bgr, imgsz=640, conf=0.25, verbose=False)[0].boxes.xyxy.numpy()
        counts.append((len(b), len(pb)))
        ious += [max(iou(q, bb) for bb in b) for q in pb if len(b)]
    same = all(x == y for x, y in counts)
    miou = float(np.mean(ious)) if ious else 0.0
    print(f"split check: counts (split, pt) {counts}, mean IoU {miou:.3f}")
    if not same or miou < 0.9:
        sys.exit("split check FAILED: the six-node cut doesn't match the .pt; don't compile")
    os.remove(split_path)

    # 2. calibration and eval images, letterboxed like the robot
    random.seed(1)
    train = sorted(glob.glob(f"{a.cv}/data/turkey_v1/images/train/*.jpg"))
    for fn, src, n in ((f"calib_{a.name}.zip", train, a.calib), (f"eval_{a.name}.zip", test, a.eval)):
        with zipfile.ZipFile(os.path.join(a.out, fn), "w", zipfile.ZIP_STORED) as z:
            for p in random.sample(src, min(n, len(src))):
                lb, _, _ = letterbox(cv2.imread(p), (640, 640))
                ok, buf = cv2.imencode(".jpg", lb, [cv2.IMWRITE_JPEG_QUALITY, 95])
                z.writestr(os.path.basename(p), buf.tobytes())
    shutil.copy(os.path.join(REPO, "Vision", "yolo_decode.py"), a.out)

    # 3. hashes, then split the big files under the Drive connector's 7 MB limit
    with open(os.path.join(a.out, "SHA256.txt"), "w") as f:
        for fn in (f"{a.name}.onnx", f"calib_{a.name}.zip"):
            h = hashlib.sha256(open(os.path.join(a.out, fn), "rb").read()).hexdigest()
            f.write(f"{h}  {fn}\n"); print(fn, h)
            subprocess.run(["split", "-b", "7000000", fn, fn + ".part-"], cwd=a.out, check=True)
    print(f"kit ready in {a.out}. Upload the parts, eval zip, yolo_decode.py and the notebook to "
          f"Drive hailo/; update SHA in the notebook's step 2 if the name or data changed.")


if __name__ == "__main__":
    main()
