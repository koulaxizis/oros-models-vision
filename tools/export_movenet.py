#!/usr/bin/env python3
"""Write movenet/movenet-lightning.onnx from Google's MoveNet TFLite release.

MoveNet SinglePose Lightning (Apache-2.0) is published only as TFLite and
TensorFlow models. orOS runs ONNX with ONNX Runtime Web, so this script:
  1. downloads the float16 TFLite file (version 4) from the first source
     that answers and checks it (TFLite magic; SHA-256 once SRC_SHA is set);
  2. converts it with tf2onnx (opset 17) and renames the graph's input to
     "input" (int32 [1, 192, 192, 3], RGB 0..255) and its output to
     "keypoints" (float32 [1, 1, 17, 3]: y, x, score per COCO keypoint,
     y and x in 0..1 of the input);
  3. refuses the file unless ONNX Runtime gives the same keypoints as the
     TFLite interpreter on test images (largest difference under 1e-4) and
     unless it finds the person in scikit-image's astronaut photo (nose,
     eyes and shoulders found, nose above the shoulders);
  4. writes movenet/export.json (sources, hashes, tool versions, check).

Usage: python3 tools/export_movenet.py   (needs network for step 1)
"""
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "movenet")
OUT = os.path.join(OUT_DIR, "movenet-lightning.onnx")
UA = {"User-Agent": "oros-models-vision-export"}

SOURCES = [
    "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite",
    "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/movenet/singlepose/lightning/tflite/float16/4.tflite",
    "https://www.kaggle.com/api/v1/models/google/movenet/tfLite/singlepose-lightning-tflite-float16/1/download",
]
# SHA-256 of the TFLite file. Empty on the first run, which records it in
# export.json; then it is written here so every later run checks it.
SRC_SHA = ""

KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle"]


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def tflite_bytes(raw):
    """The TFLite file itself, also out of a .tar.gz (Kaggle's download)."""
    if raw[4:8] == b"TFL3":
        return raw
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as t:
            names = [m for m in t.getmembers() if m.isfile() and m.name.endswith(".tflite")]
            if len(names) == 1:
                b = t.extractfile(names[0]).read()
                if b[4:8] == b"TFL3":
                    return b
    except tarfile.TarError:
        pass
    return None


def fetch():
    errors = []
    for url in SOURCES:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                raw = r.read(40 * 1000 * 1000)
        except Exception as e:  # try the next source
            errors.append("%s: %s" % (url, e))
            continue
        b = tflite_bytes(raw)
        if b is None:
            errors.append("%s: not a TFLite file" % url)
            continue
        if SRC_SHA and sha256(b) != SRC_SHA:
            raise SystemExit("TFLite hash differs from SRC_SHA: %s %s" % (url, sha256(b)))
        return url, b
    raise SystemExit("no source answered:\n  " + "\n  ".join(errors))


def convert(tfl_path):
    import onnx
    import tf2onnx
    model, _ = tf2onnx.convert.from_tflite(tfl_path, opset=17)
    g = model.graph
    if len(g.input) != 1 or len(g.output) != 1:
        raise SystemExit("unexpected graph inputs/outputs")
    old_in, old_out = g.input[0].name, g.output[0].name
    for node in g.node:
        node.input[:] = ["input" if n == old_in else n for n in node.input]
        node.output[:] = ["keypoints" if n == old_out else n for n in node.output]
    g.input[0].name, g.output[0].name = "input", "keypoints"
    model.producer_name = "oros-models-vision tools/export_movenet.py"
    model.producer_version = ""
    model.doc_string = "MoveNet SinglePose Lightning (Google, Apache-2.0), converted from TFLite float16 v4"
    onnx.checker.check_model(model)
    return model


def test_images():
    import numpy as np
    from skimage import data
    from skimage.transform import resize
    rng = np.random.default_rng(7)
    astro = (resize(data.astronaut(), (192, 192), anti_aliasing=True) * 255).round().astype(np.int32)
    grad = np.stack(np.meshgrid(np.arange(192), np.arange(192)), -1).sum(-1)
    grad = np.repeat((grad * 255 // 382)[..., None], 3, -1).astype(np.int32)
    noise = rng.integers(0, 256, (192, 192, 3), dtype=np.int32)
    return {"astronaut": astro, "gradient": grad, "noise": noise}


def main():
    import numpy as np
    import onnx
    import onnxruntime as ort
    import tensorflow as tf
    import tf2onnx
    import skimage

    url, tfl = fetch()
    print("TFLite from", url, sha256(tfl), len(tfl), "bytes", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "movenet.tflite")
        with open(p, "wb") as f:
            f.write(tfl)
        model = convert(p)
        interp = tf.lite.Interpreter(model_path=p)
        interp.allocate_tensors()
        tin, tout = interp.get_input_details()[0], interp.get_output_details()[0]
        if list(tin["shape"]) != [1, 192, 192, 3] or list(tout["shape"]) != [1, 1, 17, 3]:
            raise SystemExit("unexpected TFLite shapes %s %s" % (tin["shape"], tout["shape"]))

        data = model.SerializeToString()
        sess = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
        diffs, astro = {}, None
        for name, img in test_images().items():
            x = img[None].astype(tin["dtype"])
            interp.set_tensor(tin["index"], x)
            interp.invoke()
            want = interp.get_tensor(tout["index"])
            got = sess.run(["keypoints"], {"input": x.astype(np.int32)})[0]
            diffs[name] = float(np.abs(want - got).max())
            if name == "astronaut":
                astro = got[0, 0]
        worst = max(diffs.values())
        print("largest difference ONNX vs TFLite:", worst, diffs, flush=True)
        if not worst < 1e-4:
            raise SystemExit("ONNX differs from TFLite: %r" % diffs)
        k = {n: astro[i] for i, n in enumerate(KEYPOINTS)}
        need = ["nose", "left_eye", "right_eye", "left_shoulder", "right_shoulder"]
        low = [n for n in need if k[n][2] < 0.3]
        if low or not k["nose"][0] < min(k["left_shoulder"][0], k["right_shoulder"][0]):
            raise SystemExit("astronaut check failed: %s" % {n: k[n].tolist() for n in need})

    with open(OUT, "wb") as f:
        f.write(data)
    info = {
        "note": "Written by tools/export_movenet.py; never edit by hand.",
        "model": "MoveNet SinglePose Lightning v4 (float16 TFLite), Google, Apache-2.0",
        "source": url,
        "sourceSha256": sha256(tfl),
        "onnxSha256": sha256(data),
        "input": {"name": "input", "type": "int32", "shape": [1, 192, 192, 3], "meaning": "RGB 0..255, square, letterboxed"},
        "output": {"name": "keypoints", "type": "float32", "shape": [1, 1, 17, 3], "meaning": "y, x (0..1 of the input), score"},
        "keypoints": KEYPOINTS,
        "tools": {"python": sys.version.split()[0], "tensorflow": tf.__version__, "tf2onnx": tf2onnx.__version__,
                  "onnx": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__,
                  "scikit-image": skimage.__version__, "opset": 17},
        "check": {"largestDifference": worst, "perImage": diffs,
                  "astronaut": {n: [round(float(v), 4) for v in k[n]] for n in need}},
    }
    with open(os.path.join(OUT_DIR, "export.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
        f.write("\n")
    print("written", OUT, sha256(data), len(data), "bytes")


if __name__ == "__main__":
    main()
