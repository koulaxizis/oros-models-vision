#!/usr/bin/env python3
"""Export DDColor tiny (piddnad/DDColor, Apache-2.0) to ONNX for orOS Photo
Lab's and Atelier's "Colourize".

There is no official ONNX file, so this script makes one from the authors'
own weights and proves it right:

  1. reads the DDColor source code at a pinned commit (the workflow clones
     github.com/piddnad/DDColor at DDCOLOR_COMMIT into DDCOLOR_SRC); only the
     network's own files are imported, nothing for training;
  2. downloads the weights of Hugging Face piddnad/ddcolor_paper_tiny at a
     pinned commit and checks them against Hugging Face's LFS SHA-256;
  3. exports the network with a free picture size (height and width
     multiples of 32), weights stored as float16 with a Cast back to float32
     in the graph (half the size; every operation still runs in float32);
  4. checks the ONNX file with ONNX Runtime against the PyTorch network on
     test pictures at 256 and 512 px: the colours orOS makes from it
     (the picture's own lightness + the model's a*b*) must match PyTorch's
     by at least MIN_PSNR_DB;
  5. writes ddcolor/ddcolor_tiny.onnx (or, when it is over 95 MB, the parts
     ddcolor_tiny.onnx.part00, .part01 … that orOS joins, named as
     tools/mirror.py names a split download) and ddcolor/export.json.

The graph:
  input   input   float32 [1, 3, H, W]  the picture's lightness as grey RGB 0..1
                                        (sRGB L*a*b* with a* = b* = 0, back to RGB)
  output  output  float32 [1, 2, H, W]  a*, b* (CIE L*a*b*, D65, as OpenCV's float Lab)
  orOS gives it the picture at 512 px (256 on small devices), scales a*b* back
  to the full size and puts them on the picture's own L*.

Usage:
  DDCOLOR_SRC=<DDColor checkout> python3 tools/export_ddcolor.py
  DDCOLOR_SRC=<DDColor checkout> python3 tools/export_ddcolor.py --random
      (random weights of the same shape, into a temp folder: tests the script)
"""
import glob
import hashlib
import importlib.util
import json
import os
import struct
import sys
import tempfile
import time
import types
import urllib.request

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = "https://huggingface.co"
HF_REPO = "piddnad/ddcolor_paper_tiny"
HF_REVISION = "main"          # resolved on the first run; export.json keeps the commit
DDCOLOR_COMMIT = "2adb63f2656ac41cbdf7b894cddd94121a3faf13"
OPSET = 17
MAX_PART = 95 * 1000 * 1000
MIN_PSNR_DB = 40.0
SIZES = (256, 512)
UA = {"User-Agent": "oros-models-vision-export"}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


# ---------- the network, without the training code ----------
def ddcolor_class(src):
    """ddcolor.model.DDColor, importing only its five network files (the
    basicsr package's own __init__ files pull in OpenCV and training tools)."""
    for name, rel in (("basicsr", "basicsr"), ("basicsr.archs", "basicsr/archs"),
                      ("basicsr.archs.ddcolor_arch_utils", "basicsr/archs/ddcolor_arch_utils"),
                      ("ddcolor", "ddcolor")):
        m = types.ModuleType(name)
        m.__path__ = [os.path.join(src, rel)]
        sys.modules[name] = m
    spec = importlib.util.spec_from_file_location("ddcolor.model", os.path.join(src, "ddcolor", "model.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ddcolor.model"] = mod
    spec.loader.exec_module(mod)
    return mod.DDColor


def read_safetensors(path):
    """A .safetensors file → {name: tensor}, with the standard library and numpy."""
    kinds = {"F32": np.float32, "F16": np.float16, "BF16": None, "I64": np.int64, "I32": np.int32,
             "U8": np.uint8, "BOOL": np.bool_}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        data = f.read()
    out = {}
    for k, v in head.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        if v["dtype"] == "BF16":
            raw = np.frombuffer(data[a:b], dtype=np.uint16).astype(np.uint32) << 16
            arr = raw.view(np.float32)
        else:
            arr = np.frombuffer(data[a:b], dtype=kinds[v["dtype"]])
        out[k] = torch.from_numpy(arr.reshape(v["shape"]).copy())
    return out


def load_model(src, random_weights, work):
    DDColor = ddcolor_class(src)
    kwargs = dict(encoder_name="convnext-t", decoder_name="MultiScaleColorDecoder", input_size=[512, 512],
                  num_output_channels=2, last_norm="Spectral", do_normalize=False,
                  num_queries=100, num_scales=3, dec_layers=9)
    if random_weights:
        torch.manual_seed(0)
        return DDColor(**kwargs).eval(), {"weights": "random (test run)"}
    rev = get_json("%s/api/models/%s/revision/%s" % (HF, HF_REPO, HF_REVISION))["sha"]
    tree = {t["path"]: t for t in get_json("%s/api/models/%s/tree/%s" % (HF, HF_REPO, rev)) if t.get("type") == "file"}
    name = next((p for p in ("model.safetensors", "pytorch_model.bin", "pytorch_model.pt") if p in tree), None)
    if not name or not tree[name].get("lfs"):
        raise SystemExit("no weights file in %s@%s: %s" % (HF_REPO, rev, sorted(tree)))
    if "config.json" in tree:
        cfg = get_json("%s/%s/resolve/%s/config.json" % (HF, HF_REPO, rev))
        cfg = cfg.get("config", cfg) if isinstance(cfg, dict) else {}
        for k in kwargs:
            if k in cfg:
                kwargs[k] = cfg[k]
    if kwargs["encoder_name"] != "convnext-t" or kwargs["num_output_channels"] != 2:
        raise SystemExit("not DDColor tiny: %r" % kwargs)
    path = os.path.join(work, name)
    print("download", HF_REPO, rev, name, flush=True)
    urllib.request.urlretrieve("%s/%s/resolve/%s/%s" % (HF, HF_REPO, rev, name), path)
    sha = sha256_of(path)
    if sha != tree[name]["lfs"]["oid"] or os.path.getsize(path) != tree[name]["lfs"]["size"]:
        raise SystemExit("weights differ from Hugging Face's LFS hash: " + sha)
    if name.endswith(".safetensors"):
        state = read_safetensors(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("params", state)
    m = DDColor(**kwargs)
    res = m.load_state_dict(state, strict=False)
    if res.missing_keys:
        raise SystemExit("weights missing for: %s" % res.missing_keys[:10])
    return m.eval(), {"weights": "%s/%s/blob/%s/%s" % (HF, HF_REPO, rev, name), "weightsSha256": sha,
                      "huggingFaceRevision": rev, "kwargs": kwargs,
                      "unusedKeys": len(res.unexpected_keys)}


# ---------- export ----------
def fp16_weights(path_in, path_out):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    g = onnx.load(path_in)
    graph = g.graph
    inits, casts = [], []
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= 1024:
            arr = numpy_helper.to_array(init).astype(np.float16)
            inits.append(numpy_helper.from_array(arr, init.name + "__fp16"))
            casts.append(helper.make_node("Cast", [init.name + "__fp16"], [init.name],
                                          to=TensorProto.FLOAT, name=init.name + "__cast"))
        else:
            inits.append(init)
    del graph.initializer[:]
    graph.initializer.extend(inits)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    onnx.checker.check_model(g)
    onnx.save(g, path_out)


def export(m, out_path, work):
    f32 = os.path.join(work, "ddcolor.f32.onnx")
    with torch.no_grad():
        torch.onnx.export(m, (torch.rand(1, 3, 512, 512),), f32,
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}},
                          opset_version=OPSET, dynamo=False, do_constant_folding=True)
    fp16_weights(f32, out_path)


# ---------- colour maths (what orOS does in JavaScript) ----------
WHITE = np.array([0.950456, 1.0, 1.088754])
M = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])


def srgb_to_lab(rgb):
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ M.T / WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = np.where(xyz[..., 1] > 0.008856, 116 * f[..., 1] - 16, 903.3 * xyz[..., 1])
    return np.stack([L, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_srgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > 0.206893, f ** 3, (f - 16 / 116) / 7.787) * WHITE
    lin = np.clip(xyz @ np.linalg.inv(M).T, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def grey_input(rgb, size):
    """rgb float [h, w, 3] 0..1 → the model's input at size × size (area resize)."""
    t = torch.from_numpy(rgb.transpose(2, 0, 1)[None].astype(np.float32))
    small = torch.nn.functional.interpolate(t, size=(size, size), mode="area")[0].numpy().transpose(1, 2, 0)
    L = srgb_to_lab(small.astype(np.float64))[..., 0]
    grey = lab_to_srgb(np.stack([L, 0 * L, 0 * L], -1))
    return grey.transpose(2, 0, 1)[None].astype(np.float32)


def colourize(rgb, ab):
    """The picture's own L* + the model's a*b* scaled to its size → RGB 0..1."""
    h, w = rgb.shape[:2]
    ab = torch.nn.functional.interpolate(torch.from_numpy(ab), size=(h, w), mode="bilinear",
                                         align_corners=False)[0].numpy().transpose(1, 2, 0)
    L = srgb_to_lab(rgb.astype(np.float64))[..., :1]
    return lab_to_srgb(np.concatenate([L, ab], -1))


def test_pictures(src):
    """The DDColor repository's own black-and-white test photos (for the check
    only; not copied) and one synthetic picture."""
    pics = []
    try:
        from PIL import Image
        for p in sorted(glob.glob(os.path.join(src, "assets", "test_images", "*")))[:4]:
            im = Image.open(p).convert("RGB")
            im.thumbnail((640, 640))
            pics.append((os.path.basename(p), np.asarray(im, dtype=np.float64) / 255))
    except ImportError:
        pass
    y, x = np.mgrid[0:300, 0:400] / 400.0
    pics.append(("synthetic", np.stack([x, y, (x + y) / 2], -1)))
    return pics


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse == 0 else 10 * np.log10(1 / mse)


def check(m, onnx_path, src):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    s = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    rows, secs = [], {}
    for size in SIZES:
        for name, rgb in test_pictures(src):
            x = grey_input(rgb, size)
            with torch.no_grad():
                exp = m(torch.from_numpy(x)).numpy()
            t0 = time.time()
            got = s.run(["output"], {"input": x})[0]
            secs.setdefault(size, []).append(time.time() - t0)
            rows.append({"picture": name, "size": size,
                         "psnrDb": round(psnr(colourize(rgb, got), colourize(rgb, exp)), 1),
                         "maxAbsErrAb": round(float(np.abs(got - exp).max()), 4)})
    return {"pictures": rows, "minPsnrDb": min(r["psnrDb"] for r in rows),
            "nativeOneThreadSeconds": {str(k): round(float(np.median(v)), 2) for k, v in secs.items()}}


def split(path):
    """Leaves the file as it is when it fits; else parts of at most 95 MB."""
    size = os.path.getsize(path)
    if size <= MAX_PART:
        return [os.path.basename(path)]
    names = []
    with open(path, "rb") as f:
        i = 0
        while True:
            chunk = f.read(MAX_PART)
            if not chunk:
                break
            part = "%s.part%02d" % (path, i)
            with open(part, "wb") as o:
                o.write(chunk)
            names.append(os.path.basename(part))
            i += 1
    os.remove(path)
    return names


def main():
    src = os.environ.get("DDCOLOR_SRC")
    if not src or not os.path.exists(os.path.join(src, "ddcolor", "model.py")):
        raise SystemExit("set DDCOLOR_SRC to a checkout of github.com/piddnad/DDColor at " + DDCOLOR_COMMIT)
    random_weights = "--random" in sys.argv
    out_dir = tempfile.mkdtemp() if random_weights else os.path.join(ROOT, "ddcolor")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ddcolor_tiny.onnx")
    info_path = os.path.join(out_dir, "export.json")
    if not random_weights and os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            old = json.load(f)
        if all(os.path.exists(os.path.join(out_dir, p["name"])) and sha256_of(os.path.join(out_dir, p["name"])) == p["sha"]
               for p in old.get("files", [])):
            print("ddcolor/ is there with the hashes in export.json; nothing to do")
            return
    with tempfile.TemporaryDirectory() as work:
        m, origin = load_model(src, random_weights, work)
        export(m, out_path, work)
        whole = {"size": os.path.getsize(out_path), "sha": sha256_of(out_path)}
        result = check(m, out_path, src)
    print(json.dumps(result, indent=1))
    if result["minPsnrDb"] < MIN_PSNR_DB:
        raise SystemExit("the ONNX colours differ from PyTorch's (below %.0f dB)" % MIN_PSNR_DB)
    names = split(out_path)
    import onnx
    import onnxruntime
    info = {
        "note": "Written by tools/export_ddcolor.py; never edit by hand.",
        "model": "DDColor tiny (ddcolor_paper_tiny), Kang et al., ICCV 2023, Apache-2.0",
        "code": "https://github.com/piddnad/DDColor/tree/" + DDCOLOR_COMMIT,
        **origin,
        "whole": {"name": "ddcolor_tiny.onnx", **whole},
        "files": [{"name": n, "size": os.path.getsize(os.path.join(out_dir, n)),
                   "sha": sha256_of(os.path.join(out_dir, n))} for n in names],
        "tools": {"torch": torch.__version__, "onnx": onnx.__version__,
                  "onnxruntime": onnxruntime.__version__, "opset": OPSET},
        "graph": {"input": {"input": [1, 3, "H", "W"]}, "output": {"output": [1, 2, "H", "W"]},
                  "sizeMultipleOf": 32, "sizes": list(SIZES), "weightsFloat16": True,
                  "input_is": "grey RGB 0..1 from the picture's L* (sRGB, D65), the model normalises itself",
                  "output_is": "a*, b* in CIE L*a*b* units"},
        "check": result,
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
        f.write("\n")
    print("written", names, whole)


if __name__ == "__main__":
    main()
