#!/usr/bin/env python3
"""Real-ESRGAN general x4v3 (.pth, PyTorch) -> ONNX, without PyTorch.

The upstream weights (xinntao/Real-ESRGAN release v0.2.5.0,
realesr-general-x4v3.pth, BSD-3-Clause) are a zip of a pickle and raw
float32 storages. This script reads them with the standard library and
numpy, checks the download's SHA-256, and writes the network
SRVGGNetCompact (num_feat 64, num_conv 32, PReLU, x4) as an ONNX graph
with onnx.helper. No PyTorch export runs, so the output bytes depend only
on the weights and the onnx package version (pinned in the workflow).

Graph: input "input" float32 [1, 3, H, W] in 0..1 (RGB)
       output "output" float32 [1, 3, 4H, 4W] (not clamped)
  conv3x3(3->64) PReLU, 32 x [conv3x3(64->64) PReLU], conv3x3(64->48),
  DepthToSpace(4, CRD) (= torch pixel_shuffle), + nearest x4 of the input.

Usage: convert_realesrgan.py IN.pth OUT.onnx [--sha SHA256]
"""
import hashlib
import io
import pickle
import sys
import zipfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def read_pth(path):
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    pkl = [n for n in names if n.endswith("data.pkl")][0]
    prefix = pkl[: -len("data.pkl")]
    dtypes = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64}

    class Storage:
        def __init__(self, dtype, key):
            self.dtype, self.key = dtype, key

    def rebuild(storage, offset, size, stride, *rest):
        raw = zf.read(prefix + "data/" + storage.key)
        flat = np.frombuffer(raw, dtype=storage.dtype)
        n = int(np.prod(size)) if size else 1
        if size:
            arr = np.lib.stride_tricks.as_strided(
                flat[offset:], shape=tuple(size),
                strides=tuple(s * flat.itemsize for s in stride))
        else:
            arr = flat[offset:offset + n]
        return np.array(arr, dtype=np.float32)

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "torch._utils" and name == "_rebuild_tensor_v2":
                return rebuild
            if mod == "torch" and name in dtypes:
                return name
            if mod == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            raise pickle.UnpicklingError("not allowed: %s.%s" % (mod, name))

        def persistent_load(self, pid):
            # ('storage', storage_type, key, location, numel)
            return Storage(dtypes[pid[1]], pid[2])

    obj = U(io.BytesIO(zf.read(pkl))).load()
    return obj.get("params", obj.get("params_ema", obj))


def build(sd):
    nodes, inits = [], []
    x = "input"
    i, n = 0, 0
    convs = sorted({int(k.split(".")[1]) for k in sd if k.startswith("body.")})
    for idx in convs:
        w = sd["body.%d.weight" % idx]
        name = "body.%d" % idx
        if w.ndim == 4:
            inits.append(numpy_helper.from_array(w, name + ".weight"))
            inits.append(numpy_helper.from_array(sd[name + ".bias"], name + ".bias"))
            y = "c%d" % n
            nodes.append(helper.make_node("Conv", [x, name + ".weight", name + ".bias"], [y],
                                          kernel_shape=[3, 3], pads=[1, 1, 1, 1], name=name))
        else:
            inits.append(numpy_helper.from_array(w.reshape(-1, 1, 1), name + ".weight"))
            y = "a%d" % n
            nodes.append(helper.make_node("PRelu", [x, name + ".weight"], [y], name=name))
        x, n = y, n + 1
    nodes.append(helper.make_node("DepthToSpace", [x], ["up"], blocksize=4, mode="CRD", name="shuffle"))
    inits.append(numpy_helper.from_array(np.array([1, 1, 4, 4], dtype=np.float32), "scales"))
    nodes.append(helper.make_node("Resize", ["input", "", "scales"], ["base"], mode="nearest",
                                  coordinate_transformation_mode="asymmetric", nearest_mode="floor",
                                  name="nearest"))
    nodes.append(helper.make_node("Add", ["up", "base"], ["output"], name="add"))
    g = helper.make_graph(
        nodes, "realesr-general-x4v3",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "h4", "w4"])],
        inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)],
                          producer_name="oros-models-vision convert_realesrgan.py")
    m.ir_version = 8
    m.doc_string = "Real-ESRGAN general x4v3 (c) 2021 Xintao Wang, BSD-3-Clause. Input RGB 0..1, output x4."
    onnx.checker.check_model(m)
    return m


def main():
    a = sys.argv[1:]
    if len(a) < 2:
        raise SystemExit(__doc__)
    src, out = a[0], a[1]
    if "--sha" in a:
        want = a[a.index("--sha") + 1]
        got = hashlib.sha256(open(src, "rb").read()).hexdigest()
        if got != want:
            raise SystemExit("sha256 mismatch: %s != %s" % (got, want))
    m = build(read_pth(src))
    with open(out, "wb") as f:
        f.write(m.SerializeToString())
    print(out, hashlib.sha256(open(out, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
