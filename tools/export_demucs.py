#!/usr/bin/env python3
"""Export Demucs v4 'htdemucs' (Meta, MIT) to ONNX for orOS Studio's
"Separate vocals and instruments".

There is no official ONNX file, so this script makes one from Meta's own
checkpoint and proves it right:

  1. downloads the checkpoint demucs uses by default (hybrid_transformer/
     955717e8-8726e21a.th) and checks it: its SHA-256 must start with
     8726e21a (demucs' own check) and equal CKPT_SHA once that is pinned;
  2. exports HTDemucs.forward WITHOUT its STFT and inverse STFT (complex
     tensors do not export); orOS does those in JavaScript exactly as
     tools/demucs_spec.py does here;
  3. stores the weights as float16 with a Cast back to float32 in the graph
     (half the size; every operation still runs in float32);
  4. checks the whole separation as orOS does it (tools/demucs_ref.py: STFT,
     model, inverse STFT, 7.8 s segments, 25% overlap, triangle weights)
     against demucs' own apply_model on a 20 s test signal, and refuses an
     export below MIN_SNR_DB;
  5. writes demucs/htdemucs.onnx and demucs/export.json (where it came from,
     the check's result, the model's inputs and outputs).

The graph:
  inputs  mix  float32 [1, 2, 343980]        one 7.8 s stereo segment, 44.1 kHz,
                                             already normalised for the whole track
          mag  float32 [1, 4, 2048, 336]     the segment's STFT (L re, L im, R re, R im)
  outputs spec float32 [1, 4, 4, 2048, 336]  per source, the same layout
          wave float32 [1, 4, 2, 343980]     per source, the time branch
  sources: drums, bass, other, vocals; a source = iSTFT(spec) + wave

Usage:
  python3 tools/export_demucs.py            export (skips when the file is already
                                            there with the hash in sources.json)
  python3 tools/export_demucs.py --random   random weights of the same shape, into
                                            a temp folder: for testing the script
"""
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.request
from fractions import Fraction

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demucs_ref as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_URL = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th"
CKPT_PREFIX = "8726e21a"
CKPT_SHA = None        # pinned after the first export (full SHA-256 of the checkpoint)
MIN_SNR_DB = 50.0      # orOS's separation vs demucs' apply_model, float16 weights included
OPSET = 17
SOURCES = ["drums", "bass", "other", "vocals"]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Core(torch.nn.Module):
    """HTDemucs.forward (demucs 4.1.0) from the normalisation to the
    de-normalisation, without _spec, _mask and _ispec."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, mix, mag):
        m = self.m
        x = mag
        B, C, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)
        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = m.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + m.freq_emb_scale * emb
            saved.append(x)
        if m.crosstransformer:
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                x = m.channel_upsampler(x)
                x = x.reshape(b, -1, f, t)
                xt = m.channel_upsampler_t(xt)
            x, xt = m.crosstransformer(x, xt)
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                x = m.channel_downsampler(x)
                x = x.reshape(b, -1, f, t)
                xt = m.channel_downsampler_t(xt)
        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)
        S = len(m.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]
        xt = xt.view(B, S, -1, mix.shape[-1])
        xt = xt * stdt[:, None] + meant[:, None]
        return x, xt


def load_model(random_weights, work):
    from demucs.htdemucs import HTDemucs
    if random_weights:
        torch.manual_seed(0)
        m = HTDemucs(SOURCES, dconv_mode=3, bottom_channels=512, segment=Fraction(39, 5))
        return m, {"checkpoint": "random (test run)"}
    from demucs.states import set_state
    ckpt = os.path.join(work, "htdemucs.th")
    print("download", CKPT_URL, flush=True)
    urllib.request.urlretrieve(CKPT_URL, ckpt)
    sha = sha256_of(ckpt)
    if not sha.startswith(CKPT_PREFIX) or (CKPT_SHA and sha != CKPT_SHA):
        raise SystemExit("checkpoint hash differs: " + sha)
    pkg = torch.load(ckpt, map_location="cpu", weights_only=False)
    if getattr(pkg["klass"], "__name__", "") != "HTDemucs":
        raise SystemExit("not an HTDemucs checkpoint: %r" % pkg["klass"])
    m = HTDemucs(*pkg["args"], **pkg["kwargs"])
    set_state(m, pkg["state"])
    if list(m.sources) != SOURCES or m.samplerate != 44100 or m.audio_channels != 2:
        raise SystemExit("unexpected model: %r %r" % (m.sources, m.samplerate))
    return m, {"checkpoint": CKPT_URL, "checkpointSha256": sha,
               "kwargs": {k: (str(v) if isinstance(v, Fraction) else v) for k, v in pkg["kwargs"].items()}}


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
    m.eval()
    torch.backends.mha.set_fastpath_enabled(False)   # the fused attention kernel has no ONNX export
    L = int(m.segment * m.samplerate)
    T = -(-L // m.hop_length)
    assert (L, T, m.nfft) == (R.SEG, 336, 4096), (L, T, m.nfft)
    f32 = os.path.join(work, "htdemucs.f32.onnx")
    with torch.no_grad():
        torch.onnx.export(Core(m).eval(), (torch.randn(1, 2, L), torch.randn(1, 4, m.nfft // 2, T)), f32,
                          input_names=["mix", "mag"], output_names=["spec", "wave"],
                          opset_version=OPSET, dynamo=False, do_constant_folding=True)
    fp16_weights(f32, out_path)


def test_signal(seconds=20.3):
    """Deterministic, no copyrighted audio: chords, a pulsing bass, noise bursts."""
    L = int(seconds * 44100)
    t = np.arange(L) / 44100
    rng = np.random.default_rng(7)
    beat = (np.sin(2 * np.pi * 2 * t) > 0.95).astype(np.float64)
    left = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 277.2 * t) \
        + 0.3 * np.sin(2 * np.pi * 55 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)) \
        + 0.2 * beat * rng.standard_normal(L)
    right = 0.2 * np.sin(2 * np.pi * 330 * t) + 0.3 * np.sin(2 * np.pi * 55 * t) \
        + 0.2 * beat * rng.standard_normal(L) + 0.01 * rng.standard_normal(L)
    return np.stack([left, right]).astype(np.float32)


def check(m, onnx_path):
    import onnxruntime as ort
    from demucs.apply import apply_model
    mix = test_signal()
    ref = mix.mean(0)
    mean, std = float(ref.mean()), float(ref.std(ddof=1)) + 1e-8
    with torch.no_grad():
        exp = apply_model(m, ((torch.from_numpy(mix) - mean) / std)[None],
                          shifts=0, split=True, overlap=R.OVERLAP)[0].numpy() * std + mean
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    s = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    t0 = time.time()
    got = R.separate(s, mix)
    secs = time.time() - t0
    snr = [float(10 * np.log10((exp[k] ** 2).sum() / max(1e-30, ((got[k] - exp[k]) ** 2).sum())))
           for k in range(4)]
    return {"signalSeconds": round(mix.shape[-1] / 44100, 2), "snrDb": dict(zip(SOURCES, [round(v, 1) for v in snr])),
            "maxAbsErr": float(np.abs(got - exp).max()),
            "nativeOneThreadSecondsPerMinute": round(secs / (mix.shape[-1] / 44100) * 60, 1)}


def main():
    random_weights = "--random" in sys.argv
    out_dir = tempfile.mkdtemp() if random_weights else os.path.join(ROOT, "demucs")
    out_path = os.path.join(out_dir, "htdemucs.onnx")
    if not random_weights:
        with open(os.path.join(ROOT, "sources.json"), encoding="utf-8") as f:
            entry = next((e for e in json.load(f).get("local", []) if e.get("dir") == "demucs"), {})
        want = entry.get("sha", {}).get("htdemucs.onnx")
        if want and os.path.exists(out_path) and sha256_of(out_path) == want:
            print("demucs/htdemucs.onnx is there with the pinned hash; nothing to do")
            return
    with tempfile.TemporaryDirectory() as work:
        m, origin = load_model(random_weights, work)
        export(m, out_path, work)
        result = check(m, out_path)
    print(json.dumps(result, indent=1))
    if min(result["snrDb"].values()) < MIN_SNR_DB:
        raise SystemExit("the ONNX separation differs from demucs (below %.0f dB)" % MIN_SNR_DB)
    size = os.path.getsize(out_path)
    if size > 95 * 1000 * 1000:
        raise SystemExit("too big for GitHub: %d bytes" % size)
    import onnx
    import onnxruntime
    info = {
        "note": "Written by tools/export_demucs.py; never edit by hand.",
        "model": "htdemucs (Demucs v4, Hybrid Transformer), Meta, MIT",
        **origin,
        "file": {"name": "htdemucs.onnx", "size": size, "sha": sha256_of(out_path)},
        "tools": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__,
                  "demucs": __import__("demucs").__version__, "opset": OPSET},
        "graph": {"inputs": {"mix": [1, 2, R.SEG], "mag": [1, 4, 2048, 336]},
                  "outputs": {"spec": [1, 4, 4, 2048, 336], "wave": [1, 4, 2, R.SEG]},
                  "sources": SOURCES, "sampleRate": 44100, "nfft": 4096, "hop": 1024,
                  "segment": R.SEG, "overlap": R.OVERLAP, "weightsFloat16": True},
        "check": result,
    }
    with open(os.path.join(out_dir, "export.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
        f.write("\n")
    print("written", out_path, size, info["file"]["sha"])


if __name__ == "__main__":
    main()
