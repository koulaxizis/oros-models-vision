#!/usr/bin/env python3
"""Check a deepfilter.wasm build against the native build of the same code.

Usage: python3 check.py <deepfilter.wasm> <liboros_deepfilter.so>
Prints JSON: wasmVsNativeDb (signal to difference, test signal, no limit),
noiseCutDb (white noise in vs out with a 24 dB limit, should be ~24),
realTimeFactor (wasm, one thread), hop, delay.
"""
import ctypes
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

SR = 48000
HERE = os.path.dirname(os.path.abspath(__file__))


def test_signal(n, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.7 * t)                  # a voice-like gliding pitch
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voiced = sum(np.sin(k * phase) / k for k in range(1, 20))
    gate = (np.sin(2 * np.pi * 1.5 * t) > -0.2).astype(np.float64)  # syllables and pauses
    return (0.25 * voiced * gate + 0.05 * rng.standard_normal(n)).astype(np.float32)


def native(lib, x, atten):
    L = ctypes.CDLL(lib)
    L.dfo_create.restype = ctypes.c_void_p
    L.dfo_create.argtypes = [ctypes.c_size_t, ctypes.c_float]
    L.dfo_hop.restype = ctypes.c_size_t
    L.dfo_hop.argtypes = [ctypes.c_void_p]
    L.dfo_process.restype = ctypes.c_float
    L.dfo_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    L.dfo_destroy.argtypes = [ctypes.c_void_p]
    st = L.dfo_create(1, atten)
    hop = L.dfo_hop(st)
    y = np.zeros_like(x)
    for f in range(0, len(x) - hop + 1, hop):
        i = np.ascontiguousarray(x[f:f + hop])
        o = np.zeros(hop, np.float32)
        L.dfo_process(st, i.ctypes.data, o.ctypes.data)
        y[f:f + hop] = o
    L.dfo_destroy(st)
    return y


def wasm(path, x, atten):
    with tempfile.TemporaryDirectory() as d:
        xi, yo = os.path.join(d, "x.f32"), os.path.join(d, "y.f32")
        x.tofile(xi)
        r = subprocess.run(["node", os.path.join(HERE, "run_wasm.cjs"), path, xi, yo, str(atten)],
                           check=True, capture_output=True, text=True)
        return np.fromfile(yo, dtype=np.float32), json.loads(r.stdout)


def main():
    wasm_path, lib = sys.argv[1], sys.argv[2]
    x = test_signal(SR * 10, 7)
    yw, info = wasm(wasm_path, x, 100)
    yn = native(lib, x, 100)
    diff = float(np.sum((yw - yn).astype(np.float64) ** 2))
    db = 10 * np.log10(float(np.sum(yn.astype(np.float64) ** 2)) / max(diff, 1e-30))
    noise = (0.1 * np.random.default_rng(3).standard_normal(SR * 6)).astype(np.float32)
    nw, _ = wasm(wasm_path, noise, 24)
    tail = slice(SR, None)  # after the first second
    cut = 10 * np.log10(float(np.sum(noise[tail].astype(np.float64) ** 2)) /
                        max(float(np.sum(nw[tail].astype(np.float64) ** 2)), 1e-30))
    print(json.dumps({
        "wasmVsNativeDb": round(min(db, 300.0), 1),
        "noiseCutDb": round(cut, 1),
        "realTimeFactor": round(info["ms"] / 1000 / 10, 3),
        "hop": info["hop"],
        "delay": info["delay"],
    }))


if __name__ == "__main__":
    main()
