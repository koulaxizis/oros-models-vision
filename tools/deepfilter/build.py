#!/usr/bin/env python3
"""Build deepfilter/deepfilter.wasm: DeepFilterNet3 for orOS.

The module is the official libDF runtime (Rust, tract) of
Rikorose/DeepFilterNet with the official DeepFilterNet3 model built in,
wrapped by tools/deepfilter/src/lib.rs in a plain C ABI, compiled to
WebAssembly (SIMD) with no imports at all: no JavaScript glue, no eval, no
entropy source. orOS loads it with WebAssembly.instantiate(bytes, {}).

Steps:
  1. upstream commit: the one recorded in deepfilter/build.json, else the
     current main (recorded on the first run, kept by later runs);
  2. fetch that commit, check the model archive's SHA-256 (MODEL_SHA);
  3. drop what a library build never needs from libDF/Cargo.toml (the
     [[bin]] targets and the optional hdf5 git dependency of its dataset
     feature) so tools/deepfilter/Cargo.lock pins every crate (--locked);
  4. build for wasm32 and for this machine with the same lock;
  5. check (tools/deepfilter/check.py): the wasm's output equals the native
     build's within float rounding (>= 100 dB) on a test signal, and pure
     noise is cut by at least 20 dB with a 24 dB limit;
  6. write deepfilter/deepfilter.wasm, its licences and deepfilter/build.json.

Needs: git, Rust (rustup, the toolchain in TOOLCHAIN with wasm32), node,
python3 with numpy. Usage: python3 tools/deepfilter/build.py
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(ROOT, "tools", "deepfilter")
SRC = os.path.join(ROOT, "build", "DeepFilterNet")
OUT = os.path.join(ROOT, "deepfilter")
UPSTREAM = "https://github.com/Rikorose/DeepFilterNet"
MODEL = "models/DeepFilterNet3_onnx.tar.gz"
MODEL_SHA = "c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616"
TOOLCHAIN = "1.97.0"
WASM_TARGET = "wasm32-unknown-unknown"


def run(cmd, cwd=None, env=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def out(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pinned_rev():
    p = os.path.join(OUT, "build.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            rev = json.load(f).get("upstreamCommit", "")
        if re.fullmatch(r"[0-9a-f]{40}", rev):
            return rev
    line = out(["git", "ls-remote", UPSTREAM, "refs/heads/main"])
    return line.split()[0]


def fetch(rev):
    if not os.path.isdir(os.path.join(SRC, ".git")):
        os.makedirs(SRC, exist_ok=True)
        run(["git", "init", "-q"], cwd=SRC)
        run(["git", "remote", "add", "origin", UPSTREAM], cwd=SRC)
    run(["git", "fetch", "-q", "--depth", "1", "origin", rev], cwd=SRC)
    run(["git", "checkout", "-q", "--force", rev], cwd=SRC)
    if out(["git", "rev-parse", "HEAD"], cwd=SRC) != rev:
        raise SystemExit("upstream checkout is not " + rev)
    got = sha256_of(os.path.join(SRC, MODEL))
    if got != MODEL_SHA:
        raise SystemExit("model archive hash differs: " + got)


def trim_manifest():
    p = os.path.join(SRC, "libDF", "Cargo.toml")
    lines, keep, skip = open(p, encoding="utf-8").read().split("\n"), [], False
    for line in lines:
        if line.strip() == "[[bin]]":
            skip = True
            continue
        if skip and line.startswith("["):
            skip = False
        if skip:
            continue
        if line.startswith("hdf5") or '"dep:hdf5",' in line:
            continue
        keep.append(line)
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(keep))


def build():
    env = dict(os.environ)
    # const-random (via rust-ini) bakes a random hash seed into the code:
    # a fixed seed keeps the build reproducible.
    env["CONST_RANDOM_SEED"] = "oros-deepfilter"
    remap = "--remap-path-prefix=%s=/src --remap-path-prefix=%s=/cargo" % (
        ROOT, os.path.expanduser("~/.cargo"))
    cargo = ["cargo", "+" + TOOLCHAIN, "build", "--release", "--locked"]
    env["RUSTFLAGS"] = "-C target-feature=+simd128 " + remap
    run(cargo + ["--target", WASM_TARGET], cwd=HERE, env=env)
    env["RUSTFLAGS"] = remap
    run(cargo, cwd=HERE, env=env)
    wasm = os.path.join(HERE, "target", WASM_TARGET, "release", "oros_deepfilter.wasm")
    native = os.path.join(HERE, "target", "release", "liboros_deepfilter.so")
    return wasm, native


def main():
    rev = pinned_rev()
    fetch(rev)
    trim_manifest()
    wasm, native = build()
    report = json.loads(out([sys.executable, os.path.join(HERE, "check.py"), wasm, native]))
    if report["wasmVsNativeDb"] < 100 or report["noiseCutDb"] < 20:
        raise SystemExit("check failed: " + json.dumps(report))
    os.makedirs(OUT, exist_ok=True)
    shutil.copyfile(wasm, os.path.join(OUT, "deepfilter.wasm"))
    for name in ("LICENSE-MIT", "LICENSE-APACHE"):
        shutil.copyfile(os.path.join(SRC, name), os.path.join(OUT, name))
    info = {
        "note": "Written by tools/deepfilter/build.py; never edit by hand.",
        "upstream": UPSTREAM,
        "upstreamCommit": rev,
        "model": MODEL,
        "modelSha256": MODEL_SHA,
        "rustc": out(["rustc", "+" + TOOLCHAIN, "--version"]),
        "wasmSize": os.path.getsize(os.path.join(OUT, "deepfilter.wasm")),
        "wasmSha256": sha256_of(os.path.join(OUT, "deepfilter.wasm")),
        "check": report,
    }
    with open(os.path.join(OUT, "build.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
        f.write("\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
