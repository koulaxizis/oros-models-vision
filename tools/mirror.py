#!/usr/bin/env python3
"""Copy the model files listed in sources.json, unchanged, and write index.json.

Standard library only. For every Hugging Face entry it:
  1. pins the revision (an index.json pin wins over sources.json, so a
     re-run never moves to a newer upstream commit by itself);
  2. reads the upstream file list of that commit;
  3. downloads each file, refuses anything over 95 MB (GitHub's limit is
     100 MB per file) and checks it against upstream: LFS files by their
     SHA-256, small files by their git blob id;
  4. records size and SHA-256 in index.json (orOS writes these hashes into
     its own code and checks every download against them).
GitHub entries ("github": repo, revision, files) work the same way with
the GitHub API: the revision is pinned to a commit and every file must
match its git blob id in that commit's tree.
Files kept in "local" entries (copies of files orOS shipped itself) are
only hashed. A file already here with the recorded hash is not fetched again.

Usage: python3 tools/mirror.py [--check]   (--check: verify only, no network)
"""
import hashlib
import json
import os
import sys
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = "https://huggingface.co"
GH_API = "https://api.github.com"
GH_RAW = "https://raw.githubusercontent.com"
MAX_BYTES = 95 * 1000 * 1000
UA = {"User-Agent": "oros-models-vision-mirror"}


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_id(path):
    size = os.path.getsize(path)
    h = hashlib.sha1(b"blob %d\0" % size)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest))
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
            n = 0
            for chunk in iter(lambda: r.read(1 << 20), b""):
                n += len(chunk)
                if n > MAX_BYTES:
                    raise SystemExit("too big (> 95 MB): " + url)
                out.write(chunk)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    check_only = "--check" in sys.argv
    with open(os.path.join(ROOT, "sources.json"), encoding="utf-8") as f:
        sources = json.load(f)
    index_path = os.path.join(ROOT, "index.json")
    old = {}
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            old = json.load(f).get("models", {})
    models = {}
    bad = []

    for src in sources.get("hf", []):
        d, repo = src["dir"], src["repo"]
        prev = old.get(d, {})
        rev = prev.get("revision") if prev.get("repo") == repo else None
        files = {}
        if check_only:
            rev = rev or src["revision"]
            for p in src["files"]:
                dest = os.path.join(ROOT, d, p)
                want = prev.get("files", {}).get(p, {}).get("sha")
                if not os.path.exists(dest) or sha256_of(dest) != want:
                    bad.append(d + "/" + p)
            models[d] = prev
            continue
        if not rev or len(rev) != 40:
            rev = get_json("%s/api/models/%s/revision/%s" % (HF, repo, src["revision"]))["sha"]
        # The listing of a big repository is cut into pages, so each folder
        # that holds a wanted file is listed on its own (piper-voices has
        # thousands of files), plus the whole tree for the onnx/ summary.
        tree = get_json("%s/api/models/%s/tree/%s?recursive=true" % (HF, repo, rev))
        for folder in sorted({os.path.dirname(p) for p in src["files"] if os.path.dirname(p)}):
            tree += get_json("%s/api/models/%s/tree/%s/%s" % (HF, repo, rev, folder))
        by_path = {t["path"]: t for t in tree if t.get("type") == "file"}
        for p in src["files"]:
            t = by_path.get(p)
            if not t:
                raise SystemExit("not upstream: %s %s" % (repo, p))
            dest = os.path.join(ROOT, d, p)
            known = prev.get("files", {}).get(p, {}).get("sha")
            if not (known and os.path.exists(dest) and sha256_of(dest) == known):
                print("download", repo, p, flush=True)
                download("%s/%s/resolve/%s/%s" % (HF, repo, rev, p), dest)
            sha = sha256_of(dest)
            lfs = t.get("lfs")
            if lfs:
                if sha != lfs.get("oid") or os.path.getsize(dest) != lfs.get("size"):
                    raise SystemExit("hash differs from upstream LFS: %s %s" % (repo, p))
            elif git_blob_id(dest) != t.get("oid"):
                raise SystemExit("git blob id differs from upstream: %s %s" % (repo, p))
            files[p] = {"size": os.path.getsize(dest), "sha": sha}
        others = sorted((t["path"], t["size"]) for t in tree
                        if t.get("type") == "file" and t["path"].startswith("onnx/"))
        models[d] = {
            "repo": repo,
            "revision": rev,
            "source": "%s/%s/tree/%s" % (HF, repo, rev),
            "licence": src.get("licence", ""),
            "files": files,
            "upstreamOnnx": [{"path": p, "size": s} for p, s in others],
        }

    for src in sources.get("github", []):
        d, repo = src["dir"], src["repo"]
        prev = old.get(d, {})
        rev = prev.get("revision") if prev.get("repo") == repo else None
        files = {}
        if check_only:
            for p in src["files"]:
                dest = os.path.join(ROOT, d, os.path.basename(p))
                want = prev.get("files", {}).get(p, {}).get("sha")
                if not os.path.exists(dest) or sha256_of(dest) != want:
                    bad.append(d + "/" + p)
            models[d] = prev
            continue
        if not rev or len(rev) != 40:
            rev = get_json("%s/repos/%s/commits/%s" % (GH_API, repo, src["revision"]))["sha"]
        tree = get_json("%s/repos/%s/git/trees/%s?recursive=1" % (GH_API, repo, rev))
        if tree.get("truncated"):
            raise SystemExit("tree listing cut short: " + repo)
        by_path = {t["path"]: t for t in tree["tree"] if t.get("type") == "blob"}
        for p in src["files"]:
            t = by_path.get(p)
            if not t:
                raise SystemExit("not upstream: %s %s" % (repo, p))
            # kept flat: the folder holds the file under its own name
            dest = os.path.join(ROOT, d, os.path.basename(p))
            known = prev.get("files", {}).get(p, {}).get("sha")
            if not (known and os.path.exists(dest) and sha256_of(dest) == known):
                print("download", repo, p, flush=True)
                download("%s/%s/%s/%s" % (GH_RAW, repo, rev, p), dest)
            if git_blob_id(dest) != t.get("sha"):
                raise SystemExit("git blob id differs from upstream: %s %s" % (repo, p))
            files[p] = {"size": os.path.getsize(dest), "sha": sha256_of(dest), "file": os.path.basename(p)}
        models[d] = {
            "host": "github",
            "repo": repo,
            "revision": rev,
            "source": "https://github.com/%s/tree/%s" % (repo, rev),
            "licence": src.get("licence", ""),
            "files": files,
        }

    for src in sources.get("local", []):
        d = src["dir"]
        files = {}
        for p in src["files"]:
            dest = os.path.join(ROOT, d, p)
            if not os.path.exists(dest):
                raise SystemExit("missing local file: " + d + "/" + p)
            sha = sha256_of(dest)
            if src.get("sha", {}).get(p) and src["sha"][p] != sha:
                bad.append(d + "/" + p)
            files[p] = {"size": os.path.getsize(dest), "sha": sha}
        models[d] = {k: v for k, v in src.items() if k not in ("dir", "files", "sha")}
        models[d]["files"] = files

    if bad:
        raise SystemExit("hash mismatch: " + ", ".join(bad))
    if check_only:
        print("ok:", sum(len(m.get("files", {})) for m in models.values()), "files")
        return
    out = {"note": "Unchanged copies. size = bytes, sha = SHA-256. Written by tools/mirror.py; never edit by hand.",
           "models": models}
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("index.json written:", sum(len(m["files"]) for m in models.values()), "files")


if __name__ == "__main__":
    main()
