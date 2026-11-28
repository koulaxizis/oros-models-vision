#!/usr/bin/env python3
"""Copy the model files listed in sources.json, unchanged, and write index.json.

Standard library only. For every Hugging Face entry it:
  1. pins the revision (an index.json pin wins over sources.json, so a
     re-run never moves to a newer upstream commit by itself);
  2. reads the upstream file list of that commit;
  3. downloads each file and checks it against upstream: LFS files by their
     SHA-256, small files by their git blob id;
  4. splits a file over 95 MB (GitHub's limit is 100 MB per file) into
     "<file>.part00", "<file>.part01", ... of at most 95 MB each and keeps
     only the parts (orOS downloads the parts, joins them and checks the
     whole file's SHA-256);
  5. records size and SHA-256 in index.json, for a split file also each
     part's ("parts": [{size, sha}, ...]); orOS writes these hashes into its
     own code and checks every download against them.
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
PART_BYTES = 95 * 1000 * 1000      # largest file kept in the repository
MAX_BYTES = 1000 * 1000 * 1000     # largest upstream file accepted at all
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
                    raise SystemExit("too big (> 1 GB): " + url)
                out.write(chunk)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def part_path(dest, i):
    return "%s.part%02d" % (dest, i)


def split(dest):
    """Cut dest into parts of at most PART_BYTES, remove it, return the parts' records."""
    parts = []
    with open(dest, "rb") as f:
        i = 0
        while True:
            chunk = f.read(PART_BYTES)
            if not chunk:
                break
            with open(part_path(dest, i), "wb") as out:
                out.write(chunk)
            parts.append({"size": len(chunk), "sha": hashlib.sha256(chunk).hexdigest()})
            i += 1
    os.remove(dest)
    return parts


def parts_ok(dest, rec):
    """True when every recorded part is here with its hash and they join to the whole file's hash."""
    parts = rec.get("parts") or []
    if not parts:
        return False
    whole = hashlib.sha256()
    for i, p in enumerate(parts):
        path = part_path(dest, i)
        if not os.path.exists(path) or os.path.getsize(path) != p.get("size"):
            return False
        with open(path, "rb") as f:
            data = f.read()
        if hashlib.sha256(data).hexdigest() != p.get("sha"):
            return False
        whole.update(data)
    if os.path.exists(part_path(dest, len(parts))):
        return False
    return whole.hexdigest() == rec.get("sha") and sum(p["size"] for p in parts) == rec.get("size")


def whole_ok(dest, rec):
    return (not rec.get("parts") and bool(rec.get("sha")) and os.path.exists(dest)
            and sha256_of(dest) == rec["sha"])


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
                rec = prev.get("files", {}).get(p, {})
                if not (whole_ok(dest, rec) or parts_ok(dest, rec)):
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
            rec = prev.get("files", {}).get(p, {})
            lfs = t.get("lfs")
            if parts_ok(dest, rec):
                # Already split here; the parts join to the recorded hash,
                # which was checked against upstream when they were made.
                if lfs and (rec["sha"] != lfs.get("oid") or rec["size"] != lfs.get("size")):
                    raise SystemExit("hash differs from upstream LFS: %s %s" % (repo, p))
                files[p] = {"size": rec["size"], "sha": rec["sha"], "parts": rec["parts"]}
                continue
            if not whole_ok(dest, rec):
                print("download", repo, p, flush=True)
                download("%s/%s/resolve/%s/%s" % (HF, repo, rev, p), dest)
            sha = sha256_of(dest)
            if lfs:
                if sha != lfs.get("oid") or os.path.getsize(dest) != lfs.get("size"):
                    raise SystemExit("hash differs from upstream LFS: %s %s" % (repo, p))
            elif git_blob_id(dest) != t.get("oid"):
                raise SystemExit("git blob id differs from upstream: %s %s" % (repo, p))
            files[p] = {"size": os.path.getsize(dest), "sha": sha}
            if files[p]["size"] > PART_BYTES:
                print("split", repo, p, flush=True)
                files[p]["parts"] = split(dest)
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
            if os.path.getsize(dest) > PART_BYTES:
                raise SystemExit("too big for a github entry: %s %s" % (repo, p))
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
