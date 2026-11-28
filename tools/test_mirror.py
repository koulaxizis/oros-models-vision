#!/usr/bin/env python3
"""Offline tests for tools/mirror.py's split of big files (standard library only).

Run: python3 tools/test_mirror.py
"""
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("mirror", os.path.join(HERE, "mirror.py"))
mirror = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mirror)

BIG = bytes(range(256)) * 41          # 10,496 bytes
SMALL = b'{"a": 1}\n'
REV = "a" * 40


def blob_id(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class FakeHub:
    def __init__(self):
        self.downloads = []
        self.files = {"onnx/model.onnx": BIG, "config.json": SMALL}

    def tree(self):
        return [
            {"type": "file", "path": "onnx/model.onnx", "size": len(BIG),
             "lfs": {"oid": hashlib.sha256(BIG).hexdigest(), "size": len(BIG)}},
            {"type": "file", "path": "config.json", "size": len(SMALL), "oid": blob_id(SMALL)},
        ]

    def get_json(self, url):
        if "/revision/" in url:
            return {"sha": REV}
        if url.endswith("/tree/%s/onnx" % REV):
            return [t for t in self.tree() if t["path"].startswith("onnx/")]
        return self.tree()

    def download(self, url, dest):
        p = url.split("/resolve/%s/" % REV, 1)[1]
        self.downloads.append(p)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(self.files[p])


class SplitTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        with open(os.path.join(self.root, "sources.json"), "w") as f:
            json.dump({"hf": [{"dir": "m", "repo": "x/m", "revision": "main",
                               "files": ["onnx/model.onnx", "config.json"]}]}, f)
        self.hub = FakeHub()
        self.saved = (mirror.ROOT, mirror.PART_BYTES, mirror.get_json, mirror.download)
        mirror.ROOT, mirror.PART_BYTES = self.root, 4000
        mirror.get_json, mirror.download = self.hub.get_json, self.hub.download

    def tearDown(self):
        mirror.ROOT, mirror.PART_BYTES, mirror.get_json, mirror.download = self.saved
        shutil.rmtree(self.root)

    def run_mirror(self, *args):
        import sys
        old = sys.argv
        sys.argv = ["mirror.py"] + list(args)
        try:
            mirror.main()
        finally:
            sys.argv = old

    def index(self):
        with open(os.path.join(self.root, "index.json")) as f:
            return json.load(f)["models"]["m"]["files"]

    def test_big_file_is_split_and_recorded(self):
        self.run_mirror()
        rec = self.index()["onnx/model.onnx"]
        base = os.path.join(self.root, "m", "onnx", "model.onnx")
        self.assertFalse(os.path.exists(base))
        self.assertEqual(rec["size"], len(BIG))
        self.assertEqual(rec["sha"], hashlib.sha256(BIG).hexdigest())
        self.assertEqual([p["size"] for p in rec["parts"]], [4000, 4000, 2496])
        joined = b""
        for i, p in enumerate(rec["parts"]):
            with open("%s.part%02d" % (base, i), "rb") as f:
                data = f.read()
            self.assertEqual(hashlib.sha256(data).hexdigest(), p["sha"])
            joined += data
        self.assertEqual(joined, BIG)
        self.assertNotIn("parts", self.index()["config.json"])
        self.run_mirror("--check")

    def test_rerun_keeps_parts_without_downloading(self):
        self.run_mirror()
        self.hub.downloads.clear()
        self.run_mirror()
        self.assertEqual(self.hub.downloads, [])
        self.assertEqual(len(self.index()["onnx/model.onnx"]["parts"]), 3)

    def test_check_fails_on_changed_or_missing_or_extra_part(self):
        self.run_mirror()
        base = os.path.join(self.root, "m", "onnx", "model.onnx")
        with open(base + ".part01", "r+b") as f:
            f.write(b"\0")
        with self.assertRaises(SystemExit):
            self.run_mirror("--check")
        os.remove(base + ".part01")
        with self.assertRaises(SystemExit):
            self.run_mirror("--check")
        self.run_mirror()                      # downloads again and splits
        self.run_mirror("--check")
        with open(base + ".part03", "wb") as f:
            f.write(b"x")
        with self.assertRaises(SystemExit):
            self.run_mirror("--check")

    def test_upstream_mismatch_is_refused(self):
        self.hub.files["onnx/model.onnx"] = BIG[:-1] + b"\1"
        with self.assertRaises(SystemExit):
            self.run_mirror()

    def test_local_file_in_parts_is_recorded_like_a_split_download(self):
        with open(os.path.join(self.root, "sources.json"), "w") as f:
            json.dump({"local": [{"dir": "l", "files": ["big.onnx"],
                                  "sha": {"big.onnx": hashlib.sha256(BIG).hexdigest()}}]}, f)
        base = os.path.join(self.root, "l", "big.onnx")
        os.makedirs(os.path.dirname(base))
        for i in range(3):
            with open("%s.part%02d" % (base, i), "wb") as f:
                f.write(BIG[i * 4000:(i + 1) * 4000])
        self.run_mirror()
        with open(os.path.join(self.root, "index.json")) as f:
            rec = json.load(f)["models"]["l"]["files"]["big.onnx"]
        self.assertEqual(rec["size"], len(BIG))
        self.assertEqual(rec["sha"], hashlib.sha256(BIG).hexdigest())
        self.assertEqual([p["size"] for p in rec["parts"]], [4000, 4000, 2496])
        self.run_mirror("--check")
        with open(base + ".part02", "r+b") as f:
            f.write(b"\0")
        with self.assertRaises(SystemExit):
            self.run_mirror("--check")


if __name__ == "__main__":
    unittest.main()
