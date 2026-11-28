# orOS models: vision and speech

Model files that [orOS](https://useoros.online) downloads on the person's
device, only when the person asks for them, and runs there with ONNX Runtime
Web. Nothing a model sees or hears leaves the device. Each file is an exact
copy of its upstream release, and orOS checks it by SHA-256 before it keeps
it.

Served from GitHub Pages: `https://koulaxizis.github.io/oros-models-vision/`.
The translation models and the Mockups garments live in
[koulaxizis/oros-models](https://github.com/koulaxizis/oros-models); a second
repository keeps each site under GitHub Pages' 1 GB.

| Folder | What orOS does with it | Model | Licence | Download |
|---|---|---|---|---|
| `clip-vit-base-patch32/` | Files › Photos: search photos by description | OpenAI CLIP ViT-B/32, int8 ONNX export by Xenova | MIT (model), Apache-2.0 (export) | ~150 MB |
| `whisper-tiny/` | Montage, Lyric Sync, Notes: speech to text | OpenAI Whisper tiny, int8 ONNX export by Xenova | MIT (model), Apache-2.0 (export) | ~42 MB |
| `whisper-base/` | The Assistant's voice input (better at Greek) | OpenAI Whisper base, int8 ONNX export by Xenova | MIT (model), Apache-2.0 (export) | ~77 MB |
| `u2netp/` | Atelier, Photo Lab: background removal | U²-Net u2netp, ONNX from rembg | Apache-2.0 | 4.6 MB |
| `migan/` | Atelier: Magic eraser and Magic grab | MI-GAN 512 Places2, ONNX export of the IOPaint copy | MIT | 27 MB |
| `yunet/` | Blur faces before sharing (detection only, never recognition) | YuNet 2023mar, OpenCV Zoo | MIT | 0.2 MB |
| `demucs/` | Studio: separate vocals, drums, bass and the rest of a song | Meta Demucs v4 `htdemucs`, our own ONNX export (float16 weights) | MIT | 90 MB |
| `realesrgan/` | Photo Lab: Enlarge with AI 2×/4× | Real-ESRGAN general x4v3 (SRVGGNetCompact), ONNX written from the official weights | BSD-3-Clause | 4.9 MB |
| `piper-el-joy/` | Read aloud: the natural Greek voice Joy | Piper voice el_GR-joy-medium (rhasspy/piper-voices), unchanged | CC BY-NC 4.0, non-commercial use only (`piper-el-joy/LICENSE`) | ~63 MB |
| `depth-anything-v2-small/` | Photo Lab: Portrait blur (background blur by depth) | Depth Anything V2 Small, int8 ONNX export by onnx-community | Apache-2.0 (Small only) | ~25 MB |

Each folder has the licence of its files. `index.json` records, for every
file, its size and SHA-256, and for the Hugging Face copies the upstream
repository and the exact commit they came from.

## How the copies are made

`sources.json` lists what to copy. `tools/mirror.py` (Python standard library
only) downloads each file from the pinned upstream commit, refuses files over
95 MB (GitHub's limit is 100 MB per file), checks each one against upstream
(Hugging Face LFS files by their SHA-256, small files by their git blob id)
and writes `index.json`. The workflow `.github/workflows/mirror.yml` runs it
when `sources.json` changes on a `claude/**` branch and commits the result;
on pull requests it only checks the hashes (`python3 tools/mirror.py --check`).

`u2netp/` and `migan/` are the same bytes orOS shipped in its own `vendor/`
folder (u2netp is also byte for byte rembg's release file); the script only
hashes them.

`demucs/` is not a copy: Meta publishes Demucs only
as a PyTorch checkpoint. `tools/export_demucs.py` (workflow
`.github/workflows/export-demucs.yml`) downloads that checkpoint, checks its
hash, exports the model without its STFT (orOS does the STFT in JavaScript,
exactly as `tools/demucs_spec.py`), stores the weights as float16, and
refuses the file unless the whole separation as orOS does it
(`tools/demucs_ref.py`) matches demucs' own `apply_model` on a test signal by
at least 50 dB. `demucs/export.json` records the checkpoint's hash, the tool
versions and the check's result.

`realesrgan/` is not a copy either: its upstream release is only a PyTorch
`.pth`. `tools/convert_realesrgan.py` reads the official
`realesr-general-x4v3.pth` (release v0.2.5.0, SHA-256 checked) with the
standard library and numpy, without PyTorch, and writes the same network as
an ONNX graph (opset 17, input RGB 0..1 `[1, 3, H, W]`, output `[1, 3, 4H,
4W]`). The output is byte for byte reproducible:

    pip install numpy==2.5.5 onnx==1.23.2
    curl -LO https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
    python3 tools/convert_realesrgan.py realesr-general-x4v3.pth realesrgan/realesr-general-x4v3.onnx \
      --sha 8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292

It was checked against a plain numpy run of the PyTorch network (largest
difference 1.6e-6) with ONNX Runtime.

### Adding a model

1. Add it to `sources.json` (Hugging Face: `repo`, `revision`, `files`,
   `licence`), with a `LICENSE` file in its folder.
2. Push to a `claude/**` branch: the workflow copies the files and writes
   `index.json`. Never edit `index.json` or a model file by hand.
3. Write the hashes from `index.json` into the orOS code that downloads it.

The site is about 450 MB now; keep it under 1 GB.

Security: see [koulaxizis/oros-models SECURITY.md](https://github.com/koulaxizis/oros-models/blob/main/SECURITY.md).
