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
| `piper-el-joy/` | Read aloud: the natural Greek voice Joy | Piper voice el_GR-joy-medium (rhasspy/piper-voices), unchanged | CC BY-NC 4.0, non-commercial use only (`piper-el-joy/LICENSE`) | ~63 MB |

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

### Adding a model

1. Add it to `sources.json` (Hugging Face: `repo`, `revision`, `files`,
   `licence`), with a `LICENSE` file in its folder.
2. Push to a `claude/**` branch: the workflow copies the files and writes
   `index.json`. Never edit `index.json` or a model file by hand.
3. Write the hashes from `index.json` into the orOS code that downloads it.

The site is about 360 MB now; keep it under 1 GB.

Security: see [koulaxizis/oros-models SECURITY.md](https://github.com/koulaxizis/oros-models/blob/main/SECURITY.md).
