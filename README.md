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
| `piper-de-thorsten/` | Read aloud: the natural German voice Thorsten | Piper voice de_DE-thorsten-medium (rhasspy/piper-voices), unchanged | CC0 recordings, base model CC0 (`piper-de-thorsten/LICENSE`) | ~63 MB |
| `kokoro-82m/` | Read aloud: the natural English voices Heart and Michael | Kokoro-82M v1.0 by hexgrad, 8-bit ONNX export by onnx-community, voices af_heart and am_michael, unchanged | Apache-2.0 (`kokoro-82m/LICENSE`, with the CC BY credits of its training audio) | ~93 MB |
| `kokoro-82m-card/` | The model card of Kokoro-82M (licence and training data) | hexgrad/Kokoro-82M | Apache-2.0 | <0.1 MB |
| `misaki-en/` | Read aloud: how English words are said (for the Kokoro voices) | misaki US English dictionaries by hexgrad (us_gold, us_silver), unchanged | Apache-2.0 (`misaki-en/LICENSE`) | ~6 MB |
| `depth-anything-v2-small/` | Photo Lab: Portrait blur (background blur by depth) | Depth Anything V2 Small, int8 ONNX export by onnx-community | Apache-2.0 (Small only) | ~25 MB |
| `multilingual-e5-small/` | Search by meaning in Notes, Mail, Files, Reader and the Assistant (Greek, English, German and more) | intfloat multilingual-e5-small, int8 ONNX export by Xenova (model split into parts) | MIT (model), Apache-2.0 (export) | ~135 MB |
| `silero-vad/` | Notes dictation, Montage's Remove silences, the Assistant's voice input: finds where someone speaks | Silero VAD v6.2 (snakers4/silero-vad), unchanged | MIT | 2.3 MB |
| `movenet/` | Workouts: count repetitions with the camera (body keypoints only) | Google MoveNet SinglePose Lightning v4, ONNX converted from the float16 TFLite release | Apache-2.0 | 9.4 MB |
| `deepfilter/` | Studio, Montage, Notes: clean noise out of a voice | DeepFilterNet3 (official libDF runtime and model), our own WebAssembly build | MIT or Apache-2.0 | 14 MB |
| `basic-pitch/` | Studio: sound to MIDI notes | Spotify Basic Pitch `nmp.onnx`, unchanged | Apache-2.0 | 0.2 MB |

Each folder has the licence of its files. `index.json` records, for every
file, its size and SHA-256, and for the Hugging Face copies the upstream
repository and the exact commit they came from.

## How the copies are made

`sources.json` lists what to copy. `tools/mirror.py` (Python standard library
only) downloads each file from the pinned upstream commit, checks each one
against upstream (Hugging Face LFS files by their SHA-256, small files by their
git blob id) and writes `index.json`. The workflow `.github/workflows/mirror.yml` runs it
when `sources.json` changes on a `claude/**` branch and commits the result;
on pull requests it only checks the hashes (`python3 tools/mirror.py --check`).

### Files over 95 MB

GitHub refuses files over 100 MB, so `tools/mirror.py` cuts a file over 95 MB
into `<file>.part00`, `<file>.part01`, ... of at most 95,000,000 bytes each
and keeps only the parts. Its `index.json` record keeps the whole file's
`size` and `sha` (the upstream file's, checked before the cut) and a `parts`
list with each part's `size` and `sha`, in order. orOS downloads the parts,
checks each one, joins them and checks the whole file's SHA-256 before it
keeps it. `--check` checks every part and the joined hash, and fails on a
part left over from an older cut. `tools/test_mirror.py` tests the cut
offline (standard library only); the workflow runs it first.

`silero-vad/` is the file of the GitHub tag v6.2 of snakers4/silero-vad,
unchanged; its SHA-256 is pinned in `sources.json`.

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

`deepfilter/` is not a copy: DeepFilterNet ships its runtime as Rust.
`tools/deepfilter/build.py` (workflow `.github/workflows/build-deepfilter.yml`)
fetches the pinned upstream commit, checks the DeepFilterNet3 model archive's
SHA-256, and builds the official libDF runtime (tract) with that model inside,
behind the small C ABI in `tools/deepfilter/src/lib.rs`, for wasm32 with SIMD.
The module imports nothing: orOS runs it with `WebAssembly.instantiate(bytes,
{})`, no JavaScript glue and no eval. Every crate is pinned by
`tools/deepfilter/Cargo.lock`. The build is refused unless its output equals
a native build of the same code within float rounding (at least 100 dB) and
pure noise is cut by about 24 dB with a 24 dB limit; `deepfilter/build.json`
records the commit, the hashes and the check.

`basic-pitch/` is a GitHub source in `sources.json`: `tools/mirror.py` pins
the commit and checks each file's git blob id in that commit's tree.

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

`movenet/` is not a copy either: Google publishes MoveNet only as TFLite and
TensorFlow models. `tools/export_movenet.py` (workflow
`.github/workflows/export-movenet.yml`) downloads the float16 TFLite release,
converts it with tf2onnx (opset 17), renames the input to `input` (uint8
`[1, 192, 192, 3]`, RGB 0..255) and the output to `keypoints` (float32
`[1, 1, 17, 3]`: y, x, score), and refuses the file unless ONNX Runtime gives
the same keypoints as the TFLite interpreter (largest difference under 1e-4)
and finds the person in a test photo. `movenet/export.json` records the
source, both hashes, the tool versions and the check. tf2onnx does not write
byte-identical files from run to run, so once the hash is pinned in
`sources.json` a run keeps the file here and only checks it against TFLite
again.

### Adding a model

1. Add it to `sources.json` (Hugging Face: `repo`, `revision`, `files`,
   `licence`), with a `LICENSE` file in its folder.
2. Push to a `claude/**` branch: the workflow copies the files and writes
   `index.json`. Never edit `index.json` or a model file by hand.
3. Write the hashes from `index.json` into the orOS code that downloads it.

The site is about 615 MB with `multilingual-e5-small/`; keep it under 1 GB.
Models too big for that (Whisper small, Florence-2) go to a third repository.

Security: see [koulaxizis/oros-models SECURITY.md](https://github.com/koulaxizis/oros-models/blob/main/SECURITY.md).
