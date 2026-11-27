"""The whole separation in NumPy + ONNX Runtime, step for step as orOS does it
(studio/stems.js). Compared with demucs.apply.apply_model(split=True,
overlap=0.25, shifts=0) by export_demucs.py --check."""
import numpy as np
import demucs_spec as S

SR, SEG = 44100, 343980          # int(7.8 * 44100)
OVERLAP = 0.25
STRIDE = int((1 - OVERLAP) * SEG)


def weight():
    w = np.concatenate([np.arange(1, SEG // 2 + 1), np.arange(SEG - SEG // 2, 0, -1)]).astype(np.float64)
    return w / w.max()


def chunk(mix, offset):
    """TensorChunk(mix, offset, SEG).padded(SEG): zeros outside the track;
    a short last chunk is centred, so it takes samples before `offset`."""
    total = mix.shape[-1]
    length = min(total - offset, SEG)
    delta = SEG - length
    start = offset - delta // 2
    end = start + SEG
    cs, ce = max(0, start), min(total, end)
    out = np.zeros((mix.shape[0], SEG), np.float32)
    out[:, cs - start:cs - start + (ce - cs)] = mix[:, cs:ce]
    return out, length, delta


def run_segment(session, seg):
    z = S.spec(seg.astype(np.float64))
    spec, wave = session.run(None, {"mix": seg[None], "mag": S.mag(z)[None]})
    sp = spec[0].astype(np.float64)
    zc = sp[:, 0::2] + 1j * sp[:, 1::2]
    return S.ispec(zc, SEG) + wave[0]          # [4, 2, SEG]


def separate(session, mix):
    """mix float32 [2, L] at 44.1 kHz -> float32 [4, 2, L] (drums, bass, other, vocals)."""
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std(ddof=1) + 1e-8
    x = ((mix - mean) / std).astype(np.float32)
    L = x.shape[-1]
    out = np.zeros((4, 2, L))
    sw = np.zeros(L)
    w = weight()
    for offset in range(0, L, STRIDE):
        seg, length, delta = chunk(x, offset)
        y = run_segment(session, seg)
        y = y[..., delta // 2:delta // 2 + length]         # center_trim
        out[..., offset:offset + length] += w[:length] * y
        sw[offset:offset + length] += w[:length]
    out /= sw
    return (out * std + mean).astype(np.float32)
