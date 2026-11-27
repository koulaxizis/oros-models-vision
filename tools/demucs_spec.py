"""NumPy copy of HTDemucs._spec / _ispec (demucs 4.1.0). orOS studio/stems.js follows it line by line."""
import numpy as np
NFFT, HOP = 4096, 1024

def hann(n):  # torch.hann_window periodic
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)

def reflect(x, l, r):
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(l, r)], mode="reflect")

def spec(x):  # x [C, L] -> complex [C, 2048, T]
    L = x.shape[-1]; le = -(-L // HOP); pad = HOP // 2 * 3
    x = reflect(x, pad, pad + le * HOP - L)
    x = reflect(x, NFFT // 2, NFFT // 2)          # torch.stft center=True
    w = hann(NFFT); n = (x.shape[-1] - NFFT) // HOP + 1
    fr = np.stack([x[..., i * HOP:i * HOP + NFFT] * w for i in range(n)], -1)  # C, NFFT, n
    z = np.fft.rfft(fr, axis=-2) / np.sqrt(NFFT)  # normalized=True
    z = z[..., :-1, :]
    return z[..., 2:2 + le]

def mag(z):  # CaC [C,F,T] -> [2C,F,T]
    C, F, T = z.shape
    return np.stack([z.real, z.imag], 1).reshape(C * 2, F, T).astype(np.float32)

def ispec(z, length):  # z [..., 2048, T] complex -> [..., length]
    pad = HOP // 2 * 3
    z = np.concatenate([z, np.zeros(z.shape[:-2] + (1, z.shape[-1]), z.dtype)], -2)
    z = np.pad(z, [(0, 0)] * (z.ndim - 1) + [(2, 2)])
    le = HOP * -(-length // HOP) + 2 * pad
    w = hann(NFFT); T = z.shape[-1]
    fr = np.fft.irfft(z * np.sqrt(NFFT), n=NFFT, axis=-2) * w[:, None]
    full = (T - 1) * HOP + NFFT
    y = np.zeros(z.shape[:-2] + (full,)); ws = np.zeros(full)
    for i in range(T):
        y[..., i * HOP:i * HOP + NFFT] += fr[..., i]; ws[i * HOP:i * HOP + NFFT] += w * w
    y = y[..., NFFT // 2:NFFT // 2 + le] / ws[NFFT // 2:NFFT // 2 + le]
    return y[..., pad:pad + length]
