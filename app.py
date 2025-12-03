import sys
import os
import glob
import json
import math
import numpy as np
import soundfile as sf

from scipy.signal import resample_poly

from librosa.core.audio import load as lr_load
from librosa.core.spectrum import stft as lr_stft, istft as lr_istft

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QProgressBar,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QMessageBox,
    QSpacerItem,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QGroupBox,
)

# ----------------- DEFAULTS -----------------
DEFAULT_FRAME_LENGTH = 2048
DEFAULT_HOP_LENGTH = 256

WT_DIR_DEFAULT = r"D:\WAVETABLE"
AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".m4a")

INDEX_FILENAME = "wt_index.npz"
INDEX_FEAT_DIM = 64
INDEX_FRAME_SIZE = 2048
INDEX_SMOOTH_BINS = 9

# Target quality
TARGET_SR = 96000
OUTPUT_SUBTYPE = "PCM_24"  # 24-bit WAV

# Hit analysis defaults
DEFAULT_MIN_DIST_MS = 20
DEFAULT_THRESH = 0.25
DEFAULT_BODY_START_MS = 10
DEFAULT_BODY_DUR_MS = 45

# Synthesis defaults
DEFAULT_MAX_HIT_MS = 250
DEFAULT_ATTACK_MS = 2
DEFAULT_AIR_HPF = 8000.0
DEFAULT_CLICK_HPF = 3000.0
DEFAULT_VEL_GAMMA = 1.15

# Pitch defaults
DEFAULT_PITCH_ENABLE = True
DEFAULT_PITCH_FMIN = 40.0
DEFAULT_PITCH_FMAX = 900.0
DEFAULT_TONALITY_THR = 0.25     # más alto => más estricto
DEFAULT_PITCH_ENV_SEMI = 12.0   # caída inicial (semitonos, positivo = más alto al inicio)
DEFAULT_PITCH_ENV_MS = 15.0
DEFAULT_PITCH_ENV_STRENGTH = 0.8

# Multi-candidate defaults
DEFAULT_MIX_N = 3
DEFAULT_MIX_TEMP = 0.08  # más bajo => más “elige el mejor”; más alto => mezcla más


# ----------------- UTILS -----------------
def _to_mono_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    return x.astype(np.float32, copy=False)


def _fade_edges(frame: np.ndarray, fade: int = 16) -> np.ndarray:
    if fade <= 0 or 2 * fade >= len(frame):
        return frame
    w = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    out = frame.copy()
    out[:fade] *= w
    out[-fade:] *= w[::-1]
    return out


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(frame)) + 1e-12)
    return (frame / m).astype(np.float32, copy=False)


def _linear_resample(x: np.ndarray, new_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if new_len == n:
        return x
    src = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, new_len, endpoint=False, dtype=np.float32)
    return np.interp(dst, src, x).astype(np.float32, copy=False)


def load_mono(path: str):
    y, sr = lr_load(path, sr=None, mono=True)
    return y.astype(np.float32, copy=False), int(sr)


def peak_amp(x: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(x, dtype=np.float32))) + 1e-12)


def peak_dbfs(x: np.ndarray) -> float:
    p = peak_amp(x)
    return float(20.0 * np.log10(p))


def match_peak_to_reference(y: np.ndarray, ref_peak_amp: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    p = float(np.max(np.abs(y)) + 1e-12)
    if ref_peak_amp <= 1e-9:
        return y
    g = float(ref_peak_amp / p)
    yy = y * g
    # clamp safety
    m = float(np.max(np.abs(yy)) + 1e-12)
    if m > 0.999:
        yy = yy * (0.999 / m)
    return yy.astype(np.float32, copy=False)


def resample_to_target(y: np.ndarray, sr_in: int, sr_target: int) -> np.ndarray:
    if sr_in == sr_target:
        return y.astype(np.float32, copy=False)
    # resample_poly with rational approximation: up/down by gcd
    g = math.gcd(sr_in, sr_target)
    up = sr_target // g
    down = sr_in // g
    y_rs = resample_poly(y.astype(np.float32), up=up, down=down).astype(np.float32, copy=False)
    return y_rs


def one_pole_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 1e-6, 1.0))
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    a = alpha
    b = 1.0 - alpha
    for i in range(1, len(x)):
        y[i] = a * x[i] + b * y[i - 1]
    return y


def one_pole_lpf(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    cutoff_hz = float(np.clip(cutoff_hz, 5.0, sr * 0.45))
    a = float(np.exp(-2.0 * np.pi * cutoff_hz / sr))
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    b = 1.0 - a
    for i in range(1, len(x)):
        y[i] = b * x[i] + a * y[i - 1]
    return y


def one_pole_hpf(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    return (x - one_pole_lpf(x, sr, cutoff_hz)).astype(np.float32, copy=False)


def list_wav_files(folder: str, recursive: bool = True):
    if not folder or not os.path.isdir(folder):
        return []
    pattern = "**/*.wav" if recursive else "*.wav"
    files = glob.glob(os.path.join(folder, pattern), recursive=recursive)
    files += glob.glob(os.path.join(folder, pattern.upper()), recursive=recursive)
    files = [f for f in files if os.path.isfile(f)]
    files.sort(key=lambda p: p.lower())
    return files


def list_audio_files(folder: str):
    files = []
    for ext in AUDIO_EXTS:
        files += glob.glob(os.path.join(folder, f"*{ext}"))
        files += glob.glob(os.path.join(folder, f"*{ext.upper()}"))
    files = [f for f in files if os.path.isfile(f)]
    files.sort(key=lambda p: p.lower())
    return files


def infer_wavetable_frame_size(n_samples: int) -> int:
    for fs in (2048, 4096, 1024):
        if n_samples >= fs * 4 and (n_samples % fs) == 0:
            return fs
    return 2048


# ----------------- FEATURE EXTRACTION (for matching) -----------------
def smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    k = int(max(1, k))
    if k <= 1:
        return x.astype(np.float32, copy=False)
    ker = np.ones(k, dtype=np.float32) / float(k)
    pad = k // 2
    xp = np.pad(x.astype(np.float32), (pad, pad), mode="reflect")
    return np.convolve(xp, ker, mode="valid").astype(np.float32, copy=False)


def downsample_bins(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == n:
        return x
    idx = np.linspace(0, len(x), n + 1, dtype=np.int32)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a, b = idx[i], idx[i + 1]
        if b <= a:
            b = min(len(x), a + 1)
        out[i] = float(np.mean(x[a:b]))
    return out


def frame_to_feature(frame_2048: np.ndarray, feat_dim: int = INDEX_FEAT_DIM, smooth_bins: int = INDEX_SMOOTH_BINS) -> np.ndarray:
    x = np.asarray(frame_2048, dtype=np.float32)
    x = x - float(np.mean(x))
    x = _fade_edges(x, fade=16)
    x = _normalize_frame(x)

    win = np.hanning(len(x)).astype(np.float32)
    X = np.fft.rfft(x * win)
    mag = (np.abs(X) + 1e-8).astype(np.float32)
    logmag = np.log(mag).astype(np.float32)

    if len(logmag) > 2:
        logmag = logmag[1:]

    logmag = smooth_1d(logmag, smooth_bins)
    feat = downsample_bins(logmag, feat_dim)

    nrm = float(np.linalg.norm(feat) + 1e-12)
    feat = (feat / nrm).astype(np.float32, copy=False)
    return feat


def segment_to_feature(y_seg: np.ndarray, feat_dim: int = INDEX_FEAT_DIM) -> np.ndarray:
    y_seg = np.asarray(y_seg, dtype=np.float32)
    if len(y_seg) < INDEX_FRAME_SIZE:
        y_seg = np.pad(y_seg, (0, INDEX_FRAME_SIZE - len(y_seg)))
    else:
        y_seg = y_seg[:INDEX_FRAME_SIZE]
    return frame_to_feature(y_seg, feat_dim=feat_dim)


# ----------------- WAVETABLE INDEX -----------------
def build_wt_index(wt_folder: str, index_path: str, log_cb=None) -> dict:
    wavs = list_wav_files(wt_folder, recursive=True)
    if not wavs:
        raise RuntimeError("No se encontraron .wav en la carpeta de wavetables.")

    paths = []
    file_frame_size = []
    file_n_frames = []

    features = []
    file_ids = []
    frame_ids = []

    for fi, p in enumerate(wavs):
        if log_cb:
            log_cb(f"Indexando: {os.path.basename(p)}")
        audio, _sr = sf.read(p, always_2d=False)
        audio = _to_mono_float(audio)
        fs = infer_wavetable_frame_size(len(audio))

        if len(audio) < fs:
            audio = np.pad(audio, (0, fs - len(audio)))

        n_frames = max(1, len(audio) // fs)
        use_len = n_frames * fs
        audio = audio[:use_len]
        frames = audio.reshape(n_frames, fs)

        paths.append(p)
        file_frame_size.append(int(fs))
        file_n_frames.append(int(n_frames))

        for fr in range(n_frames):
            f = frames[fr].astype(np.float32, copy=False)
            f = f - float(np.mean(f))
            f = _fade_edges(f, fade=16)
            f = _normalize_frame(f)
            if fs != INDEX_FRAME_SIZE:
                f = _linear_resample(f, INDEX_FRAME_SIZE)
            feat = frame_to_feature(f, feat_dim=INDEX_FEAT_DIM, smooth_bins=INDEX_SMOOTH_BINS)

            features.append(feat.astype(np.float32))
            file_ids.append(fi)
            frame_ids.append(fr)

    features = np.asarray(features, dtype=np.float32)
    fn = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    features = (features / fn).astype(np.float32)

    file_ids = np.asarray(file_ids, dtype=np.int32)
    frame_ids = np.asarray(frame_ids, dtype=np.int32)
    file_frame_size = np.asarray(file_frame_size, dtype=np.int32)
    file_n_frames = np.asarray(file_n_frames, dtype=np.int32)
    paths_arr = np.asarray(paths, dtype=object)

    np.savez_compressed(
        index_path,
        features=features.astype(np.float16),
        file_ids=file_ids,
        frame_ids=frame_ids,
        paths=paths_arr,
        file_frame_size=file_frame_size,
        file_n_frames=file_n_frames,
        meta=np.asarray(
            [json.dumps({"feat_dim": INDEX_FEAT_DIM, "index_frame_size": INDEX_FRAME_SIZE, "smooth_bins": INDEX_SMOOTH_BINS})],
            dtype=object,
        ),
    )

    return {
        "features": features,
        "file_ids": file_ids,
        "frame_ids": frame_ids,
        "paths": paths,
        "file_frame_size": file_frame_size,
        "file_n_frames": file_n_frames,
    }


def load_wt_index(index_path: str) -> dict:
    z = np.load(index_path, allow_pickle=True)
    features = z["features"].astype(np.float32)
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)

    return {
        "features": features,
        "file_ids": z["file_ids"].astype(np.int32),
        "frame_ids": z["frame_ids"].astype(np.int32),
        "paths": [str(p) for p in z["paths"].tolist()],
        "file_frame_size": z["file_frame_size"].astype(np.int32),
        "file_n_frames": z["file_n_frames"].astype(np.int32),
    }


def match_best_frames(query_feat: np.ndarray, index: dict, topk: int = 8):
    feats = index["features"]  # (M,D) normalized
    q = np.asarray(query_feat, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)
    sims = feats @ q  # cosine similarity
    k = int(min(max(1, topk), len(sims)))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx.tolist(), [float(sims[i]) for i in idx.tolist()]


def softmax_w(x: np.ndarray, temp: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    t = float(max(1e-5, temp))
    z = x / t
    z = z - float(np.max(z))
    e = np.exp(z).astype(np.float32)
    s = float(np.sum(e) + 1e-12)
    return (e / s).astype(np.float32, copy=False)


# ----------------- HIT DETECTION & ANALYSIS -----------------
def smooth_freq_logmag(logmag: np.ndarray, smooth_bins: int) -> np.ndarray:
    smooth_bins = int(max(1, smooth_bins))
    if smooth_bins <= 1:
        return logmag
    k = smooth_bins
    kernel = np.ones(k, dtype=np.float32) / float(k)
    pad = k // 2
    padded = np.pad(logmag, ((pad, pad), (0, 0)), mode="reflect")
    out = np.empty_like(logmag, dtype=np.float32)
    for t in range(logmag.shape[1]):
        out[:, t] = np.convolve(padded[:, t], kernel, mode="valid").astype(np.float32)
    return out


def compute_logmag(y: np.ndarray, frame_length: int, hop_length: int):
    S = lr_stft(
        y.astype(np.float32),
        n_fft=frame_length,
        hop_length=hop_length,
        win_length=frame_length,
        center=True,
    )
    mag = np.abs(S).astype(np.float32)
    logmag = np.log(mag + 1e-8).astype(np.float32)
    return mag, logmag


def spectral_flux_from_logmag(logmag: np.ndarray) -> np.ndarray:
    d = np.diff(logmag, axis=1)
    dpos = np.maximum(d, 0.0)
    flux = np.sum(dpos, axis=0).astype(np.float32)
    flux = np.concatenate([np.array([0.0], dtype=np.float32), flux], axis=0)
    mx = float(np.max(flux) + 1e-12)
    return (flux / mx).astype(np.float32, copy=False)


def pick_triggers(flux: np.ndarray, thresh: float, min_dist_frames: int) -> np.ndarray:
    x = np.asarray(flux, dtype=np.float32)
    thr = float(np.clip(thresh, 0.0, 1.0))

    p90 = float(np.percentile(x, 90))
    if p90 < thr:
        thr = max(0.08, p90 * 0.85)

    picks = []
    last = -10**9
    for i in range(1, len(x) - 1):
        if x[i] < thr:
            continue
        if x[i] >= x[i - 1] and x[i] >= x[i + 1]:
            if i - last >= min_dist_frames:
                picks.append(i)
                last = i
    return np.asarray(picks, dtype=np.int32)


def centroid_norm_from_mag(mag: np.ndarray, sr: int) -> np.ndarray:
    n_bins, n_frames = mag.shape
    freqs = np.linspace(0.0, sr / 2.0, n_bins, dtype=np.float32)
    num = np.sum(mag * freqs[:, None], axis=0).astype(np.float32)
    den = (np.sum(mag, axis=0) + 1e-12).astype(np.float32)
    centroid = (num / den).astype(np.float32)
    c_lo, c_hi = 80.0, min(12000.0, sr / 2.0)
    cn = (centroid - c_lo) / (c_hi - c_lo + 1e-12)
    return np.clip(cn, 0.0, 1.0).astype(np.float32, copy=False)


def extract_multiband_env_and_specenv(y: np.ndarray, sr: int, frame_length: int, hop_length: int, env_smooth_alpha: float, spec_smooth_bins: int):
    mag, logmag = compute_logmag(y, frame_length=frame_length, hop_length=hop_length)
    n_bins, n_frames = mag.shape
    freqs = np.linspace(0.0, sr / 2.0, n_bins, dtype=np.float32)

    edges = [
        (20.0, 140.0),
        (140.0, 700.0),
        (700.0, 4000.0),
        (4000.0, min(16000.0, sr / 2.0 - 1.0)),
    ]
    env_bands = np.zeros((4, n_frames), dtype=np.float32)
    for bi, (lo, hi) in enumerate(edges):
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            continue
        band_mag = mag[mask, :]
        env = np.sqrt(np.mean(band_mag * band_mag, axis=0) + 1e-12).astype(np.float32)
        env = env / (np.max(env) + 1e-12)
        env_bands[bi] = env

    spec_env_log = smooth_freq_logmag(logmag, smooth_bins=spec_smooth_bins)
    spec_env = np.exp(spec_env_log).astype(np.float32)

    for bi in range(4):
        env_bands[bi] = one_pole_smooth(env_bands[bi], alpha=env_smooth_alpha)

    return env_bands, spec_env, mag, logmag


# ----------------- Pitch estimation per hit (autocorr, + tonal gate) -----------------
def estimate_f0_autocorr(y_seg: np.ndarray, sr: int, fmin: float, fmax: float) -> tuple[float, float]:
    """
    Returns (f0_hz, harmonicity) where harmonicity is 0..1-ish (higher = more tonal).
    """
    x = np.asarray(y_seg, dtype=np.float32)
    if len(x) < 64:
        return 0.0, 0.0

    # prefilter: remove DC, gentle HP to reduce rumble
    x = x - float(np.mean(x))
    x = one_pole_hpf(x, sr, 25.0)
    # window
    w = np.hanning(len(x)).astype(np.float32)
    xw = x * w

    # autocorr via FFT
    n = int(1 << (len(xw) - 1).bit_length())
    X = np.fft.rfft(xw, n=n)
    ac = np.fft.irfft(X * np.conj(X), n=n).astype(np.float32)
    ac = ac[: len(xw)]
    ac0 = float(ac[0] + 1e-12)

    # lag range
    fmin = float(np.clip(fmin, 20.0, sr * 0.45))
    fmax = float(np.clip(fmax, fmin + 10.0, sr * 0.45))
    lag_min = int(max(1, sr / fmax))
    lag_max = int(min(len(ac) - 1, sr / fmin))
    if lag_max <= lag_min + 2:
        return 0.0, 0.0

    seg = ac[lag_min:lag_max]
    i_rel = int(np.argmax(seg))
    i = lag_min + i_rel

    # parabolic refine
    if 1 <= i < len(ac) - 1:
        y0, y1, y2 = float(ac[i - 1]), float(ac[i]), float(ac[i + 1])
        denom = (y0 - 2.0 * y1 + y2)
        if abs(denom) > 1e-9:
            delta = 0.5 * (y0 - y2) / denom
            i_f = float(i) + float(np.clip(delta, -0.5, 0.5))
        else:
            i_f = float(i)
    else:
        i_f = float(i)

    peak = float(ac[i] / ac0)
    f0 = float(sr / max(1e-6, i_f))

    # harmonicity measure: peak normalized energy
    harm = float(np.clip(peak, 0.0, 1.0))
    if not np.isfinite(f0):
        return 0.0, 0.0
    return f0, harm


def pitch_envelope_freq(
    f0: float,
    sr: int,
    n: int,
    semi: float,
    env_ms: float,
    strength: float,
) -> np.ndarray:
    f0 = float(max(1.0, f0))
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 1e-6 or env_ms <= 0.1:
        return np.full(n, f0, dtype=np.float32)

    ratio0 = float(2.0 ** (semi / 12.0))
    # exponential decay of ratio to 1.0 over env_ms
    tau = float(max(1e-4, (env_ms / 1000.0) / 4.0))
    t = (np.arange(n, dtype=np.float32) / float(sr)).astype(np.float32)
    ratio = 1.0 + (ratio0 - 1.0) * np.exp(-t / tau).astype(np.float32)

    f_env = (f0 * ratio).astype(np.float32)
    # strength blend
    return ((1.0 - strength) * f0 + strength * f_env).astype(np.float32, copy=False)


# ----------------- SYNTH (wavetable osc per hit) -----------------
class WavetableCache:
    def __init__(self):
        self._audio = {}       # path -> mono float audio
        self._framesize = {}   # path -> frame_size
        self._nframes = {}     # path -> n_frames

    def _load(self, path: str):
        if path in self._audio:
            return
        audio, _sr = sf.read(path, always_2d=False)
        audio = _to_mono_float(audio)
        fs = infer_wavetable_frame_size(len(audio))
        if len(audio) < fs:
            audio = np.pad(audio, (0, fs - len(audio)))
        n_frames = max(1, len(audio) // fs)
        audio = audio[: n_frames * fs]
        self._audio[path] = audio.astype(np.float32, copy=False)
        self._framesize[path] = int(fs)
        self._nframes[path] = int(n_frames)

    def get_frame_2048(self, path: str, frame_idx: int) -> np.ndarray:
        self._load(path)
        audio = self._audio[path]
        fs = self._framesize[path]
        nfr = self._nframes[path]
        frame_idx = int(np.clip(frame_idx, 0, nfr - 1))
        start = frame_idx * fs
        fr = audio[start : start + fs].astype(np.float32, copy=False)
        fr = fr - float(np.mean(fr))
        fr = _fade_edges(fr, fade=16)
        fr = _normalize_frame(fr)
        if fs != INDEX_FRAME_SIZE:
            fr = _linear_resample(fr, INDEX_FRAME_SIZE)
        return fr.astype(np.float32, copy=False)

    def get_nframes(self, path: str) -> int:
        self._load(path)
        return int(self._nframes[path])


def table_read_linear(table_1d: np.ndarray, phase: np.ndarray) -> np.ndarray:
    n = len(table_1d)
    idx = phase * n
    i0 = np.floor(idx).astype(np.int32)
    frac = idx - i0
    i1 = (i0 + 1) % n
    return (1.0 - frac) * table_1d[i0] + frac * table_1d[i1]


def render_wavetable_osc_varfreq(table: np.ndarray, f_inst: np.ndarray, sr: int, phase0: float = 0.0):
    f_inst = np.asarray(f_inst, dtype=np.float32)
    n = len(f_inst)
    phase = np.empty(n, dtype=np.float32)
    ph = float(phase0 % 1.0)
    inv_sr = 1.0 / float(sr)
    for i in range(n):
        phase[i] = ph
        ph += float(np.clip(f_inst[i], 1.0, sr * 0.45)) * inv_sr
        ph -= math.floor(ph)
    y = table_read_linear(table, phase).astype(np.float32, copy=False)
    return y, float(ph)


def exp_env(n: int, sr: int, attack_ms: float, dur_ms: float, curve: float = 1.0):
    attack = int(max(1, (attack_ms / 1000.0) * sr))
    dur = int(max(attack + 1, (dur_ms / 1000.0) * sr))
    dur = min(dur, n)
    env = np.zeros(n, dtype=np.float32)

    a = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    env[:attack] = a

    remain = dur - attack
    if remain > 0:
        t = np.linspace(0.0, 1.0, remain, dtype=np.float32)
        k = 6.0
        d = np.exp(-k * t).astype(np.float32)
        env[attack:dur] = d

    if curve != 1.0:
        env = np.clip(env, 0.0, 1.0) ** float(curve)

    return env


def blend_tables_multi_candidate(
    wt_index: dict,
    cache: WavetableCache,
    idxs: list[int],
    sims: list[float],
    mix_n: int,
    mix_temp: float,
    pos_state: float,
    rng: np.random.Generator,
):
    n = int(min(max(1, mix_n), len(idxs)))
    use_idxs = idxs[:n]
    use_sims = np.asarray(sims[:n], dtype=np.float32)

    w = softmax_w(use_sims, temp=mix_temp)
    # tiny humanize in weights
    w = w * (0.90 + 0.20 * rng.random(n)).astype(np.float32)
    w = w / (float(np.sum(w)) + 1e-12)

    table_mix = np.zeros(INDEX_FRAME_SIZE, dtype=np.float32)

    for wi, global_idx in zip(w.tolist(), use_idxs):
        fid = int(wt_index["file_ids"][global_idx])
        wt_path = wt_index["paths"][fid]
        nfr = int(wt_index["file_n_frames"][fid])

        choose_idx = int(round(pos_state * (nfr - 1))) if nfr > 1 else 0
        tab = cache.get_frame_2048(wt_path, choose_idx)
        table_mix += float(wi) * tab

    table_mix = table_mix - float(np.mean(table_mix))
    table_mix = _fade_edges(table_mix, fade=16)
    table_mix = _normalize_frame(table_mix)
    return table_mix.astype(np.float32, copy=False)


def synth_wt_hits_recon(
    y_src: np.ndarray,
    sr: int,
    triggers_frames: np.ndarray,
    flux: np.ndarray,
    centroid_norm: np.ndarray,
    env_bands: np.ndarray,
    hop_length: int,
    frame_length: int,
    wt_index: dict,
    match_topk: int,
    mix_n: int,
    mix_temp: float,
    pos_min: float,
    pos_max: float,
    pos_walk_min: float,
    pos_walk_max: float,
    body_start_ms: float,
    body_dur_ms: float,
    max_hit_ms: float,
    attack_ms: float,
    only_hits: bool,
    # pitch
    pitch_enable: bool,
    pitch_fmin: float,
    pitch_fmax: float,
    tonality_thr: float,
    pitch_env_semi: float,
    pitch_env_ms: float,
    pitch_env_strength: float,
    rng: np.random.Generator,
    log_cb=None,
):
    n_samples = len(y_src)
    out = np.zeros(n_samples, dtype=np.float32)

    cache = WavetableCache()
    pos_state = float(np.clip(rng.uniform(pos_min, pos_max), 0.0, 1.0))

    body_start = int((body_start_ms / 1000.0) * sr)
    body_len = int((body_dur_ms / 1000.0) * sr)
    max_len = int((max_hit_ms / 1000.0) * sr)

    for k, fr in enumerate(triggers_frames.tolist()):
        t_center = int(fr * hop_length + frame_length // 2)
        t0 = int(np.clip(t_center, 0, n_samples - 1))

        vel = float(np.clip(flux[fr], 0.0, 1.0)) ** DEFAULT_VEL_GAMMA
        vel *= float(0.9 + 0.2 * rng.random())

        cn = float(np.clip(centroid_norm[fr], 0.0, 1.0))

        seg_start = int(np.clip(t0 + body_start, 0, n_samples - 1))
        seg_end = int(np.clip(seg_start + body_len, seg_start + 1, n_samples))
        seg = y_src[seg_start:seg_end]

        qfeat = segment_to_feature(seg, feat_dim=INDEX_FEAT_DIM)
        idxs, sims = match_best_frames(qfeat, wt_index, topk=match_topk)

        # pos walk based on best match frame position
        best = idxs[0]
        fid_best = int(wt_index["file_ids"][best])
        fidx_best = int(wt_index["frame_ids"][best])
        nfr_best = int(wt_index["file_n_frames"][fid_best])
        pos_target = 0.0 if nfr_best <= 1 else float(fidx_best) / float(max(1, nfr_best - 1))
        walk_std = float(np.interp(cn, [0.0, 1.0], [pos_walk_min, pos_walk_max]))
        pos_state = float(np.clip(0.85 * pos_state + 0.15 * pos_target + rng.normal(0.0, walk_std), pos_min, pos_max))

        # multi-candidate blended table
        table = blend_tables_multi_candidate(
            wt_index=wt_index,
            cache=cache,
            idxs=idxs,
            sims=sims,
            mix_n=mix_n,
            mix_temp=mix_temp,
            pos_state=pos_state,
            rng=rng,
        )

        # duration
        dur_ms = float(np.interp(cn, [0.0, 1.0], [max_hit_ms, max(25.0, 0.25 * max_hit_ms)]))
        n_hit = int(min(max_len, max(96, (dur_ms / 1000.0) * sr)))
        n_hit = int(min(n_hit, n_samples - t0))
        if n_hit <= 16:
            continue

        # pitch: estimate if tonal, else fallback to centroid mapping
        f0 = 0.0
        harm = 0.0
        if pitch_enable:
            # Use a slightly longer segment for f0 robustness
            seg_f0_end = int(np.clip(seg_start + int(0.075 * sr), seg_start + 1, n_samples))
            seg_f0 = y_src[seg_start:seg_f0_end]
            f0_est, harm = estimate_f0_autocorr(seg_f0, sr=sr, fmin=pitch_fmin, fmax=pitch_fmax)
            if harm >= tonality_thr and (pitch_fmin <= f0_est <= pitch_fmax):
                f0 = float(f0_est)

        if f0 <= 1.0:
            # fallback mapping (works well for hats/snare-ish)
            f0 = float(np.interp(cn, [0.0, 1.0], [55.0, 240.0]))

        # pitch envelope (configurable)
        f_inst = pitch_envelope_freq(
            f0=f0,
            sr=sr,
            n=n_hit,
            semi=pitch_env_semi,
            env_ms=pitch_env_ms,
            strength=pitch_env_strength,
        )
        # slight “pitch drop with velocity” (musical)
        f_inst = (f_inst * (1.0 - 0.12 * vel)).astype(np.float32)

        # amplitude envelope: incorporate band energy
        env = exp_env(n_hit, sr, attack_ms=attack_ms, dur_ms=dur_ms, curve=1.25)
        band_energy = float(np.clip(0.40 * env_bands[0][fr] + 0.35 * env_bands[1][fr] + 0.25 * env_bands[2][fr], 0.0, 1.0))
        env *= float(0.6 + 0.9 * band_energy)

        osc, _ = render_wavetable_osc_varfreq(table, f_inst=f_inst, sr=sr, phase0=0.0)
        y_hit = (osc * env * vel).astype(np.float32)

        # click + air (ataque)
        click_len = int(max(32, (min(10.0, attack_ms * 4.0) / 1000.0) * sr))
        click_len = min(click_len, n_hit)
        if click_len > 16:
            d = np.diff(env[:click_len], prepend=0.0).astype(np.float32)
            click = one_pole_hpf(d, sr, DEFAULT_CLICK_HPF) * 2.2
            y_hit[:click_len] += click.astype(np.float32)

            noise = rng.standard_normal(click_len).astype(np.float32) * 0.25
            air = one_pole_hpf(noise, sr, DEFAULT_AIR_HPF)
            y_hit[:click_len] += air * float(0.35 + 0.65 * cn) * vel

        out[t0 : t0 + n_hit] += y_hit

        if log_cb and (k % 8 == 0):
            fid0 = int(wt_index["file_ids"][idxs[0]])
            p0 = os.path.basename(wt_index["paths"][fid0])
            log_cb(
                f" Hit {k+1}/{len(triggers_frames)} @frame={fr} vel={vel:.2f} cn={cn:.2f} "
                f"f0={f0:.1f}Hz harm={harm:.2f} table≈{p0} mixN={mix_n} pos={pos_state:.2f} sim0={sims[0]:.3f}"
            )

    if not only_hits:
        out += 0.0 * y_src

    mx = float(np.max(np.abs(out)) + 1e-12)
    out = (out / mx * 0.95).astype(np.float32)
    return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)


# ----------------- SPECTRAL GLUE (optional) -----------------
def spectral_envelope_match(
    y_syn: np.ndarray,
    sr: int,
    spec_env_src: np.ndarray,
    frame_length: int,
    hop_length: int,
    spec_smooth_bins: int,
    strength: float,
    clamp_lo: float,
    clamp_hi: float,
):
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return y_syn

    S = lr_stft(
        y_syn.astype(np.float32),
        n_fft=frame_length,
        hop_length=hop_length,
        win_length=frame_length,
        center=True,
    )
    mag = np.abs(S).astype(np.float32)
    phase = np.angle(S).astype(np.float32)

    logmag = np.log(mag + 1e-8).astype(np.float32)
    spec_env_syn_log = smooth_freq_logmag(logmag, smooth_bins=spec_smooth_bins)
    spec_env_syn = np.exp(spec_env_syn_log).astype(np.float32)

    bins = min(spec_env_src.shape[0], spec_env_syn.shape[0], mag.shape[0])
    frames = min(spec_env_src.shape[1], spec_env_syn.shape[1], mag.shape[1])

    src = spec_env_src[:bins, :frames]
    syn = spec_env_syn[:bins, :frames]
    mag_use = mag[:bins, :frames]
    ph_use = phase[:bins, :frames]

    G = src / (syn + 1e-12)
    G = np.clip(G, float(clamp_lo), float(clamp_hi)).astype(np.float32)

    mag_out = mag_use * np.power(G, strength).astype(np.float32)
    S_out = mag_out * (np.cos(ph_use) + 1j * np.sin(ph_use))

    y_out = lr_istft(
        S_out,
        hop_length=hop_length,
        win_length=frame_length,
        center=True,
        length=len(y_syn),
    )
    y_out = np.asarray(y_out, dtype=np.float32)
    return np.clip(y_out, -1.0, 1.0).astype(np.float32, copy=False)


# ----------------- WORKER -----------------
class AudioWorker(QObject):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        input_path: str,
        output_path: str,
        wt_dir: str,
        seed: int,
        process_96k: bool,
        hop_length: int,
        frame_length: int,
        env_alpha: float,
        transient_alpha: float,
        only_hits: bool,
        output_gain: float,
        normalize_to_original_peak: bool,
        enable_spec_match: bool,
        spec_strength: float,
        spec_smooth_bins: int,
        spec_clamp_lo: float,
        spec_clamp_hi: float,
        # hit params
        thresh: float,
        min_dist_ms: float,
        body_start_ms: float,
        body_dur_ms: float,
        max_hit_ms: float,
        attack_ms: float,
        # match params
        match_topk: int,
        mix_n: int,
        mix_temp: float,
        pos_min: float,
        pos_max: float,
        pos_walk_min: float,
        pos_walk_max: float,
        # pitch params
        pitch_enable: bool,
        pitch_fmin: float,
        pitch_fmax: float,
        tonality_thr: float,
        pitch_env_semi: float,
        pitch_env_ms: float,
        pitch_env_strength: float,
    ):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.wt_dir = wt_dir
        self.seed = int(seed)
        self.process_96k = bool(process_96k)

        self.hop_length = int(hop_length)
        self.frame_length = int(frame_length)
        self.env_alpha = float(env_alpha)
        self.transient_alpha = float(transient_alpha)
        self.only_hits = bool(only_hits)
        self.output_gain = float(output_gain)
        self.normalize_to_original_peak = bool(normalize_to_original_peak)

        self.enable_spec_match = bool(enable_spec_match)
        self.spec_strength = float(spec_strength)
        self.spec_smooth_bins = int(spec_smooth_bins)
        self.spec_clamp_lo = float(spec_clamp_lo)
        self.spec_clamp_hi = float(spec_clamp_hi)

        self.thresh = float(thresh)
        self.min_dist_ms = float(min_dist_ms)
        self.body_start_ms = float(body_start_ms)
        self.body_dur_ms = float(body_dur_ms)
        self.max_hit_ms = float(max_hit_ms)
        self.attack_ms = float(attack_ms)

        self.match_topk = int(match_topk)
        self.mix_n = int(mix_n)
        self.mix_temp = float(mix_temp)
        self.pos_min = float(pos_min)
        self.pos_max = float(pos_max)
        self.pos_walk_min = float(pos_walk_min)
        self.pos_walk_max = float(pos_walk_max)

        self.pitch_enable = bool(pitch_enable)
        self.pitch_fmin = float(pitch_fmin)
        self.pitch_fmax = float(pitch_fmax)
        self.tonality_thr = float(tonality_thr)
        self.pitch_env_semi = float(pitch_env_semi)
        self.pitch_env_ms = float(pitch_env_ms)
        self.pitch_env_strength = float(pitch_env_strength)

        self._wt_index = None

    def _log(self, s: str):
        self.log.emit(s)

    def _ensure_index(self):
        index_path = os.path.join(self.wt_dir, INDEX_FILENAME)
        if os.path.isfile(index_path):
            self._log(f"Cargando índice: {index_path}")
            return load_wt_index(index_path)

        self._log("No existe wt_index.npz. Construyendo índice (solo 1 vez)...")
        idx = build_wt_index(self.wt_dir, index_path, log_cb=self._log)
        self._log(f"Índice guardado: {index_path}")
        return idx

    def _process_one(self, src_file: str, out_file: str, rng: np.random.Generator):
        self._log(f"Fuente: {os.path.basename(src_file)}")

        y0, sr0 = load_mono(src_file)
        ref_peak = peak_amp(y0)
        ref_db = peak_dbfs(y0)
        self._log(f" sr_in={sr0} len={len(y0)} peak={ref_db:.2f} dBFS")

        # Resample to 96k if enabled
        if self.process_96k and sr0 != TARGET_SR:
            self._log(f" Resample → {TARGET_SR} Hz (alta calidad resample_poly)...")
            y = resample_to_target(y0, sr_in=sr0, sr_target=TARGET_SR)
            sr = TARGET_SR
        else:
            y = y0
            sr = sr0

        self._log(f" sr_proc={sr} len_proc={len(y)}")

        # Analysis: env/spec + flux/triggers + centroid
        self._log(" Analizando: multiband env + spec_env + flux/triggers...")
        env_bands, spec_env, mag, logmag = extract_multiband_env_and_specenv(
            y=y,
            sr=sr,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            env_smooth_alpha=self.env_alpha,
            spec_smooth_bins=self.spec_smooth_bins,
        )

        flux = spectral_flux_from_logmag(logmag)
        flux = one_pole_smooth(flux, alpha=self.transient_alpha)
        cn = centroid_norm_from_mag(mag, sr=sr)

        min_dist_frames = int(max(1, (self.min_dist_ms / 1000.0) * sr / self.hop_length))
        triggers = pick_triggers(flux, thresh=self.thresh, min_dist_frames=min_dist_frames)
        self._log(f" Triggers: {len(triggers)} (thresh={self.thresh:.2f}, min_dist={self.min_dist_ms:.1f}ms)")

        if len(triggers) == 0:
            raise RuntimeError("No se detectaron hits. Baja thresh o reduce min_dist.")

        # Synthesis: pitch-per-hit + multi-candidate blend
        self._log(" Sintetizando: WT-match por hit + pitch-per-hit (auto) + multi-candidato...")
        y_syn = synth_wt_hits_recon(
            y_src=y,
            sr=sr,
            triggers_frames=triggers,
            flux=flux,
            centroid_norm=cn,
            env_bands=env_bands,
            hop_length=self.hop_length,
            frame_length=self.frame_length,
            wt_index=self._wt_index,
            match_topk=self.match_topk,
            mix_n=self.mix_n,
            mix_temp=self.mix_temp,
            pos_min=self.pos_min,
            pos_max=self.pos_max,
            pos_walk_min=self.pos_walk_min,
            pos_walk_max=self.pos_walk_max,
            body_start_ms=self.body_start_ms,
            body_dur_ms=self.body_dur_ms,
            max_hit_ms=self.max_hit_ms,
            attack_ms=self.attack_ms,
            only_hits=self.only_hits,
            pitch_enable=self.pitch_enable,
            pitch_fmin=self.pitch_fmin,
            pitch_fmax=self.pitch_fmax,
            tonality_thr=self.tonality_thr,
            pitch_env_semi=self.pitch_env_semi,
            pitch_env_ms=self.pitch_env_ms,
            pitch_env_strength=self.pitch_env_strength,
            rng=rng,
            log_cb=self._log,
        ).astype(np.float32)

        # Optional spectral glue
        if self.enable_spec_match:
            self._log(" Aplicando spectral glue (match espectral suave)...")
            y_syn = spectral_envelope_match(
                y_syn=y_syn,
                sr=sr,
                spec_env_src=spec_env,
                frame_length=self.frame_length,
                hop_length=self.hop_length,
                spec_smooth_bins=self.spec_smooth_bins,
                strength=self.spec_strength,
                clamp_lo=self.spec_clamp_lo,
                clamp_hi=self.spec_clamp_hi,
            )

        # Output gain
        if self.output_gain != 1.0:
            y_syn = np.clip(y_syn * float(self.output_gain), -1.0, 1.0).astype(np.float32)

        # Normalize to original peak (same peak dBFS as source)
        if self.normalize_to_original_peak:
            self._log(f" Normalizando al pico original (ref_peak={ref_db:.2f} dBFS)...")
            y_syn = match_peak_to_reference(y_syn, ref_peak_amp=ref_peak)

        # Ensure output dir
        out_dir = os.path.dirname(out_file)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # Write 24-bit WAV
        # (Si el path no termina en .wav, lo dejamos igual, pero PCM_24 aplica idealmente a WAV/AIFF)
        sf.write(out_file, y_syn, sr, subtype=OUTPUT_SUBTYPE)
        self._log(f" Guardado: {out_file} (sr={sr}, {OUTPUT_SUBTYPE})")

    def run(self):
        try:
            if not self.wt_dir or not os.path.isdir(self.wt_dir):
                raise RuntimeError("La carpeta de wavetables no existe.")

            self._wt_index = self._ensure_index()
            if self._wt_index is None:
                raise RuntimeError("No se pudo cargar/crear índice de wavetables.")

            rng = np.random.default_rng(None if self.seed == 0 else self.seed)

            inp = self.input_path
            outp = self.output_path

            is_batch = os.path.isdir(inp)
            if not is_batch:
                if not os.path.isfile(inp):
                    raise RuntimeError("Input single debe ser un archivo.")

                if os.path.isdir(outp):
                    base = os.path.splitext(os.path.basename(inp))[0]
                    out_file = os.path.join(outp, base + "__recon.wav")
                else:
                    if os.path.splitext(outp)[1].lower() != ".wav":
                        os.makedirs(outp, exist_ok=True)
                        base = os.path.splitext(os.path.basename(inp))[0]
                        out_file = os.path.join(outp, base + "__recon.wav")
                    else:
                        out_file = outp

                self.progress.emit(5)
                self._log("=== PROCESO SINGLE (WT MATCH RECON PRO) ===")
                self._log(f"Seed={'RANDOM' if self.seed==0 else self.seed} | 96k={'ON' if self.process_96k else 'OFF'}")
                self._process_one(inp, out_file, rng)
                self.progress.emit(100)
                self.finished.emit()
                return

            # BATCH
            in_dir = inp
            out_dir = outp
            if os.path.isfile(out_dir):
                out_dir = os.path.dirname(out_dir)
            if not out_dir:
                raise RuntimeError("Output batch debe ser una carpeta válida.")
            os.makedirs(out_dir, exist_ok=True)

            files = list_audio_files(in_dir)
            if not files:
                raise RuntimeError("No se encontraron audios en la carpeta input.")

            self._log("=== PROCESO BATCH (WT MATCH RECON PRO) ===")
            self._log(f"Input: {in_dir}")
            self._log(f"Output: {out_dir}")
            self._log(f"Seed={'RANDOM' if self.seed==0 else self.seed} | 96k={'ON' if self.process_96k else 'OFF'}")
            self._log(f"Archivos: {len(files)}")

            self.progress.emit(1)
            total = len(files)
            for idx, src_file in enumerate(files):
                base = os.path.splitext(os.path.basename(src_file))[0]
                out_file = os.path.join(out_dir, base + "__recon.wav")
                self._log(f"\n[{idx+1}/{total}]")
                p = int(5 + 90 * (idx / max(1, total)))
                self.progress.emit(p)
                self._process_one(src_file, out_file, rng)

            self.progress.emit(100)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


# ----------------- UI -----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WT Match Reconstructor PRO (Pitch-per-hit + Multi-candidate + 96k/24bit)")
        self.resize(1100, 820)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Input / Output
        self.in_edit = QLineEdit()
        self.out_edit = QLineEdit()

        btn_in_file = QPushButton("Input archivo…")
        btn_in_dir = QPushButton("Input carpeta…")
        btn_out_file = QPushButton("Output archivo…")
        btn_out_dir = QPushButton("Output carpeta…")

        btn_in_file.clicked.connect(self.pick_input_file)
        btn_in_dir.clicked.connect(self.pick_input_dir)
        btn_out_file.clicked.connect(self.pick_output_file)
        btn_out_dir.clicked.connect(self.pick_output_dir)

        row_in = QHBoxLayout()
        row_in.addWidget(QLabel("Input:"))
        row_in.addWidget(self.in_edit, stretch=1)
        row_in.addWidget(btn_in_file)
        row_in.addWidget(btn_in_dir)

        row_out = QHBoxLayout()
        row_out.addWidget(QLabel("Output:"))
        row_out.addWidget(self.out_edit, stretch=1)
        row_out.addWidget(btn_out_file)
        row_out.addWidget(btn_out_dir)

        layout.addLayout(row_in)
        layout.addLayout(row_out)

        # Wavetables
        gb_wt = QGroupBox("Wavetables + Index")
        g = QVBoxLayout(gb_wt)

        self.wt_dir_edit = QLineEdit(WT_DIR_DEFAULT)
        btn_wt_dir = QPushButton("Carpeta…")
        btn_wt_dir.clicked.connect(self.pick_wt_dir)

        row_wtdir = QHBoxLayout()
        row_wtdir.addWidget(QLabel("Carpeta wavetables:"))
        row_wtdir.addWidget(self.wt_dir_edit, stretch=1)
        row_wtdir.addWidget(btn_wt_dir)
        g.addLayout(row_wtdir)

        row_seed = QHBoxLayout()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_000_000_000)
        self.seed_spin.setValue(0)
        self.process_96k = QCheckBox("Procesar a 96k")
        self.process_96k.setChecked(True)

        row_seed.addWidget(QLabel("Seed (0=RANDOM):"))
        row_seed.addWidget(self.seed_spin)
        row_seed.addSpacing(10)
        row_seed.addWidget(self.process_96k)
        row_seed.addStretch()
        g.addLayout(row_seed)

        layout.addWidget(gb_wt)

        # Hit controls
        gb_an = QGroupBox("Hits / Timing")
        a = QHBoxLayout(gb_an)

        self.hop_spin = QSpinBox()
        self.hop_spin.setRange(64, 4096)
        self.hop_spin.setSingleStep(64)
        self.hop_spin.setValue(DEFAULT_HOP_LENGTH)

        self.env_alpha = QDoubleSpinBox()
        self.env_alpha.setRange(0.01, 1.0)
        self.env_alpha.setSingleStep(0.05)
        self.env_alpha.setValue(0.25)

        self.trans_alpha = QDoubleSpinBox()
        self.trans_alpha.setRange(0.01, 1.0)
        self.trans_alpha.setSingleStep(0.05)
        self.trans_alpha.setValue(0.25)

        self.only_hits_check = QCheckBox("Solo hits")
        self.only_hits_check.setChecked(True)

        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.01, 1.0)
        self.thresh_spin.setSingleStep(0.02)
        self.thresh_spin.setValue(DEFAULT_THRESH)

        self.mindist_spin = QDoubleSpinBox()
        self.mindist_spin.setRange(1.0, 300.0)
        self.mindist_spin.setSingleStep(1.0)
        self.mindist_spin.setValue(DEFAULT_MIN_DIST_MS)

        self.body_start_spin = QDoubleSpinBox()
        self.body_start_spin.setRange(0.0, 60.0)
        self.body_start_spin.setSingleStep(1.0)
        self.body_start_spin.setValue(DEFAULT_BODY_START_MS)

        self.body_dur_spin = QDoubleSpinBox()
        self.body_dur_spin.setRange(8.0, 140.0)
        self.body_dur_spin.setSingleStep(2.0)
        self.body_dur_spin.setValue(DEFAULT_BODY_DUR_MS)

        self.maxhit_spin = QDoubleSpinBox()
        self.maxhit_spin.setRange(30.0, 1000.0)
        self.maxhit_spin.setSingleStep(10.0)
        self.maxhit_spin.setValue(DEFAULT_MAX_HIT_MS)

        self.attack_spin = QDoubleSpinBox()
        self.attack_spin.setRange(0.5, 20.0)
        self.attack_spin.setSingleStep(0.5)
        self.attack_spin.setValue(DEFAULT_ATTACK_MS)

        a.addWidget(QLabel("Hop:"))
        a.addWidget(self.hop_spin)
        a.addSpacing(10)
        a.addWidget(QLabel("Env α:"))
        a.addWidget(self.env_alpha)
        a.addSpacing(10)
        a.addWidget(QLabel("Flux α:"))
        a.addWidget(self.trans_alpha)
        a.addSpacing(10)
        a.addWidget(self.only_hits_check)
        a.addSpacing(14)
        a.addWidget(QLabel("Thresh:"))
        a.addWidget(self.thresh_spin)
        a.addSpacing(10)
        a.addWidget(QLabel("MinDist(ms):"))
        a.addWidget(self.mindist_spin)
        a.addSpacing(10)
        a.addWidget(QLabel("Body start(ms):"))
        a.addWidget(self.body_start_spin)
        a.addWidget(QLabel("dur(ms):"))
        a.addWidget(self.body_dur_spin)
        a.addSpacing(10)
        a.addWidget(QLabel("Max hit(ms):"))
        a.addWidget(self.maxhit_spin)
        a.addSpacing(10)
        a.addWidget(QLabel("Attack(ms):"))
        a.addWidget(self.attack_spin)
        a.addStretch()
        layout.addWidget(gb_an)

        # Matching / Multi-candidate
        gb_mt = QGroupBox("Matching (Multi-candidato + carácter)")
        m = QHBoxLayout(gb_mt)

        self.match_topk = QSpinBox()
        self.match_topk.setRange(1, 64)
        self.match_topk.setValue(12)

        self.mix_n = QSpinBox()
        self.mix_n.setRange(1, 6)
        self.mix_n.setValue(DEFAULT_MIX_N)

        self.mix_temp = QDoubleSpinBox()
        self.mix_temp.setRange(0.01, 1.0)
        self.mix_temp.setSingleStep(0.01)
        self.mix_temp.setValue(DEFAULT_MIX_TEMP)

        self.pos_min = QDoubleSpinBox()
        self.pos_min.setRange(0.0, 1.0)
        self.pos_min.setSingleStep(0.01)
        self.pos_min.setValue(0.0)

        self.pos_max = QDoubleSpinBox()
        self.pos_max.setRange(0.0, 1.0)
        self.pos_max.setSingleStep(0.01)
        self.pos_max.setValue(1.0)

        self.pos_walk_min = QDoubleSpinBox()
        self.pos_walk_min.setRange(0.0, 0.2)
        self.pos_walk_min.setSingleStep(0.005)
        self.pos_walk_min.setValue(0.01)

        self.pos_walk_max = QDoubleSpinBox()
        self.pos_walk_max.setRange(0.0, 0.2)
        self.pos_walk_max.setSingleStep(0.005)
        self.pos_walk_max.setValue(0.03)

        m.addWidget(QLabel("TopK:"))
        m.addWidget(self.match_topk)
        m.addSpacing(10)
        m.addWidget(QLabel("MixN:"))
        m.addWidget(self.mix_n)
        m.addSpacing(10)
        m.addWidget(QLabel("Temp:"))
        m.addWidget(self.mix_temp)
        m.addSpacing(14)
        m.addWidget(QLabel("Pos min:"))
        m.addWidget(self.pos_min)
        m.addWidget(QLabel("max:"))
        m.addWidget(self.pos_max)
        m.addSpacing(10)
        m.addWidget(QLabel("Pos walk min:"))
        m.addWidget(self.pos_walk_min)
        m.addWidget(QLabel("max:"))
        m.addWidget(self.pos_walk_max)
        m.addStretch()
        layout.addWidget(gb_mt)

        # Pitch per hit
        gb_pitch = QGroupBox("Pitch por hit (auto) + Pitch envelope")
        p = QHBoxLayout(gb_pitch)

        self.pitch_enable = QCheckBox("Pitch auto")
        self.pitch_enable.setChecked(DEFAULT_PITCH_ENABLE)

        self.pitch_fmin = QDoubleSpinBox()
        self.pitch_fmin.setRange(20.0, 2000.0)
        self.pitch_fmin.setSingleStep(5.0)
        self.pitch_fmin.setValue(DEFAULT_PITCH_FMIN)

        self.pitch_fmax = QDoubleSpinBox()
        self.pitch_fmax.setRange(60.0, 8000.0)
        self.pitch_fmax.setSingleStep(10.0)
        self.pitch_fmax.setValue(DEFAULT_PITCH_FMAX)

        self.tonality_thr = QDoubleSpinBox()
        self.tonality_thr.setRange(0.05, 0.95)
        self.tonality_thr.setSingleStep(0.05)
        self.tonality_thr.setValue(DEFAULT_TONALITY_THR)

        self.pitch_env_semi = QDoubleSpinBox()
        self.pitch_env_semi.setRange(-24.0, 24.0)
        self.pitch_env_semi.setSingleStep(1.0)
        self.pitch_env_semi.setValue(DEFAULT_PITCH_ENV_SEMI)

        self.pitch_env_ms = QDoubleSpinBox()
        self.pitch_env_ms.setRange(0.0, 80.0)
        self.pitch_env_ms.setSingleStep(1.0)
        self.pitch_env_ms.setValue(DEFAULT_PITCH_ENV_MS)

        self.pitch_env_strength = QDoubleSpinBox()
        self.pitch_env_strength.setRange(0.0, 1.0)
        self.pitch_env_strength.setSingleStep(0.05)
        self.pitch_env_strength.setValue(DEFAULT_PITCH_ENV_STRENGTH)

        p.addWidget(self.pitch_enable)
        p.addSpacing(10)
        p.addWidget(QLabel("Fmin:"))
        p.addWidget(self.pitch_fmin)
        p.addWidget(QLabel("Fmax:"))
        p.addWidget(self.pitch_fmax)
        p.addSpacing(10)
        p.addWidget(QLabel("Tonal thr:"))
        p.addWidget(self.tonality_thr)
        p.addSpacing(14)
        p.addWidget(QLabel("Env semi:"))
        p.addWidget(self.pitch_env_semi)
        p.addWidget(QLabel("ms:"))
        p.addWidget(self.pitch_env_ms)
        p.addWidget(QLabel("strength:"))
        p.addWidget(self.pitch_env_strength)
        p.addStretch()
        layout.addWidget(gb_pitch)

        # Output
        gb_out = QGroupBox("Salida")
        o = QHBoxLayout(gb_out)

        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.1, 3.0)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.setValue(1.0)

        self.norm_peak = QCheckBox("Normalizar al pico original")
        self.norm_peak.setChecked(True)

        o.addWidget(QLabel("Gain:"))
        o.addWidget(self.gain_spin)
        o.addSpacing(10)
        o.addWidget(self.norm_peak)
        o.addStretch()
        layout.addWidget(gb_out)

        # Spectral glue
        gb_sm = QGroupBox("Spectral glue (opcional)")
        s = QHBoxLayout(gb_sm)

        self.spec_enable = QCheckBox("Activar")
        self.spec_enable.setChecked(True)

        self.spec_strength = QDoubleSpinBox()
        self.spec_strength.setRange(0.0, 1.0)
        self.spec_strength.setSingleStep(0.05)
        self.spec_strength.setValue(0.45)

        self.spec_smooth_bins = QSpinBox()
        self.spec_smooth_bins.setRange(1, 256)
        self.spec_smooth_bins.setValue(25)

        self.spec_clamp_lo = QDoubleSpinBox()
        self.spec_clamp_lo.setRange(0.01, 1.0)
        self.spec_clamp_lo.setSingleStep(0.05)
        self.spec_clamp_lo.setValue(0.35)

        self.spec_clamp_hi = QDoubleSpinBox()
        self.spec_clamp_hi.setRange(1.0, 20.0)
        self.spec_clamp_hi.setSingleStep(0.5)
        self.spec_clamp_hi.setValue(3.5)

        s.addWidget(self.spec_enable)
        s.addSpacing(10)
        s.addWidget(QLabel("Strength:"))
        s.addWidget(self.spec_strength)
        s.addSpacing(10)
        s.addWidget(QLabel("Smooth bins:"))
        s.addWidget(self.spec_smooth_bins)
        s.addSpacing(10)
        s.addWidget(QLabel("Clamp lo:"))
        s.addWidget(self.spec_clamp_lo)
        s.addWidget(QLabel("hi:"))
        s.addWidget(self.spec_clamp_hi)
        s.addStretch()
        layout.addWidget(gb_sm)

        # Process
        self.btn_process = QPushButton("Procesar")
        self.btn_process.clicked.connect(self.start_processing)
        layout.addWidget(self.btn_process)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        layout.addWidget(self.logs, stretch=1)

        layout.addItem(QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))
        footer = QLabel("© 2025 — WT Match Reconstructor PRO")
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)

        self.thread = None
        self.worker = None

    def log(self, msg: str):
        self.logs.append(msg)

    # --------- pickers ---------
    def pick_input_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar audio (capa)",
            "",
            "Audio files (*.wav *.flac *.ogg *.mp3 *.aiff *.m4a);;Todos (*.*)",
        )
        if path:
            self.in_edit.setText(path)

    def pick_input_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta input (batch)")
        if folder:
            self.in_edit.setText(folder)

    def pick_output_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar salida",
            "resultado__recon.wav",
            "WAV (*.wav);;Todos (*.*)",
        )
        if path:
            self.out_edit.setText(path)

    def pick_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta output")
        if folder:
            self.out_edit.setText(folder)

    def pick_wt_dir(self):
        start = self.wt_dir_edit.text().strip() or WT_DIR_DEFAULT
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de wavetables", start)
        if folder:
            self.wt_dir_edit.setText(folder)

    # --------- run ---------
    def start_processing(self):
        inp = self.in_edit.text().strip()
        outp = self.out_edit.text().strip()
        wt_dir = self.wt_dir_edit.text().strip()

        if not inp or not outp:
            QMessageBox.warning(self, "Falta info", "Completa input y output.")
            return
        if not wt_dir or not os.path.isdir(wt_dir):
            QMessageBox.warning(self, "Wavetables", "La carpeta de wavetables no existe.")
            return

        pos_min = float(self.pos_min.value())
        pos_max = float(self.pos_max.value())
        if pos_min > pos_max:
            pos_min, pos_max = pos_max, pos_min

        self.logs.clear()
        self.progress.setValue(0)
        self.btn_process.setEnabled(False)

        self.thread = QThread()
        self.worker = AudioWorker(
            input_path=inp,
            output_path=outp,
            wt_dir=wt_dir,
            seed=int(self.seed_spin.value()),
            process_96k=bool(self.process_96k.isChecked()),
            hop_length=int(self.hop_spin.value()),
            frame_length=DEFAULT_FRAME_LENGTH,
            env_alpha=float(self.env_alpha.value()),
            transient_alpha=float(self.trans_alpha.value()),
            only_hits=bool(self.only_hits_check.isChecked()),
            output_gain=float(self.gain_spin.value()),
            normalize_to_original_peak=bool(self.norm_peak.isChecked()),
            enable_spec_match=bool(self.spec_enable.isChecked()),
            spec_strength=float(self.spec_strength.value()),
            spec_smooth_bins=int(self.spec_smooth_bins.value()),
            spec_clamp_lo=float(self.spec_clamp_lo.value()),
            spec_clamp_hi=float(self.spec_clamp_hi.value()),
            thresh=float(self.thresh_spin.value()),
            min_dist_ms=float(self.mindist_spin.value()),
            body_start_ms=float(self.body_start_spin.value()),
            body_dur_ms=float(self.body_dur_spin.value()),
            max_hit_ms=float(self.maxhit_spin.value()),
            attack_ms=float(self.attack_spin.value()),
            match_topk=int(self.match_topk.value()),
            mix_n=int(self.mix_n.value()),
            mix_temp=float(self.mix_temp.value()),
            pos_min=pos_min,
            pos_max=pos_max,
            pos_walk_min=float(self.pos_walk_min.value()),
            pos_walk_max=float(self.pos_walk_max.value()),
            pitch_enable=bool(self.pitch_enable.isChecked()),
            pitch_fmin=float(self.pitch_fmin.value()),
            pitch_fmax=float(self.pitch_fmax.value()),
            tonality_thr=float(self.tonality_thr.value()),
            pitch_env_semi=float(self.pitch_env_semi.value()),
            pitch_env_ms=float(self.pitch_env_ms.value()),
            pitch_env_strength=float(self.pitch_env_strength.value()),
        )

        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)

        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.log)
        self.worker.finished.connect(self.on_done)
        self.worker.error.connect(self.on_err)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)

        self.thread.start()

    def on_done(self):
        self.btn_process.setEnabled(True)
        QMessageBox.information(self, "Listo", "Proceso completado.")

    def on_err(self, msg: str):
        self.btn_process.setEnabled(True)
        self.log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Error", msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    import qdarkstyle

    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyside6"))
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


