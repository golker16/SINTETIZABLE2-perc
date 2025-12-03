import sys
import os
import glob
import numpy as np
import soundfile as sf

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
    QListWidget,
    QListWidgetItem,
)

# ----------------- DEFAULTS -----------------
DEFAULT_FRAME_LENGTH = 2048
DEFAULT_HOP_LENGTH = 256
AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".m4a")


# ----------------- I/O -----------------
def load_mono(path: str):
    y, sr = lr_load(path, sr=None, mono=True)
    return y.astype(np.float32, copy=False), int(sr)


def list_audio_files(folder: str):
    files = []
    for ext in AUDIO_EXTS:
        files += glob.glob(os.path.join(folder, f"*{ext}"))
        files += glob.glob(os.path.join(folder, f"*{ext.upper()}"))
    files = [f for f in files if os.path.isfile(f)]
    files.sort(key=lambda p: p.lower())
    return files


# ----------------- DSP utils -----------------
def one_pole_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 1e-6, 1.0))
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


def smooth_freq(x: np.ndarray, k: int) -> np.ndarray:
    """moving average along freq bins; x: (bins,) or (bins, frames)"""
    k = int(max(1, k))
    if k <= 1:
        return x
    kernel = np.ones(k, dtype=np.float32) / float(k)
    pad = k // 2
    if x.ndim == 1:
        xp = np.pad(x, (pad, pad), mode="reflect")
        return np.convolve(xp, kernel, mode="valid").astype(np.float32)
    bins, frames = x.shape
    out = np.empty_like(x, dtype=np.float32)
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="reflect")
    for t in range(frames):
        out[:, t] = np.convolve(xp[:, t], kernel, mode="valid").astype(np.float32)
    return out


def frame_centers_samples(n_frames: int, frame_length: int, hop_length: int, n_samples: int) -> np.ndarray:
    centers = (np.arange(n_frames) * hop_length + frame_length / 2.0)
    return np.clip(centers, 0, max(0, n_samples - 1)).astype(np.int64)


def hann(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    t = np.arange(n, dtype=np.float32)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * t / (n - 1))).astype(np.float32)


def remove_dc_and_fade(x: np.ndarray, fade_ms: float, sr: int) -> np.ndarray:
    y = x.astype(np.float32, copy=True)
    y -= float(np.mean(y))
    fade = int(max(1, fade_ms * 0.001 * sr))
    fade = min(fade, len(y) // 4)
    if fade > 0:
        w = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        y[:fade] *= w
        y[-fade:] *= w[::-1]
    return y


# ----------------- Trigger detection (spectral flux) -----------------
def spectral_flux_triggers(
    y: np.ndarray,
    sr: int,
    frame_length: int,
    hop_length: int,
    flux_smooth_alpha: float,
    thresh_rel: float,
    min_dist_ms: float,
):
    """
    Returns dict with:
      trig_frames (int32), vel (float32 0..1), flux (float32 0..1),
      mag (bins, frames), freqs (bins,),
      stats: thr, med, min_dist_frames
    """
    S = lr_stft(
        y.astype(np.float32),
        n_fft=frame_length,
        hop_length=hop_length,
        win_length=frame_length,
        center=True,
    )
    mag = np.abs(S).astype(np.float32)
    logmag = np.log(mag + 1e-8).astype(np.float32)

    d = np.diff(logmag, axis=1)
    dpos = np.maximum(d, 0.0)
    flux = np.sum(dpos, axis=0).astype(np.float32)
    flux = np.concatenate([np.array([0.0], dtype=np.float32), flux], axis=0)

    flux /= (np.max(flux) + 1e-12)
    flux = one_pole_smooth(flux, alpha=flux_smooth_alpha)
    flux = np.clip(flux, 0.0, 1.0).astype(np.float32)

    med = float(np.median(flux))
    thr = max(med + (1.0 - med) * float(np.clip(thresh_rel, 0.0, 1.0)), 0.02)

    min_dist_frames = int(max(1, (min_dist_ms * 0.001 * sr) / hop_length))

    peaks = []
    last = -10**9
    for t in range(1, len(flux) - 1):
        if t - last < min_dist_frames:
            continue
        if flux[t] >= thr and flux[t] >= flux[t - 1] and flux[t] >= flux[t + 1]:
            peaks.append(t)
            last = t

    trig_frames = np.array(peaks, dtype=np.int32)
    vel = np.zeros(len(trig_frames), dtype=np.float32)
    if len(trig_frames) > 0:
        vel = flux[trig_frames].astype(np.float32)
        vel = vel / (np.max(vel) + 1e-12)
        vel = np.clip(vel, 0.0, 1.0).astype(np.float32)

    freqs = np.linspace(0.0, sr / 2.0, mag.shape[0], dtype=np.float32)

    return {
        "trig_frames": trig_frames,
        "vel": vel,
        "flux": flux,
        "mag": mag,
        "freqs": freqs,
        "stats": {"thr": thr, "med": med, "min_dist_frames": min_dist_frames},
    }


# ----------------- Per-event analysis -----------------
def extract_attack_env(mag: np.ndarray, t0: int, attack_frames: int, smooth_bins: int) -> np.ndarray:
    t1 = min(mag.shape[1], t0 + max(1, attack_frames))
    seg = mag[:, t0:t1]
    if seg.shape[1] == 0:
        E = np.mean(mag, axis=1).astype(np.float32)
    else:
        E = np.mean(seg, axis=1).astype(np.float32)
    E = smooth_freq(E, smooth_bins)
    E /= (np.max(E) + 1e-12)
    return np.clip(E, 0.0, 1.0).astype(np.float32)


def tonalness_score(mag: np.ndarray, t0: int, win_frames: int = 8) -> float:
    t1 = min(mag.shape[1], t0 + max(1, win_frames))
    seg = mag[:, t0:t1]
    if seg.size == 0:
        return 0.0
    m = np.mean(seg, axis=1).astype(np.float32)
    tot = float(np.sum(m) + 1e-12)
    K = min(32, len(m))
    top = float(np.sum(np.partition(m, -K)[-K:]))
    return float(np.clip(top / tot, 0.0, 1.0))


def pick_modal_freqs(
    mag: np.ndarray,
    freqs: np.ndarray,
    t0: int,
    win_frames: int = 10,
    f_lo: float = 30.0,
    f_hi: float = 6000.0,
    K: int = 4,
):
    t1 = min(mag.shape[1], t0 + max(1, win_frames))
    seg = mag[:, t0:t1]
    if seg.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    m = np.mean(seg, axis=1).astype(np.float32)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    idx = np.where(mask)[0]
    if len(idx) < 12:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    m2 = m[idx]
    peaks = []
    for i in range(1, len(m2) - 1):
        if m2[i] > m2[i - 1] and m2[i] > m2[i + 1]:
            peaks.append(i)
    if not peaks:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    peaks = np.array(peaks, dtype=np.int32)
    vals = m2[peaks]
    order = np.argsort(vals)[::-1][:K]
    sel = peaks[order]
    f_sel = freqs[idx[sel]].astype(np.float32)
    a_sel = vals[order].astype(np.float32)
    a_sel /= (np.max(a_sel) + 1e-12)
    return f_sel, a_sel


# ----------------- Kernel synthesis (reconstruction) -----------------
def make_click(sr: int, length_samples: int) -> np.ndarray:
    n = int(length_samples)
    t = np.arange(n, dtype=np.float32)
    c = 0.15 * (n - 1)
    sigma = max(1.0, 0.08 * n)
    g = np.exp(-0.5 * ((t - c) / sigma) ** 2).astype(np.float32)
    dg = np.diff(g, prepend=g[0]).astype(np.float32)
    dg *= hann(n)
    dg /= (np.max(np.abs(dg)) + 1e-12)
    return dg.astype(np.float32)


def shape_noise_to_env(
    sr: int,
    env_f: np.ndarray,     # (bins,) 0..1
    frame_length: int,
    hop_length: int,
    length_samples: int,
    rng: np.random.Generator,
    time_env: np.ndarray | None = None,  # (frames,)
):
    n = int(length_samples)
    noise = rng.standard_normal(n).astype(np.float32)
    noise *= hann(n)

    S = lr_stft(noise, n_fft=frame_length, hop_length=hop_length, win_length=frame_length, center=True)
    ph = np.angle(S).astype(np.float32)
    bins, frames = S.shape

    env = env_f[:bins].astype(np.float32)
    env = env / (np.max(env) + 1e-12)

    if time_env is None:
        mag_tgt = env[:, None] * np.ones((1, frames), dtype=np.float32)
    else:
        te = time_env[:frames].astype(np.float32)
        te = np.clip(te, 0.0, 1.0)
        mag_tgt = env[:, None] * te[None, :]

    # conserva: evita “harsh”
    mag_tgt = np.clip(mag_tgt, 0.0, 1.0)

    S_out = mag_tgt * (np.cos(ph) + 1j * np.sin(ph))
    y = lr_istft(S_out, hop_length=hop_length, win_length=frame_length, center=True, length=n)
    y = np.asarray(y, dtype=np.float32)
    y = remove_dc_and_fade(y, fade_ms=0.4, sr=sr)
    y /= (np.max(np.abs(y)) + 1e-12)
    return y.astype(np.float32)


def render_modal_tail(
    sr: int,
    freqs_hz: np.ndarray,
    amps: np.ndarray,
    length_samples: int,
    tau_ms: float,
    rng: np.random.Generator,
):
    n = int(length_samples)
    if len(freqs_hz) == 0:
        return np.zeros(n, dtype=np.float32)

    t = np.arange(n, dtype=np.float32) / float(sr)
    tau = max(1e-3, float(tau_ms) * 0.001)
    env = np.exp(-t / tau).astype(np.float32)

    y = np.zeros(n, dtype=np.float32)
    for f, a in zip(freqs_hz, amps):
        ph0 = float(rng.uniform(0.0, 2.0 * np.pi))
        y += (float(a) * np.sin(2.0 * np.pi * float(f) * t + ph0)).astype(np.float32)

    y *= env
    y *= hann(n)
    y = remove_dc_and_fade(y, fade_ms=0.4, sr=sr)
    y /= (np.max(np.abs(y)) + 1e-12)
    return y.astype(np.float32)


def highband_air(sr: int, length_samples: int, rng: np.random.Generator, hp_hz: float = 9000.0):
    n = int(length_samples)
    x = rng.standard_normal(n).astype(np.float32)
    x *= hann(n)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr).astype(np.float32)
    mask = (freqs >= float(hp_hz)).astype(np.float32)
    X *= mask
    y = np.fft.irfft(X, n=n).astype(np.float32)
    y = remove_dc_and_fade(y, fade_ms=0.3, sr=sr)
    y /= (np.max(np.abs(y)) + 1e-12)
    return y.astype(np.float32)


def make_event_kernel(
    sr: int,
    frame_length: int,
    hop_length: int,
    env_f_attack: np.ndarray,
    vel: float,
    tonal_strength: float,
    freqs_modal: np.ndarray,
    amps_modal: np.ndarray,
    rng: np.random.Generator,
    # ms parameters
    click_ms: float,
    noise_ms: float,
    tail_ms: float,
    air_ms: float,
    air_amount: float,
):
    vel = float(np.clip(vel, 0.0, 1.0))
    tonal_strength = float(np.clip(tonal_strength, 0.0, 1.0))

    n_click = int(max(8, click_ms * 0.001 * sr))
    n_noise = int(max(32, noise_ms * 0.001 * sr))
    n_tail = int(max(64, tail_ms * 0.001 * sr))
    n_air = int(max(32, air_ms * 0.001 * sr))

    click = make_click(sr, n_click)
    click *= (0.55 + 0.70 * vel)

    frames_noise = max(1, int(np.ceil(n_noise / hop_length)) + 2)
    te = (np.linspace(1.0, 0.0, frames_noise, dtype=np.float32) ** 1.6).astype(np.float32)

    noise = shape_noise_to_env(
        sr=sr,
        env_f=env_f_attack,
        frame_length=frame_length,
        hop_length=hop_length,
        length_samples=n_noise,
        rng=rng,
        time_env=te,
    )
    noise *= (0.30 + 0.95 * vel)

    tail = np.zeros(n_tail, dtype=np.float32)
    if tonal_strength > 0.22 and len(freqs_modal) > 0:
        tau_ms = 35.0 + 220.0 * tonal_strength
        tail = render_modal_tail(sr, freqs_modal, amps_modal, n_tail, tau_ms=tau_ms, rng=rng)
        tail *= (0.22 + 0.95 * vel) * (0.25 + 0.75 * tonal_strength)

    air = np.zeros(n_air, dtype=np.float32)
    if air_amount > 0.0:
        air = highband_air(sr, n_air, rng=rng, hp_hz=9000.0)
        air *= float(np.clip(air_amount, 0.0, 2.0)) * (0.08 + 0.35 * vel)

    n = max(n_click, n_noise, n_tail, n_air)
    out = np.zeros(n, dtype=np.float32)
    out[:n_click] += click
    out[:n_noise] += noise
    out[:n_tail] += tail
    out[:n_air] += air

    # human-ish micro variation
    out *= float(0.90 + 0.20 * rng.random())

    out = remove_dc_and_fade(out, fade_ms=0.6, sr=sr)
    out /= (np.max(np.abs(out)) + 1e-12)
    out *= 0.95
    return out.astype(np.float32)


# ----------------- Event-based reconstruction -----------------
def reconstruct_by_events(
    y: np.ndarray,
    sr: int,
    frame_length: int,
    hop_length: int,
    flux_smooth_alpha: float,
    thresh_rel: float,
    min_dist_ms: float,
    attack_frames: int,
    env_smooth_bins: int,
    only_hits: bool,
    air_amount: float,
    rng: np.random.Generator,
):
    det = spectral_flux_triggers(
        y=y,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
        flux_smooth_alpha=flux_smooth_alpha,
        thresh_rel=thresh_rel,
        min_dist_ms=min_dist_ms,
    )
    trig_frames = det["trig_frames"]
    vel = det["vel"]
    flux = det["flux"]
    mag = det["mag"]
    freqs = det["freqs"]
    stats = det["stats"]

    n_samples = len(y)
    out = np.zeros(n_samples, dtype=np.float32)
    if len(trig_frames) == 0:
        return out, det

    centers = frame_centers_samples(mag.shape[1], frame_length, hop_length, n_samples)

    # if not only_hits: option to keep low-level bed (very low) - disabled by default
    bed = np.zeros(n_samples, dtype=np.float32)
    if not only_hits:
        # tiny “bed” derived from flux (just to avoid total dryness if desired)
        # intentionally conservative
        bed_strength = 0.02
        frames_cent = centers.astype(np.float32)
        t = np.arange(n_samples, dtype=np.float32)
        env = np.interp(t, frames_cent, flux.astype(np.float32)).astype(np.float32)
        bed = (env ** 1.2) * bed_strength

    # per-event synthesis
    for k, t0 in enumerate(trig_frames):
        start_s = int(centers[t0])
        v = float(vel[k])

        env_f = extract_attack_env(mag, t0, attack_frames=attack_frames, smooth_bins=env_smooth_bins)
        ton = tonalness_score(mag, t0, win_frames=8)
        f_modal, a_modal = pick_modal_freqs(mag, freqs, t0, win_frames=10, K=4)

        # adaptive lengths (ms)
        # tonal -> longer tail, transient-> shorter
        tail_ms = 60.0 + 260.0 * float(np.clip(ton, 0.0, 1.0))
        noise_ms = 22.0 + 75.0 * float(np.clip(1.0 - ton, 0.0, 1.0))

        ker = make_event_kernel(
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
            env_f_attack=env_f,
            vel=v,
            tonal_strength=ton,
            freqs_modal=f_modal,
            amps_modal=a_modal,
            rng=rng,
            click_ms=8.0,
            noise_ms=noise_ms,
            tail_ms=tail_ms,
            air_ms=28.0,
            air_amount=air_amount,
        )

        # overlap-add
        end_s = min(n_samples, start_s + len(ker))
        if end_s > start_s:
            out[start_s:end_s] += ker[: end_s - start_s]

    out += bed.astype(np.float32)

    # peak control
    mx = float(np.max(np.abs(out)) + 1e-12)
    out = (out / mx * 0.95).astype(np.float32)
    return out, det


# ----------------- WORKER -----------------
class AudioWorker(QObject):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        input_files: list,
        output_path: str,
        seed: int,
        hop_length: int,
        frame_length: int,
        flux_alpha: float,
        trig_thresh: float,
        min_dist_ms: float,
        attack_frames: int,
        env_smooth_bins: int,
        only_hits: bool,
        air_amount: float,
        output_gain: float,
    ):
        super().__init__()
        self.input_files = list(input_files)
        self.output_path = str(output_path)
        self.seed = int(seed)
        self.hop_length = int(hop_length)
        self.frame_length = int(frame_length)
        self.flux_alpha = float(flux_alpha)
        self.trig_thresh = float(trig_thresh)
        self.min_dist_ms = float(min_dist_ms)
        self.attack_frames = int(attack_frames)
        self.env_smooth_bins = int(env_smooth_bins)
        self.only_hits = bool(only_hits)
        self.air_amount = float(air_amount)
        self.output_gain = float(output_gain)

    def _resolve_outfile(self, src_file: str) -> str:
        outp = self.output_path
        if os.path.isdir(outp) or outp.endswith(os.sep) or (os.path.splitext(outp)[1].lower() != ".wav"):
            os.makedirs(outp, exist_ok=True)
            base = os.path.splitext(os.path.basename(src_file))[0]
            return os.path.join(outp, base + "__recon.wav")
        return outp

    def _process_one(self, src_file: str, out_file: str, rng: np.random.Generator):
        self.log.emit(f"Fuente: {os.path.basename(src_file)}")
        y, sr = load_mono(src_file)
        dur = len(y) / float(sr)
        self.log.emit(f" sr={sr}  samples={len(y)}  dur={dur:.3f}s")
        self.log.emit(
            f" Params: frame={self.frame_length} hop={self.hop_length} | fluxα={self.flux_alpha:.3f} "
            f"thr={self.trig_thresh:.3f} minDist={self.min_dist_ms:.1f}ms | attackFrames={self.attack_frames} "
            f"smoothBins={self.env_smooth_bins} | onlyHits={self.only_hits} air={self.air_amount:.2f}"
        )

        y_rec, det = reconstruct_by_events(
            y=y,
            sr=sr,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            flux_smooth_alpha=self.flux_alpha,
            thresh_rel=self.trig_thresh,
            min_dist_ms=self.min_dist_ms,
            attack_frames=self.attack_frames,
            env_smooth_bins=self.env_smooth_bins,
            only_hits=self.only_hits,
            air_amount=self.air_amount,
            rng=rng,
        )

        trig_frames = det["trig_frames"]
        vel = det["vel"]
        stats = det["stats"]
        self.log.emit(
            f" Triggers: {len(trig_frames)} | medianFlux={stats['med']:.3f} thrAbs={stats['thr']:.3f} "
            f"minDistFrames={stats['min_dist_frames']}"
        )
        if len(trig_frames) > 0:
            # show first few trigger times
            centers = frame_centers_samples(det["mag"].shape[1], self.frame_length, self.hop_length, len(y))
            show = min(8, len(trig_frames))
            times_ms = (centers[trig_frames[:show]] / float(sr) * 1000.0)
            vshow = vel[:show]
            pairs = ", ".join([f"{times_ms[i]:.1f}ms(v={vshow[i]:.2f})" for i in range(show)])
            self.log.emit(f" Primeros triggers: {pairs}")

        if self.output_gain != 1.0:
            y_rec = np.clip(y_rec * float(self.output_gain), -1.0, 1.0).astype(np.float32)

        out_dir = os.path.dirname(out_file)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        sf.write(out_file, y_rec, sr)
        self.log.emit(f" Guardado: {out_file}")

    def run(self):
        try:
            files = [f for f in self.input_files if os.path.isfile(f)]
            if not files:
                raise RuntimeError("No hay archivos de entrada válidos.")

            outp = self.output_path.strip()
            if not outp:
                raise RuntimeError("Output vacío. Elige una carpeta o archivo de salida.")

            # si múltiples inputs: output debe ser carpeta o ruta sin extensión .wav
            if len(files) > 1 and os.path.splitext(outp)[1].lower() == ".wav" and not os.path.isdir(outp):
                raise RuntimeError("Para múltiples archivos de entrada, el Output debe ser una carpeta (no un .wav).")

            rng = np.random.default_rng(None if self.seed == 0 else self.seed)

            total = len(files)
            self.progress.emit(1)

            for idx, f in enumerate(files):
                self.log.emit("")
                self.log.emit(f"[{idx+1}/{total}]")
                out_file = self._resolve_outfile(f)
                p = int(2 + 96 * (idx / max(1, total)))
                self.progress.emit(p)
                self._process_one(f, out_file, rng)

            self.progress.emit(100)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


# ----------------- UI -----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reconstrucción Percusiva por Eventos (HQ) — Batch / Multi-capas")
        self.resize(1020, 760)

        self.input_files: list[str] = []

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # ---------- Input / Output ----------
        gb_io = QGroupBox("Entrada / Salida")
        io = QVBoxLayout(gb_io)

        self.in_edit = QLineEdit()
        self.in_edit.setPlaceholderText("Selecciona archivo(s) o carpeta…")
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Selecciona carpeta de salida (recomendado) o archivo .wav (solo 1 input)…")

        btn_in_files = QPushButton("Input archivo(s)…")
        btn_in_dir = QPushButton("Input carpeta…")
        btn_clear = QPushButton("Limpiar lista")

        btn_out_file = QPushButton("Output archivo…")
        btn_out_dir = QPushButton("Output carpeta…")

        btn_in_files.clicked.connect(self.pick_input_files)
        btn_in_dir.clicked.connect(self.pick_input_dir)
        btn_clear.clicked.connect(self.clear_inputs)
        btn_out_file.clicked.connect(self.pick_output_file)
        btn_out_dir.clicked.connect(self.pick_output_dir)

        row_in = QHBoxLayout()
        row_in.addWidget(QLabel("Input:"))
        row_in.addWidget(self.in_edit, stretch=1)
        row_in.addWidget(btn_in_files)
        row_in.addWidget(btn_in_dir)
        row_in.addWidget(btn_clear)

        row_out = QHBoxLayout()
        row_out.addWidget(QLabel("Output:"))
        row_out.addWidget(self.out_edit, stretch=1)
        row_out.addWidget(btn_out_file)
        row_out.addWidget(btn_out_dir)

        io.addLayout(row_in)
        io.addLayout(row_out)

        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(120)
        io.addWidget(QLabel("Archivos en cola:"))
        io.addWidget(self.file_list)

        layout.addWidget(gb_io)

        # ---------- Settings ----------
        gb_set = QGroupBox("Parámetros (calidad / triggers / aire)")
        s = QHBoxLayout(gb_set)

        # left column
        col1 = QVBoxLayout()
        col2 = QVBoxLayout()

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_000_000_000)
        self.seed_spin.setValue(0)

        self.hop_spin = QSpinBox()
        self.hop_spin.setRange(64, 4096)
        self.hop_spin.setSingleStep(64)
        self.hop_spin.setValue(DEFAULT_HOP_LENGTH)

        self.flux_alpha = QDoubleSpinBox()
        self.flux_alpha.setRange(0.01, 1.0)
        self.flux_alpha.setSingleStep(0.05)
        self.flux_alpha.setValue(0.25)

        self.trig_thresh = QDoubleSpinBox()
        self.trig_thresh.setRange(0.0, 1.0)
        self.trig_thresh.setSingleStep(0.05)
        self.trig_thresh.setValue(0.35)

        self.min_dist_ms = QDoubleSpinBox()
        self.min_dist_ms.setRange(5.0, 250.0)
        self.min_dist_ms.setSingleStep(5.0)
        self.min_dist_ms.setValue(35.0)

        self.attack_frames = QSpinBox()
        self.attack_frames.setRange(1, 10)
        self.attack_frames.setValue(3)

        self.env_smooth_bins = QSpinBox()
        self.env_smooth_bins.setRange(1, 256)
        self.env_smooth_bins.setValue(31)

        self.only_hits = QCheckBox("Solo hits (sin bed)")
        self.only_hits.setChecked(True)

        self.air_amount = QDoubleSpinBox()
        self.air_amount.setRange(0.0, 2.0)
        self.air_amount.setSingleStep(0.1)
        self.air_amount.setValue(1.0)

        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.1, 3.0)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.setValue(1.0)

        col1.addWidget(QLabel("Seed (0 = RANDOM):"))
        col1.addWidget(self.seed_spin)
        col1.addSpacing(6)
        col1.addWidget(QLabel("Hop (samples):"))
        col1.addWidget(self.hop_spin)
        col1.addSpacing(6)
        col1.addWidget(QLabel("Flux smooth α:"))
        col1.addWidget(self.flux_alpha)
        col1.addSpacing(6)
        col1.addWidget(QLabel("Trigger thresh (rel):"))
        col1.addWidget(self.trig_thresh)
        col1.addStretch()

        col2.addWidget(QLabel("Min dist (ms):"))
        col2.addWidget(self.min_dist_ms)
        col2.addSpacing(6)
        col2.addWidget(QLabel("Attack frames:"))
        col2.addWidget(self.attack_frames)
        col2.addSpacing(6)
        col2.addWidget(QLabel("Env smooth bins:"))
        col2.addWidget(self.env_smooth_bins)
        col2.addSpacing(6)
        col2.addWidget(self.only_hits)
        col2.addSpacing(6)
        col2.addWidget(QLabel("Air amount:"))
        col2.addWidget(self.air_amount)
        col2.addSpacing(6)
        col2.addWidget(QLabel("Output gain:"))
        col2.addWidget(self.gain_spin)
        col2.addStretch()

        s.addLayout(col1)
        s.addSpacing(20)
        s.addLayout(col2)
        s.addStretch()

        layout.addWidget(gb_set)

        # ---------- Process ----------
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
        footer = QLabel("© 2025 Gabriel Golker — Reconstrucción por Eventos (HQ)")
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)

        self.thread = None
        self.worker = None

    def log(self, msg: str):
        self.logs.append(msg)

    def refresh_file_list(self):
        self.file_list.clear()
        for f in self.input_files:
            it = QListWidgetItem(f)
            self.file_list.addItem(it)
        if len(self.input_files) == 0:
            self.in_edit.setText("")
        elif len(self.input_files) == 1:
            self.in_edit.setText(self.input_files[0])
        else:
            self.in_edit.setText(f"{self.input_files[0]}  (+{len(self.input_files)-1} más)")

    def clear_inputs(self):
        self.input_files = []
        self.refresh_file_list()

    # --------- pickers ---------
    def pick_input_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Seleccionar capa(s)/audio(s)",
            "",
            "Audio files (*.wav *.flac *.ogg *.mp3 *.aiff *.m4a);;Todos (*.*)",
        )
        if paths:
            # evita duplicados, mantiene orden
            s = set(self.input_files)
            for p in paths:
                if p not in s:
                    self.input_files.append(p)
                    s.add(p)
            self.refresh_file_list()

    def pick_input_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta input")
        if folder:
            files = list_audio_files(folder)
            if not files:
                QMessageBox.warning(self, "Input", "No se encontraron audios en la carpeta.")
                return
            self.input_files = files
            self.refresh_file_list()

    def pick_output_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar salida (solo recomendado si hay 1 input)",
            "resultado__recon.wav",
            "WAV (*.wav);;Todos (*.*)",
        )
        if path:
            self.out_edit.setText(path)

    def pick_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta output")
        if folder:
            self.out_edit.setText(folder)

    # --------- run ---------
    def start_processing(self):
        outp = self.out_edit.text().strip()

        if not self.input_files:
            QMessageBox.warning(self, "Falta info", "Selecciona al menos 1 archivo o una carpeta de entrada.")
            return
        if not outp:
            QMessageBox.warning(self, "Falta info", "Selecciona Output (carpeta recomendado).")
            return

        # quick validation
        bad = [f for f in self.input_files if not os.path.isfile(f)]
        if bad:
            QMessageBox.warning(self, "Input", "Hay rutas inválidas en la lista.")
            return

        self.logs.clear()
        self.progress.setValue(0)
        self.btn_process.setEnabled(False)
        self.log("=== RECONSTRUCCIÓN POR EVENTOS (HQ) ===")
        self.log(f"Inputs: {len(self.input_files)} archivo(s)")
        self.log(f"Output: {outp}")
        self.log(f"Seed: {'RANDOM' if self.seed_spin.value()==0 else self.seed_spin.value()}")

        self.thread = QThread()
        self.worker = AudioWorker(
            input_files=self.input_files,
            output_path=outp,
            seed=int(self.seed_spin.value()),
            hop_length=int(self.hop_spin.value()),
            frame_length=DEFAULT_FRAME_LENGTH,
            flux_alpha=float(self.flux_alpha.value()),
            trig_thresh=float(self.trig_thresh.value()),
            min_dist_ms=float(self.min_dist_ms.value()),
            attack_frames=int(self.attack_frames.value()),
            env_smooth_bins=int(self.env_smooth_bins.value()),
            only_hits=bool(self.only_hits.isChecked()),
            air_amount=float(self.air_amount.value()),
            output_gain=float(self.gain_spin.value()),
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

