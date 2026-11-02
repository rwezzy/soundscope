"""Direction of arrival for the reSpeaker XVF3800's four raw mics.

SRP-PHAT (steered response power with phase transform), tuned for faint and far sounds:
- each time-frequency bin is weighted by how far it rises above the room's noise floor,
  so steady noise (fans, hum) drops out while faint sounds still count;
- bins that are only a decaying echo of earlier frames get no weight, because the direct
  sound always arrives before its echoes (this is what keeps distant sounds on target);
- evidence is summed over frames, so a quiet sound gets sharper the longer it plays;
- the live tracker keeps the few frames from just before a sound is detected, since
  that first arrival carries the cleanest direction.
"""
from collections import deque

import numpy as np

FS = 16000              # Hz, rate of the 16k6ch firmware
N_FFT = 512             # 32 ms analysis frames
HOP = 256               # 16 ms between frames
SPEED_OF_SOUND = 343.0  # m/s at about 20 C
UPSAMPLE = 16           # cross-correlation lag resolution: 1/16 sample, about 1.3 mm of path

# Raw mic positions in meters, board frame seen from the LED side: x = front (cable) edge,
# y = left, z = out of the LED side. Order (channels 2-5): front-left, front-right, back-right,
# back-left; a 66 mm square. This is the chip's AEC_MIC_ARRAY_GEO table mirrored (y flipped):
# the table reads as if seen from the back of the board. Confirmed Nov 2, 2026 by the tap test
# (channels [3, 2, 5, 4] in the old order), a live camera recording, and a laptop-speaker test
# that read 45 deg right with the table as-is and 45 deg left (correct) mirrored.
MIC_XYZ = 0.033 * np.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]], float)


def direction_grid(az_step=1.0, el_step=2.5):
    """Unit vectors over the upper hemisphere (a flat array can't tell up from down)."""
    az, el = np.meshgrid(np.radians(np.arange(0, 360, az_step)),
                         np.radians(np.arange(0, 90 + el_step / 2, el_step)), indexing='ij')
    az, el = az.ravel(), el.ravel()
    dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    return dirs, np.degrees(az), np.degrees(el)


def snr_db(power_ratio):
    """Signal-to-noise ratio from (signal + noise) / noise power."""
    return 10 * np.log10(max(power_ratio - 1, 1e-3))


class Localizer:
    def __init__(self, mic_xyz=MIC_XYZ, f_lo=200.0, f_hi=7600.0, echo_decay_s=1.0):
        mic_xyz = np.asarray(mic_xyz, float)
        self.mic_xyz = mic_xyz
        # Power that fades at least as fast as a room echo (60 dB per echo_decay_s) counts as echo.
        self.echo_decay = 10 ** (-6 * HOP / FS / echo_decay_s)
        n = len(mic_xyz)
        self.pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
        freqs = np.fft.rfftfreq(N_FFT, 1 / FS)
        self.band = (freqs >= f_lo) & (freqs <= f_hi)
        self.window = np.hanning(N_FFT)
        self.dirs, self.az, self.el = direction_grid()
        # Pair (i, j) correlates best at lag (p_j - p_i) . d / c for a sound from direction d.
        lags = (mic_xyz[self.pairs[:, 1]] - mic_xyz[self.pairs[:, 0]]) @ self.dirs.T / SPEED_OF_SOUND
        self.lag_idx = np.round(lags * FS * UPSAMPLE).astype(int) % (N_FFT * UPSAMPLE)
        self.noise = None  # per-bin noise power, from calibrate() or a Tracker

    def stft(self, x):
        """(samples, mics) -> (frames, mics, bins)."""
        n = (len(x) - N_FFT) // HOP + 1
        if n < 1:
            return np.zeros((0, x.shape[1], N_FFT // 2 + 1), complex)
        frames = x[np.arange(N_FFT) + HOP * np.arange(n)[:, None]]  # (frames, N_FFT, mics)
        frames = frames - frames.mean(axis=1, keepdims=True)
        return np.fft.rfft(frames * self.window[:, None], axis=1).transpose(0, 2, 1)

    @staticmethod
    def power(X):
        return (np.abs(X) ** 2).mean(axis=-2)  # mean over mics

    def calibrate(self, x):
        """Learn the room's noise floor from a stretch with no target sound."""
        self.noise = self.power(self.stft(x)).mean(axis=0) + 1e-30

    def weights(self, P, past=None):
        """Per-bin weights for power P (frames, bins), plus the echo memory to pass to the next call.

        Weight = the inter-mic coherence the bin's SNR implies (0 at the noise floor, near 1 well
        above it) times the share of its power that is new rather than an echo of earlier frames.
        """
        xi = np.maximum(P / self.noise - 1.0, 0.0)
        w = (xi / (1.0 + xi)) ** 2
        w[..., ~self.band] = 0.0
        past = np.zeros(P.shape[-1]) if past is None else past
        for t in range(len(P)):
            w[t] *= np.clip(1 - past / (P[t] + 1e-30), 0, 1)
            past = np.maximum(P[t], past * self.echo_decay)
        return w, past

    def evidence(self, X, w):
        """Sum weighted, PHAT-normalized cross-spectra over frames. Returns (C, W, W2)."""
        G = X[:, self.pairs[:, 0]] * np.conj(X[:, self.pairs[:, 1]])  # (frames, pairs, bins)
        G /= np.abs(G) + 1e-30
        return (G * w[:, None]).sum(axis=0), w.sum(axis=0), (w ** 2).sum(axis=0)

    def gcc(self, C, W):
        """Cross-correlation of every mic pair at 1/UPSAMPLE-sample lags; each pair peaks at <= 1."""
        m = N_FFT * UPSAMPLE
        spec = np.zeros((len(self.pairs), m // 2 + 1), complex)
        spec[:, :C.shape[1]] = C
        return np.fft.irfft(spec, n=m, axis=1) * (m / 2) / W.sum()

    def srp_points(self, gcc, points):
        """How well a sound from each 3D point (N, 3; meters, array frame) fits the evidence.

        Uses the true distance from each point to each mic, so it stays right for nearby points;
        on the same scale as the direction map in peak().
        """
        dist = np.linalg.norm(points[:, None, :] - self.mic_xyz[None], axis=2)              # (N, mics)
        lags = (dist[:, self.pairs[:, 0]] - dist[:, self.pairs[:, 1]]) / SPEED_OF_SOUND     # (N, pairs)
        idx = np.round(lags * FS * UPSAMPLE).astype(int) % gcc.shape[1]
        return gcc[np.arange(len(self.pairs)), idx].mean(axis=1)

    def peak(self, C, W, W2):
        """Best direction from summed evidence, or None if nothing rose above the floor."""
        total = W.sum()
        if total <= 0:
            return None
        srp = np.take_along_axis(self.gcc(C, W), self.lag_idx, axis=1).mean(axis=0)
        k = int(np.argmax(srp))
        # How far the map would wander if every phase were random; z says how sure the peak is.
        sigma = np.sqrt(W2.sum() / 2 / len(self.pairs)) / total
        return {'azimuth': float(self.az[k]), 'elevation': float(self.el[k]),
                'direction': self.dirs[k], 'coherence': float(srp[k]),
                'z': float((srp[k] - np.median(srp)) / sigma)}

    def locate_clip(self, x):
        """Direction of the sound in a recording (samples, mics), using every frame above the floor."""
        X = self.stft(x)
        P = self.power(X)
        if self.noise is None:
            self.noise = np.percentile(P, 20, axis=0) + 1e-30
        result = self.peak(*self.evidence(X, self.weights(P)[0]))
        if result:
            result['snr_db'] = snr_db(P[:, self.band].sum(axis=1).mean() / self.noise[self.band].sum())
        return result


class Tracker:
    """Streaming detector: follows the room's noise floor, spots sounds, and localizes each one."""

    def __init__(self, loc, threshold_db=0.5, memory_s=0.6, gap_s=0.5, update_s=0.15):
        fps = FS / HOP
        self.loc = loc
        self.threshold = 10 ** (threshold_db / 10)
        self.decay = np.exp(-1 / (memory_s * fps))
        self.gap = int(gap_s * fps)
        self.update_every = max(1, int(update_s * fps))
        self.follow = 1 - np.exp(-1 / (2.0 * fps))  # the floor tracks a quiet room over ~2 s
        self.creep = 10 ** (0.5 / 10 / fps)         # and creeps up 0.5 dB/s while something plays
        self.level = 1.0
        self.quiet = self.gap
        self.past = None                            # echo memory for Localizer.weights
        self.recent = deque(maxlen=4)               # onset frames from before detection fired
        self._new_event()

    def _new_event(self):
        shape = (len(self.loc.pairs), N_FFT // 2 + 1)
        self.run = [np.zeros(shape, complex), np.zeros(shape[1]), np.zeros(shape[1])]
        self.event = [np.zeros(shape, complex), np.zeros(shape[1]), np.zeros(shape[1])]
        self.frames = 0
        self.ratios = []

    def _add(self, X, w):
        new = self.loc.evidence(X[None], w)
        for run, event, n in zip(self.run, self.event, new):
            run *= self.decay
            run += n
            event += n
        self.frames += 1

    def _result(self, acc):
        r = self.loc.peak(*acc)
        if r:
            r['seconds'] = self.frames * HOP / FS
            r['snr_db'] = snr_db(np.mean(self.ratios))
            r['evidence'] = tuple(a.copy() for a in acc)  # lets callers score any 3D point (srp_points)
        return r

    def push(self, X):
        """Feed one frame's spectrum (mics, bins). Returns None, ('update', result) or ('end', result)."""
        loc = self.loc
        P = loc.power(X)
        if loc.noise is None:
            loc.noise = P + 1e-30
            return None
        ratio = P[loc.band].sum() / loc.noise[loc.band].sum()
        self.level = 0.75 * self.level + 0.25 * ratio
        strong = np.count_nonzero(P[loc.band] > 10 * loc.noise[loc.band])  # catches faint narrow beeps
        active = self.level > self.threshold or strong >= 3
        w, self.past = loc.weights(P[None], self.past)

        out = None
        if active:
            if self.quiet >= self.gap:  # a new sound: start fresh, keeping its onset
                self._new_event()
                for Xr, wr in self.recent:
                    self._add(Xr, wr)
            self.quiet = 0
            self._add(X, w)
            self.ratios.append(ratio)
            if self.frames >= 3 and self.frames % self.update_every == 0:
                out = ('update', self._result(self.run))
        else:
            self.quiet += 1
            if self.quiet == self.gap and self.frames >= 3:
                out = ('end', self._result(self.event))
        self.recent.append((X, w))

        if active:
            loc.noise = np.where(P > loc.noise, loc.noise * self.creep, loc.noise + self.follow * (P - loc.noise))
        else:
            loc.noise = loc.noise + self.follow * (P - loc.noise)
        return out
