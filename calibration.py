"""Offline, yaw-only calibration from paired horizontal bearings.

Both bearings must already be in the same configured world frame. The returned
correction is added to the configured array yaw. This does not estimate camera or
array position, pitch, roll, microphone order, or prove that the geometry is right.
The caller must supply fresh, independent sound events and valid clicked depth.
"""
from dataclasses import dataclass, replace
import math


def wrap_deg(angle):
    """The signed shortest angle, in [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def _mean_deg(values, weights):
    x = sum(w * math.cos(math.radians(a)) for a, w in zip(values, weights))
    y = sum(w * math.sin(math.radians(a)) for a, w in zip(values, weights))
    if math.hypot(x, y) <= 1e-12 * sum(weights):
        return None
    return wrap_deg(math.degrees(math.atan2(y, x)))


def _span_deg(values):
    """Length of the smallest circular arc containing the bearings."""
    angles = sorted(a % 360.0 for a in values)
    if len(angles) < 2:
        return 0.0
    gaps = [b - a for a, b in zip(angles, angles[1:])]
    gaps.append(angles[0] + 360.0 - angles[-1])
    return 360.0 - max(gaps)


def _three_distinct(values, separation):
    """Require three pairwise separated bearings, including across the seam."""
    for i, a in enumerate(values):
        for j in range(i + 1, len(values)):
            b = values[j]
            if abs(wrap_deg(a - b)) < separation:
                continue
            for c in values[j + 1:]:
                if (abs(wrap_deg(a - c)) >= separation and
                        abs(wrap_deg(b - c)) >= separation):
                    return True
    return False


@dataclass(frozen=True)
class Observation:
    expected_deg: float
    heard_deg: float
    timestamp: float
    z: float
    coherence: float
    weight: float

    @property
    def correction_deg(self):
        return wrap_deg(self.expected_deg - self.heard_deg)


@dataclass(frozen=True)
class SampleIndicator:
    expected_deg: float
    heard_deg: float
    residual_deg: float
    inlier: bool


@dataclass(frozen=True)
class CalibrationResult:
    correction_deg: float | None
    residual_deg: float | None
    count: int
    ready: bool
    message: str
    accepted: bool = True
    samples: tuple[SampleIndicator, ...] = ()
    expected_span_deg: float = 0.0
    heard_span_deg: float = 0.0

    @property
    def outlier_count(self):
        return sum(not sample.inlier for sample in self.samples)


class CalibrationSession:
    """Collect quality observations and fit one circular yaw correction.

    At least three inliers must cover 15 degrees in both observed and expected
    bearings. Outliers can be excluded only with a 75% consensus. The residual is
    quality-weighted angular RMS over the inliers; every excluded observation is
    exposed in ``result.samples``. Quality weights are capped so a single strong
    reflection cannot dominate the fit. No state is written outside this object.
    """

    def __init__(self, *, min_z=12.0, min_coherence=0.10,
                 min_span_deg=15.0, min_distinct_deg=5.0,
                 max_residual_deg=8.0):
        settings = (min_z, min_coherence, min_span_deg, min_distinct_deg,
                    max_residual_deg)
        if not all(math.isfinite(x) and x > 0 for x in settings):
            raise ValueError('Calibration thresholds must be finite and positive.')
        self.min_z = min_z
        self.min_coherence = min_coherence
        self.min_span_deg = min_span_deg
        self.min_distinct_deg = min_distinct_deg
        self.max_residual_deg = max_residual_deg
        self._samples = []

    @property
    def samples(self):
        return tuple(self._samples)

    def add(self, expected_deg, heard_deg, *, timestamp, z, coherence):
        """Add one fresh source observation; invalid/weak records are not stored.

        ``timestamp`` uses the caller's clock. Freshness and distinct sound-event
        identity are checked by the caller, where the live event is available.
        ``accepted`` means this record passed input/quality checks; consult its
        inlier indicator to see whether it agrees with the other observations.
        """
        try:
            expected_deg, heard_deg, timestamp, z, coherence = map(
                float, (expected_deg, heard_deg, timestamp, z, coherence))
        except (TypeError, ValueError, OverflowError):
            return self._reject('Invalid calibration observation; use finite numbers.')
        if not all(math.isfinite(x) for x in
                   (expected_deg, heard_deg, timestamp, z, coherence)):
            return self._reject('Invalid calibration observation; use finite numbers.')
        if z < self.min_z or coherence < self.min_coherence:
            return self._reject(
                f'Sound too weak: need z >= {self.min_z:g} and '
                f'coherence >= {self.min_coherence:g}. Try another sound.')
        weight = math.sqrt(min(z / self.min_z, 3.0) *
                           min(coherence / self.min_coherence, 3.0))
        self._samples.append(Observation(wrap_deg(expected_deg), wrap_deg(heard_deg),
                                         timestamp, z, coherence, weight))
        return self.evaluate()

    def _reject(self, message):
        return replace(self.evaluate(), accepted=False, message=message)

    def evaluate(self):
        if not self._samples:
            return CalibrationResult(None, None, 0, False,
                                     'Collect three separated source positions.')

        deltas = [s.correction_deg for s in self._samples]
        weights = [s.weight for s in self._samples]
        radius = 1.5 * self.max_residual_deg
        seeds = list(deltas)
        average = _mean_deg(deltas, weights)
        if average is not None:
            seeds.append(average)
        candidates = []
        for seed in seeds:
            center = seed
            for _ in range(8):
                indices = [i for i, d in enumerate(deltas)
                           if abs(wrap_deg(d - center)) <= radius]
                if not indices:
                    break
                fitted = _mean_deg([deltas[i] for i in indices],
                                   [weights[i] for i in indices])
                if fitted is None:
                    break
                movement = abs(wrap_deg(fitted - center))
                center = fitted
                if movement < 1e-9:
                    break
            indices = [i for i, d in enumerate(deltas)
                       if abs(wrap_deg(d - center)) <= radius]
            if not indices:
                continue
            rms = math.sqrt(sum(weights[i] * wrap_deg(deltas[i] - center) ** 2
                                for i in indices) / sum(weights[i] for i in indices))
            candidates.append((len(indices), -rms, center, indices))

        _, negative_rms, correction, indices = max(candidates, key=lambda c: c[:2])
        residual = -negative_rms
        inliers = set(indices)
        indicators = tuple(SampleIndicator(s.expected_deg, s.heard_deg,
                                          wrap_deg(s.correction_deg - correction),
                                          i in inliers)
                           for i, s in enumerate(self._samples))
        expected = [self._samples[i].expected_deg for i in indices]
        heard = [self._samples[i].heard_deg for i in indices]
        expected_span, heard_span = _span_deg(expected), _span_deg(heard)
        count, total = len(indices), len(self._samples)
        ready = False
        if total < 3:
            message = f'{total}/3 observations; move the source to another position.'
        elif count < 3 or count / total < 0.75 or residual > self.max_residual_deg:
            message = ('Observations disagree with a single yaw correction. '
                       'Check source identity, reflections, microphone order, and geometry; '
                       'restart or collect more separated sounds.')
        elif not (_three_distinct(expected, self.min_distinct_deg) and
                  _three_distinct(heard, self.min_distinct_deg) and
                  expected_span >= self.min_span_deg and
                  heard_span >= self.min_span_deg):
            message = (f'Move the source: need three distinct horizontal bearings spanning '
                       f'at least {self.min_span_deg:g} degrees, both clicked and heard.')
        else:
            ready = True
            excluded = f' Excluded {total - count} inconsistent observation(s).' if count < total else ''
            message = (f'Yaw-only correction {correction:+.1f} deg; '
                       f'RMS residual {residual:.1f} deg across {count} positions.'
                       f'{excluded} Check a new source position after applying; '
                       'this does not validate position, pitch, roll, or microphone order.')
        return CalibrationResult(correction, residual, count, ready, message,
                                 samples=indicators, expected_span_deg=expected_span,
                                 heard_span_deg=heard_span)
