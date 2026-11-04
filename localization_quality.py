"""Conservative display heuristics for an acoustic surface-score map.

These labels are not probabilities or calibrated confidence intervals. A compact
candidate is only a surface consistent with the sound, not a confirmed source or
an acoustic distance estimate. No image semantics or range inference is used.
"""
import math

import cv2
import numpy as np


def assess(scores, fix, *, min_z=12.0, min_coherence=0.10,
           support_level=0.8, max_support_fraction=0.15, max_axis_fraction=0.5):
    """Return ``(quality, support_mask)`` without modifying either input.

    Absolute sound z/coherence and the raw map peak must pass before the map can
    select a candidate. Support means scores >= 80% of the *raw positive peak*;
    subtracting the median first would make broad elevated maps appear narrow.
    Support covering >15% of the image, or multiple disconnected 8-connected
    lobes, is ambiguous. A thin ridge spanning more than half either image axis
    is also ambiguous even when its area is small. These extent thresholds are
    display heuristics, not calibrated confidence intervals. The mask has the
    input shape and is empty for weak maps.
    Malformed/non-2D maps and missing/nonfinite quality are weak, never candidates.
    """
    if (not all(math.isfinite(x) and x > 0 for x in
                (min_z, min_coherence, support_level, max_support_fraction, max_axis_fraction)) or
            support_level > 1 or max_support_fraction > 1 or max_axis_fraction > 1):
        raise ValueError('Quality thresholds must be finite and positive; fractions <= 1.')
    try:
        values = np.asarray(scores, dtype=float)
    except (TypeError, ValueError, OverflowError):
        values = np.empty((0, 0), dtype=float)
    support = np.zeros(values.shape, dtype=bool)
    peak = float(values.max()) if values.size else float('nan')

    def result(state, reason):
        rows, cols = np.nonzero(support) if support.ndim == 2 else ([], [])
        return ({'state': state, 'reason': reason, 'peak_score': peak,
                 'support_fraction': float(support.mean()) if support.size else 0.0,
                 'vertical_span_fraction': float((max(rows)-min(rows)+1)/support.shape[0]) if len(rows) else 0.,
                 'horizontal_span_fraction': float((max(cols)-min(cols)+1)/support.shape[1]) if len(cols) else 0.},
                support)

    if values.ndim != 2 or not values.size:
        return result('weak', 'No valid two-dimensional acoustic map.')
    if not np.isfinite(values).all():
        return result('weak', 'Acoustic map contains nonfinite scores.')
    try:
        z, coherence = float(fix['z']), float(fix['coherence'])
    except (TypeError, ValueError, OverflowError, KeyError, IndexError):
        return result('weak', 'Sound quality evidence is missing or invalid.')
    if not math.isfinite(z) or not math.isfinite(coherence):
        return result('weak', 'Sound quality evidence is nonfinite.')
    if z < min_z:
        return result('weak', f'Sound evidence is weak (z < {min_z:g}).')
    if coherence < min_coherence:
        return result('weak', f'Sound coherence is low (< {min_coherence:g}).')
    if peak < min_coherence:
        return result('weak', 'No sufficiently positive acoustic peak in this view.')
    contrast = peak - float(np.median(values))
    if contrast <= max(1e-6, 0.05 * abs(peak)):
        return result('weak', 'Acoustic map is flat or has too little contrast.')

    support = values >= support_level * peak
    if fix.get('directional_ambiguous'):
        return result('ambiguous', fix.get('directional_reason') or
                      'Comparable acoustic directions remain outside this visible region.')
    if float(support.mean()) > max_support_fraction:
        return result('ambiguous', 'Acoustic support is broad; no isolated source region.')
    components, _ = cv2.connectedComponents(support.astype(np.uint8), connectivity=8)
    if components > 2:  # one background component plus one or more support lobes
        return result('ambiguous', 'Multiple separated regions have comparable acoustic support.')
    rows, cols = np.nonzero(support)
    if len(rows) and (rows.max()-rows.min()+1)/values.shape[0] > max_axis_fraction:
        return result('ambiguous', 'Vertical acoustic support is extended; height is unresolved.')
    if len(cols) and (cols.max()-cols.min()+1)/values.shape[1] > max_axis_fraction:
        return result('ambiguous', 'Horizontal acoustic support is extended; bearing is unresolved.')
    return result('candidate', 'Compact acoustic support; candidate surface, not a confirmed source.')
