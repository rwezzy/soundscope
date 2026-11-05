"""Display styles for an existing acoustic score field; no localization logic.

``strength`` is a same-size 2D relative-agreement field in [0, 1], produced by
the viewer's existing score normalization. Styles never move, sharpen, smooth,
or reinterpret its peaks. Caller must suppress stale evidence. Flat, missing,
or wholly invalid evidence produces no overlay.
"""
from functools import lru_cache
import math

import cv2
import numpy as np


HEATMAP_STYLES = {
    'smooth': 'Smooth gradient',
    'bands': 'Score bands',
    'contours': 'Score contours',
    'spotlight': 'Scene spotlight',
}
HEATMAP_PALETTES = {
    'neon': 'Pink / yellow',
    'amber': 'Cyan / amber',
    'turbo': 'Rainbow',
    'inferno': 'Inferno',
    'viridis': 'Viridis',
    'plasma': 'Plasma',
}
_CV_PALETTES = {
    'turbo': cv2.COLORMAP_TURBO,
    'inferno': cv2.COLORMAP_INFERNO,
    'viridis': cv2.COLORMAP_VIRIDIS,
    'plasma': cv2.COLORMAP_PLASMA,
}


@lru_cache(maxsize=8)
def palette_lut(palette='neon'):
    """Read-only 256 x 1 x 3 BGR color table, also usable for a UI legend.

    ``pink-yellow`` is an alias for ``neon``; only the canonical name is in the
    cycling list so the same palette does not appear twice.
    """
    if palette == 'pink-yellow':
        return palette_lut('neon')
    level = np.linspace(0, 1, 256, dtype=np.float32)
    if palette == 'amber':
        blend = np.clip((level - 0.25) / 0.75, 0, 1)[:, None]
        color = np.array((220, 207, 75)) * (1 - blend) + np.array((91, 187, 244)) * blend
    elif palette == 'neon':
        # BGR anchors: violet -> hot pink -> coral -> yellow.
        stops = (0.0, 0.25, 0.55, 0.80, 1.0)
        anchors = np.array(((106, 16, 54), (133, 21, 199), (173, 79, 255),
                            (59, 177, 255), (106, 255, 255)), dtype=float)
        color = np.column_stack([np.interp(level, stops, anchors[:, channel]) for channel in range(3)])
    elif palette in _CV_PALETTES:
        color = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1),
                                 _CV_PALETTES[palette]).reshape(256, 3)
    else:
        raise ValueError(f'Unknown heatmap palette: {palette}')
    table = np.rint(color).astype(np.uint8).reshape(256, 1, 3)
    table.flags.writeable = False
    return table


@lru_cache(maxsize=128)
def _blend_tables(style, palette, opacity, weak):
    level = np.linspace(0, 1, 256, dtype=np.float32)
    colors = palette_lut(palette).reshape(256, 3).astype(np.float32)
    # Keep weak evidence visible without discrete contours suggesting precision.
    if weak:
        style = 'smooth'
        opacity *= 0.35
    if style == 'bands':
        band = np.minimum((level * 5).astype(np.int32), 4)
        colors = colors[np.rint(band * (255 / 4)).astype(np.int32)]
    if style == 'spotlight':
        tint = opacity * 0.16 * level
        retention = (1 - opacity * (1 - level)) * (1 - tint)
        contribution = colors * tint[:, None]
    else:
        alpha = opacity * (level > 0) if style == 'contours' else opacity * level ** 1.2
        if style == 'bands':
            # Leave the lowest relative-score band transparent. High-scoring
            # broad regions stay broad; this is a display floor, not a new lock.
            alpha = np.where(level >= .2, opacity * level ** 1.5, 0)
        retention = 1 - alpha
        contribution = colors * alpha[:, None]
    return (np.rint(retention * 255).astype(np.uint8),
            np.rint(contribution).astype(np.uint8).reshape(256, 1, 3))


def blend_heatmap(image_bgr, strength, *, style='smooth', palette='amber', opacity=0.44, weak=False):
    """Blend into and return ``image_bgr`` without changing ``strength``.

    Smooth/bands tint measured relative scores (bands leave <0.2 transparent);
    contours trace their 0.2/0.4/
    0.6/0.8 level sets. Spotlight preserves the scene at high scores and dims
    context in proportion to the inverse field, never drawing a synthetic disk.
    Opacity is in [0, .85]. NaN/Inf pixels retain the original scene. Weak
    evidence always uses reduced smooth tint, irrespective of selected style.
    """
    if style not in HEATMAP_STYLES:
        raise ValueError(f'Unknown heatmap style: {style}')
    palette_lut(palette)  # validate even when the overlay is disabled
    try:
        opacity = float(opacity)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Heatmap opacity must be finite and between 0 and .85.') from exc
    if not math.isfinite(opacity) or not 0 <= opacity <= .85:
        raise ValueError('Heatmap opacity must be finite and between 0 and .85.')
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8
            or image_bgr.ndim != 3 or image_bgr.shape[2] != 3):
        raise ValueError('Heatmap image must be a uint8 BGR image.')
    if strength is None or opacity == 0 or image_bgr.size == 0:
        return image_bgr
    field = np.asarray(strength, dtype=np.float32)
    if field.ndim != 2 or field.shape != image_bgr.shape[:2]:
        raise ValueError('Heatmap strength must match the displayed image height and width.')
    valid = np.isfinite(field)
    if not valid.any():
        return image_bgr
    all_valid = bool(valid.all())
    floor, peak = cv2.minMaxLoc(field, mask=None if all_valid else valid.astype(np.uint8))[:2]
    if peak - floor <= 1e-6 or peak <= 0:
        return image_bgr
    if not all_valid or floor < 0 or peak > 1:
        field = np.where(valid, np.clip(field, 0, 1), 0).astype(np.float32)
        floor, peak = cv2.minMaxLoc(field)[:2]
        if peak - floor <= 1e-6 or peak <= 0:
            return image_bgr
    if style == 'contours' and not weak:
        coverage = np.zeros(field.shape, np.uint8)
        colors = np.zeros_like(image_bgr)
        width = max(1, int(round(min(field.shape) / 360)))
        for threshold in (.2, .4, .6, .8):
            mask = ((field >= threshold) & valid).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # Less than one display pixel of simplification removes raster
            # stair steps; it cannot manufacture a tighter source region.
            contours = [cv2.approxPolyDP(contour, .5, True) for contour in contours]
            color = tuple(int(c) for c in palette_lut(palette)[int(round(threshold * 255)), 0])
            cv2.drawContours(colors, contours, -1, color, width, cv2.LINE_AA)
            cv2.drawContours(coverage, contours, -1, 255, width, cv2.LINE_AA)
        # Native masks avoid expensive full-frame NumPy boolean assignment;
        # retain the exact contour geometry and every invalid/zero-score gap.
        support = ((field > 0) & valid).astype(np.uint8)
        cv2.multiply(colors, cv2.cvtColor(support, cv2.COLOR_GRAY2BGR), dst=colors)
        cv2.multiply(coverage, support, dst=coverage)
        retain = np.rint(255 - opacity * np.arange(256)).astype(np.uint8)
        retention = cv2.cvtColor(cv2.LUT(coverage, retain), cv2.COLOR_GRAY2BGR)
        contribution = cv2.convertScaleAbs(colors, alpha=opacity)
        cv2.multiply(image_bgr, retention, scale=1 / 255.0, dst=image_bgr)
        cv2.add(image_bgr, contribution, dst=image_bgr)
        return image_bgr
    level = cv2.convertScaleAbs(field, alpha=255)
    invalid_scene = image_bgr[~valid].copy() if not all_valid else None
    retention_table, heat_table = _blend_tables(style, palette, opacity, bool(weak))
    retention = cv2.cvtColor(cv2.LUT(level, retention_table), cv2.COLOR_GRAY2BGR)
    contribution = cv2.applyColorMap(level, heat_table)
    cv2.multiply(image_bgr, retention, scale=1 / 255.0, dst=image_bgr)
    cv2.add(image_bgr, contribution, dst=image_bgr)
    if invalid_scene is not None:
        image_bgr[~valid] = invalid_scene
    return image_bgr
