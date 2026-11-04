"""Bounded local numerical diagnostics. No audio, images, transcripts or uploads."""
import json
import math
import sqlite3
import time
from datetime import datetime, timezone

import numpy as np


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)): return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    return value


class Telemetry:
    def __init__(self, path, limit=5000):
        self.db=sqlite3.connect(path)
        self.limit=limit
        self.db.execute('CREATE TABLE IF NOT EXISTS samples (id INTEGER PRIMARY KEY, session TEXT, data TEXT)')
        self.pending=0

    def add(self, session, event, view, rig, cfg):
        fix=event['fix']
        direction=rig.array_direction_to_camera(fix['direction'])
        record=dict(timestamp=datetime.now(timezone.utc).isoformat(),
                    event_id=event.get('event_id'),kind=event.get('kind'),
                    age_ms=1000*(time.monotonic()-event['observed_at']),
                    direction_camera=direction,
                    quality=view.get('quality'),spot=view.get('spot'),
                    off_screen=view.get('off_screen'),measured=view.get('measured',False),
                    array_config=cfg.get('array'),T_camera_from_array=rig.T_camera_from_array,
                    evidence={k:fix.get(k) for k in ('z','coherence','snr_db','seconds','directional_support_radius_deg','directional_ambiguous')})
        self.db.execute('INSERT INTO samples(session,data) VALUES (?,?)',(session,json.dumps(clean(record),allow_nan=False)))
        self.pending+=1
        if self.pending>=10:self.flush()

    def flush(self):
        self.db.execute('DELETE FROM samples WHERE id NOT IN (SELECT id FROM samples ORDER BY id DESC LIMIT ?)',(self.limit,))
        self.db.commit();self.pending=0

    def close(self):
        self.flush();self.db.close()
