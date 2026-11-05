"""Local evidence-browser state; no hardware, playback or model side effects."""
from collections import OrderedDict
import copy
import json

from history_ui import hit_test


class HistoryController:
    """Newest-first snapshot and an eight-event LRU of verified disk evidence.

    ``store.load(artifact)`` must return (BGR image, WAV bytes, metadata), as
    EvidenceStore does. The caller controls capture, audio playback and search.
    ``events`` stays metadata-only; replace its selected element in a render-list
    copy with ``current()`` to display the hydrated snapshot.
    """
    def __init__(self, store):
        self.store = store
        self.active = False
        self.events = []
        self.selected = 0
        self.query = ''
        self.status = ''
        self._cache = OrderedDict()

    def open(self, events, query=''):
        self.events = [copy.deepcopy(event) for event in (events or ()) if isinstance(event, dict)]
        self.selected = 0
        self.query = str(query or '')
        self.status = ''
        self.active = True
        # Opening again explicitly retries formerly missing/unfinished artifacts.
        self._cache.clear()
        return self.current()

    def select(self, index):
        self.selected = max(0, min(int(index), len(self.events) - 1)) if self.events else 0
        self.status = ''
        return self.current()

    def next(self, delta):
        return self.select(self.selected + int(delta))

    def _entry(self, record):
        artifact = record.get('evidence')
        if not isinstance(artifact, dict) or not artifact:
            return None, None, {}, ''
        key = (str(record.get('id', '')), json.dumps(artifact, sort_keys=True, default=str))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        try:
            image, audio, metadata = self.store.load(artifact)
            if not isinstance(metadata, dict):
                raise ValueError('saved evidence metadata is invalid')
            if not isinstance(audio, (bytes, bytearray)):
                raise ValueError('saved audio is invalid')
            entry = image, bytes(audio), copy.deepcopy(metadata), ''
        except Exception as exc:
            # An unreadable recording must not take down the live viewer. Cache
            # the failure as well, so a render loop cannot hammer a broken file.
            entry = None, None, {}, f'Evidence unavailable: {str(exc)[:180]}'
        self._cache[key] = entry
        while len(self._cache) > 8:
            self._cache.popitem(last=False)
        return entry

    def current(self):
        if not self.events:
            return {}
        record = copy.deepcopy(self.events[self.selected])
        image, audio, metadata, error = self._entry(record)
        if error:
            self.status = error
        if metadata:
            # Verified sidecar fields take precedence over mutable record hints.
            record['evidence'] = {**record.get('evidence', {}), **metadata}
        record['evidence_image'] = image
        record['evidence_image_size'] = metadata.get('original_image_size')
        record['evidence_audio'] = bool(audio)
        return record

    def selected_audio(self):
        if not self.events:
            return None
        _, audio, _, error = self._entry(self.events[self.selected])
        if error:
            self.status = error
        return audio

    def handle_click(self, x, y, viewport=(1280, 720)):
        if not self.active:
            return None
        action = hit_test(x, y, viewport, len(self.events), self.selected)
        if action is None:
            return None
        name, value = action
        if name == 'select':
            self.select(value)
        elif name == 'live':
            self.active = False
        elif name == 'play' and not self.selected_audio():
            if not self.status:
                self.status = 'No saved audio for this event.'
            return None
        return action
