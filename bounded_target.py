"""Compatibility wrapper for the legacy viewer's optional target filter."""
from pathlib import Path
from meta_api import MetaError
from semantic_meta import SemanticMeta, SemanticMetaError
from sound_events import EventStore


class BoundedTarget:
    def __init__(self, max_calls=5):
        path=Path(__file__).resolve().parent/'studio_data'/'live'
        self.store=EventStore(path/'events.sqlite3')
        self.meta=SemanticMeta(self.store,path/'meta',enabled=True,max_calls=max_calls)

    def sound_matches(self,wav,target):
        try:return self.meta.match_sound(wav,target)
        except SemanticMetaError as exc:raise MetaError(str(exc)) from None

    def close(self):
        self.meta.close();self.store.close()
