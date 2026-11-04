import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
import numpy as np
from rig_geometry import RigGeometry
from telemetry import Telemetry


class DiagnosticRetention(unittest.TestCase):
    def test_bounded_numeric_evidence_survives_close_without_media(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'diagnostics.sqlite3'
            log=Telemetry(path,limit=3)
            for i in range(7):
                event=dict(event_id=i,kind='update',observed_at=time.monotonic(),wav=b'private audio',
                           fix=dict(direction=[0,0,1],z=float('nan'),coherence=.4))
                log.add('session',event,dict(quality={'state':'ambiguous'},scores=np.ones((30,30))),RigGeometry(np.eye(4)),{})
            log.close()
            with closing(sqlite3.connect(path)) as db:rows=db.execute('SELECT data FROM samples ORDER BY id').fetchall()
            self.assertEqual(len(rows),3)
            data=[json.loads(row[0]) for row in rows]
            self.assertEqual([x['event_id'] for x in data],[4,5,6])
            self.assertIsNone(data[0]['evidence']['z'])
            self.assertNotIn('wav',data[0]);self.assertNotIn('scores',data[0])
            self.assertNotIn('frame',data[0])


if __name__=='__main__':unittest.main()
