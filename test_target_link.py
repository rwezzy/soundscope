"""Offline local-file bridge tests. No camera, microphone, network or robot access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import aim
from target_link import TargetWriter, geometry_fingerprint, read_target


class TargetLinkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'candidate.json'
        self.cfg = aim.load_config(profile='carrier_v3_desk')
        self.writer = TargetWriter(self.path, self.cfg)
        self.view = {'time': 100.0, 'final': True, 'event_id': 7,
                     'quality': {'state': 'candidate'}, 'measured': True,
                     'point': np.array([0.0, 0.0, 1.0]),
                     'fix': {'direction': np.array([0.0, 0.0, 1.0]), 'z': 20.0,
                             'snr_db': 8.0, 'coherence': 0.4, 'directional_ambiguous': False,
                             'evidence': (np.zeros((100, 4)), np.ones((200,)))}}

    def publish(self, **changes):
        return self.writer.publish({**self.view, **changes}, now=100.2)

    def read(self, **kwargs):
        return read_target(self.path, self.cfg, now=100.5, **kwargs)

    def tamper(self, **changes):
        packet = json.loads(self.path.read_text())
        packet.update(changes)
        self.path.write_text(json.dumps(packet))

    def test_measured_candidate_roundtrip_and_units(self):
        result = self.publish()
        self.assertTrue(result['ready'])
        packet = self.read(not_before=99.0)
        self.assertEqual(packet['event_id'], '7')
        self.assertEqual(packet['session_id'], self.writer.session_id)
        self.assertEqual(packet['camera_point_m'], [0, 0, 1])
        np.testing.assert_allclose(packet['base_point_mm'], [620, 740, 160])
        self.assertFalse(packet['source_verified'])
        self.assertNotIn('evidence', packet['fix'])
        self.assertEqual(len(list(Path(self.temp.name).iterdir())), 1)

    def test_age_is_since_observation_not_recent_write(self):
        self.publish()
        self.tamper(published_at=101.8)
        with self.assertRaisesRegex(ValueError, 'stale'):
            read_target(self.path, self.cfg, now=102.0, max_age_s=1.5)

    def test_pre_arm_replay_rejected_and_event_identity_is_stable(self):
        self.publish()
        with self.assertRaisesRegex(ValueError, 'predates'):
            self.read(not_before=100.0)
        first, second = self.read(), self.read()
        # Repeated reads deliberately keep the identity; consumer deduplicates this pair.
        self.assertEqual((first['session_id'], first['event_id']),
                         (second['session_id'], second['event_id']))
        replacement = TargetWriter(self.path, self.cfg)
        self.assertNotEqual(replacement.session_id, first['session_id'])
        with self.assertRaisesRegex(ValueError, 'not ready'):
            self.read()

    def test_bad_updates_remove_previous_actionable_target(self):
        for changes in ({'final': False}, {'quality': 'weak'}, {'quality': {'state': 'ambiguous'}},
                        {'measured': False}, {'off_screen': 'right'}, {'event_id': None},
                        {'point': [0, 0, 0]}, {'point': [0, 0, 0.15]}, {'point': [0, 0, -1]},
                        {'point': [0, 0, float('nan')]}, {'point': [1, 2]},
                        {'time': float('inf')}, {'time': 101.0}):
            with self.subTest(changes=changes):
                self.publish()
                self.assertFalse(self.publish(**changes)['ready'])
                with self.assertRaisesRegex(ValueError, 'not ready'):
                    self.read()

    def test_bad_sound_evidence_is_never_actionable(self):
        for changes in ({'directional_ambiguous': True}, {'direction': [0, 0, 0]},
                        {'direction': [0, 0, 2]}, {'direction': [0, 0, float('nan')]},
                        {'z': float('nan')}, {'snr_db': float('inf')}, {'coherence': float('nan')}):
            with self.subTest(changes=changes):
                fix = {**self.view['fix'], **changes}
                self.assertFalse(self.publish(fix=fix)['ready'])
                with self.assertRaises(ValueError):
                    self.read()

    def test_fingerprint_changes_with_geometry_not_safety(self):
        changed = copy.deepcopy(self.cfg)
        changed['safety']['live'] = not changed['safety']['live']
        self.assertEqual(geometry_fingerprint(self.cfg), geometry_fingerprint(changed))
        changed['array']['front_yaw_deg'] += 10
        self.assertNotEqual(geometry_fingerprint(self.cfg), geometry_fingerprint(changed))
        self.publish()
        with self.assertRaisesRegex(ValueError, 'geometry differs'):
            read_target(self.path, changed, now=100.5)

    def test_writer_uses_updated_calibration_not_stale_cached_geometry(self):
        first = self.publish()['geometry_fingerprint']
        self.cfg['array']['front_yaw_deg'] += 5
        second = self.publish()['geometry_fingerprint']
        self.assertNotEqual(first, second)
        self.assertTrue(self.read()['ready'])

    def test_future_and_malformed_timestamps_are_rejected(self):
        for changes in ({'observed_at': 101}, {'published_at': 101}, {'published_at': 99},
                        {'observed_at': None}, {'observed_at': True}):
            with self.subTest(changes=changes):
                self.publish()
                self.tamper(**changes)
                with self.assertRaises(ValueError):
                    self.read()

    def test_tampered_coordinates_flags_and_units_are_rejected(self):
        for changes in ({'base_point_mm': [0, 0, 1000]}, {'camera_point_m': [0, 0, 1000]},
                        {'base_point_mm': [0, float('nan'), 0]}, {'camera_point_m': [0, 0, -1]},
                        {'final': False}, {'measured': False}, {'source_verified': True},
                        {'quality': 'ambiguous'}, {'session_id': ''}, {'event_id': 7}):
            with self.subTest(changes=changes):
                self.publish()
                self.tamper(**changes)
                with self.assertRaises(ValueError):
                    self.read()

    def test_invalidates_explicitly_and_missing_or_broken_files_fail_closed(self):
        self.publish()
        self.writer.invalidate('viewer paused')
        with self.assertRaisesRegex(ValueError, 'viewer paused'):
            self.read()
        for content in ('{', 'null', '{"schema":"wrong"}', '{"value":NaN}'):
            self.path.write_text(content)
            with self.assertRaises(ValueError):
                self.read()
        self.path.unlink()
        with self.assertRaises(ValueError):
            self.read()

    def test_reader_rejects_nonunit_or_ambiguous_evidence_even_if_file_tampered(self):
        for changes in ({'direction': [0, 0, 2]}, {'directional_ambiguous': True}):
            self.publish()
            fix = {**self.read()['fix'], **changes}
            self.tamper(fix=fix)
            with self.assertRaises(ValueError):
                self.read()

    def test_untracked_wrist_frame_is_not_published_as_world_target(self):
        self.cfg['camera']['mount'] = 'arm'
        result = self.publish()
        self.assertFalse(result['ready'])
        self.assertIn('stationary', result['reason'])


if __name__ == '__main__':
    unittest.main()
