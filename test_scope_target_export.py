"""Viewer export decisions: no devices, credentials, API calls or robot commands."""
import unittest
from unittest.mock import Mock

from scope import export_target


class TargetExport(unittest.TestCase):
    def test_paused_calibrating_and_voice_stop_clear_existing_target(self):
        for state in ({'paused': True}, {'calibrating': True}, {'hunt': {'mode': 'paused'}}):
            with self.subTest(state=state):
                writer = Mock()
                export_target(writer, {'point': [0, 0, 1]}, **state)
                writer.invalidate.assert_called_once()
                writer.publish.assert_not_called()

    def test_no_view_clears_existing_target(self):
        writer = Mock()
        export_target(writer, None)
        writer.invalidate.assert_called_once()
        writer.publish.assert_not_called()

    def test_target_filter_does_not_export_pending_negative_or_low_confidence_verdict(self):
        for verdict in (None, {'match': False, 'confidence': 1},
                        {'match': True, 'confidence': .2},
                        {'match': True, 'confidence': float('nan')},
                        {'match': 'true', 'confidence': .9}):
            with self.subTest(verdict=verdict):
                writer = Mock()
                export_target(writer, {'verdict': verdict}, hunt={'target': 'ring'})
                writer.publish.assert_not_called()
                writer.invalidate.assert_called_once()

    def test_freshness_and_geometry_still_checked_by_writer_after_semantic_match(self):
        writer = Mock()
        view = {'verdict': {'match': True, 'confidence': .8}, 'time': 1}
        export_target(writer, view, hunt={'target': 'ring'}, now=10)
        writer.publish.assert_called_once_with(view, now=10)

    def test_plain_candidate_passes_to_writer_and_no_writer_is_noop(self):
        writer = Mock()
        view = {'time': 7, 'event_id': 4}
        export_target(writer, view, now=8)
        writer.publish.assert_called_once_with(view, now=8)
        export_target(None, view)


if __name__ == '__main__':
    unittest.main()
