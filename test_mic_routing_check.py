"""Offline fake-USB traces: reads by default, guarded writes and complete rollback."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

import mic_routing_check as routing


class FakeUSB:
    def __init__(self):
        self.state = {'firmware': [2, 1, 1], 'usb_bit_depth': [16, 16], 'mic_gain': 90.0,
                      'processed': {'left': [6, 0], 'right': [6, 1]},
                      'raw_routes': {str(ch): [1, ch - 3] for ch in routing.ROUTES}}
        self.trace = []
        self.writes = []
        self.fail_write = None
        self.ignore_write = None
        self.fail_all_after = None

    def ctrl_transfer(self, request_type, request, value, resource, data, timeout):
        self.trace.append((request_type, request, value, resource))
        if request_type == 0x40:
            assert request == 0 and resource == 35 and value in range(28, 32)
            channel = str(value - 28 + 3)
            self.writes.append((channel, list(data)))
            count = len(self.writes)
            if self.fail_write == count or (self.fail_all_after and count >= self.fail_all_after):
                raise OSError('simulated USB write failure')
            if count != self.ignore_write:
                self.state['raw_routes'][channel] = list(data)
            return len(data)
        assert request_type == 0xC0 and request == 0 and value & 0x80
        command = value & 0x7F
        values = {(48, 0): ('<3B', self.state['firmware']),
                  (48, 8): ('<2B', self.state['usb_bit_depth']),
                  (35, 0): ('<f', [self.state['mic_gain']]),
                  (35, 15): ('<2B', self.state['processed']['left']),
                  (35, 19): ('<2B', self.state['processed']['right'])}
        values.update({(35, 28 + ch - 3): ('<2B', self.state['raw_routes'][str(ch)]) for ch in routing.ROUTES})
        fmt, payload = values[(resource, command)]
        result = b'\0' + struct.pack(fmt, *payload)
        assert len(result) == data
        return result


class Routing(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.snapshot = Path(self.directory.name) / 'before.json'
        self.usb = FakeUSB()
        self.control = routing.USBControl(self.usb)
        self.before = copy.deepcopy(self.usb.state)

    def test_status_only_reads_and_snapshot_contains_actual_state(self):
        state = routing.read_state(self.control)
        self.assertEqual(state, self.before)
        routing.save_snapshot(self.snapshot, state)
        self.assertEqual(routing.load_snapshot(self.snapshot), self.before)
        self.assertTrue(all(row[0] == 0xC0 for row in self.usb.trace))
        self.assertEqual(self.usb.writes, [])

    def test_enable_changes_only_four_order_preserving_raw_routes_then_restores(self):
        after = routing.enable_preamp(self.control, self.snapshot)
        self.assertEqual(after['raw_routes'], {str(ch): [11, ch - 3] for ch in routing.ROUTES})
        self.assertEqual(routing.protected_state(after), routing.protected_state(self.before))
        self.assertEqual(routing.load_snapshot(self.snapshot), self.before)
        self.assertEqual(len(self.usb.writes), 4)
        restored = routing.restore(self.control, self.snapshot)
        self.assertEqual(restored, self.before)
        self.assertEqual(len(self.usb.writes), 8)
        self.assertTrue(all(resource == 35 and value in range(28, 32)
                            for kind, _, value, resource in self.usb.trace if kind == 0x40))

    def test_partial_write_failure_attempts_all_four_rollback_routes(self):
        self.usb.fail_write = 2
        with self.assertRaisesRegex(routing.RoutingError, 'All four original routes restored and verified'):
            routing.enable_preamp(self.control, self.snapshot)
        self.assertEqual(self.usb.writes[-4:], [(str(ch), [1, ch - 3]) for ch in routing.ROUTES])
        self.assertEqual(self.usb.state, self.before)
        self.assertTrue(self.snapshot.exists())

    def test_readback_failure_also_rolls_back_every_route(self):
        self.usb.ignore_write = 3
        with self.assertRaisesRegex(routing.RoutingError, 'readback did not match'):
            routing.enable_preamp(self.control, self.snapshot)
        self.assertEqual(self.usb.writes[-4:], [(str(ch), [1, ch - 3]) for ch in routing.ROUTES])
        self.assertEqual(self.usb.state, self.before)

    def test_incomplete_rollback_is_reported_honestly(self):
        self.usb.fail_all_after = 2
        with self.assertRaisesRegex(routing.RoutingError, 'ROLLBACK INCOMPLETE'):
            routing.enable_preamp(self.control, self.snapshot)
        self.assertEqual([ch for ch, _ in self.usb.writes[-4:]], ['3', '4', '5', '6'])

    def test_guarded_preconditions_fail_before_any_write(self):
        cases = ({'firmware': [2, 0, 7]}, {'usb_bit_depth': [32, 16]}, {'mic_gain': 91.0},
                 {'raw_routes': {'3': [1, 1], '4': [1, 0], '5': [1, 2], '6': [1, 3]}})
        for change in cases:
            with self.subTest(change=change):
                self.usb.state = {**copy.deepcopy(self.before), **change}
                with self.assertRaises(routing.RoutingError):
                    routing.enable_preamp(self.control, self.snapshot)
                self.assertFalse(self.snapshot.exists())
                self.assertEqual(self.usb.writes, [])

    def test_snapshot_cannot_be_overwritten_and_must_precede_device_writes(self):
        self.snapshot.write_text('keep this backup', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            routing.enable_preamp(self.control, self.snapshot)
        self.assertEqual(self.usb.writes, [])
        self.assertEqual(self.snapshot.read_text(), 'keep this backup')

    def test_restore_refuses_changed_processed_settings(self):
        routing.enable_preamp(self.control, self.snapshot)
        self.usb.writes.clear()
        self.usb.state['processed']['right'] = [7, 0]
        with self.assertRaisesRegex(routing.RoutingError, 'changed since the snapshot'):
            routing.restore(self.control, self.snapshot)
        self.assertEqual(self.usb.writes, [])

    def test_write_adapter_cannot_change_gain_firmware_or_channel_order(self):
        for channel, route in ((2, [11, 0]), (3, [11, 3]), (3, [3, 0]), (3, [90, 0])):
            with self.subTest(channel=channel, route=route), self.assertRaises(routing.RoutingError):
                self.control.write_route(channel, route)
        self.assertEqual(self.usb.writes, [])


if __name__ == '__main__':
    unittest.main()
