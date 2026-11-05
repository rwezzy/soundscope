"""Read XVF3800 routing; optionally test reversible amplified raw-mic routing.

Default/status performs reads only. enable-preamp writes only USB CH3-6 routing;
restore returns those routes to the actual saved snapshot. Never saves firmware
configuration, changes gain/USB format, resets, flashes, detaches, or claims USB.
Close audio capture first. Restart room-noise calibration and recheck direction
after either change; a previous live calibration is not declared valid.

Protocol: https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY/blob/master/python_control/xvf_host.py
Category11: https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY/blob/master/host_control/README.md
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import struct
import sys
import time


VID, PID = 0x2886, 0x001A
SCHEMA = 'soundscope.mic-routing.v1'
ROUTES = tuple(range(3, 7))
NOTICE = ('Routing changed in device RAM only. Close and restart SoundScope, measure the room noise again, '
          'and recheck channel levels, clipping and direction before trusting localization. Re-measure the '
          'automatic Meta dBFS threshold at the new recording gain before enabling paid analysis. '
          'No calibration was marked valid.')


class RoutingError(RuntimeError):
    pass


class USBControl:
    """Small fixed-protocol adapter; constructor/import never opens USB."""
    def __init__(self, device, timeout_ms=1000):
        self.device, self.timeout_ms = device, timeout_ms

    def read(self, resource, command, fmt):
        size = struct.calcsize(fmt) + 1
        for attempt in range(3):
            packet = bytes(self.device.ctrl_transfer(0xC0, 0, command | 0x80, resource,
                                                     size, timeout=self.timeout_ms))
            if packet and packet[0] == 64 and attempt < 2:
                time.sleep(0.01)
                continue
            if len(packet) != size or packet[0] != 0:
                raise RoutingError(f'USB read {resource}/{command} returned invalid length/status.')
            return list(struct.unpack(fmt, packet[1:]))
        raise RoutingError('USB remained busy.')

    def write_route(self, channel, route):
        # Deliberately no generic write API: only this reversible mapping exists.
        if channel not in ROUTES or list(route) not in ([1, channel - 3], [11, channel - 3]):
            raise RoutingError('Only CH3-6 raw/amplified routing with preserved microphone order is allowed.')
        sent = self.device.ctrl_transfer(0x40, 0, 28 + channel - 3, 35, bytes(route), timeout=self.timeout_ms)
        if sent != 2:
            raise RoutingError(f'CH{channel} routing write was incomplete.')


def read_state(control):
    return {'firmware': control.read(48, 0, '<3B'),
            'usb_bit_depth': control.read(48, 8, '<2B'),
            'mic_gain': control.read(35, 0, '<f')[0],
            'processed': {'left': control.read(35, 15, '<2B'),
                          'right': control.read(35, 19, '<2B')},
            'raw_routes': {str(ch): control.read(35, 28 + ch - 3, '<2B') for ch in ROUTES}}


def validate_state(state, *, require_unamplified=False):
    try:
        version = state['firmware']
        if (len(version) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in version)
                or tuple(version) < (2, 1, 0)):
            raise ValueError('Firmware must be 2.1.0 or newer for this documented routing test.')
        if state['usb_bit_depth'] != [16, 16]:
            raise ValueError('This test expects the observed 16-bit USB input/output configuration.')
        if not math.isfinite(state['mic_gain']) or not math.isclose(state['mic_gain'], 90.0, abs_tol=1e-5):
            raise ValueError('This test expects the observed microphone gain 90; gain will not be changed.')
        for key in ('left', 'right'):
            route = state['processed'][key]
            if len(route) != 2 or any(type(v) is not int or not 0 <= v <= 255 for v in route):
                raise ValueError('Invalid processed-channel routing snapshot.')
        for ch in ROUTES:
            route = state['raw_routes'][str(ch)]
            allowed = ([1, ch - 3],) if require_unamplified else ([1, ch - 3], [11, ch - 3])
            if route not in allowed:
                raise ValueError(f'CH{ch} routing is not the expected ordered raw-microphone mapping.')
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RoutingError(f'Preflight refused: {exc}') from exc


def save_snapshot(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'schema': SCHEMA, 'device': {'vid': VID, 'pid': PID},
               'captured_at': datetime.now(timezone.utc).isoformat(), 'before': state}
    # Refuse overwrite: a prior rollback snapshot must never silently disappear.
    with path.open('x', encoding='utf-8') as out:
        json.dump(payload, out, indent=2, allow_nan=False)
        out.write('\n')
    return payload


def load_snapshot(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(data, dict) or data.get('schema') != SCHEMA or
            data.get('device') != {'vid': VID, 'pid': PID}):
        raise RoutingError('Not a SoundScope XVF3800 routing snapshot.')
    validate_state(data.get('before'), require_unamplified=True)
    return data['before']


def protected_state(state):
    return {key: state[key] for key in ('firmware', 'usb_bit_depth', 'mic_gain', 'processed')}


def change_routes(control, before, routes):
    """Readback every write; on any error attempt all four original routes."""
    try:
        for ch in ROUTES:
            route = routes[str(ch)]
            control.write_route(ch, route)
            if control.read(35, 28 + ch - 3, '<2B') != route:
                raise RoutingError(f'CH{ch} routing readback did not match.')
        after = read_state(control)
        if after['raw_routes'] != routes or protected_state(after) != protected_state(before):
            raise RoutingError('Final readback did not preserve routing, processed channels, gain and USB format.')
        return after
    except Exception as cause:
        errors = []
        for ch in ROUTES:
            try:
                control.write_route(ch, before['raw_routes'][str(ch)])
            except Exception as exc:
                errors.append(f'CH{ch}: {exc}')
        try:
            if read_state(control) != before:
                errors.append('rollback readback differs from the original state')
        except Exception as exc:
            errors.append(f'rollback readback failed: {exc}')
        suffix = 'All four original routes restored and verified.' if not errors else 'ROLLBACK INCOMPLETE: ' + '; '.join(errors)
        raise RoutingError(f'Routing change failed: {cause}. {suffix}') from cause


def enable_preamp(control, snapshot_path):
    before = read_state(control)
    validate_state(before, require_unamplified=True)
    save_snapshot(snapshot_path, before)  # must complete before the first device write
    routes = {str(ch): [11, ch - 3] for ch in ROUTES}
    return change_routes(control, before, routes)


def restore(control, snapshot_path):
    saved = load_snapshot(snapshot_path)
    current = read_state(control)
    validate_state(current)
    if protected_state(current) != protected_state(saved):
        raise RoutingError('Firmware, gain, USB format or processed routing changed since the snapshot; refusing restore.')
    return change_routes(control, current, saved['raw_routes'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=('status', 'enable-preamp', 'restore'), default='status')
    parser.add_argument('--snapshot', type=Path, help='New rollback JSON for enable/status; existing JSON for restore.')
    args = parser.parse_args(argv)
    if args.action != 'status' and args.snapshot is None:
        parser.error('--snapshot is required before any routing change')
    try:
        import usb.util
        if sys.platform == 'win32':
            from libusb_package import find
        else:
            from usb.core import find
        devices = list(find(find_all=True, idVendor=VID, idProduct=PID))
        if len(devices) != 1:
            raise RoutingError(f'Expected exactly one XVF3800; found {len(devices)}. No writes made.')
        device = devices[0]
        try:
            control = USBControl(device)
            if args.action == 'enable-preamp':
                state = enable_preamp(control, args.snapshot)
            elif args.action == 'restore':
                state = restore(control, args.snapshot)
            else:
                state = read_state(control)
                if args.snapshot:
                    save_snapshot(args.snapshot, state)
            print(json.dumps(state, indent=2, allow_nan=False))
            if args.action != 'status':
                print(NOTICE)
        finally:
            usb.util.dispose_resources(device)
    except (RoutingError, OSError, ValueError, ImportError) as exc:
        print(f'Routing check failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
