"""Read-only Viam pose packets for Studio's --pose-file (never drives the arm).

Example, when the user is ready to connect:
  python pose_export.py --connect-read-only --output studio_data/wrist_pose.json

The camera hand-eye calibration must use the SAME end-effector frame reported
by get_end_position(), not an assumed flange or suction-tip frame. Translation
is converted from Viam millimetres to metres. Times are this host's monotonic
clock and bracket RPC delivery; they are not hardware capture timestamps.

Orientation conversion is delegated to the official SDK, not interpreted as
axis-angle. Verified API and row-major matrix convention:
https://python.viam.dev/autoapi/viam/spatialmath/index.html
https://raw.githubusercontent.com/viamrobotics/rdk/main/spatialmath/orientationVector.go
Older SDKs without spatialmath fail closed before any connection is attempted.
"""
import argparse
import asyncio
from collections import deque
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from rig_geometry import validate_transform


class UnsupportedSDK(RuntimeError):
    pass


def pose_to_transform(pose, quaternion_type):
    """Use Viam's proto bridge: theta is degrees, not pre-converted radians."""
    values = np.asarray([getattr(pose, k) for k in ('x', 'y', 'z', 'o_x', 'o_y', 'o_z', 'theta')], float)
    if not np.isfinite(values).all() or np.linalg.norm(values[3:6]) < 1e-10:
        raise ValueError('pose has invalid coordinates or an undefined orientation')
    rotation = quaternion_type.from_pose(pose).to_rotation_matrix()
    t = np.eye(4)
    t[:3, :3] = np.asarray(rotation.elements, float).reshape(3, 3)
    t[:3, 3] = values[:3] / 1000.0
    return validate_transform(t, 'Viam end-effector pose')


def sdk_converter():
    """Load and sanity-check spatialmath locally, without opening a connection."""
    try:
        from viam.proto.common import Pose
        from viam.spatialmath import Quaternion
        # A nonzero proto angle catches degrees/radians or matrix-order mistakes.
        check = pose_to_transform(Pose(x=100, o_z=1, theta=90), Quaternion)
        expected = np.array([[0, -1, 0, .1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        if not np.allclose(check, expected, atol=1e-6):
            raise ValueError('orientation bridge does not match the documented convention')
    except Exception as exc:
        raise UnsupportedSDK('Installed viam-sdk lacks a working documented spatialmath pose bridge; '
                             'no connection was attempted. Install a compatible viam-sdk before retrying.') from exc
    return lambda pose: pose_to_transform(pose, Quaternion)


def atomic_publish(path, packet):
    """Readers see a complete old or new JSON packet, never a partial write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.tmp',
                                         prefix=path.name + '.', dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump(packet, f, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class PoseSampler:
    """Only uses is_moving/get_end_position. Stability is a preview filter, not a safety sensor."""
    def __init__(self, arm, converter, *, clock=time.monotonic, max_read_s=.12,
                 settle_s=.3, max_gap_s=.3):
        if not 0 < max_read_s <= .15 or not .3 <= settle_s <= 10 or max_gap_s <= 0:
            raise ValueError('invalid read deadline, settling interval or polling gap')
        self.arm, self.converter, self.clock = arm, converter, clock
        self.max_read_s, self.settle_s, self.max_gap_s = max_read_s, settle_s, max_gap_s
        self.previous = None
        self.previous_at = None
        self.settled_since = None

    def invalidate(self, reason, *, at=None):
        self.previous = self.previous_at = self.settled_since = None
        return {'status': 'blocked', 'reason': reason, 'captured_at': self.clock() if at is None else at,
                'moving': None, 'motion_enabled': False, 'units': 'm',
                'clock': 'same-host time.monotonic', 'T_base_from_tool': None}

    async def sample(self):
        start = self.clock()

        async def read():
            before = await self.arm.is_moving()
            if before is not False:
                return before, None, None
            pose = await self.arm.get_end_position()
            after = await self.arm.is_moving()
            return before, pose, after

        try:
            before, pose, after = await asyncio.wait_for(read(), timeout=self.max_read_s)
            end = self.clock()
            if not 0 <= end - start <= self.max_read_s:
                return self.invalidate('pose read exceeded its timing budget', at=end)
            if before is not False or after is not False:
                return self.invalidate('arm moving or motion state unknown', at=end)
            t = self.converter(pose)
            t = validate_transform(t, 'T_base_from_tool')
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self.invalidate('read unavailable: ' + type(exc).__name__)

        captured = (start + end) / 2
        changed = self.previous is None
        if self.previous is not None:
            cosine = (np.trace(self.previous[:3, :3].T @ t[:3, :3]) - 1) / 2
            angle = math.acos(float(np.clip(cosine, -1, 1)))
            changed = (np.linalg.norm(t[:3, 3] - self.previous[:3, 3]) > .001 or
                       angle > math.radians(.25) or
                       not 0 <= captured - self.previous_at <= self.max_gap_s)
        if changed:
            # Start at END of the first stationary read, never before it.
            self.settled_since = end
        self.previous, self.previous_at = t, captured
        common = {'captured_at': captured, 'read_started_at': start, 'read_finished_at': end,
                  'read_duration_s': end - start, 'moving': False,
                  'settled_since': self.settled_since, 'motion_enabled': False, 'units': 'm',
                  'clock': 'same-host time.monotonic',
                  'tool_frame': 'Viam get_end_position end-effector frame',
                  'timestamp_source': 'midpoint of host RPC bracket; not hardware synchronized'}
        if captured - self.settled_since < self.settle_s:
            return dict(common, status='blocked', reason='waiting for consecutive stationary readings',
                        T_base_from_tool=None)
        return dict(common, status='ready', T_base_from_tool=t.tolist())


class PoseHistory:
    """Small same-stationary-period history for delayed completed sound events."""
    def __init__(self, max_age_s=2., max_samples=40):
        self.max_age_s = max_age_s
        self.samples = deque(maxlen=max_samples)

    def attach(self, packet):
        # A movement, read error or fresh settling period invalidates the WHOLE
        # history. A previously stationary pose must not outlive a movement gap.
        if packet.get('status') != 'ready' or packet.get('moving') is not False:
            self.samples.clear()
        else:
            captured = packet['captured_at']
            if self.samples and captured < self.samples[-1]['captured_at']:
                self.samples.clear()
            while self.samples and captured - self.samples[0]['captured_at'] > self.max_age_s:
                self.samples.popleft()
            self.samples.append(dict(packet))
        return dict(packet, samples=list(self.samples))


async def export_connected(connection, converter, output, *, interval_s=.05, max_samples=None):
    """Connection is an existing ViamArm wrapper; never calls its control methods."""
    if not 0 < interval_s <= .15:
        raise ValueError('interval must be above zero and at most 0.15 seconds')
    sampler = PoseSampler(connection.arm, converter)
    history = PoseHistory()
    count, last_status = 0, None
    try:
        while max_samples is None or count < max_samples:
            packet = await sampler.sample()
            atomic_publish(output, history.attach(packet))
            status = (packet['status'], packet.get('reason'))
            if status != last_status:
                print('Pose export:', packet['status'], packet.get('reason', '(read-only coordinate preview)'))
                last_status = status
            count += 1
            await asyncio.sleep(interval_s)
    finally:
        try:
            atomic_publish(output, history.attach(sampler.invalidate('pose exporter closed')))
        finally:
            # Close the client connection only. In particular, do not stop the arm.
            await connection.robot.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--connect-read-only', action='store_true', help='explicitly connect to the configured Viam arm')
    parser.add_argument('--output', required=True, type=Path, help='dedicated JSON file for studio.py --pose-file')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('arm_config.toml'))
    parser.add_argument('--env-file', type=Path, default=Path(__file__).with_name('.env'))
    args = parser.parse_args(argv)
    if not args.connect_read_only:
        parser.error('pass --connect-read-only to start read-only hardware polling')
    if args.output.suffix.lower() != '.json':
        parser.error('--output must be a dedicated .json pose file')
    # SDK verification happens before credentials/config or hardware connection.
    try:
        converter = sdk_converter()
    except UnsupportedSDK as exc:
        parser.exit(2, str(exc) + '\n')
    import aim
    from arm_follow import ViamArm
    from meta_api import load_env
    load_env(args.env_file)
    cfg = aim.load_config(args.config)

    async def run():
        atomic_publish(args.output, {'status': 'blocked', 'moving': None,
                                    'T_base_from_tool': None, 'reason': 'connecting read-only'})
        connection = await ViamArm.connect(cfg)
        await export_connected(connection, converter, args.output)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
