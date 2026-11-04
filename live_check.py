"""Bounded passive hardware check of Studio. No Meta calls, playback or robot connection.

Close other sensor viewers, then: python live_check.py --seconds 120
The actual OpenCV viewer opens; a local HTTP monitor checks its dashboard.
Reports contain counters and timings, not audio, images or transcripts.
This cannot measure localization accuracy without known source positions.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from urllib import request

import cv2
import numpy as np

import aim
import scope
from rig_geometry import RigGeometry
from studio import Studio, serve


def monitor(port, stop, samples):
    opener=request.build_opener(request.ProxyHandler({}))
    while not stop.is_set():
        started=time.monotonic()
        sample={'at':started}
        try:
            def get(path):
                with opener.open(f'http://127.0.0.1:{port}{path}',timeout=2) as response:
                    return response.read(4_000_000)
            status=json.loads(get('/api/status'))
            sample.update(frame_age_s=status['frame_age_s'],meta_enabled=status['meta']['enabled'],
                          meta_calls=status['meta']['calls_used'],retained_media=len(status['media_ids']))
            if status['frame_age_s'] is not None:
                frame=cv2.imdecode(np.frombuffer(get('/frame.jpg'),np.uint8),cv2.IMREAD_COLOR)
                sample['jpeg_valid']=frame is not None
                sample['shape']=list(frame.shape) if frame is not None else None
            sample['events']=len(json.loads(get('/api/events')))
        except Exception as exc:
            sample['error']=type(exc).__name__
        sample['http_roundtrip_ms']=(time.monotonic()-started)*1000
        samples.append(sample)
        stop.wait(2)


def summarize(samples, event_count, elapsed, error=None):
    ready=[s for s in samples if s.get('frame_age_s') is not None]
    healthy=[s for s in ready if not s.get('error') and s.get('jpeg_valid')
             and 0<=s['frame_age_s']<=2 and s.get('retained_media',999)<=20]
    no_meta=all(not s.get('meta_enabled') and s.get('meta_calls',0)==0 for s in samples)
    fraction=len(healthy)/len(ready) if ready else 0
    latencies=[s['http_roundtrip_ms'] for s in healthy]
    return dict(kind='passive hardware integration check',duration_s=elapsed,
                passed=error is None and len(ready)>=5 and fraction>=.95 and no_meta,
                runtime_error=error,samples=len(samples),ready_samples=len(ready),
                healthy_fraction=fraction,history_events=event_count,
                meta_attempts=max((s.get('meta_calls',0) for s in samples),default=0),
                meta_disabled=no_meta,
                http_p50_ms=float(np.median(latencies)) if latencies else None,
                http_p95_ms=float(np.percentile(latencies,95)) if latencies else None,
                max_frame_age_s=max((s['frame_age_s'] for s in ready),default=None),
                localization_accuracy_measured=False,
                limitations='Passive, unknown room sounds. Event count is not detection accuracy; direction, calibration, model accuracy and robot motion are not validated.')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seconds',type=float,default=120)
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    if not 15<=args.seconds<=600:parser.error('--seconds must be between 15 and 600')
    if not 1024<=args.port<=65535:parser.error('invalid port')
    root=Path(__file__).resolve().parent/'studio_data'/'hardware_checks'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    cfg=aim.load_config();rig=RigGeometry.from_legacy_config(cfg)
    app=Studio(root,rig=rig,enable_meta=False)
    server=None;worker=None;stop=threading.Event();samples=[];failure=None
    started=time.monotonic()
    try:
        server=serve(app,args.port)
        worker=threading.Thread(target=monitor,args=(args.port,stop,samples),daemon=True)
        worker.start()
        print(f'Passive check: {args.seconds:g}s; Meta OFF; no robot commands or audio playback.',flush=True)
        options=SimpleNamespace(target=None,no_heatmap=False,diagnostics=False,record=None,
                                seconds=args.seconds,reach=False)
        scope.run(options,cfg,bridge=app,rig=rig)
    except (Exception,SystemExit,KeyboardInterrupt) as exc:
        failure=f'{type(exc).__name__}: {exc}'
    finally:
        stop.set()
        if worker:worker.join(timeout=7)
        report=summarize(samples,len(app.store.list_events(limit=500)),time.monotonic()-started,failure)
        report['meta_attempts']=app.meta.status()['calls_used']
        report['meta_disabled']=not app.meta.status()['enabled']
        report['passed']=report['passed'] and report['meta_attempts']==0 and report['meta_disabled']
        (root/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (root/'timings.json').write_text(json.dumps(samples,indent=2),encoding='utf-8')
        if server:server.shutdown();server.server_close()
        app.close()
        print(json.dumps(report,indent=2),flush=True)
        print(f'Report: {root / "report.json"}',flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
