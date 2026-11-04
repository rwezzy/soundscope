"""SoundScope Studio: local sound history, explicit Meta actions, optional rig geometry.

python studio.py --demo                  # synthetic scene, no sensors/API/robot
python studio.py --live                  # real sensors, Meta disabled
python studio.py --live --enable-meta --meta-max-calls 5
"""
import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import queue
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import uuid
import wave

import cv2
import numpy as np

from sound_events import EventStore, describe_event
from semantic_meta import SemanticMeta
from rig_geometry import RigGeometry, RobotPreview

HERE = Path(__file__).resolve().parent


def waveform_description(wav):
    """Simple measurable texture only, never infer object identity from an FFT."""
    if not wav:
        return 'unidentified sound'
    try:
        with wave.open(io.BytesIO(wav)) as w:
            rate = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(float)
        if len(x) < 512 or not np.any(x):
            return 'unidentified sound'
        x = x[:6*rate]
        frames = np.lib.stride_tricks.sliding_window_view(x, 512)[::256]
        energy = np.mean(frames**2, axis=1)
        spectrum = np.abs(np.fft.rfft(frames * np.hanning(512), axis=1))**2
        power = spectrum.mean(axis=0)
        peak_hz = np.argmax(power[1:]) * rate / 512 + rate / 512
        tonality = float(power.max()/max(power.sum(), 1e-12))
        crest = float(energy.max()/max(energy.mean(), 1e-12))
        if tonality > .2:
            return f'tonal sound, dominant frequency about {peak_hz:.0f} Hz'
        if crest > 4:
            return 'brief broadband transient'
        return 'broadband sound'
    except (wave.Error, ValueError, EOFError):
        return 'unidentified sound'


class MotionGate:
    """Conservative image-motion veto, NOT a robot motion/pose safety sensor."""
    def __init__(self, settle_s=.8):
        self.previous = None
        self.settle_s = settle_s
        self.unstable_at = time.monotonic()

    def check(self, frame, now):
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
        if self.previous is None or np.mean(cv2.absdiff(gray, self.previous)) > 6:
            self.unstable_at = now
        self.previous = gray
        return now-self.unstable_at >= self.settle_s


class Studio:
    def __init__(self, path, *, mode='fixed', demo=False, enable_meta=False, max_calls=5, rig=None, explicit_rig=False, pose_file=None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.store = EventStore(self.path/'events.sqlite3')
        self.meta = SemanticMeta(self.store, self.path/'meta', enabled=enable_meta and not demo, max_calls=max_calls)
        self.mode, self.demo = mode, demo
        self.rig, self.explicit_rig = rig, explicit_rig
        self.pose_file=Path(pose_file) if pose_file else None
        self.motion = MotionGate()
        self.stable = mode == 'fixed'
        self.media = OrderedDict()  # Last twenty event clips/snapshots, RAM only.
        self.seen = set()
        self.session = uuid.uuid4().hex[:8]
        self.lock = threading.RLock()
        self.jpeg = None
        self.frame_at = None
        self.paused = False
        self.jobs = OrderedDict()
        self.queue = queue.Queue(maxsize=3)
        self.stop = threading.Event()
        self.token = secrets.token_urlsafe(24)
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    def refresh_legacy_rig(self, cfg):
        self.rig = RigGeometry.from_legacy_config(cfg)
        self.rig.mode = self.mode
        return self.rig

    def check_frame(self, frame, now):
        self.stable = self.motion.check(frame, now) if self.mode == 'wrist' else True
        return self.stable

    def accept_event(self, ev):
        return self.mode != 'wrist' or (self.stable and ev.get('onset_at', -math.inf) >= self.motion.unstable_at + self.motion.settle_s)

    def observe(self, ev, view, frame, now, cfg, rig, frame_at=None):
        ident = f'{self.session}-{ev.get("event_id", ev["observed_at"])}'
        with self.lock:
            if ident in self.seen:
                return
            self.seen.add(ident)
            if len(self.seen) > 1000:
                self.seen = {ident}
        direction = np.asarray(rig.array_direction_to_camera(ev['fix']['direction']), float)
        point = view.get('point') if view.get('measured') and view['quality']['state'] != 'weak' else None
        timestamp = datetime.fromtimestamp(time.time()-(now-ev['observed_at']), timezone.utc).isoformat()
        event = dict(id=ident, timestamp=timestamp, mode='wrist' if self.mode == 'wrist' else 'standalone',
            label='unidentified sound', label_source='unknown', acoustic_description=waveform_description(ev.get('wav')),
            quality=view['quality']['state'], bearing_deg=float(np.degrees(np.arctan2(direction[0],direction[2]))),
            elevation_deg=float(np.degrees(np.arctan2(-direction[1],np.hypot(direction[0],direction[2])))),
            range_m=float(np.linalg.norm(point)) if point is not None else None,
            point_camera_m=np.asarray(point).tolist() if point is not None else None,
            pixel=list(view['spot']) if view.get('spot') is not None and point is not None else None,
            image_size=[frame.shape[1],frame.shape[0]], snr_db=ev['fix'].get('snr_db'),
            level_dbfs=ev.get('level_dbfs'), duration_s=ev['fix'].get('seconds'),
            nearby=[], source_verified=False, observed_at=ev['observed_at'],
            frame_offset_s=None if frame_at is None else frame_at-ev['observed_at'],
            measurement_frame_at=frame_at,
            media_sync='nearest RGB-D delivery timestamp; not hardware synchronized',
            coordinates='camera at observation time; not a tracked world location')
        self.store.add(event)
        ok,jpg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85])
        with self.lock:
            self.media[ident]={'wav':ev.get('wav'), 'jpeg':jpg.tobytes() if ok else None}
            while len(self.media)>20:
                self.media.popitem(last=False)

    def publish_frame(self, frame, paused=False):
        # Decouple the HTTP preview from sensor FPS; no unbounded image queue.
        now=time.monotonic()
        if self.frame_at is not None and now-self.frame_at<.1:
            return
        ok,jpg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,75])
        if ok:
            with self.lock:
                self.jpeg=jpg.tobytes();self.frame_at=now;self.paused=paused

    def status(self):
        with self.lock:
            result = dict(mode=self.mode, demo=self.demo, stable=self.stable, paused=self.paused,
                frame_age_s=None if self.frame_at is None else round(time.monotonic()-self.frame_at,2),
                jobs=[dict(job) for job in list(self.jobs.values())[-6:]], token=self.token,
                robot='Preview only; no robot connection or motion', media_ids=list(self.media))
        # Never hold the frame/media lock while querying another subsystem.
        result['meta'] = self.meta.status()
        return result

    def submit(self, action, event_id=None, text=None, wav=None):
        if action not in ('sound','context','segment','voice','interpret'):
            raise ValueError('Unknown action')
        if self.demo or not self.meta.status().get('enabled',False):
            raise ValueError('Meta is off. Restart with --enable-meta to allow bounded manual calls.')
        with self.lock:
            if any(j['state'] in ('queued','running') for j in self.jobs.values()):
                raise ValueError('A Meta request is already running; wait for its result.')
            if action in ('sound','context','segment') and event_id not in self.media:
                raise ValueError('The original event media expired from RAM. Select a newer event.')
            ident=uuid.uuid4().hex[:12]
            self.jobs[ident]=dict(id=ident,action=action,state='queued')
            self.queue.put_nowait((ident,action,event_id,text,wav))
            while len(self.jobs)>30:self.jobs.popitem(last=False)
        return {'job_id':ident}

    def _work(self):
        while not self.stop.is_set():
            try: ident,action,event_id,text,wav=self.queue.get(timeout=.2)
            except queue.Empty: continue
            try:
                with self.lock:
                    self.jobs[ident]['state']='running'
                    media=dict(self.media.get(event_id,{}))
                if action=='voice': result={'transcript':self.meta.transcribe(wav)}
                elif action=='interpret': result=self.meta.interpret_query(text)
                elif action=='sound': result=self.meta.describe_sound(media.get('wav'),event_id)
                elif action=='context': result=self.meta.describe_context(media.get('jpeg'),event_id)
                else:
                    event=self.store.get(event_id)
                    if not event or event.get('pixel') is None:raise ValueError('No valid acoustic/depth candidate to segment.')
                    result=self.meta.segment_at(media.get('jpeg'),event_id,event['pixel'])
                with self.lock:self.jobs[ident].update(state='done',result=result)
            except Exception as exc:
                with self.lock:self.jobs[ident].update(state='error',error=str(exc)[:400])
            finally:self.queue.task_done()

    def event_image(self, ident):
        with self.lock: jpg=self.media.get(ident,{}).get('jpeg')
        if not jpg:return None
        ev=self.store.get(ident) or {}
        image=cv2.imdecode(np.frombuffer(jpg,np.uint8),cv2.IMREAD_COLOR)
        segmentation=ev.get('segmentation',{})
        for obj in segmentation.get('objects',[]):
            for polygon in obj.get('polygons',[]):
                pts=np.asarray(polygon,dtype=np.int32).reshape(-1,1,2)
                if len(pts)>=3:cv2.polylines(image,[pts],True,(60,220,210),2)
        if ev.get('pixel') is not None:
            cv2.drawMarker(image,tuple(map(int,ev['pixel'])),(80,190,255),cv2.MARKER_CROSS,22,2)
        ok,encoded=cv2.imencode('.jpg',image)
        return encoded.tobytes() if ok else jpg

    def preview(self, ident):
        event=self.store.get(ident)
        if not event:raise ValueError('Unknown event')
        if not self.rig:return {'ok':False,'reason':'No calibrated robot transform; standalone mode remains available.'}
        # Robot coordinates refer to the RGB-D measurement, not processing time.
        mapped=dict(event)
        if mapped.get('measurement_frame_at') is not None:
            mapped['observed_at']=mapped['measurement_frame_at']
        pose=None
        if self.pose_file:
            if self.pose_file.stat().st_size>131072:raise ValueError('Pose packet too large')
            pose=json.loads(self.pose_file.read_text(encoding='utf-8'))
            if not isinstance(pose,dict):raise ValueError('Pose packet must be an object')
            if 'samples' in pose:
                # Completed audio arrives after a quiet tail. Match its original
                # frame against recent pose history, never the newest pose blindly.
                samples=pose['samples']
                if not isinstance(samples,list) or len(samples)>40:
                    raise ValueError('Pose history must contain at most forty samples')
                if pose.get('status')=='ready' and pose.get('moving') is False:
                    samples=[p for p in samples if isinstance(p,dict)
                             and type(p.get('captured_at')) in (int,float)
                             and math.isfinite(p['captured_at'])]
                    observed=mapped.get('observed_at')
                    if samples and type(observed) in (int,float) and math.isfinite(observed):
                        pose=min(samples,key=lambda p:abs(p['captured_at']-observed))
                # A blocked/moving exporter keeps its invalid current packet.
        return RobotPreview(self.rig).plan(mapped,pose_packet=pose,now=time.monotonic())

    def close(self):
        self.stop.set();self.worker.join(timeout=2)
        # Do not close SQLite underneath a bounded in-flight network worker.
        if not self.worker.is_alive():
            self.meta.close()
            self.store.close()


def serve(studio,port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,code,data,content_type='application/json'):
            body=json.dumps(data,allow_nan=False).encode() if content_type=='application/json' else data
            self.send_response(code);self.send_header('Content-Type',content_type)
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
            self.send_header('Content-Length',str(len(body)));self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
        def valid_host(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
        def do_GET(self):
            if not self.valid_host():return self.send(403,{'error':'Local access only'})
            u=urlparse(self.path);q=parse_qs(u.query)
            try:
                if u.path=='/':return self.send(200,(HERE/'studio_web/index.html').read_bytes(),'text/html; charset=utf-8')
                if u.path in ('/app.js','/style.css'):
                    return self.send(200,(HERE/'studio_web'/u.path[1:]).read_bytes(),'text/javascript' if u.path.endswith('.js') else 'text/css')
                if u.path=='/api/status':return self.send(200,studio.status())
                if u.path=='/api/events':return self.send(200,studio.store.list_events(limit=80))
                if u.path=='/api/search':return self.send(200,studio.store.search(q.get('q',[''])[0][:500],limit=30))
                if u.path=='/api/preview':return self.send(200,studio.preview(q.get('id',[''])[0]))
                if u.path=='/frame.jpg':
                    with studio.lock:jpg=studio.jpeg
                    return self.send(200,jpg,'image/jpeg') if jpg else self.send(404,{'error':'No frame yet'})
                if u.path=='/event.jpg':
                    jpg=studio.event_image(q.get('id',[''])[0])
                    return self.send(200,jpg,'image/jpeg') if jpg else self.send(404,{'error':'Snapshot expired'})
                return self.send(404,{'error':'Not found'})
            except (ValueError,TypeError,OSError) as exc:return self.send(400,{'error':str(exc)})
        def do_POST(self):
            if not self.valid_host() or self.headers.get('X-SoundScope-Token')!=studio.token:
                return self.send(403,{'error':'Local session token required'})
            origin=self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'):
                return self.send(403,{'error':'Cross-origin actions refused'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=512000:return self.send(413,{'error':'Upload too large or empty'})
                body=self.rfile.read(length)
                if self.path=='/api/voice':
                    with wave.open(io.BytesIO(body)) as w:
                        if (w.getframerate(),w.getnchannels(),w.getsampwidth())!=(16000,1,2) or w.getnframes()>16000*10:
                            raise ValueError('Voice input must be at most 10 seconds of 16kHz mono PCM16 WAV')
                    result=studio.submit('voice',wav=body)
                elif self.path=='/api/action':
                    obj=json.loads(body)
                    if not isinstance(obj,dict):raise ValueError('Expected a JSON object')
                    result=studio.submit(obj.get('action'),event_id=obj.get('id'),text=str(obj.get('text',''))[:500])
                else:return self.send(404,{'error':'Not found'})
                return self.send(202,result)
            except (ValueError,TypeError,KeyError,wave.Error,EOFError,queue.Full) as exc:return self.send(400,{'error':str(exc)})
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads=True
    threading.Thread(target=server.serve_forever,daemon=True).start()
    return server


def demo(studio,seconds=None):
    """Illustrated scene explicitly synthetic; never masquerades as an accuracy test."""
    start=time.monotonic();last=-1
    while not studio.stop.is_set() and (seconds is None or time.monotonic()-start<seconds):
        elapsed=time.monotonic()-start;i=int(elapsed//5)
        frame=np.full((540,960,3),(28,23,18),np.uint8)
        for x,y,label in [(190,285,'TIMER'),(480,285,'KEYS'),(770,285,'SPEAKER')]:
            cv2.rectangle(frame,(x-65,y-60),(x+65,y+60),(65,65,60),-1)
            cv2.putText(frame,label,(x-48,y+4),cv2.FONT_HERSHEY_SIMPLEX,.55,(220,220,210),1,cv2.LINE_AA)
        x=[190,480,770][i%3];r=int(55+(elapsed%1)*50)
        cv2.circle(frame,(x,285),r,(170,180,55),2,cv2.LINE_AA)
        cv2.drawMarker(frame,(x,285),(80,190,255),cv2.MARKER_CROSS,22,2)
        cv2.putText(frame,'SYNTHETIC DEMO | no camera, microphone or Meta calls',(30,45),cv2.FONT_HERSHEY_SIMPLEX,.65,(100,215,245),1,cv2.LINE_AA)
        cv2.putText(frame,'Acoustic candidate / visible surface, source not verified',(30,493),cv2.FONT_HERSHEY_SIMPLEX,.6,(215,210,200),1,cv2.LINE_AA)
        studio.publish_frame(frame)
        if i!=last:
            last=i;ident=f'{studio.session}-demo-{i}'
            labels=['ringing tone','keys jingling','broadband sound']
            event=dict(id=ident,timestamp=datetime.now(timezone.utc).isoformat(),mode='standalone',label=labels[i%3],label_source='synthetic demo',
                acoustic_description=['tonal sound','brief broadband transients','broadband sound'][i%3],quality='candidate',bearing_deg=[-25,0,25][i%3],
                elevation_deg=0,range_m=1.1,point_camera_m=[(x-480)/700,0,1.1],pixel=[x,285],image_size=[960,540],
                level_dbfs=-32,snr_db=14,nearby=['keyboard'],source_verified=False,duration_s=.8,observed_at=time.monotonic(),synthetic=True)
            studio.store.add(event)
            _,jpg=cv2.imencode('.jpg',frame)
            with studio.lock:
                studio.media[ident]={'jpeg':jpg.tobytes(),'wav':None}
                while len(studio.media)>20:studio.media.popitem(last=False)
        studio.stop.wait(.08)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--demo',action='store_true');source.add_argument('--live',action='store_true')
    p.add_argument('--rig-mode',choices=['fixed','wrist'],default='fixed')
    p.add_argument('--rig-config',help='Explicit calibrated rig JSON; see rigs.example.json')
    p.add_argument('--pose-file',help='Optional read-only synchronized tool-pose JSON for robot preview; never executes motion')
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--enable-meta',action='store_true')
    p.add_argument('--meta-max-calls',type=int,default=5)
    p.add_argument('--env-file');p.add_argument('--seconds',type=float)
    args=p.parse_args()
    if not 0<=args.meta_max_calls<=20:p.error('--meta-max-calls must be 0..20')
    if not 1024<=args.port<=65535:p.error('--port must be 1024..65535')
    import aim
    from meta_api import load_env
    cfg=aim.load_config()
    if args.rig_config:
        rig=RigGeometry.load(args.rig_config)
        if rig.mode!=args.rig_mode:p.error('--rig-mode must match the rig config')
    else:
        rig=RigGeometry.from_legacy_config(cfg);rig.mode=args.rig_mode
    if args.enable_meta and not args.demo:
        load_env()
        if args.env_file:load_env(args.env_file)
    path=args.data_dir or HERE/'studio_data'/('demo' if args.demo else 'live')
    studio=Studio(path,mode=args.rig_mode,demo=args.demo,enable_meta=args.enable_meta,max_calls=args.meta_max_calls,rig=rig,explicit_rig=bool(args.rig_config),pose_file=args.pose_file)
    server=serve(studio,args.port)
    print(f'SoundScope Studio: http://127.0.0.1:{args.port}',flush=True)
    print('No robot commands. Meta is '+('manual and capped.' if args.enable_meta and not args.demo else 'OFF.'),flush=True)
    try:
        if args.demo:demo(studio,args.seconds)
        else:
            from types import SimpleNamespace
            import scope
            viewer=SimpleNamespace(target=None,no_heatmap=False,diagnostics=False,record=None,seconds=args.seconds,reach=False)
            scope.run(viewer,cfg,bridge=studio,rig=rig)
    except KeyboardInterrupt:pass
    finally:
        server.shutdown();server.server_close();studio.close()

if __name__=='__main__':main()
