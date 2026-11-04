"""Offline end-to-end checks: saved evidence, manual actions and local HTTP guards."""
from contextlib import ExitStack
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock
import wave
from types import SimpleNamespace

import cv2
import numpy as np

import studio
from rig_geometry import RigGeometry


def wav_fixture(level=.7):
    samples=(level*np.sin(2*np.pi*1000*np.arange(1600)/16000)*32767).astype('<i2')
    data=io.BytesIO()
    with wave.open(data,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(samples.tobytes())
    return data.getvalue()


class FakeSemantic:
    def __init__(self, store, path, *, enabled=False, max_calls=5):
        self.store,self.enabled,self.max_calls=store,enabled,max_calls
        self.calls=[]
    def status(self):return {'enabled':self.enabled,'calls_used':len(self.calls),'max_calls':self.max_calls}
    def close(self):pass
    def describe_context(self,jpeg,ident):
        self.calls.append(('context',ident,jpeg));return {'nearby':['cup']}
    def describe_sound(self,wav,ident):
        self.calls.append(('sound',ident,wav));return {'label':'tone'}
    def transcribe(self,wav):
        self.calls.append(('voice',None,wav));return 'ringing left'
    def interpret_query(self,text):
        self.calls.append(('interpret',None,text));return {'query':'ringing left','action':'follow'}
    def segment_at(self,jpeg,ident,pixel):
        self.calls.append(('segment',ident,jpeg));return {'objects':[]}


class StudioIntegration(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        folder=self.stack.enter_context(tempfile.TemporaryDirectory())
        self.path=Path(folder)
        self.net=self.stack.enter_context(mock.patch('urllib.request.urlopen',side_effect=AssertionError('No internet allowed in test')))
        self.stack.enter_context(mock.patch.object(studio,'SemanticMeta',FakeSemantic))
        self.handlers={}

    def make(self, **kw):
        instance=studio.Studio(self.path/str(len(list(self.path.iterdir()))), **kw)
        self.addCleanup(instance.close)
        return instance

    def observe(self, instance, ident=1, color=(0,0,220), **extras):
        now=time.monotonic()
        ev={'event_id':ident,'observed_at':now,'onset_at':now-.2,'wav':wav_fixture(),
            'level_dbfs':-63.75,'fix':{'direction':[.1,0,1],'snr_db':11,'seconds':.1}}
        ev.update(extras)
        view={'quality':{'state':'candidate'},'point':np.array([.1,0,1]),'measured':True,'spot':(10,10)}
        frame=np.full((60,90,3),color,np.uint8)
        rig=RigGeometry(np.eye(4))
        instance.observe(ev,view,frame,now,{},rig)
        return f'{instance.session}-{ident}',ev,frame

    def finish_job(self, instance, job):
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            value=instance.jobs[job['job_id']]
            if value['state'] in ('done','error'):return value
            time.sleep(.01)
        self.fail('Worker did not finish its synthetic job')

    def server(self, instance):
        # Exercise the real HTTP parser/handler without opening a socket. Windows
        # sandbox policy may prohibit even loopback connections during unit tests.
        port=18000+len(self.handlers)
        def capture(address,handler):
            self.assertEqual(address,('127.0.0.1',0))
            self.handlers[port]=handler
            return SimpleNamespace(server_port=port,serve_forever=lambda:None)
        with mock.patch.object(studio,'ThreadingHTTPServer',side_effect=capture):
            studio.serve(instance,0)
        return port

    def request(self,port,path='/',method='GET',body=None,headers=None):
        data=body.encode() if isinstance(body,str) else body or b''
        fields={'Host':f'127.0.0.1:{port}','Connection':'close',**(headers or {})}
        if data:fields['Content-Length']=str(len(data))
        raw=(f'{method} {path} HTTP/1.1\r\n'+''.join(f'{k}: {v}\r\n' for k,v in fields.items())+'\r\n').encode()+data
        class Socket:
            def __init__(self):self.output=io.BytesIO()
            def makefile(self,*_args):return io.BytesIO(raw)
            def sendall(self,data):self.output.write(data)
        socket=Socket()
        self.handlers[port](socket,('127.0.0.1',12345),SimpleNamespace(server_port=port))
        header,payload=socket.output.getvalue().split(b'\r\n\r\n',1)
        return int(header.splitlines()[0].split()[1]),payload

    def test_default_and_demo_never_enable_paid_actions(self):
        instance=self.make()
        ident,_,_=self.observe(instance)
        self.assertFalse(instance.status()['meta']['enabled'])
        with self.assertRaises(ValueError):instance.submit('sound',event_id=ident)
        demo=self.make(demo=True,enable_meta=True)
        self.assertFalse(demo.status()['meta']['enabled'])
        self.assertEqual(instance.meta.calls,[])
        self.net.assert_not_called()

    def test_frozen_snapshot_and_wav_stay_with_the_selected_event(self):
        instance=self.make(enable_meta=True)
        first,ev,frame=self.observe(instance,color=(0,0,220))
        frame[:]=(255,0,0)  # Camera buffer reuse must not change the saved event.
        second,_,_=self.observe(instance,ident=2,color=(0,220,0))
        instance.publish_frame(frame)
        job=self.finish_job(instance,instance.submit('context',event_id=first))
        self.assertEqual(job['state'],'done')
        kind,ident,jpeg=instance.meta.calls[-1]
        self.assertEqual((kind,ident),('context',first))
        decoded=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
        self.assertGreater(decoded[:,:,2].mean(),210)
        self.assertLess(decoded[:,:,0].mean(),5)
        self.assertNotEqual(instance.media[first]['jpeg'],instance.media[second]['jpeg'])
        self.finish_job(instance,instance.submit('sound',event_id=first))
        self.assertEqual(instance.meta.calls[-1],('sound',first,ev['wav']))

    def test_input_level_not_recomputed_from_normalized_upload_wav(self):
        instance=self.make();ident,ev,_=self.observe(instance)
        event=instance.store.get(ident)
        self.assertEqual(event['level_dbfs'],-63.75)
        self.assertIn('-63.8 dBFS (uncalibrated)',event['phrase'])
        self.assertFalse(event['source_verified'])
        self.assertAlmostEqual(event['range_m'],np.linalg.norm([.1,0,1]))
        self.assertIn('candidate visible surface',event['phrase'])

    def test_audio_only_record_preserves_clip_without_visual_or_robot_claims(self):
        instance=self.make(enable_meta=True,rig=RigGeometry(np.eye(4)))
        now=time.monotonic()
        ev={'kind':'sound','event_id':91,'observed_at':now,'wav':wav_fixture(),
            'level_dbfs':-63.75,'fix':{'direction':[.1,0,1],'snr_db':11,'seconds':.1},
            'evidence_started_at':now-.8,'evidence_ended_at':now,
            'audio_started_at':now-.8,'audio_ended_at':now,'segment_seconds':3.,'final_reason':'segment_limit'}
        # Even an accidentally supplied measured view cannot create geometry
        # when the paired RGB-D frame is missing.
        view={'quality':{'state':'candidate'},'point':np.array([.1,0,1]),'measured':True,'spot':(10,10)}
        instance.observe(ev,view,None,now,{},instance.rig,pairing_reason='No RGB-D frame was buffered.')
        ident=f'{instance.session}-91';saved=instance.store.get(ident)
        for key in ('range_m','point_camera_m','pixel','image_size','measurement_frame_at','frame_offset_s'):
            self.assertIsNone(saved[key],key)
        self.assertEqual(saved['pairing_status'],'audio_only')
        self.assertIn('No RGB-D',saved['pairing_reason'])
        self.assertEqual(saved['level_dbfs'],-63.75)
        for key in ('evidence_started_at','evidence_ended_at','audio_started_at','audio_ended_at','segment_seconds','final_reason'):
            self.assertEqual(saved[key],ev[key])
        self.assertIsNone(instance.event_image(ident))
        self.assertEqual(instance.status()['media_available'][ident],{'audio':True,'image':False})
        for action in ('context','segment'):
            with self.assertRaisesRegex(ValueError,'No matched camera snapshot'):
                instance.submit(action,event_id=ident)
        self.assertEqual(instance.meta.calls,[])
        job=self.finish_job(instance,instance.submit('sound',event_id=ident))
        self.assertEqual(job['state'],'done')
        self.assertEqual(instance.meta.calls,[('sound',ident,ev['wav'])])
        self.assertEqual(instance.preview(ident)['status'],'blocked')

    def test_finished_events_deduplicate_and_media_is_bounded(self):
        instance=self.make()
        first,ev,frame=self.observe(instance)
        view={'quality':{'state':'weak'},'measured':False}
        instance.observe(ev,view,frame,time.monotonic(),{},RigGeometry(np.eye(4)))
        self.assertEqual(len(instance.store.list_events()),1)
        for i in range(2,23):self.observe(instance,ident=i)
        self.assertEqual(len(instance.media),20)
        self.assertNotIn(first,instance.media)
        self.assertIsNotNone(instance.store.get(first))

    def test_wrist_gate_rejects_sound_starting_before_settling(self):
        instance=self.make(mode='wrist')
        frame=np.zeros((60,90,3),np.uint8)
        self.assertFalse(instance.check_frame(frame,100))
        self.assertTrue(instance.check_frame(frame,101))
        self.assertFalse(instance.accept_event({'onset_at':100.5}))
        self.assertTrue(instance.accept_event({'onset_at':100.9}))
        self.assertFalse(instance.check_frame(np.full_like(frame,255),101.1))
        self.assertFalse(instance.accept_event({'onset_at':102}))
        self.assertFalse(instance.accept_event({}))

    def test_http_host_and_action_token_guards(self):
        instance=self.make(enable_meta=True);port=self.server(instance)
        self.assertEqual(self.request(port,'/api/status',headers={'Host':'attacker.example'})[0],403)
        body=json.dumps({'action':'interpret','text':'find ringing'})
        self.assertEqual(self.request(port,'/api/action','POST',body,{'Content-Type':'application/json'})[0],403)
        headers={'Content-Type':'application/json','X-SoundScope-Token':instance.token,'Origin':'https://attacker.example'}
        self.assertEqual(self.request(port,'/api/action','POST',body,headers)[0],403)
        self.assertEqual(instance.meta.calls,[])
        headers['Origin']=f'http://127.0.0.1:{port}'
        status,payload=self.request(port,'/api/action','POST',body,headers)
        self.assertEqual(status,202)
        job=self.finish_job(instance,json.loads(payload))
        self.assertEqual(job['result']['action'],'follow')
        self.assertEqual([c[0] for c in instance.meta.calls],['interpret'])
        # A model's suggested action is data only, never a robot dispatch.
        self.assertIn('no robot connection',instance.status()['robot'])

    def test_get_queries_do_not_dispatch_meta_or_motion(self):
        instance=self.make(enable_meta=True);port=self.server(instance)
        ident,_,_=self.observe(instance)
        for path in ['/api/search?q=home','/api/search?q=follow%20ringing','/api/events',f'/api/preview?id={ident}']:
            self.assertEqual(self.request(port,path)[0],200)
        self.assertEqual(instance.meta.calls,[])
        self.assertEqual(instance.jobs,{})
        self.net.assert_not_called()

    def test_invalid_action_json_shape_returns_error(self):
        instance=self.make(enable_meta=True);port=self.server(instance)
        headers={'Content-Type':'application/json','X-SoundScope-Token':instance.token}
        for body in ('[]','null','"sound"'):
            self.assertEqual(self.request(port,'/api/action','POST',body,headers)[0],400)
        self.assertEqual(instance.meta.calls,[])

    def test_example_rig_config_is_accepted_by_cli(self):
        with mock.patch('sys.argv',['studio.py','--demo','--rig-config',str(studio.HERE/'rigs.example.json'),
                                    '--data-dir',str(self.path/'cli')]), \
             mock.patch.object(studio,'serve') as serve, mock.patch.object(studio,'demo') as demo:
            studio.main()
            self.assertEqual(demo.call_count,1)
            self.assertEqual(demo.call_args.args[0].rig.mode,'fixed')
            serve.return_value.shutdown.assert_called_once()


if __name__=='__main__':unittest.main()
