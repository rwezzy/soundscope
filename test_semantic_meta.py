"""Offline tests: synthetic media and fake transports; never call Meta."""
import copy
import io
import json
import tempfile
import threading
import time
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import cv2
import numpy as np

from meta_api import MetaError
from semantic_meta import SemanticMeta, SemanticMetaError, _BoundedMeta, parse_segmentation


def audio(rate=16000, samples=1600, silent=False):
    out=io.BytesIO()
    with wave.open(out,'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(rate)
        values=np.zeros(samples,dtype='<i2') if silent else (
            np.sin(np.arange(samples)*0.3)*1000).astype('<i2')
        wav.writeframes(values.tobytes())
    return out.getvalue()


def image():
    ok, encoded=cv2.imencode('.jpg',np.zeros((100,100,3),np.uint8))
    assert ok
    return encoded.tobytes()


# Official meta-sam-parser encoder output for a 10x10 all-foreground mask.
SAM_TEXT="<0f>1<|box;x1=10;y1=10;x2=19;y2=19;w=100;h=100|><|mask;x=0;y=0;data=10,10,!!!!!'y:Htq(J|>"


class Store:
    def __init__(self):
        self.events={'a':{'id':'a','pixel':[15,15],'label':'original'},
                     'b':{'id':'b','pixel':[90,90],'label':'other'}}

    def get(self,key):
        return copy.deepcopy(self.events.get(key))

    def update(self,key,values):
        self.events[key].update(copy.deepcopy(values))


class Client:
    def __init__(self):
        self.calls=[]
        self.failure=None
        self.reply={'query':'ringing','action':'search'}

    def chat_json(self,*args,**kwargs):
        self.calls.append(('chat',args,kwargs))
        if self.failure: raise self.failure
        return copy.deepcopy(self.reply)

    def transcribe(self,wav):
        self.calls.append(('transcribe',wav))
        if self.failure: raise self.failure
        return 'find the ringing phone'

    def segment_text(self,payload):
        self.calls.append(('segment',payload))
        if self.failure: raise self.failure
        return SAM_TEXT


class SemanticMetaTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store=Store()
        self.client=Client()
        self.snapshot=image()
        self.wav=audio()
        # Rendering is irrelevant to network budgets, so avoid plotting cost here.
        self.patcher=patch('semantic_meta.spectrogram_jpeg',return_value='data:image/jpeg;base64,ZmFrZQ==')
        self.patcher.start(); self.addCleanup(self.patcher.stop)

    def adapter(self,enabled=True,max_calls=5,client=None):
        result=SemanticMeta(self.store,self.temp.name,enabled,max_calls,client=client or self.client)
        self.addCleanup(result.close)
        return result

    def test_off_by_default_even_when_key_is_present(self):
        with patch.dict('os.environ',{'MODEL_API_KEY':'test-key'}), patch('semantic_meta._BoundedMeta') as ctor:
            meta=SemanticMeta(self.store,self.temp.name)
            self.addCleanup(meta.close)
            self.assertFalse(meta.status()['enabled'])
            with self.assertRaisesRegex(SemanticMetaError,'disabled'):
                meta.interpret_query('find something')
            ctor.assert_not_called()
            self.assertEqual(meta.status()['daily_used'],0)

    def test_missing_key_does_not_reserve_a_request(self):
        meta=self.adapter(); meta._client=None
        with patch('semantic_meta._BoundedMeta',side_effect=MetaError('missing')):
            with self.assertRaisesRegex(SemanticMetaError,'MODEL_API_KEY'):
                meta.interpret_query('find')
        self.assertEqual(meta.used,0)

    def test_failed_requests_count_and_do_not_retry_or_cache(self):
        meta=self.adapter(max_calls=2)
        self.client.failure=MetaError('test failure')
        for _ in range(2):
            with self.assertRaisesRegex(SemanticMetaError,'test failure'):
                meta.interpret_query('same query')
        with self.assertRaisesRegex(SemanticMetaError,'session'):
            meta.interpret_query('same query')
        self.assertEqual(len(self.client.calls),2)
        self.assertEqual(meta.status()['daily_used'],2)

    def test_invalid_response_spends_one_request(self):
        meta=self.adapter()
        self.client.reply={'action':'move_robot','query':'go'}
        with self.assertRaises(SemanticMetaError): meta.interpret_query('go')
        self.assertEqual(meta.used,1)
        self.client.reply={'action':'search','query':'go'}
        self.assertEqual(meta.interpret_query('go')['action'],'search')
        self.assertEqual(meta.used,2)

    def test_text_token_cap_and_no_automatic_robot_action(self):
        meta=self.adapter()
        self.client.reply={'action':'follow','query':'ringing phone'}
        self.assertEqual(meta.interpret_query('follow my phone')['action'],'follow')
        call=self.client.calls[0]
        self.assertEqual(call[2]['max_tokens'],1200)
        self.assertEqual(call[1][0],'muse-spark-1.3')
        self.assertIn('never execute robot',call[1][4])

    def test_cache_survives_restart_and_is_free_when_disabled(self):
        meta=self.adapter(max_calls=1)
        self.assertEqual(meta.transcribe(self.wav),'find the ringing phone')
        second=self.adapter(enabled=False,max_calls=0)
        self.assertEqual(second.transcribe(self.wav),'find the ringing phone')
        self.assertEqual(len(self.client.calls),1)
        self.assertEqual(second.used,0)
        self.assertEqual(second.status()['daily_used'],1)

    def test_daily_cap_survives_restart(self):
        for batch in range(4):
            meta=self.adapter(max_calls=5)
            for index in range(5): meta.interpret_query(f'event {batch}-{index}')
        fresh=self.adapter()
        with self.assertRaisesRegex(SemanticMetaError,'daily'):
            fresh.interpret_query('new attempt')
        self.assertEqual(len(self.client.calls),20)
        self.assertEqual(fresh.used,0)

    def test_daily_reservation_is_atomic_across_instances(self):
        adapters=[self.adapter(max_calls=20,client=Client()) for _ in range(2)]
        def worker(index):
            count=0
            for j in range(20):
                try: adapters[index].interpret_query(f'parallel {index}:{j}'); count+=1
                except SemanticMetaError: pass
            return count
        with ThreadPoolExecutor(max_workers=2) as pool:
            counts=list(pool.map(worker,range(2)))
        self.assertEqual(sum(counts),20)
        self.assertEqual(adapters[0].status()['daily_used'],20)

    def test_cached_sound_applies_to_current_event_without_old_id(self):
        meta=self.adapter()
        self.client.reply={'label':'ringing','description':'regular harmonics','uncertain':False}
        meta.describe_sound(self.wav,'a')
        meta.describe_sound(self.wav,'b')
        self.assertEqual(len(self.client.calls),1)
        self.assertEqual(self.store.events['b']['label'],'ringing')
        self.assertEqual(self.store.events['b']['id'],'b')
        self.assertTrue(self.store.events['b']['label_uncertain'])
        self.assertFalse(self.store.events['b']['source_verified'])

    def test_uncertain_sound_falls_back_to_unknown(self):
        meta=self.adapter()
        self.client.reply={'label':'alarm','description':'brief pulses','uncertain':True}
        result=meta.describe_sound(self.wav,'a')
        self.assertEqual(result['label'],'unidentified sound')

    def test_invalid_or_silent_media_spends_no_requests(self):
        meta=self.adapter()
        for wav in (audio(rate=8000), b'broken', audio(silent=True)):
            with self.assertRaises(SemanticMetaError): meta.describe_sound(wav,'a')
        with self.assertRaises(SemanticMetaError): meta.describe_context(b'bad','a')
        with self.assertRaises(SemanticMetaError): meta.transcribe(audio(samples=160001))
        with self.assertRaises(SemanticMetaError): meta.describe_sound(self.wav,'missing')
        self.assertEqual(meta.used,0)
        self.assertEqual(self.client.calls,[])

    def test_context_uses_cached_image_but_current_candidate(self):
        meta=self.adapter()
        self.client.reply={'objects':[{'label':'box','box':[0.1,0.1,0.2,0.2]}]}
        first=meta.describe_context(self.snapshot,'a')
        second=meta.describe_context(self.snapshot,'b')
        self.assertEqual(first['nearby'],['box'])
        self.assertEqual(second['nearby'],[])
        self.assertNotIn('label',first)
        self.assertEqual(self.store.events['a']['label'],'original')
        self.assertEqual(len(self.client.calls),1)

    def test_context_rejects_invalid_geometry(self):
        meta=self.adapter()
        self.client.reply={'objects':[{'label':'box','box':[0,0,float('nan'),1]}]}
        with self.assertRaises(SemanticMetaError): meta.describe_context(self.snapshot,'a')
        self.assertEqual(self.store.events['a']['label'],'original')

    def test_sound_match_is_budgeted_and_uncertain_cannot_match(self):
        meta=self.adapter()
        self.client.reply={'match':True,'heard':'maybe bell','uncertain':True,'confidence':0.9}
        result=meta.match_sound(self.wav,'bell')
        self.assertFalse(result['match']); self.assertFalse(result['source_verified'])
        meta.match_sound(self.wav,'bell')
        self.assertEqual(meta.used,1)
        meta.match_sound(self.wav,'keys')
        self.assertEqual(meta.used,2)
        self.assertFalse(meta.match_sound(audio(silent=True),'keys')['match'])
        self.assertEqual(meta.used,2)

    def test_manual_sam_uses_original_pixels_and_only_marks_snapshot(self):
        meta=self.adapter()
        result=meta.segment_at(self.snapshot,'a',[15,15])
        payload=self.client.calls[0][1]
        self.assertEqual(payload['model'],'sam-3.1')
        self.assertEqual(payload['input'][0]['content'][0]['text'],'<0f>1<|point;x=15;y=15;w=100;h=100|>')
        self.assertEqual(payload['metadata']['mask_encoding'],'one_bit')
        self.assertEqual(result['segmentation']['objects'][0]['box_xyxy'],[10,10,20,20])
        self.assertTrue(result['segmentation']['snapshot_only'])
        self.assertFalse(result['source_verified'])
        self.assertGreaterEqual(len(result['segmentation']['objects'][0]['polygons'][0]),3)
        meta.segment_at(self.snapshot,'b',[15,15])
        self.assertEqual(meta.used,1)
        self.assertIn('segmentation',self.store.events['b'])

    def test_sam_requires_decoder_and_valid_pixel_without_spending(self):
        meta=self.adapter()
        with patch('semantic_meta.importlib.util.find_spec',return_value=None):
            with self.assertRaisesRegex(SemanticMetaError,'requirements-semantic'):
                meta.segment_at(self.snapshot,'a',[15,15])
        for point in ([-0.5,10],[100,10],[True,1],[float('nan'),0]):
            with self.assertRaises(SemanticMetaError): meta.segment_at(self.snapshot,'a',point)
        self.assertEqual(meta.used,0)

    def test_sam_discards_mask_not_containing_acoustic_point(self):
        result=parse_segmentation(SAM_TEXT,100,100,(80,80))
        self.assertEqual(result['segmentation']['objects'],[])

    def test_sam_rejects_malformed_oversized_and_mismatched_masks(self):
        for bad in (SAM_TEXT.replace('w=100','w=200'),
                    SAM_TEXT.replace('data=10,10','data=1000000,1000000'),
                    SAM_TEXT.replace("!!!!!'y:Htq(J",'garbage'),
                    SAM_TEXT.replace('x2=19','x2=119')):
            with self.assertRaises((SemanticMetaError,ValueError)):
                parse_segmentation(bad,100,100,(15,15))

    def test_ledger_contains_no_raw_audio_or_image(self):
        meta=self.adapter(); meta.transcribe(self.wav)
        with open(self.temp.name+'/meta_budget.sqlite3','rb') as fp: data=fp.read()
        self.assertNotIn(self.wav,data)
        self.assertNotIn(self.snapshot,data)
        self.assertNotIn(b'MODEL_API_KEY',data)

    def test_studio_status_and_frame_updates_continue_during_slow_request(self):
        from studio import Studio
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        class SlowClient(Client):
            def chat_json(self,*args,**kwargs):
                started.set()
                if not release.wait(5): raise MetaError('offline test timed out')
                return super().chat_json(*args,**kwargs)
        instance=Studio(self.temp.name+'/studio',enable_meta=True)
        instance.meta._client=SlowClient()
        status_values=[]
        errors=[]
        def poll_and_publish():
            try:
                status_values.append(instance.status())
                instance.publish_frame(np.zeros((100,100,3),np.uint8))
            except Exception as exc: errors.append(exc)
            finally: finished.set()
        poll=None
        try:
            instance.submit('interpret',text='ringing')
            self.assertTrue(started.wait(1),'Fake network call never started')
            poll=threading.Thread(target=poll_and_publish)
            before=time.monotonic(); poll.start()
            self.assertTrue(finished.wait(.5),'Status/frame publication blocked behind network I/O')
            self.assertLess(time.monotonic()-before,.5)
            self.assertEqual(errors,[])
            self.assertEqual(status_values[0]['meta']['calls_used'],1)
            self.assertEqual(status_values[0]['meta']['daily_used'],1)
            self.assertEqual(status_values[0]['jobs'][0]['state'],'running')
            self.assertIsNotNone(instance.jpeg)
            self.assertFalse(release.is_set(),'Fake request finished before responsiveness was tested')
        finally:
            release.set()
            if poll: poll.join(2)
            instance.close()


class TransportTests(unittest.TestCase):
    def client(self):
        with patch.dict('os.environ',{'MODEL_API_KEY':'offline-placeholder'}):
            return _BoundedMeta()

    def test_sam_stream_requires_completion(self):
        client=self.client()
        client._open=Mock(return_value=io.BytesIO(b'data: {"type":"response.output_text.delta","delta":"mask"}\n'))
        with self.assertRaisesRegex(MetaError,'completed'): client.segment_text({})
        client._open.assert_called_once()

    def test_sam_stream_parses_deltas_with_completed_response(self):
        client=self.client()
        stream=b'data: {"type":"response.output_text.delta","delta":"mask"}\n\ndata: {"type":"response.completed"}\n'
        client._open=Mock(return_value=io.BytesIO(stream))
        self.assertEqual(client.segment_text({}),'mask')
        client._open.assert_called_once()

    def test_failure_error_does_not_expose_network_payloads(self):
        client=self.client()
        client._open=Mock(side_effect=OSError('private request contents'))
        with self.assertRaises(MetaError) as caught: client._post('/chat/completions',b'{}')
        self.assertNotIn('private',str(caught.exception))
        client._open.assert_called_once()


if __name__=='__main__':
    unittest.main()
