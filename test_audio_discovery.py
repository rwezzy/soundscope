"""Mocked PortAudio discovery: endpoint visibility is not a firmware diagnosis."""
import unittest
from unittest.mock import patch

import mic_test


def device(name, api, channels):
    return {'name':name, 'hostapi':api, 'max_input_channels':channels}


class AudioDiscoveryTests(unittest.TestCase):
    def discover(self, devices, *, names=None, check=None):
        names=names or ['Windows WDM-KS','Windows WASAPI','MME']
        self.apis=[{'name':name} for name in names]
        self.patchers=[patch.object(mic_test.sd,'query_hostapis',return_value=self.apis),
                       patch.object(mic_test.sd,'query_devices',return_value=devices),
                       patch.object(mic_test.sd,'check_input_settings',side_effect=check),
                       patch.object(mic_test.sd,'InputStream',side_effect=AssertionError('No hardware in test'))]
        mocks=[p.start() for p in self.patchers]
        for p in self.patchers:self.addCleanup(p.stop)
        return mocks[2]

    def test_two_channel_wasapi_with_empty_wdm_is_visibility_not_firmware(self):
        check=self.discover([device('reSpeaker XVF3800',1,2),device('Laptop microphone',1,1),
                             device('Speakers',1,0),device('reSpeaker XVF3800',2,2)])
        with self.assertRaises(SystemExit) as caught:mic_test.find_array()
        text=str(caught.exception)
        self.assertIn('No 6-channel reSpeaker input is exposed to this process',text)
        self.assertIn('Windows WDM-KS: 0 devices, 0 inputs; reSpeaker input channels: none',text)
        self.assertIn('Windows WASAPI: 3 devices, 2 inputs; reSpeaker input channels: 2',text)
        self.assertIn('Firmware version is unknown',text)
        self.assertIn('unrestricted',text)
        self.assertIn('do not flash based on channel count alone',text)
        self.assertNotIn('on the stock',text)
        check.assert_not_called()

    def test_wdm_six_channel_selected_even_if_wasapi_appears_first(self):
        check=self.discover([device('reSpeaker XVF3800',1,2),device('reSpeaker raw',0,6),
                             device('reSpeaker MME',2,6)])
        self.assertEqual(mic_test.find_array(),1)
        check.assert_called_once_with(device=1,channels=6,samplerate=mic_test.FS,dtype='int32')

    def test_failed_preferred_api_falls_back_without_changing_policy(self):
        attempted=[]
        def check(**settings):
            attempted.append(settings['device'])
            if settings['device']==1:raise mic_test.sd.PortAudioError('Device unavailable')
        self.discover([device('reSpeaker WASAPI',1,6),device('reSpeaker WDM',0,6)],check=check)
        self.assertEqual(mic_test.find_array(),0)
        self.assertEqual(attempted,[1,0])

    def test_six_channel_setting_failure_does_not_claim_two_channel_firmware(self):
        check=self.discover([device('reSpeaker raw',0,6)],
                            check=mic_test.sd.PortAudioError('Unsupported sample format'))
        with self.assertRaises(SystemExit) as caught:mic_test.find_array()
        text=str(caught.exception)
        self.assertIn('Six-channel reSpeaker inputs are visible',text)
        self.assertIn(f'6-channel {mic_test.FS} Hz int32 capture settings',text)
        self.assertIn('[0] Windows WDM-KS: Unsupported sample format',text)
        self.assertIn('Firmware version is unknown',text)
        self.assertNotIn('only offers',text)
        check.assert_called_once()

    def test_missing_array_reports_inventory_without_opening_other_microphones(self):
        check=self.discover([device('Laptop microphone',1,2),device('reSpeaker output',0,0)])
        with self.assertRaises(SystemExit) as caught:mic_test.find_array()
        text=str(caught.exception)
        self.assertIn('No reSpeaker input is visible',text)
        self.assertIn('Windows WDM-KS: 1 devices, 0 inputs; reSpeaker input channels: none',text)
        check.assert_not_called()

    def test_empty_portaudio_error_still_reports_capture_failure(self):
        self.discover([device('reSpeaker raw',0,6)],check=mic_test.sd.PortAudioError())
        with self.assertRaises(SystemExit) as caught:mic_test.find_array()
        self.assertIn('PortAudio rejected the capture settings',str(caught.exception))

    def test_unknown_host_api_remains_available_after_ranked_apis(self):
        check=self.discover([device('reSpeaker ALSA',0,6)],names=['ALSA'])
        self.assertEqual(mic_test.find_array(),0)
        check.assert_called_once()

    def test_empty_host_inventory_has_clear_diagnostic(self):
        check=self.discover([])
        with patch.object(mic_test.sd,'query_hostapis',return_value=[]):
            with self.assertRaises(SystemExit) as caught:mic_test.find_array()
        self.assertIn('No host APIs were reported',str(caught.exception))
        check.assert_not_called()


if __name__=='__main__':unittest.main()
