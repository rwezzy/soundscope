"""Report interpretation tests; do not open devices or network connections."""
import unittest
from live_check import summarize


class PassiveCheckReport(unittest.TestCase):
    def sample(self,**changes):
        return dict(frame_age_s=.1,jpeg_valid=True,retained_media=20,
                    meta_enabled=False,meta_calls=0,http_roundtrip_ms=20,**changes)

    def test_no_events_can_pass_transport_but_cannot_claim_accuracy(self):
        result=summarize([self.sample() for _ in range(10)],0,30)
        self.assertTrue(result['passed'])
        self.assertFalse(result['localization_accuracy_measured'])

    def test_frozen_frames_fail_even_with_history(self):
        sample=self.sample();sample['frame_age_s']=5
        self.assertFalse(summarize([sample]*10,25,30)['passed'])

    def test_paid_request_or_runtime_error_fails(self):
        sample=self.sample();sample['meta_calls']=1
        self.assertFalse(summarize([sample]*10,10,30)['passed'])
        self.assertFalse(summarize([self.sample()]*10,10,30,'Camera failed')['passed'])

    def test_startup_samples_are_not_mistaken_for_healthy_frames(self):
        self.assertFalse(summarize([{'frame_age_s':None}]*10,0,30)['passed'])


if __name__=='__main__':unittest.main()
