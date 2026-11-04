import copy
import unittest
import numpy as np
import aim
from calibrate_rig import fit_fixed_calibration


class FixedCalibration(unittest.TestCase):
    def setUp(self):
        a=np.random.default_rng(42).uniform([-.2,-.15,.5],[.25,.2,.9],(12,3))
        self.r=aim.rot_z(31);self.t=np.array([.1,.25,.12])
        self.data={'units':'m','camera_points_m':a.tolist(),'base_points_m':(a@self.r.T+self.t).tolist(),'heldout_indices':[9,10,11]}

    def test_recovers_transform_without_authorizing_robot(self):
        result=fit_fixed_calibration(self.data)
        matrix=np.asarray(result['T_base_from_camera'])
        np.testing.assert_allclose(matrix[:3,:3],self.r,atol=1e-12)
        np.testing.assert_allclose(matrix[:3,3],self.t,atol=1e-12)
        self.assertTrue(result['numerical_checks_passed'])
        self.assertFalse(result['robot_calibrated']);self.assertFalse(result['motion_enabled'])

    def test_holdout_error_is_not_hidden_by_refitting(self):
        altered=copy.deepcopy(self.data)
        altered['base_points_m'][9][0]+=.04
        result=fit_fixed_calibration(altered)
        self.assertLess(result['fit_rmse_m'],1e-12)
        self.assertGreater(result['heldout_max_error_m'],.039)
        self.assertFalse(result['numerical_checks_passed'])

    def test_coplanar_rejected(self):
        data=copy.deepcopy(self.data)
        for p in data['camera_points_m']:p[2]=.5
        with self.assertRaisesRegex(ValueError,'coplanar'):fit_fixed_calibration(data)

    def test_units_and_independent_check_required(self):
        for patch in [{'units':'mm'},{'heldout_indices':[]},{'heldout_indices':[1,1,2]},{'heldout_indices':[1,2,12]}]:
            with self.subTest(patch=patch),self.assertRaises(ValueError):fit_fixed_calibration(dict(self.data,**patch))


if __name__=='__main__':unittest.main()
