"""Offline fixed-camera -> robot-base calibration from independently measured points.

Usage: python calibrate_rig.py measured_points.json --output calibration_report.json

Input JSON: units="m", camera_points_m=[[x,y,z],...], base_points_m=[[x,y,z],...],
heldout_indices=[6,7,8]. Collect at least six FIT points and three DIFFERENT CHECK
points, with spatial spread in width, depth and height, and matched physical
points in both frames. The camera must remain fixed relative to the robot base.
Camera coordinates: right/down/forward. Base coordinates: the robot's base frame.

This fits a rigid transform, not a scale factor, and never moves the arm or
changes configuration. It is not wrist hand-eye calibration. The held-out check
points are never used in the fit. A small residual cannot prove the physical
correspondences, robot tool-tip offset, or collision safety are correct.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from rig_geometry import validate_transform


def fit_fixed_calibration(data, *, max_check_rmse_m=.005, max_check_error_m=.01):
    if data.get('units') != 'm':
        raise ValueError('Input must explicitly specify units: m; convert robot millimetres first')
    a=np.asarray(data.get('camera_points_m'),dtype=float)
    b=np.asarray(data.get('base_points_m'),dtype=float)
    if a.ndim!=2 or a.shape[1:]!=(3,) or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('camera_points_m and base_points_m must be matching finite N x 3 arrays')
    if np.any(a[:,2]<=0) or np.max(np.abs(a))>20 or np.max(np.abs(b))>20:
        raise ValueError('Camera points need positive depth and all coordinates must be in metres within 20 m')
    held=data.get('heldout_indices')
    if not isinstance(held,list) or len(held)<3 or any(type(i) is not int or not 0<=i<len(a) for i in held) or len(set(held))!=len(held):
        raise ValueError('Provide at least three unique valid heldout_indices; these are not fitted')
    fit=np.array([i for i in range(len(a)) if i not in held],dtype=int)
    if len(fit)<6:
        raise ValueError('Need at least six fit points plus three separate check points')
    ac=a[fit]-a[fit].mean(axis=0);bc=b[fit]-b[fit].mean(axis=0)
    spreads=[]
    for cloud in [ac,bc]:
        s=np.linalg.svd(cloud,compute_uv=False)
        if s[-1]<.005 or s[-1]/s[0]<.02:
            raise ValueError('Fit points are too concentrated, collinear or coplanar; vary width, depth AND height')
        spreads.append(s.tolist())
    u,_,vt=np.linalg.svd(ac.T@bc)
    r=vt.T@u.T
    if np.linalg.det(r)<0:
        vt[-1]*=-1;r=vt.T@u.T
    t=b[fit].mean(axis=0)-r@a[fit].mean(axis=0)
    matrix=np.eye(4);matrix[:3,:3]=r;matrix[:3,3]=t
    validate_transform(matrix)
    errors=np.linalg.norm(a@r.T+t-b,axis=1)
    train_rmse=float(np.sqrt(np.mean(errors[fit]**2)))
    check_rmse=float(np.sqrt(np.mean(errors[held]**2)));check_max=float(errors[held].max())
    passed=check_rmse<=max_check_rmse_m and check_max<=max_check_error_m
    return {'kind':'fixed_camera_calibration_candidate','units':'m',
            'T_base_from_camera':matrix.tolist(),'fit_indices':fit.tolist(),'heldout_indices':held,
            'fit_rmse_m':train_rmse,'heldout_rmse_m':check_rmse,'heldout_max_error_m':check_max,
            'errors_m':errors.tolist(),'fit_singular_values_m':spreads,
            'check_thresholds_m':{'rmse':max_check_rmse_m,'maximum':max_check_error_m},
            'numerical_checks_passed':bool(passed),'robot_calibrated':False,'motion_enabled':False,
            'note':'Review physical correspondences and tool-tip calibration before adopting this candidate. No configuration or robot state changed.'}


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('measurements',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.resolve()==args.measurements.resolve():p.error('Output must differ from the measurement file')
    try:report=fit_fixed_calibration(json.loads(args.measurements.read_text(encoding='utf-8')))
    except (ValueError,TypeError,KeyError) as exc:p.error(str(exc))
    # Exclusive creation avoids replacing an existing calibration or measurement.
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(f"Fit RMS: {report['fit_rmse_m']*1000:.2f} mm; independent check RMS: {report['heldout_rmse_m']*1000:.2f} mm; maximum: {report['heldout_max_error_m']*1000:.2f} mm")
    print('Numerical checks: '+('PASS' if report['numerical_checks_passed'] else 'FAIL'))
    print('Candidate saved; robot calibration and motion remain disabled.')


if __name__=='__main__':main()
