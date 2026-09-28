#!/bin/sh
# auto-generated OBS RTM shot loop
set -e
# Build irregular OBS receivers (variable z)
# Prefer: python -c with m8r, or sfmath + cat as in SConstruct_bath_rtm
nobs=5
# obs[0] x=0 z=2.4
# obs[1] x=5 z=2.5
# obs[2] x=10 z=2.6
# obs[3] x=15 z=2.7
# obs[4] x=20 z=2.8
# -> rr.rsf shape [2, nobs]


# -------- per-shot RTM (implement inside your propagator) --------
# S[:,:,:] = 0; R[:,:,:] = 0; I[:,:] = 0
# # forward
# for it in 0..nt-1:
#     inject wavelet at (ix_s, iz_s) into S
#     step_forward(S, vel)
#     save_boundary(S, it)          # or checkpoint
# # backward + image
# for it in nt-1 .. 0:
#     reconstruct_or_reload S at it
#     for iobs in 0..nobs-1:
#         inject data[it, iobs] at (ix_r[iobs], iz_r[iobs]) into R
#     step_forward(R, vel)          # reverse-time via reverse data order
#     I += S * R                    # zero-lag cross-correlation
#     # Illum += S * S
# # I /= (Illum + eps)
# # optional: mute water column I[z < bath(x)] = 0


# -------- shot 000  xs=0 zs=0.005  ix=20 iz=1 --------
# bandpass + mute (mute picks: use tomography first-arrival)
sfbandpass < shots\shot_000.rsf fhi=8 flo=3 > d:\python-learn\pyAOBS\pyAOBS\modeling\madagascar_obs_rtm\rtm_work\shot_000_bp.rsf
# TODO: rtm2d vel=vels.rsf data=d:\python-learn\pyAOBS\pyAOBS\modeling\madagascar_obs_rtm\rtm_work\shot_000_bp.rsf sou=(0,0.005) rec=rr_all.txt -> d:\python-learn\pyAOBS\pyAOBS\modeling\madagascar_obs_rtm\rtm_work\img_000.npy
# inject receivers at variable iz: 96,100,104,108,112

# -------- shot 001  xs=0.5 zs=0.005  ix=21 iz=1 --------
# bandpass + mute (mute picks: use tomography first-arrival)
sfbandpass < shots\shot_001.rsf fhi=8 flo=3 > d:\python-learn\pyAOBS\pyAOBS\modeling\madagascar_obs_rtm\rtm_work\shot_001_bp.rsf
# TODO: rtm2d vel=vels.rsf data=d:\python-learn\pyAOBS\pyAOBS\modeling\madagascar_obs_rtm\rtm_work\shot_001_bp.rsf sou=(0.5,0.005) rec=rr_all.txt -> d:\python-learn\pyAOBS\pyAOBS\modeling\madagascar_obs_rtm\rtm_work\img_001.npy
# inject receivers at variable iz: 96,100,104,108,112

# stack
# sfadd d:\python-learn\pyAOBS\pyAOBS\modeling\madagascar_obs_rtm\rtm_work/img_*.rsf > d:\python-learn\pyAOBS\pyAOBS\modeling\madagascar_obs_rtm\rtm_work/img_stack.rsf
# sflaplac < img_stack.rsf > img_lap.rsf
