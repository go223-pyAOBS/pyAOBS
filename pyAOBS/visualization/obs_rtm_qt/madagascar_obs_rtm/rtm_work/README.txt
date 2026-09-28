OBS RTM 输出目录（本文件夹）
========================
本目录由 GUI 生成 SConstruct，并写入全部中间/成像结果：
  SConstruct_obs_rtm, wav.rsf, wfls_*.rsf, img_obs_*.rsf, img_lap.rsf …
互易：sou=OBS, rec=炮点

输入在上一级工区根（请勿删）：
  ../vel.rsf  ../bath1d.rsf  ../rr.rsf
  ../shots/ 或 ../shots_proc/
  ../shots_xz.txt  ../obs_xz.txt
说明见 ../WORKDIR_LAYOUT.txt（diag/=诊断）

手跑（在本目录，OBS 为源互易）：
  scons -f SConstruct_obs_rtm img_lap.rsf

vel=vel.rsf
shots=/mnt/d/python-learn/pyAOBS/pyAOBS/visualization/obs_rtm_qt/madagascar_obs_rtm/shots_proc
