stage=apply_select_final
pipeline=raw->bp->mute->gain
bandpass=True 3-15 Hz
gain=True
mute_tp=0.15
order=bp_then_mute_then_gain
no_double_filter=1
purge_extra=1
