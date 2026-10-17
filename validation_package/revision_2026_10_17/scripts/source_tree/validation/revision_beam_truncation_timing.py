"""Interleaved timing repair after detecting a cold sequential first block."""
from revision_beam_truncation import DEST,axes,field,spec,projection
from revision_common import id57_data,dump,save_csv
from revision_gpu_fourier import GPUCZT2
import cupy as cp
import numpy as np

cfg,data,_=id57_data();q=data["qaxis"];rows=[];summaries=[]
for n in [384,768,1536]:
    P,_=projection("smooth_sphere",n);_,(X,Y)=axes(n)
    E=field(spec("edge","edge",fwhm_nm=200.,normal_deg=35.,cut_nm=0.,amplitude_edge_10_90_nm=0.),X,Y)
    pg=cp.asarray(P,dtype=cp.complex128);eg=cp.asarray(E);fg=cp.fft.fft2(pg);plan=GPUCZT2(n,144.,q)
    kinds=["projection_mask_czt","cached_plane_ifft_mask_czt","ifft_only"]
    def action(kind):
        if kind==kinds[0]:return plan.amplitude(pg*eg)
        if kind==kinds[1]:return plan.amplitude(cp.fft.ifft2(fg)*eg)
        return cp.fft.ifft2(fg)
    for _ in range(25):
        for kind in kinds:res=action(kind)
    cp.cuda.Stream.null.synchronize()
    for rep in range(30):
        order=kinds if rep%2==0 else kinds[::-1]
        for kind in order:
            a,b=cp.cuda.Event(),cp.cuda.Event();a.record();res=action(kind);b.record();b.synchronize()
            rows.append({"grid":n,"pair":rep,"operation":kind,"gpu_ms":float(cp.cuda.get_elapsed_time(a,b))})
    for kind in kinds:
        vals=[r["gpu_ms"] for r in rows if r["grid"]==n and r["operation"]==kind]
        summaries.append({"grid":n,"operation":kind,"median_gpu_ms":float(np.median(vals)),"p10_gpu_ms":float(np.percentile(vals,10)),"p90_gpu_ms":float(np.percentile(vals,90)),"repetitions":30})
    print(n,summaries[-3:],flush=True)
save_csv(DEST/"same_plane_tail_timings_paired_raw.csv",rows)
save_csv(DEST/"same_plane_tail_timings_paired.csv",summaries)
dump(DEST/"timing_scope.json",{"warmup":"25 interleaved cycles, allthreeoperations","measurement":"30 alternating-order pairs pergrid; GPUCUDA events; complex128; masks/dataGPUresident",
  "excluded":"projectiongeneration,3Dcachegeneration/interpolation, beamconstruction,setup,hostI/O",
  "interpretation":"Extra2DIFFT tail overhead only, not end-to-end3DCZT advantage. Fourier-space convolution is a valid alternative.",
  "initial_sequential_measurement":"Retained same_plane_tail_timings.csv; first384direct block unstable/cold and not used for claims."})
