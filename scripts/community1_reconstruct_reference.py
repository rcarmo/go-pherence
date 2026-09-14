#!/usr/bin/env python3
"""Extracted pyannote powerset reconstruction and frame-centre turn oracles.

Synthetic CPU arrays only. No model, pyannote.audio pipeline import or media.
Tied cutoff selection also records an explicit stable alternative; it is not
mislabelled as NumPy default argsort parity.
"""
import argparse
import ast
import hashlib
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

CORE_HASHES = {"segment":"3d462a735e2a0a29a861373fbe94fd57301025c91a1b4b50c44a5f5268ef169f",
               "feature":"bbff5b817108e0fe4d1fac08c70a253341a6408878d08eaf926c09801c279c3e",
               "annotation":"2e60c2ef242d53bb2df52bfcce02a7506f62fb0457e97cd1297b2a32140fb2a6",
               "timeline":"29962a1c763a1c8c5a41a89220b1d655be18b2a2f0e7642504ce2390223d2a22"}
HASHES = {"diarization":"cbb358abedef5042fcc71bb970b12a2936a16686be4bda15691a224f172656fd",
          "mixin":"14740a7286606c208db993d29fde5703d8323205959b7e7658069f385c2a63a5",
          "inference":"c29f525c93a1a5c6bd0a9475ae4fb8c73ce9b48cf9735301302eb99886ad7d41",
          "signal":"334e6b9ca1a6e472dbc9b1562e6cdeb27940d13bd6eef8d4636a692a6ac4ab16"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in HASHES: parser.add_argument("--"+name, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for name,sha in HASHES.items():
        if hashlib.sha256(Path(getattr(args,name)).read_bytes()).hexdigest()!=sha:
            raise SystemExit("source checksum mismatch: "+name)
    import numpy as np
    import inspect
    from pyannote.core import Annotation, Segment, Timeline, SlidingWindow, SlidingWindowFeature
    for name,cls in [("segment",Segment),("feature",SlidingWindowFeature),("annotation",Annotation),("timeline",Timeline)]:
        if hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()!=CORE_HASHES[name]:
            raise SystemExit("core source checksum mismatch: "+name)
    from pyannote.core.utils.generators import string_generator
    scope = dict(np=np,warnings=warnings,Annotation=Annotation,Segment=Segment,SlidingWindow=SlidingWindow,SlidingWindowFeature=SlidingWindowFeature,string_generator=string_generator)
    def method(path,cls,name):
        tree=ast.parse(Path(path).read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
        fn=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name);fn.decorator_list=[]
        return fn
    def install(fn,path):
        module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),fn],type_ignores=[]);ast.fix_missing_locations(module);exec(compile(module,path,"exec"),scope);return scope[fn.name]
    aggregate=install(method(args.inference,"Inference","aggregate"),args.inference)
    trim=install(method(args.inference,"Inference","trim"),args.inference)
    scope["Inference"]=SimpleNamespace(aggregate=aggregate,trim=trim)
    countfn=install(method(args.mixin,"SpeakerDiarizationMixin","speaker_count"),args.mixin)
    diarfn=install(method(args.mixin,"SpeakerDiarizationMixin","to_diarization"),args.mixin)
    reconstruct=install(method(args.diarization,"SpeakerDiarization","reconstruct"),args.diarization)
    cls=next(n for n in ast.parse(Path(args.signal).read_text()).body if isinstance(n,ast.ClassDef) and n.name=="Binarize")
    binarize=install(cls,args.signal)
    owner=SimpleNamespace(to_diarization=diarfn)
    flat=lambda x:[None if np.isnan(v) else float(v) for v in np.asarray(x).flatten()]
    cases=[];turns=[]
    rng=np.random.default_rng(92771)
    # Regular binary-exact and non-aligned decimal grids, duplicate labels,
    # discarded/inactive speakers, unknown observations, gaps and cap changes.
    for i in range(18):
        chunks,frames,speakers=1+i%4,5+i%6,1+i%4
        frame_step=[.125,.1,.0625][i%3];frame_duration=2*frame_step
        chunk_duration=frames*frame_step;chunk_step=chunk_duration*[.5,1.,1.5][i%3]
        start=[0.,.125,1.3][i%3]
        seg=rng.integers(0,2,size=(chunks,frames,speakers)).astype(np.float32)
        labels=np.tile(np.arange(speakers),(chunks,1)).astype(int)
        if i%4==0: labels[:,-1]=0
        if i%4==1: labels[0,0]=-2
        if i%4==2: seg[0,:,0]=0
        if i%5==0: seg[-1,0,0]=np.nan
        if i==17: seg[:]=0
        maximum=1+i%4
        windows=SlidingWindow(start=start,duration=chunk_duration,step=chunk_step)
        grid=SlidingWindow(start=.777,duration=frame_duration,step=frame_step) # upstream ignores grid.start
        count=countfn(SlidingWindowFeature(seg.copy(),windows),grid,warm_up=(0.,0.))
        count.data=np.minimum(count.data,maximum).astype(np.int8)
        effective=labels.copy();effective[seg.sum(1)==0]=-2
        classes=max(0,int(effective.max()+1))
        clustered=np.full((chunks,frames,classes),np.nan)
        for c in range(chunks):
            for label in np.unique(effective[c]):
                if label==-2: continue
                clustered[c,:,label]=np.max(seg[c,:,effective[c]==label],axis=0)
        activation=aggregate(SlidingWindowFeature(clustered,windows),count.sliding_window,missing=0.,skip_average=True)
        classes=max(classes,int(count.data.max()))
        scores=np.pad(activation.data,((0,0),(0,classes-activation.data.shape[1])))
        # Match shared-grid crop (duration>=step ensures both identical grids
        # retain their full arrays under the reference loose extent crop).
        if classes:
            full=reconstruct(owner,SlidingWindowFeature(seg.copy(),windows),effective.copy(),count).data
            exclusive_count=SlidingWindowFeature(np.minimum(count.data,1),count.sliding_window)
            exclusive=reconstruct(owner,SlidingWindowFeature(seg.copy(),windows),effective.copy(),exclusive_count).data
            if exclusive.shape[1]<classes: exclusive=np.pad(exclusive,((0,0),(0,classes-exclusive.shape[1])))
        else:
            full=np.empty((len(scores),0));exclusive=full.copy()
        stable=np.argsort(-scores,axis=1,kind="stable")
        deterministic=np.zeros_like(scores);exclusive_deterministic=np.zeros_like(scores);ambiguous=[]
        for frame,c in enumerate(count.data[:,0]):
            c=int(c)
            if not c: continue
            deterministic[frame,stable[frame,:c]]=1;exclusive_deterministic[frame,stable[frame,0]]=1
            if (c<classes and scores[frame,stable[frame,c-1]]==scores[frame,stable[frame,c]]) or (classes>1 and scores[frame,stable[frame,0]]==scores[frame,stable[frame,1]]): ambiguous.append(frame)
        cases.append(dict(name=f"case_{i}",config=dict(Chunks=chunks,Frames=frames,Speakers=speakers,Start=start,ChunkDuration=chunk_duration,ChunkStep=chunk_step,FrameDuration=frame_duration,FrameStep=frame_step,MaxSpeakers=maximum,TiePolicy=1),segmentations=flat(seg),labels=labels.flatten().tolist(),output_frames=len(scores),classes=classes,counts=count.data[:,0].tolist(),scores=flat(scores),full=full.flatten().astype(int).tolist(),exclusive=exclusive.flatten().astype(int).tolist(),stable_full=deterministic.flatten().astype(int).tolist(),stable_exclusive=exclusive_deterministic.flatten().astype(int).tolist(),ambiguous_frames=ambiguous))
    activities=[[[1],[0],[1],[1],[0],[1]],[[0],[0],[0]],[[1],[1],[1]],[[0],[0],[1]],[[1,0],[1,1],[0,1],[0,0],[1,1]],[[0,1],[1,0],[1,0],[0,1],[0,1],[0,0]]]
    for index,activity in enumerate(activities):
        data=np.asarray(activity,dtype=float)
        for collar in [0.,.125,.126]:
            grid=SlidingWindow(start=.25,duration=.25,step=.125)
            annotation=binarize(onset=.5,offset=.5,min_duration_on=0.,min_duration_off=collar)(SlidingWindowFeature(data,grid))
            intervals=[dict(Start=s.start,End=s.end,Speaker=int(label)) for s,_,label in annotation.itertracks(yield_label=True)]
            intervals.sort(key=lambda x:(x["Start"],x["End"],x["Speaker"]))
            turns.append(dict(name=f"turn_{index}_{collar}",config=dict(Frames=len(data),Speakers=data.shape[1],Start=.25,FrameDuration=.25,FrameStep=.125,MinDurationOn=0.,MinDurationOff=collar),activity=data.flatten().astype(int).tolist(),turns=intervals))
    output=dict(schema=1,source_hashes=HASHES,core_hashes=CORE_HASHES,scope="binary/NaN zero-warmup shared-grid reconstruction and Binarize intervals; ties explicitly audited",numpy=np.__version__,cases=cases,turns=turns)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(f"wrote {len(cases)} reconstruction cases, {len(turns)} turn cases; {sum(bool(c['ambiguous_frames']) for c in cases)} ambiguous cases")


if __name__=="__main__": main()
