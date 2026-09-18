#!/usr/bin/env python3
"""Synthetic connected Community-1 postprocessing oracle.

Extracts pinned filter/AHC/VBx/assignment/reconstruction methods; supplies toy
float32 embeddings widened to float64 and prepared PLDA coefficients. No neural
model or media. Stable ties are an explicit alternative, separately recorded.
"""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

from community1_reconstruct_reference import HASHES, CORE_HASHES
CLUSTER_SHA = "6031fb7c21277a7e9901ef2cdaed7d5cd69f7ef45508dc4b45e82ce0da3c8fba"
VBX_SHA = "a8c644feea4b381f9c1e7da72e0e47775c1fd482067e686801ddc16e5cac3c0e"
SKLEARN_VERSION = "1.9.0"
KMEANS_SHA = "7d9cd3c75f1c40616223fbceb23bc1e115de043b355ab90716898301756d746c"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in list(HASHES)+["clustering","vbx"]:parser.add_argument("--"+key,required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    hashes=dict(HASHES,clustering=CLUSTER_SHA,vbx=VBX_SHA)
    for key,value in hashes.items():
        if hashlib.sha256(Path(getattr(args,key)).read_bytes()).hexdigest()!=value:raise SystemExit("source mismatch "+key)
    import numpy as np
    import scipy
    import scipy.version
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    from scipy.special import logsumexp,softmax
    from sklearn.cluster import KMeans
    import sklearn
    import sklearn.cluster._kmeans as sklearn_kmeans
    if sklearn.__version__ != SKLEARN_VERSION:raise SystemExit("scikit-learn version mismatch")
    if hashlib.sha256(Path(sklearn_kmeans.__file__).read_bytes()).hexdigest()!=KMEANS_SHA:raise SystemExit("scikit-learn KMeans source mismatch")
    from einops import rearrange
    from pyannote.core import Annotation,Segment,Timeline,SlidingWindow,SlidingWindowFeature
    from pyannote.core.utils.generators import string_generator
    for key,cls in [("segment",Segment),("feature",SlidingWindowFeature),("annotation",Annotation),("timeline",Timeline)]:
        if hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()!=CORE_HASHES[key]:raise SystemExit("core mismatch")
    if scipy.version.git_revision!="e4e854eaa8f18d807cd3496028e257e36caa93cc":raise SystemExit("scipy revision mismatch")
    scope=dict(np=np,warnings=warnings,linkage=linkage,fcluster=fcluster,cdist=cdist,linear_sum_assignment=linear_sum_assignment,logsumexp=logsumexp,softmax=softmax,rearrange=rearrange,Annotation=Annotation,Segment=Segment,SlidingWindow=SlidingWindow,SlidingWindowFeature=SlidingWindowFeature,string_generator=string_generator)
    def method(path,cls,name):
        tree=ast.parse(Path(path).read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
        fn=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name);fn.decorator_list=[];return fn
    def install(fn,path):
        module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),fn],type_ignores=[]);ast.fix_missing_locations(module);exec(compile(module,path,"exec"),scope);return scope[fn.name]
    for name in ["VBx","cluster_vbx"]:
        node=next(n for n in ast.parse(Path(args.vbx).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name);install(node,args.vbx)
    aggregate=install(method(args.inference,"Inference","aggregate"),args.inference)
    trim=install(method(args.inference,"Inference","trim"),args.inference);scope["Inference"]=SimpleNamespace(aggregate=aggregate,trim=trim)
    countfn=install(method(args.mixin,"SpeakerDiarizationMixin","speaker_count"),args.mixin)
    diarfn=install(method(args.mixin,"SpeakerDiarizationMixin","to_diarization"),args.mixin)
    stable_node=method(args.mixin,"SpeakerDiarizationMixin","to_diarization")
    for node in ast.walk(stable_node):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=="argsort":node.keywords.append(ast.keyword(arg="kind",value=ast.Constant("stable")))
    stable_node.name="stable_diarization";stablefn=install(stable_node,args.mixin)
    reconstruct=install(method(args.diarization,"SpeakerDiarization","reconstruct"),args.diarization)
    filterfn=install(method(args.clustering,"BaseClustering","filter_embeddings"),args.clustering)
    assignfn=install(method(args.clustering,"BaseClustering","constrained_argmax"),args.clustering)
    clusterfn=install(method(args.clustering,"VBxClustering","__call__"),args.clustering)
    binarize=install(next(n for n in ast.parse(Path(args.signal).read_text()).body if isinstance(n,ast.ClassDef) and n.name=="Binarize"),args.signal)
    class ToyPLDA:
        phi=np.array([.8,.6,.4,.2])
        def __call__(self,x):
            y=2*x/np.linalg.norm(x,axis=1,keepdims=True)
            return 2*y/np.linalg.norm(y,axis=1,keepdims=True)
    scope["KMeans"]=KMeans
    flat=lambda x:[None if np.isnan(v) else float(v) for v in np.asarray(x).flatten()]
    turns=lambda x:[dict(Start=s.start,End=s.end,Speaker=int(label)) for s,_,label in x.itertracks(yield_label=True)]
    cases=[]
    for name,constrained,fa,fb,mode in [("clustered_constrained",True,.07,.8,0),("clustered_unconstrained",False,1.,.1,0),("clustered_forced_three",True,.07,.8,7),("clustered_forced_one",True,.07,.8,8),("overlap_stable",True,1.,.1,1),("single",True,.07,.8,2),("single_unmet_count",True,.07,.8,2),("silence",True,.07,.8,3),("rounded_silence",True,.07,.8,4),("unknown_count",True,.07,.8,5),("unknown_embedding_single",True,.07,.8,6)]:
        chunks,frames,speakers,dim=3,8,2,4
        seg=np.zeros((chunks,frames,speakers),dtype=np.float32)
        seg[:,:4,0]=1;seg[:,4:,1]=1
        emb=np.array([[[1,.125,.25,0],[-1,.125,0,.25]],[[1,.25,.125,0],[-1,.25,0,.125]],[[1,.125,0,.25],[-1,0,.125,.25]]],dtype=np.float32)
        step=1.;duration=1.;frame_step=.125
        if mode==1:seg[:,3:5,:]=1
        if mode==2:seg[:]=0;seg[0,:4,0]=1
        if mode==3:seg[:]=0;emb[:]=np.nan
        if mode==4:seg[:]=0;seg[0,1,0]=1;step=.125
        if mode==5:seg[0,0,1]=np.nan
        if mode==6:seg[:]=0;seg[0,:4,0]=1;emb[1:]=np.nan
        minimum,maximum,num=1,4,0
        if name in ("single_unmet_count","clustered_forced_three"):num=3;minimum=maximum=3
        if name=="clustered_forced_one":num=1;minimum=maximum=1
        cfg=dict(Reconstruction=dict(Chunks=chunks,Frames=frames,Speakers=speakers,Start=0.,ChunkDuration=duration,ChunkStep=step,FrameDuration=.25,FrameStep=frame_step,MaxSpeakers=maximum,TiePolicy=1),EmbeddingDimension=dim,MinSpeakers=minimum,NumSpeakers=num,AHCThreshold=.6,Fa=fa,Fb=fb,MinDurationOff=0.,Constrained=constrained)
        windows=SlidingWindow(start=0.,duration=duration,step=step);grid=SlidingWindow(start=.123,duration=.25,step=frame_step)
        count=countfn(SlidingWindowFeature(seg.copy(),windows),grid,warm_up=(0.,0.));count.data=np.minimum(count.data,maximum).astype(np.int8)
        silent=not count.data.max()
        entry=dict(name=name,config=cfg,segmentations=flat(seg),embeddings=flat(emb),counts=count.data.flatten().astype(int).tolist(),output_frames=len(count.data))
        if silent:
            entry.update(path="silence",training_chunks=[],training_speakers=[],training_rows=0,clusters=0,centroids=[],scores=[],labels=[-2]*(chunks*speakers),classes=0,full=[],exclusive=[],full_turns=[],exclusive_turns=[],constraint_satisfied=False,reference_full=[],reference_exclusive=[])
        else:
            owner=SimpleNamespace(constrained_assignment=constrained,metric="cosine",threshold=.6,Fa=fa,Fb=fb,plda=ToyPLDA())
            # Exact same source filtering on float32 input; widen only selected
            # embeddings for float64 clustering contract used by Go.
            train,ci,si=filterfn(None,emb,SlidingWindowFeature(seg,None))
            owner.filter_embeddings=lambda *a,**kw:(train.astype(np.float64),ci,si)
            owner.constrained_argmax=lambda scores:assignfn(None,scores)
            hard,scores,centers=clusterfn(owner,emb.astype(np.float64),SlidingWindowFeature(seg,None),num_clusters=num or None,min_clusters=minimum,max_clusters=maximum)
            hard[seg.sum(1)==0]=-2
            source_full=reconstruct(SimpleNamespace(to_diarization=diarfn),SlidingWindowFeature(seg.copy(),windows),hard.copy(),count).data
            ex_count=SlidingWindowFeature(np.minimum(count.data,1),count.sliding_window)
            source_ex=reconstruct(SimpleNamespace(to_diarization=diarfn),SlidingWindowFeature(seg.copy(),windows),hard.copy(),ex_count).data
            full=reconstruct(SimpleNamespace(to_diarization=stablefn),SlidingWindowFeature(seg.copy(),windows),hard.copy(),count)
            exclusive=reconstruct(SimpleNamespace(to_diarization=stablefn),SlidingWindowFeature(seg.copy(),windows),hard.copy(),ex_count)
            if exclusive.data.shape[1]<full.data.shape[1]:exclusive.data=np.pad(exclusive.data,((0,0),(0,full.data.shape[1]-exclusive.data.shape[1])))
            path="single-training-row" if len(train)==1 else "clustered"
            kmeans_labels=[]
            if len(train)>1 and num and num!=2:
                path="clustered-kmeans"
                normed=train.astype(np.float64);normed/=np.linalg.norm(normed,axis=1,keepdims=True)
                kmeans_labels=KMeans(n_clusters=num,n_init=3,random_state=42,copy_x=False).fit_predict(normed).astype(int).tolist()
            entry.update(path=path,training_rows=len(train),training_chunks=ci.tolist(),training_speakers=si.tolist(),clusters=len(centers),centroids=flat(centers),scores=flat(scores),labels=hard.flatten().astype(int).tolist(),kmeans_labels=kmeans_labels,classes=full.data.shape[1],full=full.data.flatten().astype(int).tolist(),exclusive=exclusive.data.flatten().astype(int).tolist(),full_turns=turns(binarize()(full)),exclusive_turns=turns(binarize()(exclusive)),constraint_satisfied=minimum<=len(centers)<=maximum,reference_full=flat(source_full),reference_exclusive=flat(source_ex))
        cases.append(entry)
    output=dict(schema=2,source_hashes=hashes,core_hashes=CORE_HASHES,sklearn={"version":sklearn.__version__,"kmeans_sha256":hashlib.sha256(Path(sklearn_kmeans.__file__).read_bytes()).hexdigest(),"random_state":42,"n_init":3},scope="supplied features only; source monolithic clustering including forced-count KMeans plus source/stable reconstruction; no neural inference",cases=cases)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print("postprocess cases:",[(c['name'],c['path'],c['clusters']) for c in cases])


if __name__=="__main__":main()
