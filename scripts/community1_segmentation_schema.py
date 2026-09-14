#!/usr/bin/env python3
"""Synthetic source-derived PyanNet state_dict loader fixtures.

Extract actual PyanNet/SincNet classes with a minimal nn.Module base replacing
framework task infrastructure. No trained weights, PCM inference or GPU.
Forward only synthetic 60-channel features through recurrent/head components.
"""
import argparse
import ast
from functools import lru_cache
import gzip
import hashlib
import importlib.util
from itertools import pairwise
import json
from pathlib import Path
from types import SimpleNamespace

HASHES={"pyannet":"3ceebc8c00e83d96747a706212102e7e99c44732ff1fc769e241e8de47d3d6af",
        "sincnet":"3f151a2482c3f8c266b1efd9bdcab297238c52b4792a7265261f2da063e5dade",
        "receptive_field":"021df3fc249a3c5146e15f879740a0ca02ad13fad6269191c8392a1ad61cd488"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in HASHES:parser.add_argument("--"+name.replace("_","-"),required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    for name,sha in HASHES.items():
        if hashlib.sha256(Path(getattr(args,name)).read_bytes()).hexdigest()!=sha:raise SystemExit("source mismatch")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from einops import rearrange
    import asteroid_filterbanks.param_sinc_fb as fb
    import asteroid_filterbanks.enc_dec as enc
    if hashlib.sha256(Path(fb.__file__).read_bytes()).hexdigest()!="2df0d1e6f109985c00efcc60970ebceff6a9665c3e65ebae73ea4848e48d8eae":raise SystemExit("filterbank mismatch")
    if hashlib.sha256(Path(enc.__file__).read_bytes()).hexdigest()!="4b385912861a60c6ebcadc170a3ef747637952e03ab6eab303c523788cb3beee":raise SystemExit("encoder mismatch")
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_device("cpu");torch.backends.mkldnn.enabled=False
    class Model(nn.Module):
        def __init__(self,**kwargs):super().__init__()
        def save_hyperparameters(self,*names):
            import inspect
            local=inspect.currentframe().f_back.f_locals
            self.hparams=SimpleNamespace(**{name:local[name] for name in names})
        def default_activation(self):return nn.LogSoftmax(dim=-1)
    def merge_dict(default,override):return dict(default,**(override or {}))
    scope=dict(torch=torch,nn=nn,F=F,Model=Model,lru_cache=lru_cache,pairwise=pairwise,rearrange=rearrange,merge_dict=merge_dict,Encoder=enc.Encoder,ParamSincFB=fb.ParamSincFB)
    spec=importlib.util.spec_from_file_location("rf",args.receptive_field);rf=importlib.util.module_from_spec(spec);spec.loader.exec_module(rf)
    for name in ["multi_conv_num_frames","multi_conv_receptive_field_center","multi_conv_receptive_field_size"]:scope[name]=getattr(rf,name)
    for path,name in [(args.sincnet,"SincNet"),(args.pyannet,"PyanNet")]:
        cls=next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.ClassDef) and n.name==name)
        mod=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),cls],type_ignores=[]);ast.fix_missing_locations(mod);exec(compile(mod,path,"exec"),scope)
    cases=[]
    with torch.no_grad():
        for index,(mono,bidir,hidden,layers,head_layers,head_hidden,stride) in enumerate([(True,True,3,2,2,5,10),(False,True,4,2,1,6,10),(True,False,2,1,0,0,1)]):
            model=scope["PyanNet"](sincnet={"stride":stride},lstm={"hidden_size":hidden,"num_layers":layers,"bidirectional":bidir,"monolithic":mono},linear={"hidden_size":head_hidden,"num_layers":head_layers})
            model.specifications=SimpleNamespace(powerset=True,num_powerset_classes=7)
            model.build();model.eval()
            for i,(name,param) in enumerate(model.named_parameters()):
                v=torch.arange(param.numel(),dtype=torch.float32)
                if name.endswith("low_hz_"):value=30+v*187
                elif name.endswith("band_hz_"):value=65+v*4
                elif "norm" in name and name.endswith("weight"):value=.8+torch.sin(v*.3)*.1
                else:value=torch.sin(v*.17+i)*.03
                param.copy_(value.reshape(param.shape))
            tensors={name:dict(dtype="F32",shape=list(value.shape),values=value.flatten().tolist()) for name,value in model.state_dict().items()}
            x=(torch.sin(torch.arange(7*60,dtype=torch.float32)*.13)*.2).reshape(1,7,60)
            if mono:y,_=model.lstm(x)
            else:
                y=x
                for recurrent in model.lstm:y,_=recurrent(y)
            if head_layers:
                for linear in model.linear:y=F.leaky_relu(linear(y))
            y=model.activation(model.classifier(y))
            cfg=dict(SincNetStride=stride,LSTM=dict(InputSize=60,HiddenSize=hidden,NumLayers=layers,Bidirectional=bidir),Head=dict(InputSize=hidden*(2 if bidir else 1),HiddenSize=head_hidden,NumLayers=head_layers,Speakers=3,MaxActive=2),SplitLSTM=not mono)
            cases.append(dict(config=cfg,tensors=tensors,frames=7,features=x.flatten().tolist(),log_probabilities=y.flatten().tolist()))
    output=dict(schema=1,source_hashes=HASHES,torch_commit=torch.version.git_version,scope="synthetic actual class state_dict and feature-only recurrent/head path, no SincNet forward",cases=cases)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_bytes(gzip.compress((json.dumps(output,separators=(",",":"))+"\n").encode(),mtime=0))
    print("tensor counts:",[len(c["tensors"]) for c in cases])
    print("buffer keys:",[k for k in cases[0]["tensors"] if "filterbank" in k])


if __name__=="__main__":main()
