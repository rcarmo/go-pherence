#!/usr/bin/env python3
"""Reference-only prepared prompt exporter while native codec encode is pending.
Python runs once to tokenize/encode reference; native Go does iterative synthesis.
"""
import argparse
import json
from pathlib import Path
import torch
from omnivoice import OmniVoice
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',required=True,type=Path)
p.add_argument('--voice',required=True,type=Path)
p.add_argument('--text',required=True)
p.add_argument('--frames',required=True,type=int)
p.add_argument('--output',required=True,type=Path)
a=p.parse_args()
if a.output.exists():p.error('output exists')
torch.set_num_threads(2)
voice=json.loads(a.voice.read_text())
model=OmniVoice.from_pretrained(str(a.model),device_map='cpu',dtype=torch.float32,load_asr=False,local_files_only=True).eval()
prompt=model.create_voice_clone_prompt(ref_audio=str(a.voice.parent/voice['audio']),ref_text=voice['transcript'])
prepared=model._prepare_inference_inputs(text=a.text,num_target_tokens=a.frames,ref_text=prompt.ref_text,ref_audio_tokens=prompt.ref_audio_tokens,lang='en',instruct=None,denoise=True)
ids,mask=prepared['input_ids'],prepared['audio_mask']
books=ids.shape[1];u=ids[:,:,-a.frames:]
def payload(ids,mask):return {'tokens':ids.shape[-1],'ids':ids.flatten().tolist(),'audio_mask':mask.flatten().tolist()}
data={'conditional':payload(ids,mask),'unconditional':payload(u,torch.ones(1,a.frames,dtype=torch.bool)),'target_frames':a.frames,'text':a.text,'reference':voice['id'],'prepared_by':'Python upstream tokenizer and reference codec; inference/decoding done by native Go','ref_rms':prompt.ref_rms}
with a.output.open('x') as out:json.dump(data,out,indent=2)
print(a.output,ids.shape)
