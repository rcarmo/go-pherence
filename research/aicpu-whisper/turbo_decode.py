import numpy as np, json, sys, time, onnxruntime as ort
DEC=sys.argv[1]; HNPY=sys.argv[2]
SOT,EN,TRANSCRIBE,NOTS,EOS=50258,50259,50360,50364,50257
def b2u():
    bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
    for b in range(256):
        if b not in bs: bs.append(b);cs.append(256+n);n+=1
    return {chr(c):b for b,c in zip(bs,cs)}
BD=b2u();tok=json.load(open('tokenizer.json'));id2={i:t for t,i in tok['model']['vocab'].items()}
sp={a['id'] for a in tok.get('added_tokens',[])}
def detok(ids):
    s=''.join(id2.get(i,'') for i in ids if i not in sp and i<50257)
    return bytes([BD.get(c,ord(c)&255) for c in s]).decode('utf-8','replace')
so=ort.SessionOptions();so.intra_op_num_threads=6
dec=ort.InferenceSession(DEC,so,providers=['CPUExecutionProvider'])
NL=sum(1 for i in dec.get_inputs() if i.name.endswith('.decoder.key'))
H=np.load(HNPY).astype(np.float32);nh,hd=20,64;el=H.shape[1]
kinds=['decoder.key','decoder.value','encoder.key','encoder.value']
past={f'past_key_values.{l}.{k}':np.zeros((1,nh,0 if 'decoder' in k else el,hd),np.float32) for l in range(NL) for k in kinds}
outAll=['logits']+[f'present.{l}.{k}' for l in range(NL) for k in kinds]
outStep=['logits']+[f'present.{l}.{k}' for l in range(NL) for k in ['decoder.key','decoder.value']]
ko=[(l,k) for l in range(NL) for k in kinds]; ks=[(l,k) for l in range(NL) for k in ['decoder.key','decoder.value']]
t=time.time()
r=dec.run(outAll,{'input_ids':np.array([[SOT,EN,TRANSCRIBE,NOTS]],np.int64),'encoder_hidden_states':H,'use_cache_branch':np.array([False]),**past})
nxt=int(r[0][0,-1].argmax())
for i,(l,k) in enumerate(ko): past[f'past_key_values.{l}.{k}']=r[1+i]
out=[] if nxt==EOS else [nxt]
for _ in range(220):
    if nxt==EOS: break
    r=dec.run(outStep,{'input_ids':np.array([[nxt]],np.int64),'encoder_hidden_states':H,'use_cache_branch':np.array([True]),**past})
    nxt=int(r[0][0,-1].argmax())
    for i,(l,k) in enumerate(ks): past[f'past_key_values.{l}.{k}']=r[1+i]
    if nxt==EOS: break
    out.append(nxt)
print('layers=%d decode %.2fs %d tokens'%(NL,time.time()-t,len(out)))
print('TRANSCRIPT:',detok(out))
