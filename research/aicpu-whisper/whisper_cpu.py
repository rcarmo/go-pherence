import numpy as np, json, sys, time
import onnxruntime as ort
from mel import mel30

ENC = sys.argv[1] if len(sys.argv)>1 else 'f32/encoder_model.onnx'
DEC = sys.argv[2] if len(sys.argv)>2 else 'dec_q.onnx'
WAV = sys.argv[3] if len(sys.argv)>3 else '/tmp/pod_30.wav'

SOT, EN, TRANSCRIBE, NOTS, EOS = 50258, 50259, 50360, 50364, 50257

# ---- byte-level BPE decoder (GPT2/whisper) ----
def bytes_to_unicode():
    bs=list(range(33,127))+list(range(161,173))+list(range(174,256))
    cs=bs[:]; n=0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n+=1
    return {c:b for b,c in zip(bs,[chr(c) for c in cs])}
BYTE_DEC=bytes_to_unicode()
tok=json.load(open('tokenizer.json'))
vocab=tok['model']['vocab']           # token_str -> id
id2tok={v:k for k,v in vocab.items()}
# added tokens (special) map
for at in tok.get('added_tokens',[]):
    id2tok[at['id']]=at['content']
SPECIAL={at['id'] for at in tok.get('added_tokens',[])}

def detok(ids):
    s=''.join(id2tok.get(i,'') for i in ids if i not in SPECIAL and i<50257)
    bys=bytes([BYTE_DEC.get(ch, ord(ch)&0xff) for ch in s])
    return bys.decode('utf-8', errors='replace')

so=ort.SessionOptions(); so.intra_op_num_threads=6
print('loading encoder...'); enc=ort.InferenceSession(ENC, so, providers=['CPUExecutionProvider'])
print('loading decoder...'); dec=ort.InferenceSession(DEC, so, providers=['CPUExecutionProvider'])
dec_inputs=[i.name for i in dec.get_inputs()]
n_layers=sum(1 for n in dec_inputs if n.endswith('.decoder.key'))
print('layers',n_layers)

mel=mel30(WAV)
import os
hcache=WAV.replace('/','_')+'.H.npy'
if os.path.exists(hcache):
    H=np.load(hcache); print('encoder cached', H.shape)
else:
    t0=time.time()
    H=enc.run(['last_hidden_state'], {'input_features':mel})[0]
    print('encoder %.2fs'%(time.time()-t0), H.shape)
    np.save(hcache,H)

B=1; nh=20; hd=64
enc_len=H.shape[1]
def empty(): return np.zeros((B,nh,0,hd), np.float32)
past={}
for l in range(n_layers):
    past[f'past_key_values.{l}.decoder.key']=empty()
    past[f'past_key_values.{l}.decoder.value']=empty()
    past[f'past_key_values.{l}.encoder.key']=np.zeros((B,nh,enc_len,hd),np.float32)
    past[f'past_key_values.{l}.encoder.value']=np.zeros((B,nh,enc_len,hd),np.float32)

out_names=['logits']+[f'present.{l}.{k}' for l in range(n_layers) for k in ['decoder.key','decoder.value','encoder.key','encoder.value']]
# cached steps: encoder KV is fixed, so don't re-request it (its present output
# triggers a broken reshape in this export). Only refresh decoder KV.
step_out_names=['logits']+[f'present.{l}.{k}' for l in range(n_layers) for k in ['decoder.key','decoder.value']]
kv_order=[(l,k) for l in range(n_layers) for k in ['decoder.key','decoder.value','encoder.key','encoder.value']]
step_kv_order=[(l,k) for l in range(n_layers) for k in ['decoder.key','decoder.value']]
def present_to_past(res):
    p={}
    for i,(l,k) in enumerate(kv_order): p[f'past_key_values.{l}.{k}']=res[1+i]
    return p

prompt=[SOT,EN,TRANSCRIBE,NOTS]
out_tokens=[]
t1=time.time()
res=dec.run(out_names, {'input_ids':np.array([prompt],np.int64),'encoder_hidden_states':H,
                        'use_cache_branch':np.array([False]), **past})
logits=res[0]; nxt=int(logits[0,-1].argmax())
past=present_to_past(res)  # includes encoder KV (fixed from here on)
if nxt!=EOS: out_tokens.append(nxt)
for step in range(220):
    if nxt==EOS: break
    res=dec.run(step_out_names, {'input_ids':np.array([[nxt]],np.int64),'encoder_hidden_states':H,
                            'use_cache_branch':np.array([True]), **past})
    logits=res[0]; nxt=int(logits[0,-1].argmax())
    for i,(l,k) in enumerate(step_kv_order): past[f'past_key_values.{l}.{k}']=res[1+i]
    if nxt==EOS: break
    out_tokens.append(nxt)
print('decode %.2fs, %d tokens'%(time.time()-t1,len(out_tokens)))
print('TRANSCRIPT:', detok(out_tokens))
