"""Tiny complete encoder oracle; requires torch and transformers."""
import json
from pathlib import Path
import torch
import transformers
from transformers import DebertaV2Config, DebertaV2Model

torch.manual_seed(23)
c=DebertaV2Config(vocab_size=19,hidden_size=8,num_hidden_layers=2,num_attention_heads=2,
 intermediate_size=12,hidden_act='gelu',relative_attention=True,pos_att_type=['p2c','c2p'],
 share_att_key=True,position_buckets=4,max_relative_positions=16,max_position_embeddings=16,
 norm_rel_ebd='layer_norm',position_biased_input=False,type_vocab_size=0,layer_norm_eps=1e-7,
 hidden_dropout_prob=0,attention_probs_dropout_prob=0)
m=DebertaV2Model(c).eval()
ids=torch.tensor([[1,3,5,7,2,0,0]])
mask=torch.tensor([[1,1,1,1,1,0,0]])
with torch.no_grad():
 out=m(input_ids=ids,attention_mask=mask).last_hidden_state[0]
 weights={'encoder.'+k:{'shape':list(v.shape),'values':v.flatten().tolist()} for k,v in m.state_dict().items()}
 data={'config':c.to_dict(),'ids':ids[0].tolist(),'mask':[bool(x) for x in mask[0]],'expected':out.tolist(),'weights':weights,'transformers_version':transformers.__version__}
Path(__file__).with_name('deberta_encoder_reference.json').write_text(json.dumps(data,indent=2)+'\n')
