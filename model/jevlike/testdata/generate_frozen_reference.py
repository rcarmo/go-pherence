"""PYTHONPATH=jevlike-upstream python this.py /path/to/Qwen2.5-0.5B
Offline oracle only; native inference does not import Python.
"""
import json,sys
from pathlib import Path
import torch, transformers
from transformers import AutoModel, AutoTokenizer
from jevlike.model import AttentionHead
from jevlike.data import ChoiceExample,HuggingFaceCollator

torch.set_num_threads(6);torch.manual_seed(97)
encoder=AutoModel.from_pretrained(sys.argv[1],torch_dtype=torch.float32,attn_implementation='eager').eval()
tokenizer=AutoTokenizer.from_pretrained(sys.argv[1])
examples=[ChoiceExample('Choose the city mentioned: Ada lived in London.',('London','Paris','Ada'),0),ChoiceExample('The café is in Lisboa.',('Lisboa','coffee'),0)]
batch=HuggingFaceCollator(tokenizer,64,16)(examples)
head=AttentionHead(encoder.config.hidden_size,8).eval()
with torch.no_grad():
 context=encoder(input_ids=batch['context_ids'],attention_mask=batch['context_mask']).last_hidden_state
 shape=batch['option_ids'].shape
 ids=batch['option_ids'].reshape(-1,shape[-1]);mask=batch['option_token_mask'].reshape(-1,shape[-1])
 hidden=encoder(input_ids=ids,attention_mask=mask).last_hidden_state
 pooled=(hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
 options=pooled.reshape(shape[0],shape[1],-1)
 logits=head(context,batch['context_mask'],options,batch['option_mask'])
 parameters=[{'name':'head.'+k,'shape':list(v.shape),'values':v.flatten().tolist()} for k,v in head.state_dict().items()]
 config={'width':encoder.config.hidden_size,'rank':8,'context_tokens':64,'option_tokens':16}
 data={'backbone':'Qwen/Qwen2.5-0.5B','transformers':transformers.__version__,'examples':[{'context':e.context,'options':list(e.options),'label':e.label} for e in examples],
 'checkpoint':{'version':1,'encoder':'frozen','encoder_reference':'Qwen/Qwen2.5-0.5B','config':config,'parameters':parameters},
 'context_ids':[batch['context_ids'][i][batch['context_mask'][i].bool()].tolist() for i in range(len(examples))],
 'hidden':[context[i][batch['context_mask'][i].bool()].tolist() for i in range(len(examples))],
 'logits':[logits[i,:len(e.options)].tolist() for i,e in enumerate(examples)]}
Path(__file__).with_name('frozen_reference.json').write_text(json.dumps(data,separators=(',',':'),ensure_ascii=False)+'\n')
