"""PYTHONPATH=pinned-upstream python this.py checkpoint-directory"""
import json,sys
from pathlib import Path
import torch
from gliner2.models.boundary.model import BoundaryExtractorModel

torch.set_num_threads(6)
m=BoundaryExtractorModel.from_pretrained(sys.argv[1],use_flashdeberta=False).float().eval()
text='Ada Lovelace lived in London.'
schema={'entities':{'person':[],'location':[]},'relations':[{'lives_in':{'head':'','tail':''}}]}
b=m.processor.collate_fn_inference([(text,schema)],architecture='boundary',error_policy='raise')
with torch.no_grad():
 core=m._encode_core(b);out=m.boundary_head(core['text_states'],core['text_mask'],core['query_states'],core['query_mask'])
 specs=core['rel_specs'][0]
 pairs=m.relation_pair_generator.generate_batched(out.candidates,b.query_layouts,[[s['spec'] for s in specs]])
 query=torch.stack([s['query_state'] for s in specs]).unsqueeze(0)
 logits=m.relation_scorer(core['text_states'],query,out.candidates,pairs)
 data={'text':text,'ids':b.input_ids[0].tolist(),'pairs':[{'span':[int(pairs.head_start[i]),int(pairs.head_end[i]),int(pairs.tail_start[i]),int(pairs.tail_end[i])],'logit':float(logits[i])} for i in range(len(logits)) if pairs.pair_mask is None or bool(pairs.pair_mask[i])]}
Path(__file__).with_name('mixed_relation_reference.json').write_text(json.dumps(data,indent=2)+'\n')
