"""PYTHONPATH=upstream python this.py checkpoint-directory"""
import json,sys
from pathlib import Path
import torch
from gliner2.models.boundary.model import BoundaryExtractorModel
from gliner2.models.boundary.relations import RelationTypeSpec

torch.set_num_threads(6)
m=BoundaryExtractorModel.from_pretrained(sys.argv[1],use_flashdeberta=False).float().eval()
text='Ada Lovelace lived in London.'
relation='lives_in'
b=m.processor.collate_fn_inference([(text,{'relations':[{relation:{'head':'','tail':''}}]})],architecture='boundary',error_policy='raise')
with torch.no_grad():
 core=m._encode_core(b)
 out=m.boundary_head(core['text_states'],core['text_mask'],core['query_states'],core['query_mask'])
 specs=[[RelationTypeSpec(relation,(0,),(1,))]]
 pairs=m.relation_pair_generator.generate_batched(out.candidates,b.query_layouts,specs)
 query=torch.stack([x['query_state'] for x in core['rel_specs'][0]]).unsqueeze(0)
 logits=m.relation_scorer(core['text_states'],query,out.candidates,pairs)
 rows=[]
 for i in range(len(logits)):
  if pairs.pair_mask is None or bool(pairs.pair_mask[i]):
   rows.append({'span':[int(pairs.head_start[i]),int(pairs.head_end[i]),int(pairs.tail_start[i]),int(pairs.tail_end[i])],'logit':float(logits[i])})
 data={'text':text,'relation':relation,'ids':b.input_ids[0].tolist(),'pairs':rows}
Path(__file__).with_name('published_relation_reference.json').write_text(json.dumps(data,indent=2)+'\n')
