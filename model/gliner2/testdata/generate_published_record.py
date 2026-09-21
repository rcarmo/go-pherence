"""PYTHONPATH=pinned-upstream python this.py checkpoint-directory"""
import json,sys
from pathlib import Path
import torch
from gliner2.models.boundary.model import BoundaryExtractorModel
from gliner2.processing.records import RecordSpec,RecordFieldSpec
from gliner2.models.boundary.records import decode_group

torch.set_num_threads(6)
m=BoundaryExtractorModel.from_pretrained(sys.argv[1],use_flashdeberta=False).float().eval()
text='Ada Lovelace lived in London.'
schema={'json_structures':[{'person':{'name':'','city':''}}]}
b=m.processor.collate_fn_inference([(text,schema)],architecture='boundary',error_policy='raise')
with torch.no_grad():
 core=m._encode_core(b)
 out=m.boundary_head(core['text_states'],core['text_mask'],core['query_states'],core['query_mask'])
 spec=RecordSpec(0,'person','json_structures','natural',(RecordFieldSpec(0,'name',0,is_anchor=True),RecordFieldSpec(1,'city',1)),anchor_query_id=0)
 group=m.record_decoder.forward_group_dense(spec,core['query_states'][0],out.candidates,0)
 data={'text':text,'ids':b.input_ids[0].tolist(),'spans':group.pool_spans.tolist(),'mask':group.instance_mask.tolist(),'object':group.object_logits.tolist(),'assign':{str(i):group.assign_logits[i].tolist() for i in range(len(group.instance_mask)) if bool(group.instance_mask[i])}}
Path(__file__).with_name('published_record_reference.json').write_text(json.dumps(data,indent=2)+'\n')
# Upstream decoding uses compact per-field candidates rather than padded rows.
with torch.no_grad():
 compact=m.record_decoder.forward_group(spec,core['query_states'][0],out.candidates,0)
 decoded=decode_group(compact,anchor_threshold=.5,object_threshold=.5,field_threshold=.5,temperature=1.)
 result={'text':text,'candidate_logits':out.candidates.pair_logits[0].T.tolist(),
         'decoded':[{'score':r.score,'anchor':r.anchor_span,'fields':r.fields,'field_scores':r.field_scores} for r in decoded]}
Path(__file__).with_name('published_record_decode_reference.json').write_text(json.dumps(result,indent=2)+'\n')
