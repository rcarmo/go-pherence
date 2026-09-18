"""PYTHONPATH=pinned-upstream python this.py checkpoint-directory"""
import json,sys
from pathlib import Path
import torch
from gliner2.models.boundary.model import BoundaryExtractorModel
from gliner2.processing.records import RecordSpec,RecordFieldSpec

torch.set_num_threads(6)
m=BoundaryExtractorModel.from_pretrained(sys.argv[1],use_flashdeberta=False).float().eval()
text='Ada Lovelace lived in London.'
schema={'entities':{'location':[]},'json_structures':[{'person':{'name':'','city':''}}]}
b=m.processor.collate_fn_inference([(text,schema)],architecture='boundary',error_policy='raise')
with torch.no_grad():
 core=m._encode_core(b);out=m.boundary_head(core['text_states'],core['text_mask'],core['query_states'],core['query_mask'])
 specs=core['ext_specs'][0]
 record=[s for s in specs if s['task_type']=='json_structures']
 qids=[i for i,s in enumerate(specs) if s['task_type']=='json_structures']
 spec=RecordSpec(0,'person','json_structures','natural',tuple(RecordFieldSpec(q,s['field_name'],i,is_anchor=i==0) for i,(q,s) in enumerate(zip(qids,record))),anchor_query_id=qids[0])
 g=m.record_decoder.forward_group_dense(spec,core['query_states'][0],out.candidates,0)
 data={'text':text,'ids':b.input_ids[0].tolist(),'types':b.task_types[0],'query_ids':qids,'object':g.object_logits.tolist(),'mask':g.instance_mask.tolist(),'assign':{str(i):g.assign_logits[i].tolist() for i in range(len(g.instance_mask)) if bool(g.instance_mask[i])}}
Path(__file__).with_name('mixed_record_reference.json').write_text(json.dumps(data,indent=2)+'\n')
