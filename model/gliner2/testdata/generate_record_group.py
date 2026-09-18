"""PYTHONPATH=pinned-upstream python generate_record_group.py"""
import json
from pathlib import Path
import torch
from gliner2.models.boundary.records import RecordHead
from gliner2.processing.records import RecordSpec,RecordFieldSpec
from gliner2.models.outputs import CandidateTensorBatch

torch.manual_seed(41)
h=RecordHead(4,2,2).eval()
queries=torch.randn(2,4);states=torch.randn(3,4)
indices=torch.tensor([[[[0,1],[1,2],[2,3]],[[0,1],[1,2],[2,3]]]])
valid=torch.tensor([[[True,True,False],[True,True,False]]])
logits=torch.randn(1,2,3)
c=CandidateTensorBatch(proposal_logits=None,indices=indices,pair_logits=logits,valid_mask=valid,query_mask=torch.ones(1,2,dtype=torch.bool),candidate_states=states[None,None].expand(1,2,3,4))
data={'weights':{'record_decoder.'+k:{'shape':list(v.shape),'values':v.flatten().tolist()} for k,v in h.state_dict().items()},'queries':queries.tolist(),'states':states.tolist(),'pair_logits':logits[0].T.tolist(),'cases':[]}
with torch.no_grad():
 for mode in ['natural','latent','anchorless']:
  spec=RecordSpec(0,'event','json',mode,(RecordFieldSpec(0,'name',0,is_anchor=mode=='natural'),RecordFieldSpec(1,'place',1)),anchor_query_id=0 if mode=='natural' else None)
  out=h.forward_group_dense(spec,queries,c,0)
  data['cases'].append({'mode':mode,'object':out.object_logits.tolist(),'assign':out.assign_logits.tolist(),'mask':out.instance_mask.tolist()})
Path(__file__).with_name('record_group_reference.json').write_text(json.dumps(data,indent=2)+'\n')
