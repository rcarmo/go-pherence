"""Usage: PYTHONPATH=upstream python this.py checkpoint-directory"""
import json,sys
from pathlib import Path
import torch
from gliner2.models.boundary.model import BoundaryExtractorModel

torch.set_num_threads(6)
m=BoundaryExtractorModel.from_pretrained(sys.argv[1],use_flashdeberta=False).float().eval()
text='The service was excellent.'
task='sentiment';labels=['positive','negative','neutral']
schema={'classifications':[{'task':task,'labels':labels,'true_label':[]}]}
b=m.processor.collate_fn_inference([(text,schema)],architecture='boundary',error_policy='raise')
with torch.no_grad():
 core=m._encode_core(b)
 choice=core['cls_specs'][0][0]['choice_states']
 logits=m.classifier(choice).squeeze(-1)
 data={'text':text,'task':task,'labels':labels,'ids':b.input_ids[0].tolist(),'logits':logits.tolist()}
Path(__file__).with_name('published_classification_reference.json').write_text(json.dumps(data,indent=2)+'\n')
