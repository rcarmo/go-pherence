"""Run with PYTHONPATH pointing to the pinned GLiNER2 checkout.
Usage: python generate_published_entities.py /path/to/gliner2.5-base-v1
"""
import json
import sys
from pathlib import Path
import torch
from gliner2.models.boundary.model import BoundaryExtractorModel

torch.set_num_threads(6)
m=BoundaryExtractorModel.from_pretrained(sys.argv[1],use_flashdeberta=False).float().eval()
text='Ada Lovelace lived in London.'
labels=['person','location']
b=m.processor.collate_fn_inference([(text,{'entities':{x:[] for x in labels}})],architecture='boundary',error_policy='raise')
with torch.no_grad():
 core=m._encode_core(b)
 out=m.boundary_head(core['text_states'],core['text_mask'],core['query_states'],core['query_mask'])
 c=out.candidates
 data={'text':text,'labels':labels,'ids':b.input_ids[0].tolist(),
       'indices':c.indices[0,0].tolist(),'valid':c.valid_mask[0,0].tolist(),
       'logits':c.pair_logits[0].T.tolist(),
       'null':out.null_logits[0].tolist(),'count':out.count_log_rates[0].tolist()}
 Path(__file__).with_name('published_entities_reference.json').write_text(json.dumps(data,indent=2)+'\n')
