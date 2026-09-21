#!/usr/bin/env python3
"""Generate a CPU-only released Nimble prompt/logit oracle."""
import argparse, json, sys
from pathlib import Path
import torch
SOURCE_PIN="f136b3f75721fda4ea961f73993cc50b08488835"
MODEL_PIN="594dfdcfb6f94e3d0c0db7535180d3c71689169a"
BASE_PIN="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
def main():
 p=argparse.ArgumentParser();p.add_argument('--upstream',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--adapter',type=Path);p.add_argument('--merged',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.upstream));
 if bool(a.adapter)==bool(a.merged):raise SystemExit('choose exactly one of --adapter or --merged')
 from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
 from peft import PeftModel
 from nimble.scoring.parallel_schema import prepare_prompts
 if torch.cuda.is_available():raise SystemExit('CUDA hidden required')
 torch.set_num_threads(2);context='The payment service is down for all customers.';schema={'priority':{'type':'enum','choices':['HIGH','LOW'],'description':'Urgency based on current business impact.','choice_descriptions':{'HIGH':'A critical business operation is currently blocked.','LOW':'An optional enhancement with no current business impact.'}},'requires_review':{'type':'boolean','description':'Whether customers are unable to complete a purchase.'}}
 tok=AutoTokenizer.from_pretrained(a.base,local_files_only=True);prep=prepare_prompts(tok,context,schema,2048);model=Qwen3_5ForConditionalGeneration.from_pretrained(a.merged or a.base,device_map='cpu',torch_dtype='auto',low_cpu_mem_usage=True,local_files_only=True).eval();model=PeftModel.from_pretrained(model,a.adapter,local_files_only=True).eval() if a.adapter else model;rows=[]
 with torch.inference_mode():
  for ids,candidates in zip(prep.full_ids,prep.candidate_ids):
   raw=model(input_ids=torch.tensor([ids]),use_cache=False,logits_to_keep=1).logits[0,-1].float();values=raw[candidates];rows.append({'ids':ids,'candidate_ids':candidates,'logits':values.tolist(),'probabilities':values.softmax(-1).tolist()})
 out={'source_pin':SOURCE_PIN,'model_pin':MODEL_PIN,'base_pin':BASE_PIN,'context':context,'schema':schema,'names':prep.names,'choices':prep.choices,'prefix_ids':prep.prefix_ids,'suffix_ids':prep.suffix_ids,'mode':'merged_bf16' if a.merged else 'unmerged_peft','rows':rows};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
