#!/usr/bin/env python3
"""CPU-only released Laya reference with token/marker/logit evidence."""
import argparse,json,sys
from pathlib import Path
import torch
PIN='42626c348753fbb17572a813127df2278a1ec527'
MODEL='1c5edc17a7acd8701df6fc341c0d179f1c62c982'
def main():
 p=argparse.ArgumentParser();p.add_argument('--upstream',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 sys.path.insert(0,str(a.upstream))
 from laya.agent import Agent
 from laya.common import build_sequence,QTYPES,collate_items
 if torch.cuda.is_available():raise SystemExit('CUDA hidden required')
 torch.set_num_threads(1)
 agent=Agent(str(a.model),device='cpu')
 state='The bicycle is red.'
 questions={'color':{'type':'choice','instructions':'What color is the bicycle?','criteria':{'red':None,'blue':'A blue bicycle'}},'support':{'type':'score','instructions':'How strongly is redness supported?','criteria':['Unsupported','Partly supported','Supported']},'is_red':{'type':'noul','instructions':'Is the bicycle red?'}}
 items=[]
 for qid,qdef in questions.items():
  q=agent._to_internal(qdef);ids,markers=build_sequence(agent.tok,state,q,agent.cfg['max_len'],agent.cfg['head_max_len']);items.append({'ids':ids,'markers':markers,'qtype':QTYPES[q['t']]})
 b=collate_items([items],agent.tok.pad_token_id)
 with torch.no_grad():logits,actions=agent.model(b['input_ids'],b['attention_mask'],b['marker_pos'],b['marker_mask'],b['qtype'])
 response=agent.system_one(state,questions)
 out={'source_pin':PIN,'model_pin':MODEL,'state':state,'questions':questions,'config':agent.cfg,'pad_id':agent.tok.pad_token_id,'items':items,'input_ids':b['input_ids'].tolist(),'attention_mask':b['attention_mask'].tolist(),'marker_pos':b['marker_pos'].tolist(),'marker_mask':b['marker_mask'].tolist(),'qtype':b['qtype'].tolist(),'logits':logits.float().tolist(),'action_logits':actions.float().tolist(),'response':response}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print('wrote',a.output,a.output.stat().st_size,response)
if __name__=='__main__':main()
