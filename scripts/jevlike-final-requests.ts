#!/usr/bin/env bun
/** Explicit final admission; retains the frozen pilot normal rendering verbatim. */
import {createHash} from 'node:crypto';
import {readFileSync,writeFileSync,mkdirSync,existsSync} from 'node:fs';
import {resolve,relative,isAbsolute} from 'node:path';
import {parseArgs} from 'node:util';
const hash=(s:Buffer|string)=>createHash('sha256').update(s).digest('hex');
export function generateFinal(freezePath:string,output:string,root=resolve(import.meta.dir,'..')) {
 root=resolve(root);const safe=(p:string)=>{const full=resolve(root,p),rel=relative(root,full);if(rel.startsWith('..')||isAbsolute(rel))throw Error('freeze path outside project');return full};
 const frozen=readFileSync(freezePath);const f=JSON.parse(frozen.toString());
 if(f.version!==1||f.scope!=='final-once-normal-v1'||!Number.isSafeInteger(f.expected_examples)||f.expected_examples<1||!f.files||!Object.keys(f.files).length)throw Error('explicit final freeze required');
 for(const [p,h]of Object.entries(f.files))if(hash(readFileSync(safe(p)))!==h)throw Error('frozen file changed: '+p);
 if(existsSync(output))throw Error('output exists');
 const dataset=safe(f.dataset),manifestBytes=readFileSync(dataset+'/manifest.json');if(hash(manifestBytes)!==f.dataset_sha256)throw Error('dataset identity');
 const manifest=JSON.parse(manifestBytes.toString()),test=readFileSync(dataset+'/test.jsonl');
 if(hash(test)!==f.partition_sha256||manifest.artifacts.find((a:any)=>a.path==='test.jsonl')?.sha256!==f.partition_sha256)throw Error('test identity');
 const prov=readFileSync(dataset+'/provenance.jsonl');if(hash(prov)!==manifest.artifacts.find((a:any)=>a.path==='provenance.jsonl')?.sha256)throw Error('provenance identity');
 const choices=test.toString().trim().split('\n').map(JSON.parse);if(choices.length!==f.expected_examples)throw Error('unexpected final count');
 const meta=new Map<number,any>();for(const p of prov.toString().trim().split('\n').map(JSON.parse)){if(p.partition==='test'&&!meta.has(p.output_row))meta.set(p.output_row,p)}
 if(meta.size!==choices.length)throw Error('incomplete provenance coverage');
 const groups=new Map<string,any[]>();
 for(const [row,p]of meta){if(!Number.isInteger(row)||row<1||row>choices.length||!manifest.inputs[p.input])throw Error('invalid provenance index');const input=manifest.inputs[p.input],task=input.source+(input.config?'/'+input.config:'');const a=groups.get(task)??[];a.push({p,ex:choices[row-1]});groups.set(task,a)}
 const requests:any[]=[];const seen=new Set<string>();
 for(const [task,list]of [...groups].sort()){
  list.sort((a,b)=>hash('direct-pilot-v1:'+a.p.source_id).localeCompare(hash('direct-pilot-v1:'+b.p.source_id)));
  for(const x of list){const context=x.ex.context;let evidence='',question=context;
   if(task.startsWith('multi_nli')){const start=context.indexOf('Premise: ')+9,end=context.indexOf('\nHypothesis: ');if(start<9||end<start)throw Error('NLI context contract');evidence=context.slice(start,end);question='Given the evidence, classify this hypothesis: '+context.slice(end+13).replace(/\nAnswer:$/,' ').trim()}
   else if(task.startsWith('clinc_oos')){evidence=context.split('Utterance: ')[1]?.replace(/\nIntent:$/,'');if(evidence===undefined)throw Error('CLINC context contract');question='Which intent best matches the utterance?'}
   else{question=context.replace(/^Task:[^\n]*\nQuestion: /,'').replace(/\nAnswer:$/,'')}
   const candidates=x.ex.options.map((s:string)=>({id:hash(s).slice(0,16),text:s}));if(new Set(candidates.map((c:any)=>c.id)).size!==candidates.length||!Number.isInteger(x.ex.label)||!candidates[x.ex.label])throw Error('candidate/label invalid');
   const id=task+'\0'+x.p.source_id;if(seen.has(id))throw Error('duplicate stable source id');seen.add(id);
   requests.push({id:x.p.source_id,task,gold_id:candidates[x.ex.label].id,request:{evidence,question,candidates,temperature:1},variant:'normal'});
  }
 }
 const payload=requests.map(x=>JSON.stringify(x)).join('\n')+'\n';mkdirSync(output,{recursive:true});
 writeFileSync(output+'/requests.jsonl',payload,{flag:'wx'});
 writeFileSync(output+'/selection.json',JSON.stringify({version:2,partition:'test',source_manifest_sha256:f.dataset_sha256,partition_sha256:f.partition_sha256,requests:requests.length,request_sha256:hash(payload),final_freeze_sha256:hash(frozen),policies:['all originals, original candidates, normal only','frozen pilot normal rendering and deterministic source hash ordering','no prompt/model/calibration tuning; final evaluation authorised by owner'],counts:[...groups].map(([task,a])=>({task,selected:a.length}))},null,2)+'\n',{flag:'wx'});
 return {output,requests:requests.length,request_sha256:hash(payload),freeze_sha256:hash(frozen)};
}
if(import.meta.main){const {values}=parseArgs({args:Bun.argv.slice(2),options:{freeze:{type:'string'},output:{type:'string'}},strict:true});if(!values.freeze||!values.output)throw Error('--freeze and --output required');console.log(JSON.stringify(generateFinal(resolve(values.freeze),resolve(values.output))))}
