#!/usr/bin/env bun
/** Build partition-specific study requests and controls; never reads final test. */
import { createHash } from "node:crypto";
import { mkdirSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
const hash=(s:string)=>createHash("sha256").update(s).digest("hex");
const {values}=parseArgs({args:Bun.argv.slice(2),options:{dataset:{type:"string",default:"checkpoints/jevlike-qwen3/pilot-v4"},output:{type:"string",default:"checkpoints/jevlike-qwen3/direct-pilot-v1"},"per-task":{type:"string",default:"12"},partition:{type:"string",default:"validation"}}});
const partition=values.partition!;if(!["train","validation","calibration"].includes(partition))throw new Error("only train/validation/calibration selection is supported; final test stays untouched");
const source=resolve(values.dataset!),out=resolve(values.output!);const n=Number(values["per-task"]);if(!Number.isInteger(n)||n<2||n>100)throw new Error("per-task must be 2..100");if(existsSync(out))throw new Error("output exists");
const manifest=JSON.parse(readFileSync(source+"/manifest.json","utf8"));
const text=readFileSync(source+"/"+partition+".jsonl","utf8");const artifact=manifest.artifacts.find((a:any)=>a.path===partition+".jsonl");if(hash(text)!==artifact.sha256)throw new Error("validation identity changed");
const provenance=readFileSync(source+"/provenance.jsonl","utf8");if(hash(provenance)!==manifest.artifacts.find((a:any)=>a.path==="provenance.jsonl").sha256)throw new Error("provenance identity changed");
const choices=text.trim().split("\n").map(JSON.parse);const meta=new Map<number,any>();
for(const p of provenance.trim().split("\n").map(JSON.parse)){if(p.partition===partition&&!meta.has(p.output_row))meta.set(p.output_row,p)}
const groups=new Map<string,any[]>();
for(const [row,p]of meta){const input=manifest.inputs[p.input];const task=input.source+(input.config?"/"+input.config:"");const ex=choices[row-1];const a=groups.get(task)??[];a.push({row,p,task,ex});groups.set(task,a)}
let requests:any[]=[];
for(const [task,list]of [...groups].sort()){
 list.sort((a,b)=>hash("direct-pilot-v1:"+a.p.source_id).localeCompare(hash("direct-pilot-v1:"+b.p.source_id)));
 const selected=list.slice(0,n);
 for(let i=0;i<selected.length;i++){
  const x=selected[i],context=x.ex.context;
  let evidence="",question=context;
  if(task.startsWith("multi_nli")){
   const start=context.indexOf("Premise: ")+9,end=context.indexOf("\nHypothesis: ");
   if(start<9||end<start)throw new Error("NLI context contract");
   evidence=context.slice(start,end);question="Given the evidence, classify this hypothesis: "+context.slice(end+13).replace(/\nAnswer:$/," ").trim();
  }else if(task.startsWith("clinc_oos")){
   evidence=context.split("Utterance: ")[1]?.replace(/\nIntent:$/,"");if(evidence===undefined)throw new Error("CLINC context contract");question="Which intent best matches the utterance?";
  }else{question=context.replace(/^Task:[^\n]*\nQuestion: /,"").replace(/\nAnswer:$/,"")}
  const candidates=x.ex.options.map((s:string)=>({id:hash(s).slice(0,16),text:s}));const gold=candidates[x.ex.label].id;
  const base={id:x.p.source_id,task,gold_id:gold,request:{evidence,question,candidates,temperature:1}};
  requests.push({...base,variant:"normal"},{...base,variant:"reverse-options",request:{...base.request,candidates:[...candidates].reverse()}});
  requests.push({...base,variant:"option-only",request:{...base.request,evidence:"No evidence supplied.",question:"Select the most plausible answer from these alternatives."}});
  if(evidence){
   const other=selected[(i+1)%selected.length].ex.context;
   let swapped=task.startsWith("multi_nli")?other.split("Premise: ")[1].split("\nHypothesis: ")[0]:other.split("Utterance: ")[1].replace(/\nIntent:$/,"");
   if(swapped===evidence)throw new Error("shuffled evidence unchanged");
   requests.push({...base,variant:"shuffled-evidence",request:{...base.request,evidence:swapped}});
   requests.push({...base,variant:"no-evidence",request:{...base.request,evidence:"No evidence supplied."}});
  }
 }
}
// Calibration fits only original requests; do not spend GPU time scoring
// ablations which cannot contribute to that fit.
if(partition!=="validation")requests=requests.filter(x=>x.variant==="normal");
mkdirSync(out,{recursive:true});const payload=requests.map(x=>JSON.stringify(x)).join("\n")+"\n";writeFileSync(out+"/requests.jsonl",payload);
writeFileSync(out+"/selection.json",JSON.stringify({version:2,partition,source_manifest_sha256:hash(readFileSync(source+"/manifest.json","utf8")),partition_sha256:artifact.sha256,per_task:n,requests:requests.length,request_sha256:hash(payload),policies:["lowest sha256(direct-pilot-v1:source_id) per source/config","reverse candidate order preserves stable text IDs","task-matched different-example evidence; evidence controls only NLI/CLINC","QA option-only is not an evidence-shuffle metric","no final-test reads"],counts:[...groups].map(([task,x])=>({task,available:x.length,selected:Math.min(x.length,n)}))},null,2)+"\n");
console.log(JSON.stringify({output:out,requests:requests.length,perTask:n}));
