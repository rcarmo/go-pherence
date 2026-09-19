#!/usr/bin/env bun
/** Freeze existing artefacts before test admission. Never opens test.jsonl. */
import {createHash} from 'node:crypto';
import {readFileSync,writeFileSync} from 'node:fs';
import {parseArgs} from 'node:util';
import {resolve} from 'node:path';
const {values}=parseArgs({args:Bun.argv.slice(2),options:{output:{type:'string'}},strict:true});
if(!values.output)throw Error('--output required');
const root=resolve(import.meta.dir,'..');const rel='checkpoints/jevlike-qwen3/';
const hash=(b:Buffer|string)=>createHash('sha256').update(b).digest('hex');
const read=(p:string)=>readFileSync(resolve(root,p));
const dataset=rel+'pilot-v4',assets=rel+'instruction-model-verified.json',contract=rel+'features-f32-v1/contract.json';
const manifest=JSON.parse(read(dataset+'/manifest.json').toString());
const test=manifest.artifacts.find((x:any)=>x.path==='test.jsonl');if(!test||!/^[a-f0-9]{64}$/.test(test.sha256))throw Error('test hash absent from existing manifest');
const directCal=rel+'direct-calibration-v1/instruction-report.json';
const heads=[7,17,27].map(seed=>({seed,checkpoint:rel+`head-runs-v1/seed-${seed}.json`,calibration:rel+`head-runs-v1/seed-${seed}-calibration-report.json`}));
const files=[assets,contract,dataset+'/manifest.json',directCal,...heads.flatMap(x=>[x.checkpoint,x.calibration]),'model/jevlike/direct.go','cmd/jevlike/cached.go','loader/tokenizer/tokenizer.go','loader/tokenizer/sidecar.go','scripts/jevlike-direct-pilot.ts','docs/experiments/jevlike-qwen3/final-evaluation-policy.md'];
const hashes=Object.fromEntries(files.map(p=>[p,hash(read(p))]));const cal=JSON.parse(read(directCal).toString()),feature=JSON.parse(read(contract).toString());
if(cal.model_id!==feature.model_id||cal.source_manifest_sha256!==hashes[dataset+'/manifest.json'])throw Error('model/dataset identity mismatch');
for(const h of heads){const c=JSON.parse(read(h.calibration).toString());if(!c.temperature_fitted||c.partition!=='calibration'||c.source_manifest_sha256!==cal.source_manifest_sha256)throw Error('head calibration mismatch')}
const record={version:1,scope:'final-once-normal-v1',created_utc:new Date().toISOString(),dataset,dataset_sha256:hashes[dataset+'/manifest.json'],partition_sha256:test.sha256,expected_examples:1440,model_id:feature.model_id,verified_assets:assets,feature_contract:contract,direct_calibration:directCal,heads,files:hashes,limits:{direct_tokens:512,context_tokens:512,option_tokens:128,cache_bytes:12*2**30,free_disk_bytes:30*2**30},policy:'All original final-test rows; normal variant; original candidates; no retuning; fixed calibration; all seeds; reject overlength, no truncation; deferred human-reviewed/broader matrix; close as experiment not promotion'};
writeFileSync(resolve(root,values.output),JSON.stringify(record,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify({output:values.output,sha256:hash(read(values.output)),test_rows_read:false}));
