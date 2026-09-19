import {expect,test} from 'bun:test';
import {createHash} from 'node:crypto';
import {mkdtempSync,writeFileSync,rmSync,readFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
const hash=(b:string)=>createHash('sha256').update(b).digest('hex');
test('selection is reproducible, partition-specific and never requires test data',async()=>{
 const dir=mkdtempSync(join(tmpdir(),'direct-selection-'));
 try{
  const rows=JSON.stringify({context:'Task: Answer science.\nQuestion: Does ice melt?\nAnswer:',options:['yes','no'],label:0})+'\n';
  const provenance=['validation','calibration'].map(partition=>JSON.stringify({partition,output_row:1,input:0,source_id:partition+'-id'})).join('\n')+'\n';
  const artifacts=['validation','calibration'].map(p=>({path:p+'.jsonl',sha256:hash(rows)}));artifacts.push({path:'provenance.jsonl',sha256:hash(provenance)});
  writeFileSync(join(dir,'manifest.json'),JSON.stringify({inputs:[{source:'ai2_arc',config:'ARC-Easy'}],artifacts}));
  writeFileSync(join(dir,'validation.jsonl'),rows);writeFileSync(join(dir,'calibration.jsonl'),rows);writeFileSync(join(dir,'provenance.jsonl'),provenance);
  const run=async(partition:string,out:string)=>{const p=Bun.spawn(['bun',resolve(import.meta.dir,'jevlike-direct-pilot.ts'),'--dataset',dir,'--output',join(dir,out),'--partition',partition],{stdout:'pipe',stderr:'pipe'});return await p.exited};
  expect(await run('validation','v1')).toBe(0);expect(await run('validation','v2')).toBe(0);expect(readFileSync(join(dir,'v1/requests.jsonl'),'utf8')).toBe(readFileSync(join(dir,'v2/requests.jsonl'),'utf8'));
  expect(await run('calibration','c')).toBe(0);
  const cal=readFileSync(join(dir,'c/requests.jsonl'),'utf8').trim().split('\n').map(JSON.parse);expect(cal.length).toBe(1);expect(cal[0].id).toBe('calibration-id');expect(cal[0].variant).toBe('normal');
  expect(await run('test','forbidden')).not.toBe(0);
  writeFileSync(join(dir,'validation.jsonl'),'changed');expect(await run('validation','tampered')).not.toBe(0);
 }finally{rmSync(dir,{recursive:true,force:true})}
});
