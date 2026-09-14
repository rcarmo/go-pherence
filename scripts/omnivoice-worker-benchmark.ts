/**
 * Benchmark uncached sequential OmniVoice requests, separating worker startup.
 * Usage: bun scripts/omnivoice-worker-benchmark.ts <binary> <model-dir> <reference.json> <new-output-dir> [requests=2]
 * Runs shared-streamed, resident and prepacked column-worker modes sequentially.
 * Writes private raw logs plus a report without prompt/reference content.
 */
import {mkdir, writeFile, readFile, appendFile} from 'node:fs/promises';
import {resolve, join} from 'node:path';
import {createHash} from 'node:crypto';

const [binaryArg,modelArg,referenceArg,outArg,countArg='2']=process.argv.slice(2);
const count=Number(countArg);
if(!binaryArg||!modelArg||!referenceArg||!outArg||!Number.isInteger(count)||count<2||count>10) throw Error('usage: <binary> <model-dir> <reference.json> <new-output-dir> [requests=2..10]');
const binary=resolve(binaryArg), model=resolve(modelArg), reference=resolve(referenceArg), out=resolve(outArg);
await mkdir(out,{mode:0o700}); // Refuse overwrite.
const modes=[{name:'shared',flags:['-shared-traversal']},{name:'resident',flags:['-resident-mib','2048']},{name:'prepacked',flags:['-resident-mib','4096','-prepack']}];
const results:any[]=[];
let baselineHash:string|undefined;
for(const mode of modes){
 const dir=join(out,mode.name);await mkdir(dir,{mode:0o700});
 const args=['-mode','serve','-model',model,'-reference-tokens',reference,'-output-dir',dir,'-frames','75','-steps','8','-threads','2','-gemm-workers','2','-gemm-columns','-cache-mib','0',...mode.flags];
 const proc=Bun.spawn([binary,...args],{stdin:'pipe',stdout:'pipe',stderr:'pipe'});
 const stderrPromise=new Response(proc.stderr).text();
 let peakRSSKiB=0, sampling=false;
 const sample=async()=>{if(sampling)return;sampling=true;try {const stat=await readFile(`/proc/${proc.pid}/status`,'utf8');const m=/^VmHWM:\s+(\d+) kB/m.exec(stat);if(m)peakRSSKiB=Math.max(peakRSSKiB,Number(m[1]));}catch{}finally{sampling=false;}};
 const timer=setInterval(()=>void sample(),50);
 let timedOut=false;const watchdog=setTimeout(()=>{timedOut=true;proc.kill();},(count+1)*180000);
 const events:any[]=[];let pending='',ready:any,completed=0;
 const decoder=new TextDecoder();
 const send=async()=>{proc.stdin.write(JSON.stringify({id:`run-${completed+1}`,text:'The evidence is insufficient, Captain.',frames:75})+'\n');await proc.stdin.flush();};
 await writeFile(join(dir,'events.jsonl'),'',{flag:'wx',mode:0o600});
 try {
  for await(const bytes of proc.stdout){
   pending+=decoder.decode(bytes,{stream:true});
   for(let at;(at=pending.indexOf('\n'))>=0;){
    const line=pending.slice(0,at);pending=pending.slice(at+1);if(!line.trim())continue;
    const event=JSON.parse(line);events.push(event);
    await appendFile(join(dir,'events.jsonl'),line+'\n');
    if(!['ready','start','chunk','done','error'].includes(event.event))throw Error('unknown worker event');
    if(event.event==='error')throw Error(`worker error: ${JSON.stringify(event)}`);
    if(event.event==='ready'){if(ready)throw Error('duplicate ready');if(event.cache_budget_bytes!==0)throw Error('cache must be disabled');ready=event;await send();}
    if(event.event==='done'){completed++;await sample();if(completed<count)await send();else proc.stdin.end();}
   }
  }
  if(await proc.exited!==0||timedOut||pending.trim()||!ready||completed!==count)throw Error('incomplete worker run');
  const chunks=events.filter(e=>e.event==='chunk'),done=events.filter(e=>e.event==='done');
  if(chunks.length!==count||done.some(e=>e.cache_hits!==0||e.cache_payload_bytes!==0||e.chunks!==1)||chunks.some(e=>e.cache_hit!==false))throw Error('expected one uncached chunk per request');
  const runs=[];
  for(let i=0;i<count;i++){
   const d=done[i],c=chunks[i];if(c.id!==d.id)throw Error('chunk request mismatch');
   const sha256=createHash('sha256').update(await readFile(c.path)).digest('hex');
   if(baselineHash&&sha256!==baselineHash)throw Error('worker audio parity failed');baselineHash??=sha256;
   runs.push({id:d.id,request_seconds:d.request_seconds,denoise_seconds:c.stages.denoise_seconds,decode_seconds:c.stages.decode_seconds,audio_seconds:c.audio_seconds,cache_hit:c.cache_hit,sha256});
  }
  results.push({mode:mode.name,flags:mode.flags,peak_rss_kib_sampled:peakRSSKiB,startup_seconds:ready.startup_seconds,runner_setup_seconds:ready.runner_setup_seconds,resident_cache_bytes:ready.resident_cache_bytes,prepacked_bytes:ready.prepacked_bytes,runs});
  console.log(JSON.stringify(results.at(-1)));
 }finally{
  clearInterval(timer);clearTimeout(watchdog);proc.kill();await proc.exited;
  await writeFile(join(dir,'stderr.txt'),await stderrPromise,{flag:'wx',mode:0o600});
 }
}
const report={configuration:{steps:8,guidance:2,frames:75,threads:2,workers:2,columns:true,cache_mib:0,requests_per_process:count},notes:['One process per mode; requests sequential after ready/done.','RSS is sampled /proc VmHWM, not exact wait4 peak.','Audio hashes compare worker outputs only; worker boundary processing differs from single-shot synthesis.'],results};
await writeFile(join(out,'report.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx',mode:0o600});
