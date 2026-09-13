// Generate a provisional SVG chart and sanitised evidence from a serve JSONL log.
// bun scripts/omnivoice-timing-chart.ts <input.jsonl> <new-output-directory>
import {mkdirSync,writeFileSync} from 'node:fs';
const [input,out]=Bun.argv.slice(2);if(!input||!out)throw Error('input JSONL and new output directory required');
const events=(await Bun.file(input).text()).trim().split('\n').map(s=>JSON.parse(s));
const ready=events.find(e=>e.event==='ready'),done=events.find(e=>e.event==='done');
if(!ready||!done||events.some(e=>e.event==='error'))throw Error('complete successful run required');
const chunks=events.filter(e=>e.event==='chunk'&&e.sequence===done.sequence);
if(chunks.length!==done.chunks||chunks.some(e=>e.cache_hit))throw Error('uncached complete run required');
const finite=(n:number)=>{if(!Number.isFinite(n)||n<0)throw Error('invalid timing');return n;};
const sum=(key:string)=>chunks.reduce((s,e)=>s+finite(e.stages[key]),0);
const stage={preparation:sum('prepare_seconds'),denoising:sum('denoise_seconds'),decoding:sum('decode_seconds'),postprocessing:sum('postprocess_seconds'),writing:chunks.reduce((s,e)=>s+finite(e.write_seconds),0)};
const request=finite(done.request_seconds),startup=finite(ready.startup_seconds);
const residual=request-Object.values(stage).reduce((s,n)=>s+n,0);if(residual< -1e-6)throw Error('stage times exceed request');
let previous=0;for(const c of chunks){if(c.elapsed_seconds<previous||c.elapsed_seconds>request||c.chunk_seconds>c.elapsed_seconds-previous+.001)throw Error('inconsistent chunk intervals');previous=c.elapsed_seconds;}
const evidence={status:'provisional; hardware and listening gates open',steps:ready.steps,workers:ready.gemm_workers,frames:ready.max_frames,resident_bytes:ready.resident_cache_bytes,startup_seconds:startup,request_seconds:request,process_to_done_seconds:startup+request,first_chunk_request_seconds:done.first_chunk_seconds,first_chunk_process_seconds:startup+done.first_chunk_seconds,audio_seconds:done.audio_seconds,stages_seconds:{...stage,planning_boundaries_protocol_other:Math.max(0,residual)},startup_stages_seconds:{reference_tokenizer:ready.reference_tokenizer_seconds,weight_open:ready.weights_seconds,codec_load:ready.codec_load_seconds,runner_and_resident_setup:ready.runner_setup_seconds},chunks:chunks.map(e=>({index:e.index,target_frames:e.target_frames,available_seconds:e.elapsed_seconds,chunk_seconds:e.chunk_seconds,stages:e.stages,write_seconds:e.write_seconds}))};
const width=1100,left=175,plot=855,max=Math.ceil(request/20)*20;
const x=(s:number)=>left+plot*s/max;const num=(n:number)=>n.toFixed(2);
const elements:string[]=[];const text=(x:number,y:number,s:string,size=15,color='#cbd5e1')=>elements.push(`<text x="${x}" y="${y}" font-size="${size}" fill="${color}">${s.replaceAll('&','&amp;').replaceAll('<','&lt;')}</text>`);
const rect=(x:number,y:number,w:number,h:number,color:string)=>elements.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="2" fill="${color}"/>`);
rect(0,0,width,785,'#101827');text(35,45,'OmniVoice generation stages — provisional measurements',25,'#f8fafc');
text(35,76,`${ready.steps} steps · ${ready.gemm_workers} SIMD workers · resident F32 · N100 / 2 vCPUs · ${num(done.audio_seconds)} s audio`);
text(35,101,'Single recorded run; filesystem caches not flushed. Time shown is file availability, not playback.');
text(35,143,`Startup ${num(startup)} s + request ${num(request)} s = ${num(startup+request)} s to completion`,19,'#f8fafc');
rect(35,163,17,13,'#60a5fa');text(60,175,'Denoising');rect(182,163,17,13,'#34d399');text(207,175,'Codec decode');text(370,175,'Tiny preparation / postprocessing / writes listed below');
for(let tick=0;tick<=max;tick+=40){const px=x(tick);elements.push(`<path d="M${px} 197V425" stroke="#334155"/>`);text(px-10,448,String(tick));}
chunks.forEach((c,i)=>{const y=208+i*53;const start=c.elapsed_seconds-c.chunk_seconds;const denStart=start+c.stages.prepare_seconds;const decStart=denStart+c.stages.denoise_seconds;text(35,y+20,`Chunk ${i+1} (${c.target_frames} fr)`);rect(x(denStart),y,plot*c.stages.denoise_seconds/max,30,'#60a5fa');rect(x(decStart),y,plot*c.stages.decode_seconds/max,30,'#34d399');text(x(start)+5,y+20,`${num(c.stages.denoise_seconds)} s + ${num(c.stages.decode_seconds)} s`,13,'#0f172a');text(790,y+45,`Ready ${num(c.elapsed_seconds)} s`,12);});
text(435,476,'Seconds after request receipt',14);text(35,512,`First audio: ${num(done.first_chunk_seconds)} s after request; ${num(startup+done.first_chunk_seconds)} s including startup`,17,'#f8fafc');
text(35,550,`Denoising: ${num(stage.denoising)} s (${(stage.denoising/request*100).toFixed(1)}%)`);text(550,550,`Decoding: ${num(stage.decoding)} s (${(stage.decoding/request*100).toFixed(1)}%)`);
text(35,579,`Postprocessing: ${(stage.postprocessing*1000).toFixed(2)} ms · WAV writes: ${(stage.writing*1000).toFixed(2)} ms`);
text(35,607,`Preparation: ${(stage.preparation*1000).toFixed(3)} ms · Planning / boundaries / protocol / other: ${(residual*1000).toFixed(2)} ms`);
text(35,647,'Startup detail (separate scale; all in seconds)',17,'#f8fafc');
text(35,674,`Reference + tokenizer ${num(ready.reference_tokenizer_seconds)} · Weight open ${ready.weights_seconds.toFixed(4)} · Codec load ${num(ready.codec_load_seconds)}`);
text(35,702,`Runner + resident setup ${num(ready.runner_setup_seconds)}. Setup combines allocations, cache conversion and codec preparation.`);
text(35,742,'Not a final goal-completion chart. Regional accent, prosody and GPU execution are not verified.',14,'#fbbf24');
const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="785" viewBox="0 0 ${width} 785" role="img"><title>Provisional OmniVoice generation timing</title><g font-family="system-ui, sans-serif">${elements.join('')}</g></svg>`;
mkdirSync(out);writeFileSync(`${out}/timing.svg`,svg,{flag:'wx'});writeFileSync(`${out}/timing.json`,JSON.stringify(evidence,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify(evidence,null,2));
