import {test, expect} from 'bun:test';
import {mkdtemp,writeFile,chmod,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';

test('benchmark consumes event-based ready, sends flushed requests and preserves stages',async()=>{
 const root=await mkdtemp(join(tmpdir(),'omnivoice-benchmark-test-'));
 try {
  const fake=join(root,'fake-worker');
  await writeFile(fake,`#!/usr/bin/env bun
import {join} from 'node:path';
const args=process.argv.slice(2),dir=args[args.indexOf('-output-dir')+1];
const wave=join(dir,'fake.wav');await Bun.write(wave,'fixed test waveform');
console.log(JSON.stringify({event:'ready',startup_seconds:.1,runner_setup_seconds:.05,cache_budget_bytes:0,resident_cache_bytes:0,prepacked_bytes:0}));
let pending='';const dec=new TextDecoder();
for await(const bytes of Bun.stdin.stream()){
pending+=dec.decode(bytes,{stream:true});let at;
while((at=pending.indexOf('\\n'))>=0){const line=pending.slice(0,at);pending=pending.slice(at+1);if(!line)continue;const req=JSON.parse(line);
console.log(JSON.stringify({event:'chunk',id:req.id,path:wave,audio_seconds:3,cache_hit:false,stages:{denoise_seconds:1,decode_seconds:.2}}));
console.log(JSON.stringify({event:'done',id:req.id,chunks:1,cache_hits:0,cache_payload_bytes:0,request_seconds:1.3}));
}}
`,{mode:0o700});await chmod(fake,0o700);
  const output=join(root,'output');
  const p=Bun.spawn(['bun',join(import.meta.dir,'omnivoice-worker-benchmark.ts'),fake,root,join(root,'reference.json'),output,'2'],{stdout:'pipe',stderr:'pipe'});
  const deadline=setTimeout(()=>p.kill(),10000);
  const [stdout,stderr,code]=await Promise.all([new Response(p.stdout).text(),new Response(p.stderr).text(),p.exited]);clearTimeout(deadline);
  expect(code,stderr+stdout).toBe(0);
  const report=await Bun.file(join(output,'report.json')).json();expect(report.results.length).toBe(3);
  for(const mode of report.results){expect(mode.runs.length).toBe(2);expect(mode.runner_setup_seconds).toBe(.05);for(const run of mode.runs){expect(run.denoise_seconds).toBe(1);expect(run.decode_seconds).toBe(.2);expect(run.cache_hit).toBe(false);}}
 } finally {await rm(root,{recursive:true,force:true});}
},15000);
