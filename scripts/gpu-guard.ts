#!/usr/bin/env bun
/**
 * @summary Record GPU health without launching work by default. Explicitly authorised
 * workloads run in an owned process group and stop on telemetry/driver failure.
 * @usage bun scripts/gpu-guard.ts --out NEW_DIR [--seconds 30] [--uuid GPU-...]
 * @usage bun scripts/gpu-guard.ts --out NEW_DIR --run-authorized -- command args...
 */
import { spawn, type ChildProcess } from 'node:child_process';
import { appendFileSync, closeSync, fsyncSync, mkdirSync, openSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { startOwnedWorkload } from './gpu-guard-process';
import { evaluateHealth, faultFromJournal, parseComputeCSV, parseGPUCSV, type GPUBaseline } from './gpu-guard-state';

export type QueryResult = { code: number | null; stdout: string; stderr: string };
export type Options = { out: string; seconds: number; intervalMs: number; queryMs: number; maxTemperatureC: number; uuid?: string; runAuthorized: boolean; command: string[] };
export const gpuFields = 'uuid,pci.bus_id,driver_version,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,clocks.mem';

export function parseOptions(args: string[]): Options {
  const o: Options = { out: '', seconds: 30, intervalMs: 1000, queryMs: 2000, maxTemperatureC: 83, runAuthorized: false, command: [] };
  const numeric = (s: string, min: number, max: number) => {
    if (!/^\d+$/.test(s)) throw Error('expected an unsigned integer');
    const n = Number(s); if (!Number.isSafeInteger(n) || n < min || n > max) throw Error(`value outside ${min}..${max}`); return n;
  };
  for (let i=0; i<args.length; i++) {
    const a=args[i]; if (a==='--') { o.command=args.slice(i+1); break; }
    if (a==='--run-authorized') { o.runAuthorized=true; continue; }
    if (!['--out','--seconds','--interval-ms','--query-ms','--max-temperature','--uuid'].includes(a)) throw Error(`unknown option ${a}`);
    const v=args[++i]; if (!v || v.startsWith('--')) throw Error(`missing value for ${a}`);
    if(a==='--out')o.out=resolve(v);
    if(a==='--seconds')o.seconds=numeric(v,1,3600);
    if(a==='--interval-ms')o.intervalMs=numeric(v,250,5000);
    if(a==='--query-ms')o.queryMs=numeric(v,100,5000);
    if(a==='--max-temperature')o.maxTemperatureC=numeric(v,40,83);
    if(a==='--uuid') { if(!/^GPU-[\da-fA-F-]+$/.test(v))throw Error('invalid GPU UUID'); o.uuid=v; }
  }
  if(!o.out)throw Error('--out NEW_DIR is required');
  if(o.runAuthorized!==Boolean(o.command.length))throw Error('workload requires both --run-authorized and -- command');
  return o;
}

function signalGroup(child: ChildProcess, signal: NodeJS.Signals) {
  // For query processes, signal only a leader still owned and unreaped by
  // this ChildProcess. Never send a follow-up signal after its exit/reuse.
  if(!child.pid || child.exitCode!==null || child.signalCode!==null)return;
  try { process.kill(-child.pid,signal); } catch(e) { if((e as NodeJS.ErrnoException).code!=='ESRCH')throw e; }
}

// Never use a shell. Each query has its own group; even a hung driver query is
// bounded, and a failed query aborts the guard rather than being retried.
export function boundedQuery(command: string[], timeoutMs: number, maxBytes=8*1024*1024): Promise<QueryResult> {
  return new Promise((resolveResult,reject)=>{
    const child=spawn(command[0],command.slice(1),{detached:true,stdio:['ignore','pipe','pipe']});
    let stdout='',stderr='',bytes=0,settled=false;
    let killTimer: ReturnType<typeof setTimeout> | undefined;
    const finish=(err?:Error,code:number|null=null)=>{
      if(settled)return;settled=true;clearTimeout(timer);if(killTimer)clearTimeout(killTimer);
      if(err)reject(err);else resolveResult({code,stdout,stderr});
    };
    const stop=(why:string)=>{
      if(settled||killTimer)return;
      try{signalGroup(child,'SIGTERM');}catch{}
      killTimer=setTimeout(()=>{
        try{signalGroup(child,'SIGKILL');}catch{}
        child.stdout?.destroy();child.stderr?.destroy();child.unref();
        finish(Error(why));
      },250);
      // Preserve failure even when TERM leads to a successful exit code.
      failure=why;
    };
    let failure:string|undefined;
    const timer=setTimeout(()=>stop(`query timeout: ${command[0]}`),timeoutMs);
    const take=(which:'stdout'|'stderr',data:Buffer)=>{
      bytes+=data.length;if(bytes>maxBytes){stop(`query output limit: ${command[0]}`);return;}
      if(which==='stdout')stdout+=data.toString();else stderr+=data.toString();
    };
    child.stdout?.on('data',d=>take('stdout',d));child.stderr?.on('data',d=>take('stderr',d));
    child.once('error',e=>finish(e));
    child.once('close',code=>finish(failure?Error(failure):undefined,code));
  });
}

export function journalRows(text: string, bootId: string): {cursor:string; message:string; timestamp:string}[] {
  return text.split('\n').filter(l=>l.trim()).map(line=>{
    const r=JSON.parse(line);
    if(typeof r.__CURSOR!=='string'||!r.__CURSOR||typeof r.MESSAGE!=='string'||typeof r.__REALTIME_TIMESTAMP!=='string'||r._BOOT_ID!==bootId.replaceAll('-',''))throw Error('malformed or wrong-boot kernel journal entry');
    return {cursor:r.__CURSOR,message:r.MESSAGE,timestamp:r.__REALTIME_TIMESTAMP};
  });
}

export interface GuardIO {
  query(command:string[],timeoutMs:number):Promise<QueryResult>;
  bootId():string;
  now():number;
  sleep(ms:number):Promise<void>;
  processGroup(pid:number):number;
  record(event:Record<string,unknown>):void;
  start(command:string[]): {pid:number; ended:Promise<{code:number|null; signal:string|null}>; stop():Promise<void>} | Promise<{pid:number; ended:Promise<{code:number|null; signal:string|null}>; stop():Promise<void>}>;
}

export async function guard(o:Options,io:GuardIO):Promise<Record<string,unknown>> {
  const began=io.now();let fault:string|null=null,cursor='',baseline:GPUBaseline|undefined;
  let child:Awaited<ReturnType<GuardIO['start']>>|undefined,exit:{code:number|null;signal:string|null}|undefined;
  let samples=0,phase='preflight',cancelled=false;
  let loggingFailed=false,stopError:string|undefined;
  const record=(type:string,data:Record<string,unknown>={})=>{
    const entry={type,at:new Date().toISOString(),elapsedMs:io.now()-began,...data};
    try{io.record(entry);}catch(e){loggingFailed=true;fault??=`evidence write failed: ${e}`;console.error(JSON.stringify(entry),String(e));}
  };
  const onSignal=()=>{cancelled=true;if(child)void child.stop().catch(e=>{stopError=String(e);});};
  process.on('SIGINT',onSignal);process.on('SIGTERM',onSignal);
  const checked=async(command:string[])=>{
    const r=await io.query(command,o.queryMs);
    if(r.code!==0)throw Error(`query failed (${r.code}): ${command[0]}: ${r.stderr.slice(0,1000)}`);
    // Permission/partial journal access must not masquerade as no Xids.
    if(r.stderr.trim())throw Error(`query warning: ${command[0]}: ${r.stderr.slice(0,1000)}`);
    return r.stdout;
  };
  const journal=async(initial:boolean)=>{
    const boot=io.bootId();
    if(baseline&&boot!==baseline.bootId)throw Error('BootIdChanged');
    const args=['/usr/bin/journalctl','-k','--boot=0','--no-pager','--output=json','--quiet'];
    if(!initial)args.push(`--after-cursor=${cursor}`);
    const rows=journalRows(await checked(args),boot);
    if(initial&&!rows.length)throw Error('no readable kernel journal; refusing unobserved run');
    for(const r of rows){cursor=r.cursor;const kind=faultFromJournal(r.message);if(kind){record('kernel-fault',{...r,kind});throw Error(`kernel fault: ${kind}: ${r.message}`);}}
    record('journal',{entries:rows.length,cursor,bootId:boot});
  };
  const sample=async()=>{
    const raw=await checked(['/usr/bin/nvidia-smi',`--query-gpu=${gpuFields}`,'--format=csv,noheader,nounits']);
    const all=parseGPUCSV(raw),uuid=baseline?.gpu.uuid??o.uuid;
    const matches=uuid?all.filter(s=>s.uuid===uuid):all;
    if(matches.length!==1){
      const sameSlot=baseline&&all.find(s=>s.pciBusId===baseline.gpu.pciBusId);
      if(sameSlot&&sameSlot.uuid!==baseline!.gpu.uuid)throw Error('UUIDChanged: GPU missing at its original identity');
      throw Error('GPU missing, ambiguous or duplicated; specify --uuid when multiple GPUs exist');
    }
    const gpu=matches[0],boot=io.bootId();
    const base=baseline??{bootId:boot,gpu};
    const error=evaluateHealth(gpu,boot,base,o.maxTemperatureC);
    record('sample',{gpu,bootId:boot,phase});samples++;
    if(error)throw Error(error);
    const processes=parseComputeCSV(await checked(['/usr/bin/nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'])).filter(p=>p.uuid===gpu.uuid);
    record('compute-processes',{processes});
    if(o.runAuthorized){for(const p of processes){if(!child||io.processGroup(p.pid)!==child.pid)throw Error(`competing GPU process ${p.pid}; only owned workload may run`);}}
    baseline=base;
  };
  try {
    record('start',{mode:o.runAuthorized?'workload':'monitor',options:o});
    await journal(true);await sample();await journal(false);
    if(cancelled)throw Error('interrupted before workload');
    if(loggingFailed)throw Error('evidence unavailable before workload');
    if(o.runAuthorized){child=await io.start(o.command);child.ended.then(x=>{exit=x;});record('workload-start',{pid:child.pid});}
    phase='running';const runStart=io.now();
    while(true){
      if(cancelled)throw Error(stopError??'interrupted');
      if(loggingFailed)throw Error('evidence unavailable');
      if(child&&exit)break;
      if(io.now()-runStart>=o.seconds*1000){if(child)throw Error('workload deadline exceeded');break;}
      await sample();await journal(false);
      const delay=io.sleep(o.intervalMs);
      if(child)await Promise.race([delay,child.ended]);else await delay;
    }
    // A fast child may exit between samples. Drain the journal and obtain final
    // health even when it reports success; child exit alone is not a GPU pass.
    phase='final';await journal(false);await sample();await journal(false);
    if(cancelled)throw Error(stopError??'interrupted during final health check');
    if(child&&exit?.code!==0)throw Error(`workload exit code=${exit?.code} signal=${exit?.signal}`);
  } catch(e) {
    fault??=String(e);record('fault',{phase,reason:fault});
  } finally {
    process.off('SIGINT',onSignal);process.off('SIGTERM',onSignal);
    if(child){try{await child.stop();record('workload-group-stopped',{pid:child.pid});}catch(e){fault??=String(e);record('cleanup-fault',{reason:String(e)});}}
  }
  const result={ok:fault===null,fault,mode:o.runAuthorized?'workload':'monitor',samples,baseline,exit,elapsedMs:io.now()-began};
  record('result',result);
  if(loggingFailed){result.ok=false;result.fault=fault??'evidence unavailable';}
  return result;
}

export async function main(args:string[]) {
  if(process.platform!=='linux')throw Error('GPU guard requires Linux process groups and kernel journal');
  const options=parseOptions(args);mkdirSync(options.out,{mode:0o700}); // fail if it exists
  const eventPath=resolve(options.out,'events.jsonl');
  const eventFD=openSync(eventPath,'wx',0o600);
  const handles:number[]=[];
  try{
    const result=await guard(options,{
      query:boundedQuery,bootId:()=>readFileSync('/proc/sys/kernel/random/boot_id','utf8').trim(),
      now:()=>performance.now(),sleep:ms=>Bun.sleep(ms),
      processGroup:pid=>{const stat=readFileSync(`/proc/${pid}/stat`,'utf8');return Number(stat.slice(stat.lastIndexOf(')')+2).split(' ')[2]);},
      record:event=>{
        appendFileSync(eventFD,JSON.stringify(event)+'\n');
        // Evidence is small; flush each event so a crash/reboot does not hide
        // the last observed state behind the filesystem write cache.
        fsyncSync(eventFD);
      },
      start:command=>{
        const out=openSync(resolve(options.out,'workload.stdout'),'wx',0o600),err=openSync(resolve(options.out,'workload.stderr'),'wx',0o600);handles.push(out,err);
        return startOwnedWorkload(command,out,err);
      },
    });
    try{const fd=openSync(resolve(options.out,'result.json'),'wx',0o600);try{writeFileSync(fd,JSON.stringify(result,null,2));fsyncSync(fd);}finally{closeSync(fd);}}
    catch(e){result.ok=false;result.fault=`final evidence write failed: ${e}`;}
    console.log(JSON.stringify(result));return result.ok?0:1;
  } finally {for(const fd of handles)closeSync(fd);closeSync(eventFD);}
}
if(import.meta.main){main(process.argv.slice(2)).then(code=>{process.exitCode=code;}).catch(e=>{console.error(String(e));process.exitCode=1;});}
