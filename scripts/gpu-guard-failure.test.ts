import {test,expect} from 'bun:test';
import {guard,parseOptions,type GuardIO} from './gpu-guard';

function ioForFailure(failType:string):GuardIO & {starts:number;stops:number} {
 let time=0,journals=0,resolveExit!:(x:{code:number|null;signal:string|null})=>void;
 const ended=new Promise<{code:number|null;signal:string|null}>(r=>resolveExit=r);
 const io={starts:0,stops:0,
  query:async(c:string[])=>({code:0,stderr:'',stdout:c[0].endsWith('journalctl')?(journals++===0?JSON.stringify({__CURSOR:'x',MESSAGE:'boot ok',__REALTIME_TIMESTAMP:'1',_BOOT_ID:'aaaa'})+'\n':''):c[1].startsWith('--query-compute')?'':'GPU-aaaa,00000000:00:10.0,610.57.04,40,0,1,12288,20,210,405'}),
  bootId:()=> 'aaaa',now:()=>time,sleep:async(ms:number)=>{time+=ms;},processGroup:()=>123,
  record:(x:Record<string,unknown>)=>{if(x.type===failType)throw Error('simulated disk full');},
  start:()=>{io.starts++;return{pid:123,ended,stop:async()=>{io.stops++;resolveExit({code:null,signal:'SIGTERM'});}}},
 };
 return io;
}
test('failed initial persistence prevents workload',async()=>{
 const io=ioForFailure('start');const r=await guard(parseOptions(['--out','/unused','--run-authorized','--','fake']),io);
 expect(r.ok).toBeFalse();expect(io.starts).toBe(0);expect(String(r.fault)).toContain('evidence write');
});
test('failed persistence after launch stops owned workload',async()=>{
 const io=ioForFailure('workload-start');const r=await guard(parseOptions(['--out','/unused','--run-authorized','--','fake']),io);
 expect(r.ok).toBeFalse();expect(io.starts).toBe(1);expect(io.stops).toBe(1);
});
test('final result write failure is not reported as success',async()=>{
 const io=ioForFailure('result');const r=await guard(parseOptions(['--out','/unused','--seconds','1']),io);
 expect(r.ok).toBeFalse();expect(io.starts).toBe(0);
});
