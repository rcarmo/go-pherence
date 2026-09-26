import {test,expect} from 'bun:test';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {mkdtempSync,openSync,closeSync,rmSync,readFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {startOwnedWorkload,groupMembers,processIdentity} from './gpu-guard-process';

async function fixture(fn:(out:number,err:number,dir:string)=>Promise<void>){
 const dir=mkdtempSync(join(tmpdir(),'gpu-guard-process-'));
 const out=openSync(join(dir,'out'),'w'),err=openSync(join(dir,'err'),'w');
 try{await fn(out,err,dir);}finally{closeSync(out);closeSync(err);rmSync(dir,{recursive:true,force:true});}
}
test('successful command leaves stable supervisor until explicit stop',async()=>fixture(async(out,err,dir)=>{
 const child=await startOwnedWorkload([process.execPath,'-e','console.log("ok")'],out,err);
 try{expect((await child.ended).code).toBe(0);expect(processIdentity(child.pid)?.pgid).toBe(child.pid);expect(readFileSync(join(dir,'out'),'utf8').trim()).toBe('ok');}
 finally{await child.stop();}expect(groupMembers(child.pid)).toHaveLength(0);await child.stop();
}),5000);
test('nonzero command exit retained',async()=>fixture(async(out,err)=>{
 const child=await startOwnedWorkload([process.execPath,'-e','process.exit(9)'],out,err);
 try{expect((await child.ended).code).toBe(9);}finally{await child.stop();}expect(groupMembers(child.pid)).toHaveLength(0);
}),5000);
test('owned descendants and TERM-ignoring processes are killed as a group',async()=>fixture(async(out,err)=>{
 const script=`const {spawn}=require('child_process');process.on('SIGTERM',()=>{});spawn(process.execPath,['-e','process.on("SIGTERM",()=>{});setInterval(()=>{},1000)'],{stdio:'ignore'});setInterval(()=>{},1000);`;
 const child=await startOwnedWorkload([process.execPath,'-e',script],out,err);
 await Bun.sleep(150);expect(groupMembers(child.pid).length).toBeGreaterThanOrEqual(3);
 await child.stop();expect(groupMembers(child.pid)).toHaveLength(0);
}),5000);
test('supervisor stops owned descendants if its parent pipe disappears',async()=>fixture(async(out,err)=>{
 const leader=spawn(process.execPath,[fileURLToPath(new URL('./gpu-guard-process.ts',import.meta.url)),'--supervisor',JSON.stringify([process.execPath,'-e','setInterval(()=>{},1000)'])],{detached:true,stdio:['pipe','pipe','ignore',out,err]});
 const pid=leader.pid!;
 const exited=new Promise<void>(r=>leader.once('exit',()=>r()));
 await new Promise<void>((r,j)=>{leader.once('error',j);leader.stdout!.once('data',()=>r());});
 expect(groupMembers(pid).length).toBeGreaterThanOrEqual(1);
 leader.stdin!.destroy();
 const gone=await Promise.race([exited.then(()=>true),Bun.sleep(3000).then(()=>false)]);
 expect(gone).toBeTrue();
 for(let i=0;i<20&&groupMembers(pid).length;i++)await Bun.sleep(25);
 expect(groupMembers(pid)).toHaveLength(0);
}),5000);
test('failed workload spawn still permits cleanup',async()=>fixture(async(out,err)=>{
 await expect(startOwnedWorkload(['/no/such/gpu-guard-command'],out,err)).rejects.toThrow('workload did not start');
}),5000);
