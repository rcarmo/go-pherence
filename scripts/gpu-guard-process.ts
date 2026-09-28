import { spawn, type ChildProcess } from 'node:child_process';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export type Exit = { code: number | null; signal: string | null };
export function processIdentity(pid: number) {
  try {
    const s=readFileSync(`/proc/${pid}/stat`,'utf8');
    const f=s.slice(s.lastIndexOf(')')+2).split(' ');
    const pgid=Number(f[2]),session=Number(f[3]);
    if(f.length<20 || !f[0] || !/^\d+$/.test(f[19]) || !Number.isSafeInteger(pgid) || !Number.isSafeInteger(session))throw Error(`malformed process identity ${pid}`);
    return {pid,state:f[0],pgid,session,start:f[19]};
  } catch(e) {
    if(['ENOENT','ESRCH'].includes((e as NodeJS.ErrnoException).code??''))return null;
    throw e; // Missing permission/telemetry is not proof the group is empty.
  }
}
export function groupMembers(pgid:number) {
  return readdirSync('/proc').filter(s=>/^\d+$/.test(s)).map(s=>processIdentity(Number(s)))
    .filter((x):x is NonNullable<typeof x>=>!!x&&x.pgid===pgid&&x.session===pgid&&x.state!=='Z'&&x.state!=='X');
}

// The group leader is a supervisor that deliberately stays alive after the
// actual command exits. It signals its own group; parent code never signals a
// stale/recycled PGID. This contains ordinary descendants, not daemons which
// deliberately escape with setsid; such commands are outside this harness.
export async function startOwnedWorkload(command:string[],stdoutFD:number,stderrFD:number) {
  const leader=spawn(process.execPath,[fileURLToPath(import.meta.url),'--supervisor',JSON.stringify(command)],{
    detached:true,stdio:['pipe','pipe','pipe',stdoutFD,stderrFD],env:process.env,
  });
  if(!leader.pid)throw Error('supervisor failed to start');
  const pid=leader.pid;
  let settled=false,stopping=false,isReady=false,tail='',stderr='',stopPromise:Promise<void>|undefined;
  let resolveExit!:(x:Exit)=>void;
  const ended=new Promise<Exit>(r=>resolveExit=r);
  const finish=(x:Exit)=>{if(!settled){settled=true;resolveExit(x);}};
  let readyResolve!:()=>void,readyReject!:(e:Error)=>void;
  const ready=new Promise<void>((a,b)=>{readyResolve=a;readyReject=b;});
  const timeout=setTimeout(()=>readyReject(Error('supervisor startup timeout')),2000);
  leader.stdout!.on('data',(data:Buffer)=>{
    tail+=data.toString();if(tail.length>65536){readyReject(Error('supervisor output overflow'));return;}
    let idx:number;while((idx=tail.indexOf('\n'))>=0){
      const line=tail.slice(0,idx);tail=tail.slice(idx+1);
      try{const x=JSON.parse(line);if(x.type==='ready'){isReady=true;readyResolve();}
        else if(x.type==='exit'){if(!isReady)readyReject(Error(`workload did not start: ${x.signal}`));finish({code:x.code,signal:x.signal});}
      }catch{readyReject(Error('invalid supervisor protocol'));}
    }
  });
  leader.stderr!.on('data',(d:Buffer)=>{stderr=(stderr+d.toString()).slice(-4096);});
  leader.once('error',e=>{readyReject(e);finish({code:null,signal:String(e)});});
  const leaderExit=new Promise<void>(r=>leader.once('exit',()=>{
    if(!stopping){readyReject(Error(`supervisor exited unexpectedly: ${stderr}`));finish({code:null,signal:'supervisor exited unexpectedly'});}r();
  }));
  const stop=()=>stopPromise??=(async()=>{
    stopping=true;
    // EOF also triggers self-group termination if parent disappears.
    if(leader.stdin&&!leader.stdin.destroyed)leader.stdin.end('stop\n');
    const gone=await Promise.race([leaderExit.then(()=>true),Bun.sleep(3000).then(()=>false)]);
    if(!gone){leader.unref();throw Error('supervisor did not exit; manual inspection required');}
    const deadline=performance.now()+1500;
    while(performance.now()<deadline){if(!groupMembers(pid).length)return;await Bun.sleep(25);}
    throw Error(`owned group ${pid} still has live members; manual inspection required`);
  })();
  try{await ready;}catch(e){await stop().catch(()=>{});throw e;}finally{clearTimeout(timeout);}
  const identity=processIdentity(pid);
  if(!identity||identity.pgid!==pid||identity.session!==pid){await stop();throw Error('supervisor has no dedicated session');}
  return {pid,ended,stop,identity};
}

function supervisor(command:string[]) {
  let stopping=false;let child:ChildProcess|undefined;
  const report=(x:unknown)=>{try{process.stdout.write(JSON.stringify(x)+'\n');}catch{stop();}};
  const stop=()=>{
    if(stopping)return;stopping=true;
    // Remain alive through grace period even after TERM. Our PID/session
    // cannot be reused while we anchor and kill this group ourselves.
    try{process.kill(-process.pid,'SIGTERM');}catch{}
    setTimeout(()=>{try{process.kill(-process.pid,'SIGKILL');}finally{process.exit(1);}},750);
  };
  process.on('SIGTERM',()=>{if(!stopping)stop();});process.on('SIGINT',stop);
  process.stdout.on('error',stop);process.stdin.on('error',stop);
  process.stdin.on('data',stop);process.stdin.on('end',stop);process.stdin.resume();
  setInterval(()=>{},1000);
  try{
    child=spawn(command[0],command.slice(1),{detached:false,stdio:['ignore',3,4],env:process.env});
    child.once('error',e=>report({type:'exit',code:null,signal:String(e)}));
    child.once('exit',(code,signal)=>report({type:'exit',code,signal}));
    child.once('spawn',()=>report({type:'ready'}));
  }catch(e){report({type:'exit',code:null,signal:String(e)});}
}
if(import.meta.main){if(process.argv[2]!=='--supervisor')throw Error('internal supervisor only');supervisor(JSON.parse(process.argv[3]));}
