// Bun test preload: CPU sampling plus heap snapshots at suite boundaries.
// JSC exposes live heaps, not V8 alloc_space; do not mislabel retained objects
// as cumulative allocation evidence. Spawned children need separate capture.
import { afterAll } from "bun:test";
import { Session } from "node:inspector";
import { mkdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { heapStats } from "bun:jsc";

const root = process.env.PROFILE_EVIDENCE_ROOT || (process.env.GITHUB_WORKSPACE ? join(process.env.GITHUB_WORKSPACE,"artifacts/go-pherence-bun-profiles") : "/workspace/exports/go-pherence-bun-profiles");
if (!root.startsWith("/") || (process.env.PROJECT_TMP_ROOT && (root === process.env.PROJECT_TMP_ROOT || root.startsWith(process.env.PROJECT_TMP_ROOT+"/")))) throw new Error("Bun profiles need an absolute retained artifact directory outside disposable root");
const dir=join(root,`bun-${Date.now()}-${process.pid}`);
mkdirSync(dir,{recursive:true});
const session=new Session();session.connect();
const post=(method:string)=>new Promise<any>((ok,no)=>session.post(method as any,{},(err,result)=>err?no(err):ok(result)));
const before=heapStats();
writeFileSync(join(dir,"heap-before.json"),JSON.stringify(before));
await post("Profiler.enable");await post("Profiler.start");
afterAll(async()=>{
 const result=await post("Profiler.stop");writeFileSync(join(dir,"cpu.cpuprofile"),JSON.stringify(result.profile));session.disconnect();
 writeFileSync(join(dir,"heap-after.json"),JSON.stringify(heapStats()));
 writeFileSync(join(dir,"heap.heapsnapshot"),JSON.stringify(Bun.generateHeapSnapshot()));
 writeFileSync(join(dir,"capture.txt"),`Bun ${Bun.version}; cwd=${resolve(".")}; argv=${JSON.stringify(process.argv)}\nCPU samples + JSC live heap. Not cumulative allocations; subprocess activity unprofiled. Engineering analysis required.\n`);
 console.log(`Retained Bun test evidence: ${dir}`);
});
