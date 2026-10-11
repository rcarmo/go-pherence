// Bun test preload: CPU sampling plus heap snapshots at suite boundaries.
// JSC exposes live heaps, not V8 alloc_space; do not mislabel retained objects
// as cumulative allocation evidence. Spawned children need separate capture.
import { afterAll } from "bun:test";
import { Session } from "node:inspector";
import { mkdirSync, realpathSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { join, resolve } from "node:path";
import { heapStats } from "bun:jsc";

// Use the same portable resolver even when invoked directly by bun test.
const env = spawnSync("bash", ["-c", 'source "$1"; printf "%s\\n%s\\n" "$PROJECT_TMP_ROOT" "$PROJECT_RUN_DIR"', "profile-env", join(import.meta.dir,"project-env.sh")], { env: process.env, encoding:"utf8" });
if (env.status !== 0) throw new Error(`Project profile environment: ${env.stderr}`);
const [projectRoot,runDir] = env.stdout.trim().split("\n");
const root = resolve(process.env.PROFILE_EVIDENCE_ROOT || join(runDir,"profiles"));
if (!root.startsWith(projectRoot+"/")) throw new Error("Bun profile scratch must be inside PROJECT_TMP_ROOT");
mkdirSync(root,{recursive:true});
if (!realpathSync(root).startsWith(realpathSync(projectRoot)+"/")) throw new Error("Bun profile scratch escapes project root");
if (process.env.PROFILE_TESTS === "1") {
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
 console.log(`Bun profile scratch (dispose after analysis): ${dir}`);
});
}
