#!/usr/bin/env bun
import { mkdirSync, writeFileSync, appendFileSync, existsSync } from "node:fs";
import { join, resolve } from "node:path";

const args = new Map<string,string>();
for (let i=2;i<process.argv.length;i++) {
  const a=process.argv[i];
  if (!a.startsWith("--")) continue;
  const [k,v0]=a.slice(2).split("=",2);
  const next=process.argv[i+1];
  const v=v0 ?? (next && !next.startsWith("--") ? next : "true");
  args.set(k,v); if (v0===undefined && next && !next.startsWith("--")) i++;
}
const root=resolve(import.meta.dir,"..");
const stamp=new Date().toISOString().replace(/[:.]/g,"-");
const out=resolve(args.get("out") ?? join(root,"benchmarks","matmul",stamp));
const selected=(args.get("suite") ?? "dense,quant-cpu,gguf").split(",").filter(Boolean);
const procs=(args.get("procs") ?? "1,2,all").split(",");
const benchtime=args.get("benchtime") ?? "500ms";
const count=args.get("count") ?? "5";
const dry=args.get("dry-run")==="true";
mkdirSync(out,{recursive:true});
const jsonl=join(out,"records.jsonl");

function exec(cmd:string[], env:Record<string,string>={}) {
  const p=Bun.spawnSync(cmd,{cwd:root,env:{...process.env,...env},stdout:"pipe",stderr:"pipe"});
  return {code:p.exitCode,stdout:new TextDecoder().decode(p.stdout),stderr:new TextDecoder().decode(p.stderr),command:cmd.join(" ")};
}
function sh(cmd:string[], env:Record<string,string>={}) {
  return dry ? {code:0,stdout:"",stderr:"",command:cmd.join(" ")} : exec(cmd,env);
}
function value(cmd:string[]) { const r=exec(cmd); return r.code===0?r.stdout.trim():`unavailable: ${r.stderr.trim()}`; }
const cpuCount=Number(value(["getconf","_NPROCESSORS_ONLN"]))||1;
const meta={type:"metadata",timestamp:new Date().toISOString(),root,commit:value(["git","rev-parse","HEAD"]),dirty:value(["git","status","--porcelain"]),uname:value(["uname","-a"]),go:value(["go","version"]),lscpu:value(["lscpu"]),cpu_count:cpuCount,governor:existsSync("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")?value(["cat","/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"]):"unavailable",gpu:value(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"]),benchtime,count,selected,procs};
appendFileSync(jsonl,JSON.stringify(meta)+"\n");
writeFileSync(join(out,"metadata.json"),JSON.stringify(meta,null,2)+"\n");

type Suite={pkg:string,bench:string,hardware?:"nvidia"|"spacemit"};
const suites:Record<string,Suite[]>={
  "dense":[{pkg:"./backends/simd/runtime",bench:"Benchmark(Article|BF16Dot|Sdotx4)"}],
  "quant-cpu":[{pkg:"./model",bench:"BenchmarkCPUHot(GemvMLQ|GemmMLX|GemvQ4|GemmNVFP4|GemvNVFP4)"}],
  "gguf":[{pkg:"./loader/gguf",bench:"Benchmark(DotQ|ProjectBatchQ4K)"},{pkg:"./model/diffusiongemma",bench:"BenchmarkGGUF(Q|SelectedExpert)"}],
  "nvidia":[{pkg:"./backends/nvidia/runtime",bench:"Benchmark(Gemm|Gemv|Sgemm|FP8|NVFP4|MLX|Q4)",hardware:"nvidia"}],
  "spacemit":[{pkg:"./backends/spacemit/...",bench:"Benchmark(Gemm|MatMul|K3|Q4K|Q8|Pack|Pool)",hardware:"spacemit"}],
  "models":[{pkg:"./model",bench:"Benchmark(Gemma4MTPGraphCycle|CPUHot)"},{pkg:"./model/ideogram4",bench:"BenchmarkK3"}],
};

for (const name of selected) {
  const cases=suites[name];
  if (!cases) { appendFileSync(jsonl,JSON.stringify({type:"suite",suite:name,status:"skipped",reason:"unknown suite"})+"\n"); continue; }
  for (const c of cases) {
    if (c.hardware==="nvidia" && !meta.gpu.trim()) { appendFileSync(jsonl,JSON.stringify({type:"suite",suite:name,pkg:c.pkg,status:"skipped",reason:"NVIDIA unavailable"})+"\n"); continue; }
    if (c.hardware==="spacemit" && !meta.lscpu.toLowerCase().includes("riscv")) { appendFileSync(jsonl,JSON.stringify({type:"suite",suite:name,pkg:c.pkg,status:"skipped",reason:"riscv/SpacemiT unavailable"})+"\n"); continue; }
    for (const p0 of procs) {
      const p=p0==="all"?String(cpuCount):p0;
      const file=`${name}-${c.pkg.replace(/[^a-z0-9]+/gi,"_")}-p${p}.txt`;
      let cmd=["go","test",c.pkg,"-run","^$","-bench",c.bench,"-benchmem",`-benchtime=${benchtime}`,`-count=${count}`];
      if (c.hardware==="nvidia") cmd=["flock","/tmp/go-pherence-gpu.lock","env",`GOMAXPROCS=${p}`,...cmd];
      else if (existsSync("/usr/bin/taskset")) cmd=["taskset","-c",`0-${Math.max(0,Math.min(cpuCount,Number(p))-1)}`,"env",`GOMAXPROCS=${p}`,...cmd];
      const r=sh(cmd); writeFileSync(join(out,file),r.stdout+r.stderr);
      appendFileSync(jsonl,JSON.stringify({type:"benchmark",suite:name,pkg:c.pkg,benchmark:c.bench,gomaxprocs:Number(p),status:r.code===0?"completed":"failed",exit_code:r.code,command:r.command,output:file})+"\n");
    }
  }
}
console.log(out);
