#!/usr/bin/env bun
/**
SCRIPT_JDOC:
{
  "summary": "Validate embedded Vulkan SPIR-V and compare normalised GLSL rebuilds offline.",
  "aliases": ["vulkan-static-check"],
  "domains": ["vulkan", "speech"],
  "verbs": ["validate", "compare"],
  "nouns": ["shader", "SPIR-V"],
  "keywords": ["spirv-val", "glslangValidator", "offline"],
  "guidance": ["Use --output with a new report directory. Set SPIRV_VAL and GLSLANG_VALIDATOR to offline tools; no Vulkan device is opened."],
  "examples": ["bun scripts/check-vulkan-shaders.ts --output /tmp/new-vulkan-report"],
  "kind": "mixed",
  "weight": "standard",
  "role": "entrypoint"
}
 */
import { createHash } from 'node:crypto';
import { mkdtempSync, mkdirSync, existsSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, resolve, join } from 'node:path';

const root=resolve(import.meta.dir,'..');
export const shaderNames=['attention_f32','attention_score','conv1d3_f32','gelu_erf_f32','gelu_tanh_mul_f32','gemv_bf16_mixed','gemv_f32','layer_norm_f32','linear_f32','rms_norm_bf16','rms_norm_f32','rms_norm_no_scale_f32','rope_partial_f32','silu_mul_f32','vec_add_bf16','vec_add_f32'];
const sha=(b:Uint8Array|string)=>createHash('sha256').update(b).digest('hex');

export function embeddedShaders(text:string):Map<string,Uint8Array>{
 const result=new Map<string,Uint8Array>();
 for(const match of text.matchAll(/var (spirv_\w+) = \[\]byte\{([\s\S]*?)\n\}/g)){
  const name=match[1].slice(6);
  if(!shaderNames.includes(name)||result.has(name))throw Error(`unexpected/duplicate embedded shader ${name}`);
  const tokens=match[2].split(',').map(s=>s.trim()).filter(Boolean);
  if(tokens.some(s=>!/^0x[0-9a-fA-F]{2}$/.test(s)))throw Error(`invalid byte literal in ${name}`);
  result.set(name,Uint8Array.from(tokens.map(s=>Number(s))));
 }
 if(result.size!==shaderNames.length||shaderNames.some(n=>!result.has(n)))throw Error('incomplete embedded inventory');
 return result;
}

// NOT semantic equivalence or SPIR-V validation. Keep every instruction/operand,
// ID/header field and non-decoration order, except ignore the generator word and
// compare OpDecorate/OpMemberDecorate as a sorted multiset. Both inputs must also
// pass spirv-val externally. Names/source debug records remain part of identity.
export function normalisedWords(input:Uint8Array):Uint8Array{
 if(input.length<20||input.length>1<<20||input.length%4)throw Error('SPIR-V byte bound');
 const dv=new DataView(input.buffer,input.byteOffset,input.byteLength);
 const words=Array.from({length:input.length/4},(_,i)=>dv.getUint32(i*4,true));
 if(words[0]!==0x07230203)throw Error('SPIR-V magic');
 const out=words.slice(0,5),decorations:number[][]=[];out[2]=0;
 for(let i=5;i<words.length;){const count=words[i]>>>16,op=words[i]&65535;
  if(!count||count>words.length-i)throw Error('SPIR-V word framing');
  const instruction=words.slice(i,i+count);
  if(op===71||op===72)decorations.push(instruction);else out.push(...instruction);
  i+=count;
 }
 decorations.sort((a,b)=>{for(let i=0;i<Math.min(a.length,b.length);i++){if(a[i]!==b[i])return a[i]-b[i]}return a.length-b.length});
 // Include lengths so no concatenation can conflate split instructions.
 for(const instruction of decorations)out.push(...instruction);
 const bytes=new Uint8Array(out.length*4),view=new DataView(bytes.buffer);
 out.forEach((w,i)=>view.setUint32(i*4,w,true));return bytes;
}

async function run(command:string,args:string[]){
 const p=Bun.spawn([command,...args],{cwd:root,stdout:'pipe',stderr:'pipe'});
 const [stdout,stderr,status]=await Promise.all([new Response(p.stdout).text(),new Response(p.stderr).text(),p.exited]);
 return {status,stdout,stderr};
}
async function tool(path:string){
 const executable=Bun.which(path);if(!executable)throw Error(`missing offline tool: ${path}`);
 const version=await run(executable,['--version']);if(version.status!==0)throw Error(`cannot query ${path}`);
 return {path:executable,sha256:sha(await Bun.file(executable).bytes()),version:version.stdout.trim()};
}

async function main(){
 const args=process.argv.slice(2);
 if(args.length!==2||args[0]!=='--output')throw Error('usage: bun scripts/check-vulkan-shaders.ts --output <new-report-directory>');
 const output=resolve(args[1]);if(existsSync(output))throw Error('output directory already exists');
 // If a writable new report directory cannot be created, fail without
 // touching an existing one. All later failures produce a failed report.
 mkdirSync(dirname(output),{recursive:true});mkdirSync(output);
 const report:any={created:new Date().toISOString(),target:'vulkan1.3',compilerTarget:'-V -S comp (default Vulkan1.0/SPIR-V1.0)',normalisation:'Generator word=0; sort OpDecorate/OpMemberDecorate multiset; all other words unchanged. Not semantic equivalence.',nativeVulkanOpened:false,shaders:[],limitations:['No driver/shader/model execution or numerical/performance qualification.','spirv-val does not prove race freedom or host shader interface correctness.','Byte rebuild identity is diagnostic; normalised identity and validation are gates.','This check does not clear any existing wrapper holds.']};
 let failed=false,temporary:string|undefined;
 try{
  const validator=await tool(process.env.SPIRV_VAL||'spirv-val');report.validator=validator;
  const compiler=await tool(process.env.GLSLANG_VALIDATOR||'glslangValidator');report.compiler=compiler;
  const goPath=join(root,'backends/vulkan/vulkan_spirv_embedded.go');
  const goText=await Bun.file(goPath).text(),embedded=embeddedShaders(goText);
  report.embeddedSourceSHA256=sha(goText);
  temporary=mkdtempSync(join(tmpdir(),'go-pherence-spirv-'));
  for(const name of shaderNames){
   const bytes=embedded.get(name)!,source=join(root,`backends/vulkan/shaders/${name}.glsl`),stored=await Bun.file(join(root,`backends/vulkan/shaders/${name}.spv`)).bytes();
   const exact=join(temporary,`${name}.embedded.spv`),rebuilt=join(temporary,`${name}.rebuilt.spv`);
   await Bun.write(exact,bytes);
   const validation=await run(validator.path,['--target-env','vulkan1.3',exact]);
   const compilation=await run(compiler.path,['-V','-S','comp',source,'-o',rebuilt]);
   const row:any={name,bytes:bytes.length,embeddedSHA256:sha(bytes),sourceSHA256:sha(await Bun.file(source).bytes()),storedMatchesEmbedded:sha(stored)===sha(bytes),validation,compilation};
   if(compilation.status===0){
    const rebuiltBytes=await Bun.file(rebuilt).bytes();row.rebuiltSHA256=sha(rebuiltBytes);row.byteRebuildMatches=sha(bytes)===sha(rebuiltBytes);
    row.rebuiltValidation=await run(validator.path,['--target-env','vulkan1.3',rebuilt]);
    row.normalisedEmbeddedSHA256=sha(normalisedWords(bytes));row.normalisedRebuiltSHA256=sha(normalisedWords(rebuiltBytes));row.normalisedMatches=row.normalisedEmbeddedSHA256===row.normalisedRebuiltSHA256;
   }
   row.pass=row.storedMatchesEmbedded&&validation.status===0&&compilation.status===0&&row.rebuiltValidation.status===0&&row.normalisedMatches;
   failed ||= !row.pass;report.shaders.push(row);
   console.log(`${row.pass?'PASS':'FAIL'} ${name}: stored=${row.storedMatchesEmbedded} validator=${validation.status} compiler=${compilation.status} rebuiltValidator=${row.rebuiltValidation?.status??'not-run'} byte=${row.byteRebuildMatches??false} normalised=${row.normalisedMatches??false}`);
  }
 }catch(error){failed=true;report.error=String(error)}finally{
  report.pass=!failed;
  try{await Bun.write(join(output,'verification.json'),JSON.stringify(report,null,2)+'\n')}
  finally{if(temporary)rmSync(temporary,{recursive:true,force:true})}
 }
 if(failed)throw Error(`Vulkan static check failed; see ${output}`);
}
if(import.meta.main)await main();
