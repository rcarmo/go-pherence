import { describe, test, expect } from 'bun:test';
import { embeddedShaders, normalisedWords, shaderNames } from './check-vulkan-shaders';

function binary(words:number[]){const bytes=new Uint8Array(words.length*4),view=new DataView(bytes.buffer);words.forEach((w,i)=>view.setUint32(i*4,w,true));return bytes}
const header=[0x07230203,0x10000,77,32,0];
const binding=[4<<16|71,4,33,0],offset=[5<<16|72,5,0,35,0],type=[4<<16|21,8,32,0];
describe('offline SPIR-V comparison',()=>{
 test('ignores only generator and decoration order',()=>{
  const a=binary([...header,...binding,...offset,...type]);
  const b=binary([header[0],header[1],999,header[3],header[4],...type,...offset,...binding]);
  expect(normalisedWords(a)).toEqual(normalisedWords(b));
  const saved=a.slice();normalisedWords(a);expect(a).toEqual(saved);
 });
 test('preserves changed declaration, type and header values',()=>{
  const baseline=normalisedWords(binary([...header,...binding,...offset,...type]));
  for(const words of [
   [...header,...binding.slice(0,-1),1,...offset,...type],
   [...header,...binding,...offset,...type.slice(0,-2),16,0],
   [header[0],header[1],header[2],33,0,...binding,...offset,...type],
   [...header,...binding,...binding,...offset,...type],
  ])expect(normalisedWords(binary(words))).not.toEqual(baseline);
 });
 test('retains non-decoration order',()=>{
  const a=binary([...header,...type,2<<16|19,9]);
  const b=binary([...header,2<<16|19,9,...type]);
  expect(normalisedWords(a)).not.toEqual(normalisedWords(b));
 });
 test('rejects malformed framing',()=>{
  for(const b of [new Uint8Array(),new Uint8Array(21),new Uint8Array((1<<20)+4),binary([0,...header.slice(1)]),binary([...header,71]),binary([...header,4<<16|71,1])])expect(()=>normalisedWords(b)).toThrow();
 });
 test('extracts exact checked-in inventory and rejects duplicate/missing bytes',async()=>{
  const source=await Bun.file(`${import.meta.dir}/../backends/vulkan/vulkan_spirv_embedded.go`).text();
  const values=embeddedShaders(source);expect(values.size).toBe(22);
  for(const name of shaderNames){
   expect(values.get(name)).toEqual(await Bun.file(`${import.meta.dir}/../backends/vulkan/shaders/${name}.spv`).bytes());
   expect(()=>normalisedWords(values.get(name)!)).not.toThrow();
  }
  expect(()=>embeddedShaders('')).toThrow();
  expect(()=>embeddedShaders(source+source)).toThrow();
  expect(()=>embeddedShaders(source.replace('0x03','wrong'))).toThrow();
 });
});

describe('checker failure reporting',()=>{
 test('missing validator fails and writes a failed report',async()=>{
  const {mkdtempSync,rmSync}=await import('node:fs');const {tmpdir}=await import('node:os');const {join}=await import('node:path');
  const root=mkdtempSync(join(tmpdir(),'spirv-check-test-')),output=join(root,'report');
  try{
   const child=Bun.spawn([process.execPath,`${import.meta.dir}/check-vulkan-shaders.ts`,'--output',output],{env:{...process.env,SPIRV_VAL:join(root,'not-installed')},stdout:'pipe',stderr:'pipe'});
   const [status]=await Promise.all([child.exited,new Response(child.stdout).text(),new Response(child.stderr).text()]);
   expect(status).not.toBe(0);
   const report=await Bun.file(join(output,'verification.json')).json();
   expect(report.pass).toBe(false);expect(report.error).toContain('missing offline tool');expect(report.shaders).toEqual([]);
   // A second invocation must not overwrite the first report.
   const original=await Bun.file(join(output,'verification.json')).text();
   const again=Bun.spawn([process.execPath,`${import.meta.dir}/check-vulkan-shaders.ts`,'--output',output],{stdout:'pipe',stderr:'pipe'});
   const [againStatus]=await Promise.all([again.exited,new Response(again.stdout).text(),new Response(again.stderr).text()]);
   expect(againStatus).not.toBe(0);expect(await Bun.file(join(output,'verification.json')).text()).toBe(original);
  }finally{rmSync(root,{recursive:true,force:true})}
 });
});
