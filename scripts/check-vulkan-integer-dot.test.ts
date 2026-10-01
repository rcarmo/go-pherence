import { test, expect } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { join as pathJoin } from "node:path";
import { tmpdir } from "node:os";
import { verifyIntegerDot } from "./check-vulkan-integer-dot";

test("explicit shader gate refuses overwrite and reports unavailable tools",()=>{
 const dir=mkdtempSync(pathJoin(tmpdir(),"integer-dot-gate-"));const old=[process.env.GLSLC,process.env.SPIRV_VAL,process.env.SPIRV_OPT];try{
 expect(()=>verifyIntegerDot(dir)).toThrow("new report");
 process.env.GLSLC="/missing/glslc";process.env.SPIRV_VAL="/missing/spirv-val";process.env.SPIRV_OPT="/missing/spirv-opt";
 const output=pathJoin(dir,"new");const report=verifyIntegerDot(output);expect(report.pass).toBe(false);expect(report.shaders.length).toBe(4);expect(report.shaders.every(s=>s.fixtureEqual&&s.validate.code!==0)).toBe(true);expect(JSON.parse(readFileSync(pathJoin(output,"verification.json"),"utf8")).pass).toBe(false);
 }finally{for(const [i,key] of ["GLSLC","SPIRV_VAL","SPIRV_OPT"].entries()){if(old[i]===undefined)delete process.env[key];else process.env[key]=old[i]};rmSync(dir,{recursive:true,force:true})}
});
