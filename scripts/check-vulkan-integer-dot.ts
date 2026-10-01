/**
 * SCRIPT_JDOC
 * @summary Validate the two explicitly embedded integer-dot shaders and regenerate with modern shaderc.
 * @description Separate optional arithmetic gate; never opens a Vulkan device. Requires a fresh evidence directory.
 * @usage bun scripts/check-vulkan-integer-dot.ts --output <new-directory>
 * @arg --output New report directory (required; never overwritten).
 * @env GLSLC Modern glslc executable (default glslc).
 * @env SPIRV_OPT spirv-opt executable (default spirv-opt).
 * @env SPIRV_VAL spirv-val executable (default spirv-val).
 * @writes <output>/verification.json and regenerated shader artifacts.
 */
import { resolve } from "node:path";
import { mkdirSync, existsSync, writeFileSync, readFileSync } from "node:fs";
import { normalisedWords } from "./check-vulkan-shaders";
function runCaptured(command:string[]){try{const r=Bun.spawnSync(command,{stdout:"pipe",stderr:"pipe"});return {code:r.exitCode,stdout:new TextDecoder().decode(r.stdout),stderr:new TextDecoder().decode(r.stderr)}}catch(e){return {code:127,stdout:"",stderr:String(e)}}}

export function verifyIntegerDot(output:string){
 if(!output||existsSync(output))throw Error("required new report directory");
 const root=resolve(import.meta.dir,"..");output=resolve(output);mkdirSync(output,{recursive:true});
 const results=[];
 for(const name of ["q8","linear"]){
  const source=resolve(root,"backends/vulkan/shaders/integer-dot",name+".glsl"),stored=resolve(root,"backends/vulkan/shaders/integer-dot",name+"-stripped.spv");
  const compiled=resolve(output,name+".spv"),stripped=resolve(output,name+"-stripped.spv");
  const bytes=new Uint8Array(readFileSync(stored));
  const validate=runCaptured([process.env.SPIRV_VAL||"spirv-val","--target-env","vulkan1.3",stored]);
  const compile=runCaptured([process.env.GLSLC||"glslc","--target-env=vulkan1.1","-fshader-stage=compute",source,"-o",compiled]);
  const strip=compile.code===0?runCaptured([process.env.SPIRV_OPT||"spirv-opt","--strip-debug",compiled,"-o",stripped]):{code:127,stdout:"",stderr:"compile failed"};
  const rebuiltValidate=strip.code===0?runCaptured([process.env.SPIRV_VAL||"spirv-val","--target-env","vulkan1.3",stripped]):{code:127,stdout:"",stderr:"strip failed"};
  const rebuilt=existsSync(stripped)?new Uint8Array(readFileSync(stripped)):new Uint8Array();
  const normalised=rebuilt.length>0&&Buffer.from(normalisedWords(bytes)).equals(Buffer.from(normalisedWords(rebuilt)));
  const fixture=resolve(root,"backends/vulkan/testdata/integer-dot",name+"-stripped.spv");
  const fixtureEqual=Buffer.from(bytes).equals(readFileSync(fixture));
  results.push({name,source,stored,validate,compile,strip,rebuiltValidate,normalised,fixtureEqual,pass:validate.code===0&&compile.code===0&&strip.code===0&&rebuiltValidate.code===0&&normalised&&fixtureEqual});
 }
 const report={schema:1,shaders:results,pass:results.every(r=>r.pass)};writeFileSync(resolve(output,"verification.json"),JSON.stringify(report,null,2)+"\n");return report;
}
if(import.meta.main){const at=process.argv.indexOf("--output");if(at<0||!process.argv[at+1])throw Error("--output required");const r=verifyIntegerDot(process.argv[at+1]);for(const s of r.shaders)console.log(`${s.pass?"PASS":"FAIL"} explicit-${s.name}`);if(!r.pass)process.exit(1);}
