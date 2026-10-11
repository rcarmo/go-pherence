import { expect, test } from "bun:test";
import { mkdtempSync, mkdirSync, symlinkSync, rmSync, writeFileSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";

const repo = resolve(import.meta.dir,"..");
const resolver=join(repo,"scripts/project-tmp.sh");
const environment=join(repo,"scripts/project-env.sh");
function invoke(script:string,env:Record<string,string|undefined>={}){
 const child={...process.env,...env};
 for(const key of ["PROJECT_TMP_BASE","PROJECT_TMP_ROOT","PROJECT_ORIGINAL_TMPDIR","RUNNER_TEMP","TMPDIR","CI","GITHUB_ACTIONS","GITLAB_CI","TF_BUILD","CIRCLECI"]){if(env[key]===undefined)delete child[key]}
 return Bun.spawnSync(["bash","-c",script],{cwd:repo,env:child,stdout:"pipe",stderr:"pipe"});
}
function fixture(fn:(dir:string)=>void){const dir=mkdtempSync(join(tmpdir(),"path-policy-"));try{fn(dir)}finally{rmSync(dir,{recursive:true,force:true})}}

test("validated override, unsafe overrides fail without fallback",()=>fixture(dir=>{
 const root=join(dir,"go-pherence");
 const ok=invoke(`source '${resolver}';project_tmp_resolve go-pherence`,{PROJECT_TMP_ROOT:root});expect(ok.exitCode).toBe(0);expect(ok.stdout.toString().trim()).toBe(root);
 for(const value of ["relative/go-pherence",join(dir,"wrong-name"),`${dir}/bad/../go-pherence`]){expect(invoke(`source '${resolver}';project_tmp_resolve go-pherence`,{PROJECT_TMP_ROOT:value}).exitCode).not.toBe(0)}
 expect(invoke(`source '${resolver}'; PROJECT_TMP_ROOT='';project_tmp_resolve go-pherence`).exitCode).not.toBe(0);
 mkdirSync(join(dir,"target"));symlinkSync(join(dir,"target"),root);
 expect(invoke(`source '${resolver}';project_tmp_resolve go-pherence`,{PROJECT_TMP_ROOT:root}).exitCode).not.toBe(0);
}));

test("CI prefers runner/original temp even with workspace; local workspace/system only",()=>fixture(dir=>{
 const absent=join(dir,"not-present/mount/tmp");const runner=join(dir,"runner"),tmp=join(dir,"tmp");mkdirSync(runner);mkdirSync(tmp);
 const call=(env:Record<string,string|undefined>)=>invoke(`source '${resolver}';project_tmp_resolve go-pherence '${absent}'`,{CI:"true",...env});
 expect(call({RUNNER_TEMP:runner,TMPDIR:tmp}).stdout.toString().trim()).toBe(join(runner,"go-pherence"));
 const blocked=join(dir,"blocked");symlinkSync(tmp,blocked);
 expect(call({RUNNER_TEMP:blocked,TMPDIR:tmp}).stdout.toString().trim()).toBe(join(tmp,"go-pherence"));
 expect(call({TMPDIR:tmp}).stdout.toString().trim()).toBe(join(tmp,"go-pherence"));
 expect(call({}).stdout.toString().trim()).toBe("/tmp/go-pherence");
 expect(invoke(`source '${resolver}';project_tmp_resolve go-pherence '${tmp}'`,{CI:"true",RUNNER_TEMP:runner,TMPDIR:tmp}).stdout.toString().trim()).toBe(join(runner,"go-pherence"));
 expect(invoke(`source '${resolver}';project_tmp_resolve go-pherence '${tmp}'`,{RUNNER_TEMP:runner,TMPDIR:runner}).stdout.toString().trim()).toBe(join(tmp,"go-pherence"));
 expect(invoke(`source '${resolver}';project_tmp_resolve go-pherence '${absent}'`,{RUNNER_TEMP:runner,TMPDIR:tmp}).stdout.toString().trim()).toBe("/tmp/go-pherence");
}));

test("environment resolves once, paths propagate, temp roots stay isolated",()=>fixture(dir=>{
 const root=join(dir,"go-pherence");
 const result=invoke(`source '${environment}'; first="$PROJECT_TMP_ROOT"; source '${environment}';test "$first" = "$PROJECT_TMP_ROOT"; printf '%s\\n' "$PROJECT_TMP_ROOT" "$GOCACHE" "$GOMODCACHE" "$TMPDIR" "$GOTMPDIR"; bash -c 'source "${environment}";printf "%s\\n" "$PROJECT_TMP_ROOT"'`,{PROJECT_TMP_ROOT:root,GO_PHERENCE_RUN_PURPOSE:"fixture",GO_PHERENCE_RUN_ID:"isolated"});
 expect(result.exitCode,result.stderr.toString()).toBe(0);
 expect(result.stdout.toString().trim().split("\n")).toEqual([root,join(root,"cache/go-build"),join(root,"cache/go-mod"),join(root,"runs/fixture/isolated/tmp"),join(root,"runs/fixture/isolated/go-tmp"),root]);
 const bad=invoke(`source '${environment}'`,{PROJECT_TMP_ROOT:root,GO_PHERENCE_RUN_ID:"../bad"});expect(bad.exitCode).not.toBe(0);
}));

test("Make command-line override is checked and root propagates to recipes",()=>fixture(dir=>{
 const root=join(dir,"go-pherence");
 const env={...process.env};delete env.PROJECT_TMP_ROOT;delete env.PROJECT_TMP_BASE;delete env.PROJECT_ORIGINAL_TMPDIR;
 const call=(value:string)=>Bun.spawnSync(["make","project-paths",`PROJECT_TMP_ROOT=${value}`,"GO_PHERENCE_RUN_PURPOSE=fixture","GO_PHERENCE_RUN_ID=make"],{cwd:repo,env,stdout:"pipe",stderr:"pipe"});
 const valid=call(root);expect(valid.exitCode,valid.stderr.toString()).toBe(0);expect(valid.stdout.toString()).toContain(`PROJECT_TMP_ROOT=${root}`);
 expect(call("relative/go-pherence").exitCode).not.toBe(0);
 const baseCall=(...vars:string[])=>Bun.spawnSync(["make","project-paths",...vars,"GO_PHERENCE_RUN_PURPOSE=fixture","GO_PHERENCE_RUN_ID=base"],{cwd:repo,env,stdout:"pipe",stderr:"pipe"});
 expect(baseCall(`PROJECT_TMP_BASE=${dir}`).stdout.toString()).toContain(`PROJECT_TMP_ROOT=${root}`);
 expect(baseCall(`PROJECT_TMP_BASE=${dir}`,`PROJECT_TMP_ROOT=${root}`).exitCode).toBe(0);
 expect(baseCall(`PROJECT_TMP_BASE=${dir}`,`PROJECT_TMP_ROOT=${dir}/different/go-pherence`).exitCode).not.toBe(0);
 expect(baseCall("PROJECT_TMP_BASE=relative").exitCode).not.toBe(0);
}));

test("test wrapper uses project scratch, profiling is a pre-release opt-in",()=>fixture(dir=>{
 const root=join(dir,"go-pherence");mkdirSync(root);
 const bin=join(dir,"bin");mkdirSync(bin);
 // Mock the Go CLI: this test checks routing/flags, not generated profiles.
 writeFileSync(join(bin,"go"),`#!/bin/bash
 case "$1" in
 version) echo 'go mock';;
 list) echo 'example.test/package tests';;
 tool) echo 'mock cumulative profile table';;
 test)
  printf 'go arguments:';printf ' %s' "$@";printf '\\n'
  for arg in "$@";do case "$arg" in -cpuprofile=*|-memprofile=*) touch "\${arg#*=}";; esac;done
  while [ "$#" -gt 0 ];do if [ "$1" = -o ];then touch "$2";break;fi;shift;done;;
 esac
 `,{mode:0o700});
 const wrapper=join(repo,"scripts/test-profile.sh");
 const call=(profile:string,output:string)=>invoke(`bash '${wrapper}' ./fixture`,{PROJECT_TMP_ROOT:root,GO_PHERENCE_RUN_PURPOSE:"profile-fixture",GO_PHERENCE_RUN_ID:"one",PROFILE_TESTS:profile,PROFILE_EVIDENCE_ROOT:output,PATH:bin+":"+process.env.PATH});
 const raw=join(root,"runs/profile-fixture/one/profiles");
 const ordinary=call("0",raw);expect(ordinary.exitCode,ordinary.stderr.toString()).toBe(0);expect(ordinary.stdout.toString()).not.toContain("-cpuprofile=");expect(readdirSync(raw)).toEqual([]);
 const profiled=call("1",raw);expect(profiled.exitCode,profiled.stderr.toString()).toBe(0);expect(profiled.stdout.toString()).toContain("-cpuprofile=");expect(readdirSync(raw).some(x=>x.startsWith("invocation-"))).toBe(true);
 expect(call("1",join(dir,"outside")).exitCode).not.toBe(0);
 const link=join(root,"escape");symlinkSync(dir,link);expect(call("1",link+"/profiles").exitCode).not.toBe(0);
}));

test("explicit BASE/ROOT agreement, original temp snapshot never nests",()=>fixture(dir=>{
 const script=`source '${resolver}';project_tmp_resolve go-pherence`;
 expect(invoke(script,{PROJECT_TMP_BASE:dir}).stdout.toString().trim()).toBe(join(dir,"go-pherence"));
 expect(invoke(script,{PROJECT_TMP_BASE:dir,PROJECT_TMP_ROOT:join(dir,"go-pherence")}).exitCode).toBe(0);
 expect(invoke(script,{PROJECT_TMP_BASE:dir,PROJECT_TMP_ROOT:join(dir,"other/go-pherence")}).exitCode).not.toBe(0);
 expect(invoke(script,{PROJECT_TMP_BASE:"relative"}).exitCode).not.toBe(0);
 expect(invoke(`source '${resolver}';PROJECT_TMP_BASE='';project_tmp_resolve go-pherence`).exitCode).not.toBe(0);
 const result=invoke(`source '${environment}';original="$PROJECT_ORIGINAL_TMPDIR";source '${environment}';test "$PROJECT_ORIGINAL_TMPDIR" = "$original";printf '%s\\n' "$PROJECT_TMP_ROOT" "$PROJECT_ORIGINAL_TMPDIR"`,{CI:"true",TMPDIR:dir,GO_PHERENCE_RUN_PURPOSE:"snapshot",GO_PHERENCE_RUN_ID:"fixed"});
 expect(result.exitCode,result.stderr.toString()).toBe(0);expect(result.stdout.toString().trim().split("\n")).toEqual([join(dir,"go-pherence"),dir]);
}));
