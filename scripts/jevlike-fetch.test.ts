import { expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { validatePinnedSource, verifyPinnedFile } from "./jevlike-fetch";

test("pinned assets reject unsafe paths and floating revisions", () => {
  const file = {path:"config.json",size:2,sha256:null,git_blob:"a".repeat(40)};
  expect(() => validatePinnedSource({repository:"Qwen/Qwen3-4B-Base",revision:"main",files:[file]})).toThrow();
  for (const path of ["../model/file", "/absolute", "a/../file", "a\\file"]) {
    expect(() => validatePinnedSource({repository:"owner/repo",revision:"b".repeat(40),files:[{...file,path}]})).toThrow();
  }
});
test("files verify LFS SHA256 or exact Git blob identity, not just length", async () => {
  const dir=mkdtempSync(join(tmpdir(),"jevlike-fetch-"));
  try {
    const path=join(dir,"config.json"), data=Buffer.from("{}");writeFileSync(path,data);
    const sha256=createHash("sha256").update(data).digest("hex");
    const git_blob=createHash("sha1").update("blob 2\0").update(data).digest("hex");
    const file={path:"config.json",size:2,sha256,git_blob};
    expect(await verifyPinnedFile(path,file)).toBe(sha256);
    expect(await verifyPinnedFile(path,{...file,sha256:null})).toBe(sha256);
    writeFileSync(path,"[]");
    await expect(verifyPinnedFile(path,file)).rejects.toThrow("checksum");
    await expect(verifyPinnedFile(path,{...file,size:3})).rejects.toThrow("size");
  } finally {rmSync(dir,{recursive:true,force:true});}
});
