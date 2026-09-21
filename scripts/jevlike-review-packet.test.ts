import {test,expect} from 'bun:test';
import {mkdtempSync,rmSync,readFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {createHash} from 'node:crypto';
test('review packet remains unreviewed, deterministic and never overwrites',async()=>{
 const root=mkdtempSync(join(tmpdir(),'review-packet-'));
 try{
  for(const name of ['a','b']){const p=Bun.spawn(['bun',resolve(import.meta.dir,'jevlike-review-packet.ts'),'--output',join(root,name)],{stdout:'pipe',stderr:'pipe'});expect(await p.exited).toBe(0)}
  const a=readFileSync(join(root,'a/cases.jsonl'),'utf8'),b=readFileSync(join(root,'b/cases.jsonl'),'utf8');expect(a).toBe(b)
  const manifest=JSON.parse(readFileSync(join(root,'a/manifest.json'),'utf8'));expect(manifest.human_review_completed).toBe(false);expect(manifest.model_scored).toBe(false);expect(manifest.sha256).toBe(createHash('sha256').update(a).digest('hex'))
  const rows=a.trim().split('\n').map(JSON.parse);expect(rows.length).toBe(24);expect(new Set(rows.map(r=>r.id)).size).toBe(24)
  for(const r of rows){expect(r.review_status).toBe('pending');expect(r.reviewer).toBe(null);expect(r.candidates.some((c:any)=>c.id===r.proposed_gold_id)).toBe(true)}
  const p=Bun.spawn(['bun',resolve(import.meta.dir,'jevlike-review-packet.ts'),'--output',join(root,'a')],{stdout:'pipe',stderr:'pipe'});expect(await p.exited).not.toBe(0)
 }finally{rmSync(root,{recursive:true,force:true})}
});
