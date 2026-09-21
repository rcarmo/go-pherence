#!/usr/bin/env bun
/** Write bounded fresh development cases for HUMAN review, never auto-approve. */
import {createHash} from 'node:crypto';
import {mkdirSync,writeFileSync,existsSync} from 'node:fs';
import {parseArgs} from 'node:util';
import {resolve} from 'node:path';
const {values}=parseArgs({args:Bun.argv.slice(2),options:{output:{type:'string'}},strict:true});
if(!values.output)throw Error('--output required');const out=resolve(values.output);if(existsSync(out))throw Error('output exists');
const locations=['blue drawer','green cabinet','silver locker','yellow crate'];
const names=['Mara','Ivo','Nora','Teo'];const objects=['amber key','copper coin','violet card','wooden disk'];
const cases:any[]=[];
for(let i=0;i<4;i++){
 const who=names[i],object=objects[i],place=locations[i],other=locations[(i+1)%4];
 const choices=[{id:'first',text:`The ${object} is in the ${place}.`},{id:'second',text:`The ${object} is in the ${other}.`},{id:'unknown',text:`The ${object}'s location is not determined by the evidence.`}];
 const add=(kind:string,evidence:string,question:string,candidates:any[],gold:string)=>cases.push({id:`fresh-review-v1-${i+1}-${kind}`,family:`entity-location-${i+1}`,kind,partition:'fresh-development-pending-human-review',candidate_contract:'self-contained-v1',review_status:'pending',reviewer:null,reviewed_at:null,proposed_gold_id:gold,evidence,question,candidates,temperature:1});
 add('stated',`${who} put the ${object} in the ${place}. The ${other} is empty.`,`Where is the ${object}?`,choices,'first');
 add('changed-fact',`${who} put the ${object} in the ${other}. The ${place} is empty.`,`Where is the ${object}?`,choices,'second');
 add('negation',`${who} did not put the ${object} in the ${place}. No location for the ${object} was reported.`,`Where is the ${object}?`,choices,'unknown');
 add('ambiguous',`The ${object} is either in the ${place} or in the ${other}. Nobody has checked which.`,`Which location is established for the ${object}?`,choices,'unknown');
 add('missing-answer',`${who} put the ${object} in a black bag. It is not in the ${place} or the ${other}.`,`Which alternative matches the stated location of the ${object}?`,[{id:'first',text:choices[0].text},{id:'second',text:choices[1].text},{id:'neither',text:`The ${object} is in neither the ${place} nor the ${other}.`}],'neither');
 add('two-choice',`${who} inspected both containers: the ${object} was in the ${place}, not the ${other}.`,`Which statement matches the inspection?`,choices.slice(0,2),'first');
}
mkdirSync(out,{recursive:true});const data=cases.map(x=>JSON.stringify(x)).join('\n')+'\n';writeFileSync(out+'/cases.jsonl',data);
const sha=createHash('sha256').update(data).digest('hex');writeFileSync(out+'/manifest.json',JSON.stringify({version:1,status:'pending-human-review',cases:cases.length,sha256:sha,authorship:'machine-authored synthetic mechanical probe packet',human_review_completed:false,model_scored:false,partition:'fresh-development-pending-human-review',limitations:['24 cases, not the original 200–500 target','six related templates with four entity substitutions; not broad independent-domain coverage','all descendants share family; never split for tuning versus evaluation','proposed labels are NOT verified ground truth','not a final-test replacement']},null,2)+'\n');
let md='## Fresh development cases -- pending human review\n\nThese 24 machine-authored cases have **not** been scored or approved. Please check each proposed label, wording and whether every candidate is understandable in isolation. Record corrections by ID, plus reviewer name/date. Related variants belong to the same family; this small mechanics packet does not satisfy the original broader 200–500-case target.\n\n';
for(const c of cases){md+=`### ${c.id}\n\nEvidence: ${c.evidence}\n\nQuestion: ${c.question}\n\n`;for(const a of c.candidates)md+=`* \`${a.id}\`: ${a.text}\n`;md+=`\nProposed answer: \`${c.proposed_gold_id}\`. Human decision: **pending**.\n\n`}
writeFileSync(out+'/review.md',md);console.log(JSON.stringify({out,cases:cases.length,sha256:sha,status:'pending-human-review'}));
