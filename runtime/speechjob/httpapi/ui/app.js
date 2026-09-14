'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const ID = /^[a-f0-9]{32}$/;
  const SHA = /^[a-f0-9]{64}$/;
  const MIME = {transcript:'application/json',vtt:'text/vtt','speaker-transcript':'application/json','speaker-vtt':'text/vtt'};
  const STATES = new Set(['queued','running','failed','cancelled','complete']);
  let token = '', epoch = 0, selected = null, next = '', runID = '', uploadBusy = false, downloading = false, uploadLimit = 0;
  const requests = new Set();
  const urls = new Set();
  function text(tag, value, cls) { const el = document.createElement(tag); el.textContent = value; if (cls) el.className = cls; return el; }
  function notice(value) { $('notice').textContent = value; }
  function failure(error, version) { if (version !== epoch) return; notice(error instanceof Error ? error.message : 'Operation failed. Inspect status before retrying.'); }
  function reset() {
    epoch++; token = ''; selected = null; next = ''; runID = ''; uploadBusy = false; downloading = false;
    for (const c of requests) c.abort(); requests.clear();
    for (const u of urls) URL.revokeObjectURL(u); urls.clear();
    $('token').value = ''; $('file').value = ''; $('jobs').replaceChildren(); $('inventory').replaceChildren(); $('artifacts').replaceChildren(); $('profile').replaceChildren();
    $('workspace').hidden = true; $('selection').hidden = true; $('inventory-section').hidden = true;
    $('connection').textContent = 'Disconnected. Token stays in page memory only.';
    $('connect').disabled = false; $('upload-button').disabled = false;
  }
  async function bounded(response, cap) {
    if (response.headers.get('Content-Encoding')) throw new Error('Encoded response rejected.');
    const length = response.headers.get('Content-Length');
    if (length !== null && (!/^\d+$/.test(length) || Number(length) > cap)) throw new Error('Response exceeds allowed size.');
    if (!response.body) return new Uint8Array();
    const reader = response.body.getReader(); const chunks = []; let total = 0;
    try {
      for (;;) { const {value,done} = await reader.read(); if (done) break; total += value.byteLength; if (total > cap) throw new Error('Response exceeds allowed size.'); chunks.push(value); }
    } catch (e) { await reader.cancel().catch(() => {}); throw e; } finally { reader.releaseLock(); }
    const result = new Uint8Array(total); let offset = 0;
    for (const chunk of chunks) { result.set(chunk,offset); offset += chunk.byteLength; }
    return result;
  }
  async function request(path, method, expected, body, consume) {
    if (!token) throw new Error('Connect first.');
    const control = method==='POST' && /^\/v1\/jobs\/[a-f0-9]{32}\/cancel$/.test(path);
    const controlCount = [...requests].filter(c=>c.control).length;
    if (control ? controlCount>=1 : requests.size-controlCount>=6) throw new Error('Response busy: wait for an active request.');
    const version = epoch, controller = new AbortController(); controller.control=control; requests.add(controller);
    try {
      const headers = {Authorization:'Bearer '+token}; if (body) headers['Content-Type'] = 'application/octet-stream';
      const r = await fetch(path,{method,headers,body,signal:controller.signal,credentials:'omit',mode:'same-origin',redirect:'error',cache:'no-store'});
      if (r.status !== expected) {
        await r.body?.cancel();
        if ([401,403,421].includes(r.status) && version===epoch) { reset(); notice('Access rejected (HTTP '+r.status+'). Token cleared; reconnect after checking server credentials/origin.'); }
        throw new Error('HTTP '+r.status+'. Inspect job status/inventory before retrying.');
      }
      const result = await consume(r);
      if (version !== epoch) throw new Error('Session changed.');
      return result;
    } catch (e) {
      if (e instanceof Error && /^(HTTP |Response |Encoded |Invalid |Artifact |Session )/.test(e.message)) throw e;
      throw new Error('Request interrupted or failed. Inspect job status/inventory before retrying.');
    } finally { controller.abort(); requests.delete(controller); }
  }
  function api(path, method='GET', expected=200, body) {
    return request(path,method,expected,body,async r => {
      if (expected===204) return null;
      if (r.headers.get('Content-Type')?.split(';')[0] !== 'application/json') throw new Error('Invalid response type.');
      return JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(await bounded(r,4<<20)));
    });
  }
  function validJob(j) {
    if (!j || !ID.test(j.id) || typeof j.name!=='string' || j.name.length>256 || !STATES.has(j.status) || typeof j.profile_available!=='boolean' || !Number.isSafeInteger(j.attempts) || j.attempts<0 || !Array.isArray(j.artifacts) || j.artifacts.length>4) throw new Error('Invalid job metadata.');
    const seen=new Set();
    for (const a of j.artifacts) { if (!a || !Object.hasOwn(MIME,a.name) || seen.has(a.name) || !SHA.test(a.sha256) || !Number.isSafeInteger(a.bytes) || a.bytes<0 || a.bytes>16<<20) throw new Error('Invalid artifact metadata.'); seen.add(a.name); }
    return j;
  }
  function button(label, action) { const b = text('button',label); b.type='button'; b.addEventListener('click',action); return b; }
  function controls() {
    $('run').disabled = !selected || !selected.profile_available || !!runID || selected.status==='running' || selected.status==='complete';
    $('cancel').disabled = !selected || (selected.status!=='running' && runID!==selected.id);
    $('delete').disabled = !selected || !!runID || selected.status==='running';
    $('upload-button').disabled = uploadBusy || !!runID;
    $('next').disabled = !next;
    for (const b of $('artifacts').querySelectorAll('button')) b.disabled = downloading;
    $('run-status').textContent = runID ? 'Run request active. Use Refresh status to inspect committed stages; keep this page open.' : 'No run request owned by this page.';
  }
  function renderJob(job) {
    selected = validJob(job); $('selection').hidden=false; const dl=$('job-info'); dl.replaceChildren();
    for (const [k,v] of [['ID',job.id],['Name',job.name],['Status',job.status],['Active stage',job.active_stage||'—'],['Attempts',String(job.attempts)],['Profile',job.profile_available ? job.profile : 'Unavailable: configuration changed']]) dl.append(text('dt',k),text('dd',v));
    $('artifacts').replaceChildren();
    for (const a of job.artifacts) $('artifacts').append(button('Download '+a.name,()=>download(job.id,a.name)));
    controls();
  }
  async function select(id) { const version=epoch; try { const j=validJob(await api('/v1/jobs/'+id)); if(j.id!==id)throw new Error('Invalid job identity.'); if (version===epoch) renderJob(j); } catch(e) {failure(e,version);} }
  async function list(after='') {
    const version=epoch; try {
      const result=await api('/v1/jobs'+(after?'?after='+after:''));
      if (!Array.isArray(result.jobs)||result.jobs.length>100||result.next && !ID.test(result.next)) throw new Error('Invalid job page.');
      result.jobs.forEach(validJob); if (version!==epoch) return;
      next=result.next||''; $('jobs').replaceChildren();
      for (const j of result.jobs) {const el=text('article','','job');el.append(text('p',j.name),text('p',j.id,'job-id'),text('p',j.status+(j.active_stage?' · '+j.active_stage:'')),button('Inspect',()=>select(j.id)));el.dataset.jobId=j.id;$('jobs').append(el);}
      $('page-info').textContent=result.jobs.length+' jobs on this page. Pages are live, ordered by random job ID.';controls();
    } catch(e){failure(e,version);}
  }
  async function download(id,name) {
    if (downloading) return; const version=epoch; downloading=true;controls();
    try {
      const j=validJob(await api('/v1/jobs/'+id)); if(j.id!==id)throw new Error('Invalid job identity.'); const a=j.artifacts.find(x=>x.name===name);
      if (!a) throw new Error('Artifact unavailable.');
      const bytes=await request('/v1/jobs/'+id+'/artifacts/'+name,'GET',200,undefined,async r=>{
        if (r.headers.get('Content-Type')?.split(';')[0]!==MIME[name] || r.headers.get('ETag')!=='"sha256-'+a.sha256+'"' || r.headers.get('Content-Length')!==String(a.bytes)) throw new Error('Artifact response metadata mismatch.');
        const data=await bounded(r,16<<20);if(data.byteLength!==a.bytes) throw new Error('Artifact length mismatch.');return data;
      });
      const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
      if(digest!==a.sha256) throw new Error('Artifact SHA256 mismatch; no download created.');
      if(version!==epoch)return;
      // Retain at most one verified blob until the next download or session reset.
      // Do not revoke on a short timer while a browser may still be accepting it.
      for (const old of urls) URL.revokeObjectURL(old); urls.clear();
      const url=URL.createObjectURL(new Blob([bytes],{type:MIME[name]}));urls.add(url);
      const anchor=document.createElement('a');anchor.href=url;anchor.download=id+'-'+name+(name.includes('vtt')?'.vtt':'.json');document.body.append(anchor);anchor.click();anchor.remove();
      notice('Verified '+a.bytes+' bytes; download handed to browser. Experimental labels, if present, remain unqualified.');
    } catch(e){failure(e,version);}finally{if(version===epoch){downloading=false;controls();}}
  }
  async function remove(id) {
    if(!window.confirm('Permanently delete job '+id+' and all its media/checkpoints?')) return;
    const version=epoch;
    try {await api('/v1/jobs/'+id,'DELETE',204);if(version!==epoch)return;if(selected?.id===id){selected=null;$('selection').hidden=true;}notice('Job deleted.');await list();if(!$('inventory-section').hidden)await inventory();}catch(e){failure(e,version);}
  }
  async function inventory(){
    const version=epoch;try{const result=await api('/v1/inventory');if(!Array.isArray(result.jobs)||result.jobs.length>20000)throw new Error('Invalid inventory.');
      for(const j of result.jobs)if(!ID.test(j.id)||!Number.isSafeInteger(j.bytes)||j.bytes<0||typeof j.manifest_present!=='boolean'||typeof j.deleting!=='boolean'||typeof j.corrupt!=='boolean')throw new Error('Invalid inventory entry.');
      if(version!==epoch)return;$('inventory').replaceChildren();$('inventory-section').hidden=false;
      for(const j of result.jobs){const el=text('div','','inventory-entry');el.append(text('p',j.id+' · '+j.bytes+' bytes · '+(j.deleting?'deletion incomplete':j.corrupt?'corrupt':j.manifest_present?'acknowledged':'upload incomplete')),button('Delete retained job',()=>remove(j.id)));$('inventory').append(el);}
    }catch(e){failure(e,version);}
  }
  $('auth').addEventListener('submit',async event=>{
    event.preventDefault();const entered=$('token').value;reset();
    if(!/^[!-~]{32,256}$/.test(entered)){notice('Token must be 32–256 printable non-space ASCII characters.');return;}
    if(!window.isSecureContext||!crypto.subtle){notice('Use HTTPS or a trusted loopback origin with Web Crypto.');return;}
    token=entered;const version=epoch;$('connect').disabled=true;
    try{const result=await api('/v1/profiles');if(!Array.isArray(result.profiles)||result.profiles.length<1||result.profiles.length>32||!result.profiles.every(x=>/^[a-z0-9-]{1,48}$/.test(x))||!Number.isSafeInteger(result.max_upload_bytes)||result.max_upload_bytes<1||result.max_upload_bytes>512<<20)throw new Error('Invalid profile metadata.');
      if(version!==epoch)return;uploadLimit=result.max_upload_bytes;for(const id of result.profiles){const opt=text('option',id);opt.value=id;$('profile').append(opt);}
      $('workspace').hidden=false;$('connection').textContent='Connected. Token held only in this page.';$('upload-limit').textContent='Maximum upload: '+uploadLimit+' bytes (server limits also apply).';notice('Connected. Uploading never starts inference automatically.');await list();
    }catch(e){if(version===epoch){reset();failure(e,epoch);}}finally{if(version===epoch)$('connect').disabled=false;}
  });
  $('forget').addEventListener('click',()=>{reset();notice('Token forgotten and page requests aborted. Server callbacks may still be draining; inspect status after reconnecting.');});
  $('upload').addEventListener('submit',async event=>{
    event.preventDefault();if(uploadBusy||runID)return;const file=$('file').files[0];if(!file||file.size<1||file.size>uploadLimit){notice('Choose a nonempty recording within the upload limit.');return;}
    const profile=$('profile').value,version=epoch;uploadBusy=true;controls();
    try{const q=new URLSearchParams({profile,name:file.name});const j=validJob(await api('/v1/jobs?'+q,'POST',201,file));if(j.status!=='queued')throw new Error('Invalid upload acknowledgement; inspect inventory.');if(version!==epoch)return;$('file').value='';renderJob(j);notice('Upload durably acknowledged. Select Run to start.');await list();}catch(e){failure(e,version);}finally{if(version===epoch){uploadBusy=false;controls();}}
  });
  $('run').addEventListener('click',async()=>{
    if(!selected||runID)return;const id=selected.id,version=epoch;runID=id;controls();notice('Run started. Closing this page can cancel it.');
    try{const j=validJob(await api('/v1/jobs/'+id+'/run','POST',200));if(j.id!==id||j.status!=='complete')throw new Error('Invalid run acknowledgement; inspect status.');if(version!==epoch)return;if(selected?.id===id)renderJob(j);notice('Run completed.');}
    catch(e){failure(e,version);if(version===epoch&&selected?.id===id)await select(id);}
    finally{if(version===epoch){runID='';controls();await list();}}
  });
  $('cancel').addEventListener('click',async()=>{if(!selected)return;const id=selected.id,version=epoch;try{const r=await api('/v1/jobs/'+id+'/cancel','POST',202);if(!r.cancellation_requested||r.id!==id)throw new Error('Invalid cancellation acknowledgement.');if(version===epoch)notice('Cancellation requested, not yet confirmed. Refresh status to check drain.');}catch(e){failure(e,version);}});
  $('status').addEventListener('click',()=>{if(selected)select(selected.id);});
  $('delete').addEventListener('click',()=>{if(selected)remove(selected.id);});
  $('refresh').addEventListener('click',()=>list());$('next').addEventListener('click',()=>{if(next)list(next);});$('inventory-button').addEventListener('click',inventory);
  window.addEventListener('beforeunload',e=>{if(runID||uploadBusy){e.preventDefault();e.returnValue='';}});
  window.addEventListener('pagehide',reset);
})();
