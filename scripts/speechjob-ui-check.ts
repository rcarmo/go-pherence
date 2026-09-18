/** SCRIPT_JDOC
 * {"summary":"Exercise the opt-in speech-job browser UI against model-free Go fixtures","domains":["speech","browser","testing"],"kind":"mixed","weight":"standard","role":"entrypoint"}
 */
// Requires Playwright + Chromium installed separately. No trained model/service
// access. Build the HTTP test binary first; all listeners/stores are temporary.
import {mkdtemp,rm,mkdir,readFile,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {resolve,join} from 'node:path';
import {createHash} from 'node:crypto';
const args=process.argv.slice(2),arg=(name:string)=>{const i=args.indexOf(name);if(i<0||!args[i+1])throw Error('required '+name);return args[i+1];};
const binary=resolve(arg('--binary')),out=resolve(arg('--out'));
const modulePath=process.env.PLAYWRIGHT_MODULE||'playwright';
const {chromium}=await import(modulePath);
const root=await mkdtemp(join(tmpdir(),'speechjob-browser-'));await mkdir(out,{recursive:true});
const token='browser-fixture-token-0123456789abcdef';
const child=Bun.spawn([binary,'-test.run=^TestBrowserFixtureServer$','-test.timeout=120s'],{env:{...process.env,SPEECHJOB_BROWSER_STORE:join(root,'jobs'),SPEECHJOB_BROWSER_TOKEN:token},stdout:'pipe',stderr:'pipe'});
let browser:any;const checks:string[]=[];const errors:string[]=[];let closed=false;
const fail=(condition:unknown,message:string)=>{if(!condition)throw Error(message);checks.push(message);console.log('PASS '+message);};
const timeout=setTimeout(()=>{if(!closed)child.kill('SIGKILL');},110000);
try{
 const reader=child.stdout.getReader();let text='';let origin='';
 while(!origin){const {done,value}=await reader.read();if(done)throw Error('fixture ended before ready');text+=new TextDecoder().decode(value);for(const line of text.split('\n')){try{const obj=JSON.parse(line);if(obj.origin)origin=obj.origin;}catch{}}}
 const stdoutDrain=(async()=>{try{for(;;){const {done}=await reader.read();if(done)break;}}finally{reader.releaseLock();}})();
 browser=await chromium.launch({headless:true});const context=await browser.newContext({acceptDownloads:true,viewport:{width:1120,height:900}});const page=await context.newPage();
 page.on('pageerror',(e:Error)=>errors.push(e.message));
 page.on('request',(r:any)=>{if(r.url().includes(token))errors.push('token in URL');if(!r.url().startsWith(origin)&&!r.url().startsWith('blob:'))errors.push('external request');});
 const response=await page.goto(origin+'/ui');fail(response.status()===200,'static UI loads without credentials');
 fail((await response.headerValue('content-security-policy')).includes("script-src 'self'"),'CSP permits only self-hosted scripts');
 fail(await response.headerValue('cache-control')==='no-store','UI assets are not cached');
 await page.locator('#token').fill('wrong-token-01234567890123456789012');await page.locator('#connect').click();await page.waitForFunction(()=>document.querySelector('#notice')!.textContent!.includes('Access rejected'));
 fail(await page.locator('#workspace').isHidden(),'wrong token exposes no job UI');
 async function connect(){await page.locator('#token').fill(token);await page.locator('#connect').click();await page.waitForFunction(()=>document.querySelector('#connection')!.textContent!.startsWith('Connected.'));}
 await connect();fail(await page.locator('#token').inputValue()==='','token input cleared after connect');
 fail(await page.evaluate(()=>localStorage.length===0&&sessionStorage.length===0&&document.cookie===''),'token not persisted in browser storage');
 async function upload(profile:string,name:string){await page.locator('#profile').selectOption(profile);await page.locator('#file').setInputFiles({name,mimeType:'audio/wav',buffer:Buffer.from('synthetic audio')});await page.locator('#upload-button').click();await page.waitForFunction(()=>document.querySelector('#notice')!.textContent!.includes('durably acknowledged'));const id=await page.locator('#job-info dd').first().textContent();if(!/^[a-f0-9]{32}$/.test(id))throw Error('bad uploaded ID');return id;}
 const attack='<img src=x onerror="window.UI_XSS=1">.wav';const id=await upload('asr',attack);
 fail(await page.locator('#job-info').textContent().then((s:string)=>s.includes(attack)),'hostile upload name rendered literally');
 fail(await page.locator('#job-info img').count()===0&&await page.evaluate(()=>!(window as any).UI_XSS),'upload name cannot inject DOM');
 await page.locator('#run').click();await page.waitForFunction(()=>document.querySelector('#notice')!.textContent==='Run completed.');
 fail(await page.getByRole('button',{name:'Download vtt',exact:true}).count()===1,'completed job exposes transcript download');
 let downloadPromise=page.waitForEvent('download');await page.getByRole('button',{name:'Download vtt',exact:true}).click();const download=await downloadPromise;
 const file=join(out,'fixture.vtt');await download.saveAs(file);const bytes=await readFile(file);
 fail(bytes.toString().includes('Olá &lt;&amp;&gt;'),'browser releases verified VTT bytes');
 fail(download.suggestedFilename()===id+'-vtt.vtt','generated download filename uses validated job ID');
 // Corrupt the artifact response while retaining original metadata/ETag: no blob
 // is released until the computed digest matches the status snapshot.
 await page.route('**/artifacts/vtt',async(route:any)=>{const real=await route.fetch();const b=await real.body();const wrong=Buffer.from(b);wrong[0]^=1;await route.fulfill({response:real,body:wrong});});
 let count=0;const onDownload=()=>count++;page.on('download',onDownload);
 await page.getByRole('button',{name:'Download vtt',exact:true}).click();await page.waitForFunction(()=>document.querySelector('#notice')!.textContent!.includes('SHA256 mismatch'));
 fail(count===0,'mismatched SHA256 never creates a browser download');page.off('download',onDownload);await page.unroute('**/artifacts/vtt');
 await page.screenshot({path:join(out,'browser-ui.png'),fullPage:true});
 page.once('dialog',(d:any)=>d.dismiss());await page.locator('#delete').click();fail(await page.locator('#selection').isVisible(),'delete confirmation dismissal preserves job');
 page.once('dialog',(d:any)=>d.accept());await page.locator('#delete').click();await page.waitForFunction(()=>document.querySelector('#notice')!.textContent==='Job deleted.');
 const failed=await upload('fail','failed.wav');await page.locator('#run').click();await page.waitForFunction(()=>document.querySelectorAll('#job-info dd')[2]?.textContent==='failed' && document.querySelector('#run-status')!.textContent!.startsWith('No run'));
 fail(await page.getByRole('button',{name:'Download transcript',exact:true}).count()===1,'later-stage failure preserves transcript download');
 fail(!(await page.locator('body').textContent()).includes('private error must not escape'),'server callback error remains private');
 const slow=await upload('slow','slow.wav');await page.locator('#run').click();await page.waitForFunction(()=>!document.querySelector<HTMLButtonElement>('#cancel')!.disabled);await page.locator('#cancel').click();
 await page.waitForFunction(()=>document.querySelectorAll('#job-info dd')[2]?.textContent==='cancelled' && document.querySelector('#run-status')!.textContent!.startsWith('No run'));
 fail(await page.locator('#run').isEnabled(),'cancelled run becomes explicitly retryable after drain');
 const cancelledState=await context.request.get(origin+'/v1/jobs/'+slow,{headers:{Authorization:'Bearer '+token}});
 fail((await cancelledState.json()).attempts===1,'browser does not replay cancelled POST run');
 // Raw media never gets a browser link and still requires denial at the API.
 fail(await page.locator('#artifacts').getByText('decode',{exact:true}).count()===0,'raw PCM absent from download UI');
 const raw=await context.request.get(origin+'/v1/jobs/'+failed+'/artifacts/input',{headers:{Authorization:'Bearer '+token}});fail(raw.status()===404,'raw source API download denied');
 // Lost upload acknowledgement: fixture stores the job, network drops response;
 // UI must not retry or claim queued success. Inventory allows explicit recovery.
 await page.route('**/v1/jobs?*',async(route:any)=>{if(route.request().method()==='POST'){await route.fetch();await route.abort();}else await route.continue();});
 await page.locator('#profile').selectOption('asr');await page.locator('#file').setInputFiles({name:'uncertain.wav',mimeType:'audio/wav',buffer:Buffer.from('uncertain fixture')});await page.locator('#upload-button').click();await page.waitForFunction(()=>document.querySelector('#notice')!.textContent!.includes('Inspect job status/inventory'));
 await page.unroute('**/v1/jobs?*');await page.locator('#inventory-button').click();await page.waitForFunction(()=>document.querySelectorAll('.inventory-entry').length===3);
 fail(await page.locator('.inventory-entry').count()===3,'lost upload response creates no automatic retry; inventory shows retained job');
 await page.reload();fail(await page.locator('#workspace').isHidden(),'reload clears token and authenticated state');await connect();
 // Auth rotation clears the page instead of leaving a misleading Connected label.
 await page.route('**/v1/jobs',async(route:any)=>route.fulfill({status:401,contentType:'application/json',body:'{"error":"unauthorized"}'}));await page.locator('#refresh').click();await page.waitForFunction(()=>document.querySelector('#connection')!.textContent!.startsWith('Disconnected'));
 fail(await page.locator('#workspace').isHidden(),'API auth rejection clears credentials/session');await page.unroute('**/v1/jobs');
 await connect();await page.locator('#forget').click();fail(await page.locator('#workspace').isHidden(),'explicit forget clears session');
 await page.setViewportSize({width:390,height:844});await page.screenshot({path:join(out,'browser-mobile.png'),fullPage:true});
 fail(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),'mobile layout has no horizontal overflow');
 fail(errors.length===0,'no page errors or external credential-bearing requests');
 await writeFile(join(out,'browser-checks.json'),JSON.stringify({status:'PASS',checks,errors,fixture:'synthetic HTTP stages only',downloadSHA256:createHash('sha256').update(bytes).digest('hex'),browser:await browser.version()},null,2)+'\n');
 await context.close();await browser.close();browser=null;child.kill('SIGTERM');if(await child.exited!==0)throw Error('fixture failed graceful shutdown');await stdoutDrain;
 console.log(JSON.stringify({status:'PASS',checks:checks.length,browserEvidence:out}));
}catch(e){if(browser){const pages=browser.contexts().flatMap((c:any)=>c.pages());if(pages[0]){await writeFile(join(out,'failure-page.txt'),await pages[0].locator('body').innerText());await pages[0].screenshot({path:join(out,'failure.png'),fullPage:true});}}throw e;
}finally{
 clearTimeout(timeout);closed=true;if(browser)await browser.close();if(child.exitCode===null)child.kill('SIGKILL');await child.exited;await rm(root,{recursive:true,force:true});
 const stderr=await new Response(child.stderr).text();await writeFile(join(out,'fixture-stderr.log'),stderr);
}
