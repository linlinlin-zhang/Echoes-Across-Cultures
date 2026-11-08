// Direct generation, real score binding and cache refresh regressions; all API responses are mocked.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict'),fs=require('node:fs');
const base=process.env.ECHO_TEST_URL||'http://127.0.0.1:18040';
const out=process.env.ECHO_CHECK_OUTPUT||'tmp/recommendation-metrics';
const seed={track_id:'seed',title:'Seed song',artist:'Seed artist',culture:'west',country:'US',city:'Jacksonville',lat:30.3,lng:-81.6,
  location_precision:'artist_city',source_dataset:'itunes',label:'Rock',description:'摇滚歌曲。',description_en:'A rock song.'};
const rows=[{...seed,track_id:'rec_0',title:'Recommendation 1',culture:'japan',score:.8,score_components:{novelty:.2,relevance:.8,minority:.4}},
 {...seed,track_id:'rec_1',title:'Recommendation 2',culture:'brazil',score:.6,score_components:{novelty:.8,relevance:.6,minority:.2}}];
async function main(){
 fs.mkdirSync(out,{recursive:true});const report={checks:[],errors:[],paidCalls:0};
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH,args:['--enable-unsafe-swiftshader']});
 try{
  for(const language of ['zh-CN','en','de','es','pt','ja','ko']){
   const context=await browser.newContext();let requests=[],ai=[],fail=false;
   await context.addInitScript(language=>{localStorage.setItem('echo-language',language);localStorage.setItem('echo-app-config',JSON.stringify({startupTrackMode:'none',recommendCount:2}));},language);
   await context.route('**/api/**',async route=>{
    const req=route.request(),url=new URL(req.url());let body={ok:true};
    if(url.pathname.endsWith('/status'))body={ok:true,n_tracks:30000,cultures:{west:10,japan:10,brazil:10}};
    else if(url.pathname.includes('/tracks/')){const id=decodeURIComponent(url.pathname.split('/').at(-1));const found=rows.find(x=>x.track_id===id)||seed;const {score,score_components,...metadata}=found;body=metadata;}
    else if(url.pathname.includes('/favorites'))body={ok:true,items:[]};
    else if(url.pathname.endsWith('/recommend')||url.pathname.endsWith('/upload_recommend')){
     requests.push({path:url.pathname,body:req.postData()});
     if(fail)return route.fulfill({status:503,json:{detail:'Simulated overload'}});
     body={ok:true,seeds:[seed],recommendations:rows,algorithm:{weights:{relevance:.48,novelty:.10}},metrics:{style_novelty_mean:.5,listening_relevance_mean:.7}};
    }else if(url.pathname.startsWith('/api/ai/')&&req.method()==='POST'){
     ai.push(req.postDataJSON());return route.fulfill({contentType:'text/event-stream',body:'event: content\ndata: {"delta":"Measured music explanation."}\n\nevent: done\ndata: {"ok":true}\n\n'});
    }
    return route.fulfill({json:body});
   });
   const page=await context.newPage();page.on('pageerror',e=>report.errors.push({language,error:e.message}));
   const novelty=page.locator('[data-metric="styleNovelty"] .parameter-value'),relevance=page.locator('[data-metric="listeningRelevance"] .parameter-value');
   await page.goto(base+'/music.html?seed=seed',{waitUntil:'domcontentloaded'});await page.waitForFunction(()=>document.querySelector('#top-title').textContent==='Seed song');
   assert.equal(await page.locator('#recommend-picker').count(),0);assert.equal(await novelty.innerText(),'—');assert.equal(await relevance.innerText(),'—');
   for(const width of [320,390,768,1440]){
    await page.setViewportSize({width,height:950});const rect=await novelty.boundingBox();assert(rect.width>0);
    const bad=await page.locator('.parameter-metric').evaluateAll(ns=>ns.filter(n=>n.scrollWidth>n.clientWidth+2).map(n=>n.textContent));assert.deepEqual(bad,[]);
   }
   await page.locator('#recommend-button').click();await page.waitForSelector('.rec-card[data-rec-index]');
   assert.equal(requests.length,1);assert.equal(JSON.parse(requests[0].body).target_culture,undefined);
   assert.equal(await novelty.innerText(),'50%');assert.equal(await relevance.innerText(),'70%');
   assert.equal(await page.locator('[data-metric="styleNovelty"]').getAttribute('data-scope'),'mean');
   report.checks.push(`${language}: one click generates directly, no target fields or panel; measured means 50/70`);
   await page.locator('.rec-card[data-rec-index="0"]').click();
   assert.equal(await novelty.innerText(),'20%');assert.equal(await relevance.innerText(),'80%');
   report.checks.push(`${language}: selecting a result displays its 20/80 measured scores`);
   await page.evaluate(()=>history.replaceState(null,'',location.pathname));
   await page.reload({waitUntil:'domcontentloaded'});await page.waitForSelector('.rec-card[data-rec-index]');await page.waitForTimeout(350);
   assert.equal(await novelty.innerText(),'20%');assert.equal(await relevance.innerText(),'80%');
   report.checks.push(`${language}: versioned reload and unscored metadata hydration preserve measurements`);
   if(language==='zh-CN')await page.screenshot({path:`${out}/metrics-zh.png`,fullPage:true});
   await page.locator('#ai-run-button').click();await page.waitForFunction(()=>!document.querySelector('#ai-run-button').disabled);
   assert.equal(ai.length,1);const prompt=JSON.stringify(ai[0].messages);
   assert(prompt.includes('styleNovelty')&&prompt.includes('listeningRelevance'));assert(!prompt.includes('calibrationKl')&&!prompt.includes('targetAffinity'));
   fail=true;await page.locator('#recommend-button').click();await page.waitForFunction(()=>!document.querySelector('#library-picker-button').disabled);
   assert.equal(await novelty.innerText(),'—');assert.equal(await relevance.innerText(),'—');
   assert.equal(await page.locator('.rec-card[data-rec-index]').count(),0);
   fail=false;await page.locator('#recommend-button').click();await page.waitForSelector('.rec-card[data-rec-index]');
   report.checks.push(`${language}: explanation uses new fields and failed generation cannot display fabricated scores`);
   if(language==='en'){
    const bytes=Buffer.alloc(44+8000*2);bytes.write('RIFF');bytes.writeUInt32LE(bytes.length-8,4);bytes.write('WAVEfmt ',8);bytes.writeUInt32LE(16,16);bytes.writeUInt16LE(1,20);bytes.writeUInt16LE(1,22);bytes.writeUInt32LE(8000,24);bytes.writeUInt32LE(16000,28);bytes.writeUInt16LE(2,32);bytes.writeUInt16LE(16,34);bytes.write('data',36);bytes.writeUInt32LE(bytes.length-44,40);
    await page.locator('#music-upload').setInputFiles({name:'measurement.wav',mimeType:'audio/wav',buffer:bytes});await page.waitForFunction(()=>document.querySelector('#audio-player').src.startsWith('blob:'));
    await page.locator('#recommend-button').click();await page.waitForFunction(()=>!document.querySelector('#library-picker-button').disabled);
    assert(requests.at(-1).path.endsWith('/upload_recommend'));assert(!requests.at(-1).body.includes('name="target_culture"'));
    report.checks.push('Uploaded audio also generates directly without sending target culture');
   }
   await context.close();
  }
  const legacy=await browser.newContext();await legacy.addInitScript(seed=>{
   localStorage.setItem('echo-language','en');const old={...seed,trackId:seed.track_id,metrics:{calibrationKl:2.2,targetAffinity:.9,relevance:.99,unexpectedness:.99}};
   localStorage.setItem('echo-music-station-state',JSON.stringify({version:1,currentTrack:old,primaryTrack:old,activeRecommendations:[old]}));
  },seed);await legacy.route('**/api/**',route=>route.fulfill({json:{ok:true,items:[]}}));
  const page=await legacy.newPage();await page.goto(base+'/music.html',{waitUntil:'domcontentloaded'});
  assert.equal(await page.locator('[data-metric="styleNovelty"] .parameter-value').innerText(),'—');
  assert.equal(await page.locator('[data-metric="listeningRelevance"] .parameter-value').innerText(),'—');
  report.checks.push('Legacy synthetic cached values are not relabeled as new measurements');await legacy.close();
  assert.deepEqual(report.errors,[]);console.log(`PASS ${report.checks.length} direct-generation and real-metric UI checks; zero provider calls`);
 }finally{fs.writeFileSync(`${out}/report.json`,JSON.stringify(report,null,2));await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
