// Verify deployed catalog data and the exact song reported by the user. No AI provider calls.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict'),fs=require('node:fs');
const base=process.env.ECHO_TEST_URL||'https://resonance.website';
const out=process.env.ECHO_CHECK_OUTPUT||'tmp/live-genre-check';
async function main(){
 fs.mkdirSync(out,{recursive:true});const report={checks:[],pageErrors:[],paidRequests:[]};
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH,args:['--enable-unsafe-swiftshader']});
 try{
  const context=await browser.newContext({viewport:{width:1440,height:950}});
  await context.addInitScript(()=>{localStorage.setItem('echo-language','zh-CN');localStorage.setItem('echo-app-config',JSON.stringify({startupTrackMode:'none'}));});
  await context.route('**/api/ai/**',async route=>{
   if(route.request().method()==='POST'){report.paidRequests.push(route.request().url());return route.fulfill({status:409,json:{detail:'Disabled during test'}});}
   return route.continue();
  });
  const page=await context.newPage();page.on('pageerror',e=>report.pageErrors.push(e.message));
  const rock=await (await page.request.get(base+'/api/mainline/tracks/itunes_691546555')).json();
  assert.equal(rock.label_original,'록');assert.equal(rock.label,'Rock');assert.equal(rock.label_en,'Rock');assert.equal(rock.genre_labels['zh-CN'],'摇滚');
  report.checks.push('Reported Peter & Gordon track preserves Korean source and exposes Rock / 摇滚');
  const unknown=await (await page.request.get(base+'/api/mainline/tracks/jamendo_2123343')).json();
  assert.equal(unknown.duration_ms,null);assert.equal(unknown.duration_known,false);
  report.checks.push('Known zero source duration is represented as unknown, not a zero-second song');
  const visible=await (await page.request.get(base+'/api/mainline/catalog?q='+encodeURIComponent('广播剧')+'&limit=100')).json();
  const all=await (await page.request.get(base+'/api/mainline/catalog?q='+encodeURIComponent('广播剧')+'&limit=100&exclude_low_signal=false')).json();
  assert.equal(visible.total_available,0);assert.equal(all.total_available,30);assert(all.items.every(x=>x.content_kind==='spoken_drama'));
  report.checks.push('Exactly 30 radio-drama chapters are retained but excluded from default music browsing');
  const search=await (await page.request.get(base+'/api/mainline/catalog?q='+encodeURIComponent('摇滚')+'&limit=100')).json();
  assert(search.items.some(x=>x.label==='Rock'&&/[록ロック]/.test(x.label_original)));
  report.checks.push('Chinese genre search matches original Korean/Japanese Rock labels');
  await page.goto(base+'/music.html?seed=itunes_691546555',{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>document.querySelector('#readout-style').textContent==='摇滚');
  assert.equal(await page.locator('[data-ai-mode]').count(),0);
  await page.locator('#recommend-button').click();assert(await page.locator('#recommend-picker').isVisible());await page.keyboard.press('Escape');
  await page.screenshot({path:`${out}/reported-song-zh.png`,fullPage:true});
  for(const [language,label] of [['en','Rock'],['ja','ロック'],['ko','록'],['zh-CN','摇滚']]){
   await page.evaluate(language=>EchoI18n.setLanguage(language),language);
   assert.equal(await page.locator('#readout-style').textContent(),label);
   report.checks.push(`Live language ${language}: ${label}`);
  }
  assert.deepEqual(report.pageErrors,[]);assert.deepEqual(report.paidRequests,[]);
 }finally{fs.writeFileSync(`${out}/report.json`,JSON.stringify(report,null,2));await browser.close();}
 console.log(`PASS ${report.checks.length} live data/locale checks`);
}
main().catch(e=>{console.error(e);process.exitCode=1;});
