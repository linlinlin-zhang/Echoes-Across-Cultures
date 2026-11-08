// Real DOM checks for cached multilingual genres, charts and favorites, with all APIs mocked.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict'),fs=require('node:fs');
const base=process.env.ECHO_TEST_URL||'http://127.0.0.1:18040';
const out=process.env.ECHO_CHECK_OUTPUT||'tmp/genre-ui';
const expected={'zh-CN':'摇滚',en:'Rock',de:'Rock',es:'Rock',pt:'Rock',ja:'ロック',ko:'록'};
const seed={track_id:'fixture_rock',title:'A World Without Love',artist:'Peter & Gordon',culture:'west',country:'GB',city:'London',
  lat:51.5072,lng:-0.1276,location_precision:'artist_city',source_dataset:'itunes',label:'록',label_en:'華語流行樂',
  description:'摇滚歌曲。',description_en:'A rock song.',release_year:1964};
async function main(){
 fs.mkdirSync(out,{recursive:true});const report={checks:[],errors:[],paidRequests:[]};
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH,args:['--enable-unsafe-swiftshader']});
 try{
  for(const language of Object.keys(expected)){
   const context=await browser.newContext({viewport:{width:1440,height:1000}});
   await context.addInitScript(({language,seed})=>{
    localStorage.setItem('echo-language',language);localStorage.setItem('echo-app-config',JSON.stringify({startupTrackMode:'none'}));
    const track={...seed,trackId:seed.track_id,genre:seed.label,genreEn:seed.label_en,sourceDataset:'itunes',descriptionEn:seed.description_en,locationPrecision:'artist_city'};
    const items=['록','ロック','Rock'].map((label,i)=>({...track,trackId:'rec_'+i,genre:label,label,raw:{...seed,label,track_id:'rec_'+i}}));
    localStorage.setItem('echo-music-station-state',JSON.stringify({version:1,currentTrack:track,primaryTrack:track,activeRecommendations:items}));
   },{language,seed});
   await context.route('**/api/**',async route=>{
    const path=new URL(route.request().url()).pathname;let body={ok:true};
    if(path.startsWith('/api/ai/')&&route.request().method()==='POST')report.paidRequests.push(path);
    if(path.endsWith('/status'))body={ok:true,n_tracks:30000,cultures:{west:5,japan:5,brazil:5}};
    else if(path.includes('/tracks/')||path.endsWith('/random'))body=seed;
    else if(path.includes('/favorites')||path.includes('/catalog'))body={ok:true,items:[seed],total_available:1};
    await route.fulfill({json:body});
   });
   const page=await context.newPage();page.on('pageerror',e=>report.errors.push({language,error:e.message}));
   await page.goto(base+'/music.html',{waitUntil:'domcontentloaded'});
   await page.waitForSelector('.genre-bar-label');
   assert.equal(await page.locator('#readout-style').innerText(),expected[language]);
   assert.equal(await page.locator('.genre-bar-label').count(),1,'Same genre in three scripts must form one chart group');
   assert.equal(await page.locator('.genre-bar-label').innerText(),expected[language]);
   assert.equal(await page.locator('.genre-percent').innerText(),'100%');
   assert.equal(await page.locator('[data-ai-mode]').count(),0);
   assert.equal(await page.locator('#recommend-picker').count(),0);
   await page.keyboard.press('Escape');
   report.checks.push(`${language}: legacy Korean genre, invalid English field, merged chart group and restored features`);
   if(language==='zh-CN')await page.screenshot({path:`${out}/music-zh.png`,fullPage:true});
   await page.goto(base+'/favorites.html',{waitUntil:'domcontentloaded'});
   await page.waitForSelector('[data-track-id]');
   assert((await page.locator('[data-track-id]').first().textContent()).includes(expected[language]));
   await page.evaluate(()=>EchoI18n.setLanguage('en'));
   assert((await page.locator('[data-track-id]').first().textContent()).includes('Rock'));
   assert(!(await page.locator('[data-track-id]').first().textContent()).includes('華語流行樂'));
   report.checks.push(`${language}: favorite cache and language switch use canonical genre`);
   await context.close();
  }
  assert.deepEqual(report.errors,[]);assert.deepEqual(report.paidRequests,[]);
  console.log(`PASS ${report.checks.length} music/chart/favorite locale checks`);
 }finally{fs.writeFileSync(`${out}/report.json`,JSON.stringify(report,null,2));await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
