// Validate the restored repository target picker, request payloads and fast-only AI. All APIs mocked.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const base=process.env.ECHO_TEST_URL||'http://127.0.0.1:18040';
const out=process.env.ECHO_CHECK_OUTPUT||'tmp/recommendation-targets';
const seed={track_id:'seed',title:'Seed song',artist:'Seed artist',culture:'west',country:'US',city:'Jacksonville',
  lat:30.3,lng:-81.6,location_precision:'artist_city',country_source:'artist_location',source_dataset:'itunes',
  description:'吉他音乐。',description_en:'Guitar music.',label:'摇滚',label_en:'Rock'};
const tracks=['brazil','japan','celtic'].map((culture,i)=>({...seed,culture,track_id:'rec_'+i,title:'Track '+i}));
const cultures=Object.fromEntries(['west','china','southeast_asia','eastern_europe','japan','africa','middle_east',
  'celtic','korea','caribbean','brazil','india','nordic','latin','balkans','central_asia','andean'].map(x=>[x,50]));
async function main(){
  fs.mkdirSync(out,{recursive:true});const report={checks:[],errors:[],paidAI:0};
  const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH,args:['--enable-unsafe-swiftshader']});
  try{
    for(const language of ['en','de','es','pt','ja','ko','zh-CN']){
      const context=await browser.newContext();let calls=[],ai=[],fail=false;
      await context.addInitScript(language=>{
        localStorage.setItem('echo-language',language);
        localStorage.setItem('echo-app-config',JSON.stringify({startupTrackMode:'none',recommendCount:3}));
      },language);
      await context.route('**/api/**',async route=>{
        const req=route.request(),path=new URL(req.url()).pathname;let data={ok:true};
        if(path==='/api/mainline/status')data={ok:true,n_tracks:30000,cultures};
        else if(path.includes('/tracks/')||path.endsWith('/random'))data=seed;
        else if(path==='/api/favorites')data={ok:true,items:[]};
        else if(path.endsWith('/recommend')||path.endsWith('/upload_recommend')){
          calls.push({path,body:req.postData()});
          if(fail)return route.fulfill({status:503,json:{detail:'Simulated queue overload'}});
          data={ok:true,recommendations:tracks,seeds:[seed]};
        }else if(path.startsWith('/api/ai/')&&req.method()==='POST'){
          ai.push(req.postDataJSON());
          return route.fulfill({contentType:'text/event-stream',body:'event: content\ndata: {"delta":"Fast analysis."}\n\nevent: done\ndata: {"ok":true}\n\n'});
        }
        return route.fulfill({json:data});
      });
      const page=await context.newPage();page.on('pageerror',e=>report.errors.push({language,error:e.message}));
      await page.goto(base+'/music.html?seed=seed',{waitUntil:'domcontentloaded'});
      await page.waitForFunction(()=>document.querySelector('#top-title').textContent==='Seed song');
      assert.equal(await page.locator('[data-ai-mode],.ai-mode-toggle').count(),0);
      for(const width of [320,390,768,1440]){
        await page.setViewportSize({width,height:950});
        await page.locator('#recommend-button').click();await page.waitForSelector('#recommend-picker:not([hidden])');
        assert.equal(calls.length,0,'Opening picker must not start computation');
        const box=await page.locator('.recommend-dialog').boundingBox();
        assert(box.x>=0&&box.x+box.width<=width+1&&box.y>=0&&box.y+box.height<=950+1,`${language}/${width}: picker bounds`);
        assert.equal(await page.locator('#recommend-region-select option').count(),10);
        assert.equal(await page.locator('#recommend-culture-select option').count(),18);
        if(['en','de','es','pt'].includes(language))assert(!/[\u3400-\u9fff]/.test(await page.locator('#recommend-picker').innerText()));
        if(language==='en'&&[390,1440].includes(width))await page.locator('#recommend-picker').screenshot({path:`${out}/target-${width}.png`});
        await page.keyboard.press('Escape');assert(await page.locator('#recommend-picker').isHidden());
        report.checks.push(`${language}/${width}: original target dialog and cancel`);
      }
      await page.locator('#recommend-button').click();
      await page.locator('#recommend-region-select').selectOption('east_asia');
      assert.deepEqual(await page.locator('#recommend-culture-select option').evaluateAll(ns=>ns.map(n=>n.value).sort()),['china','japan','korea','random']);
      await page.locator('#recommend-culture-select').selectOption('japan');
      await page.locator('#recommend-confirm-button').click();
      await page.waitForSelector('.rec-card[data-rec-index]');
      assert.equal(JSON.parse(calls.at(-1).body).target_culture,'japan');
      report.checks.push(`${language}: explicit culture and region filtering reach recommendation request`);
      await page.locator('#recommend-button').click();await page.locator('#recommend-culture-select').selectOption('random');
      await page.locator('#recommend-confirm-button').click();await page.waitForFunction(()=>!document.querySelector('#library-picker-button').disabled);
      assert(['china','japan','korea'].includes(JSON.parse(calls.at(-1).body).target_culture));
      report.checks.push(`${language}: region-only choice resolves within its culture pool`);
      await page.locator('#recommend-button').click();await page.locator('#recommend-random-button').click();
      await page.waitForFunction(()=>!document.querySelector('#library-picker-button').disabled);
      assert.notEqual(JSON.parse(calls.at(-1).body).target_culture,'west','Random exploration should cross the seed culture');
      fail=true;await page.locator('#recommend-button').click();await page.locator('#recommend-confirm-button').click();
      await page.waitForFunction(()=>!document.querySelector('#library-picker-button').disabled);
      fail=false;await page.locator('#recommend-button').click();await page.locator('#recommend-confirm-button').click();
      await page.waitForFunction(()=>!document.querySelector('#library-picker-button').disabled);
      report.checks.push(`${language}: random generation and overload recovery`);
      if(language==='en'){
        const bytes=Buffer.alloc(44+8000*2);bytes.write('RIFF');bytes.writeUInt32LE(bytes.length-8,4);bytes.write('WAVEfmt ',8);
        bytes.writeUInt32LE(16,16);bytes.writeUInt16LE(1,20);bytes.writeUInt16LE(1,22);bytes.writeUInt32LE(8000,24);
        bytes.writeUInt32LE(16000,28);bytes.writeUInt16LE(2,32);bytes.writeUInt16LE(16,34);bytes.write('data',36);bytes.writeUInt32LE(bytes.length-44,40);
        await page.locator('#music-upload').setInputFiles({name:'target-check.wav',mimeType:'audio/wav',buffer:bytes});
        await page.waitForFunction(()=>document.querySelector('#audio-player').src.startsWith('blob:'));
        await page.locator('#recommend-button').click();await page.locator('#recommend-region-select').selectOption('latin_america');
        await page.locator('#recommend-culture-select').selectOption('brazil');await page.locator('#recommend-confirm-button').click();
        await page.waitForFunction(()=>!document.querySelector('#library-picker-button').disabled);
        assert(calls.at(-1).path.endsWith('/upload_recommend'));assert(/name="target_culture"\r\n\r\nbrazil/.test(calls.at(-1).body));
        report.checks.push('Upload recommendation retains chosen target culture in multipart form');
      }
      await page.locator('#ai-run-button').click();await page.waitForFunction(()=>!document.querySelector('#ai-run-button').disabled);
      await page.locator('#ai-question').fill('Explain the rhythm.');await page.locator('#ai-send-button').click();
      await page.waitForFunction(()=>!document.querySelector('#ai-send-button').disabled);
      assert.equal(ai.length,2);assert(ai.every(x=>x.thinking_mode==='fast'));
      report.checks.push(`${language}: analysis and follow-up fixed to fast mode`);
      await context.close();
    }
    assert.deepEqual(report.errors,[]);console.log(`PASS ${report.checks.length} restored target / fast-only checks; zero browser errors and paid calls`);
  }finally{fs.writeFileSync(`${out}/report.json`,JSON.stringify(report,null,2));await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
