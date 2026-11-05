// Regression checks for translated chart text and the original repository library picker. No paid APIs.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const base = process.env.ECHO_TEST_URL || 'http://127.0.0.1:18040';
const out = process.env.ECHO_CHECK_OUTPUT || 'tmp/music-layout-check';
const real = process.env.ECHO_REAL_CATALOG === '1';
const track = {track_id:'fixture_br',title:'歌曲原名 / Original title',artist:'Test Artist',culture:'brazil',country:'BR',
  source_dataset:'itunes',label:'流行',label_en:'Pop',description:'轻柔吉他。',description_en:'Gentle guitar.',
  city:'João Pessoa',lat:-7.12,lng:-34.88,location_precision:'artist_city',country_source:'artist_location',
  release_year:2021};
const fixtures = Array.from({length:10}, (_,i)=>({...track,track_id:`fixture_${i}`,release_year:i<5?2015:i<9?2021:2005,
  label:i<2?'Contemporary instrumental music with extended acoustic arrangements':`Style ${i}`,
  label_en:i<2?'Contemporary instrumental music with extended acoustic arrangements':`Style ${i}`}));
async function main() {
  const browser = await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH,
    args:['--enable-unsafe-swiftshader']});
  const report = {base,checks:[],errors:[],unexpectedAI:[]};
  fs.mkdirSync(out,{recursive:true});
  try {
    for (const language of real?['en','pt','zh-CN']:['en','de','es','pt','ja','ko','zh-CN']) {
      const context = await browser.newContext();
      await context.addInitScript(({language,fixtures})=>{
        localStorage.setItem('echo-language',language);
        localStorage.setItem('echo-app-config',JSON.stringify({startupTrackMode:'none'}));
        const items=fixtures.map(item=>({...item,trackId:item.track_id,genre:item.label,genreEn:item.label_en,
          releaseYear:item.release_year,descriptionEn:item.description_en,sourceDataset:item.source_dataset,
          locationPrecision:'artist_city'}));
        localStorage.setItem('echo-music-station-state',JSON.stringify({version:1,
          activeRecommendations:items,currentTrack:items[0],primaryTrack:items[0]}));
      },{language,fixtures});

      await context.route('**/api/**',async route=>{
        const url = new URL(route.request().url());
        if(url.pathname.startsWith('/api/ai/')&&route.request().method()==='POST') {
          report.unexpectedAI.push(url.pathname); return route.fulfill({status:409,json:{detail:'Disabled by self-check'}});
        }
        if(real&&!url.pathname.includes('/tracks/fixture'))return route.continue();
        if(url.pathname.endsWith('/catalog')) {
          const q=url.searchParams.get('q')||'';
          if(q==='failure')return route.fulfill({status:503,json:{detail:'Temporary test failure'}});
          const offset=Number(url.searchParams.get('offset')||0),limit=Number(url.searchParams.get('limit')||24);
          const all=Array.from({length:q==='nothing'?0:60},(_,i)=>({...track,track_id:`catalog_${i}`,title:`${q||'Library song'} ${i}`}));
          const items=all.slice(offset,offset+limit),next=offset+items.length;
          return route.fulfill({json:{ok:true,items,total_available:all.length,offset,next_offset:next,has_more:next<all.length}});
        }
        if(url.pathname.includes('/mainline/status'))return route.fulfill({json:{ok:true,n_tracks:30000}});
        if(url.pathname.includes('/favorites'))return route.fulfill({json:{ok:true,items:[]}});
        if(url.pathname.includes('/tracks/'))return route.fulfill({json:track});
        return route.fulfill({json:{ok:true,has_local_key:false}});
      });
      const page = await context.newPage();
      page.on('pageerror',error=>report.errors.push({language,error:error.message}));
      await page.goto(base+'/music.html',{waitUntil:'domcontentloaded'});
      await page.waitForSelector('.insight-card-title');
      await page.waitForTimeout(350);
      for(const width of [320,390,768,1024,1440]) {
        await page.setViewportSize({width,height:950});
        const failures = await page.evaluate(()=>{
          const result=[];
          for(const node of document.querySelectorAll('.insight-card-title,.era-label,.era-count,.genre-bar-label,.genre-percent')) {
            const r=node.getBoundingClientRect(),card=node.closest('.insight-card').getBoundingClientRect();
            if(node.scrollWidth>node.clientWidth+2||node.scrollHeight>node.clientHeight+2||r.right>card.right+2||r.left<card.left-2)
              result.push({class:node.className,text:node.textContent,width:r.width,scroll:node.scrollWidth});
          }
          if(document.documentElement.scrollWidth>innerWidth+2)result.push('horizontal page overflow');
          return result;
        });
        assert.deepEqual(failures,[],`${language}/${width}: chart text must fit`);
        assert(await page.locator('#library-picker-button').isVisible());
        await page.locator('#library-picker-button').click();
        await page.waitForSelector('.library-result');
        assert(await page.locator('.library-result').count()>0,'Catalog must load');
        const box=await page.locator('.library-dialog').boundingBox();
        assert(box.x>=0&&box.x+box.width<=width+1,'Picker must fit viewport');
        await page.waitForFunction(()=>document.activeElement.id==='library-search-input');
        if(language==='en'&&[390,1440].includes(width))await page.locator('#library-picker').screenshot({path:`${out}/original-library-${width}.png`});
        await page.keyboard.press('Escape');
        assert(await page.locator('#library-picker').isHidden());
        report.checks.push(`${language}/${width}: chart labels and catalog modal fit`);
        if(language==='en'&&[390,1440].includes(width))await page.locator('#recommendation-insights').screenshot({path:`${out}/charts-${width}.png`});
      }
      if(real&&language==='en') {
        await page.locator('#library-picker-button').click();
        const before=await page.locator('.library-result').count();
        await page.locator('#library-results').evaluate(n=>{n.scrollTop=n.scrollHeight;n.dispatchEvent(new Event('scroll'));});
        await page.waitForFunction(before=>document.querySelectorAll('.library-result').length>before,before);
        await page.locator('#library-picker-close').click();
        const first=await (await page.request.get(base+'/api/mainline/catalog?limit=24&offset=0&random_seed=42')).json();
        const second=await (await page.request.get(base+'/api/mainline/catalog?limit=24&offset=24&random_seed=42')).json();
        assert.equal(first.next_offset,24);assert.equal(second.offset,24);
        assert.equal(new Set([...first.items,...second.items].map(x=>x.track_id)).size,48);
        report.checks.push('live: original infinite scroll and non-repeating catalog pagination');
      }
      await page.locator('#library-picker-button').click();
      const input=page.locator('#library-search-input');
      const query=real?'Lynyrd Skynyrd':'Selected song';
      await input.fill(query);await page.locator('#library-search-form').evaluate(form=>form.requestSubmit());
      await page.waitForFunction(q=>document.querySelector('#library-results')?.textContent.toLowerCase().includes(q.toLowerCase()),query);
      const selectedTitle=await page.locator('.library-result-title').first().innerText();
      await page.locator('.library-result').first().click();
      assert(await page.locator('#library-picker').isHidden());
      assert.equal(await page.locator('#top-title').innerText(),selectedTitle,'Selecting library song must update player');
      assert.equal(await page.locator('.rec-card[data-rec-index]').count(),0,'Selecting a new seed must clear old recommendations');
      report.checks.push(`${language}: search and select new recommendation seed`);
      if(real&&language==='en') {
        const audio=await page.locator('#audio-player').getAttribute('src');
        assert(audio.includes('/api/mainline/audio/'));
        const range=await page.request.get(new URL(audio,base).href,{headers:{Range:'bytes=0-4095'}});
        assert.equal(range.status(),206);assert.equal((await range.body()).length,4096);
        const response=page.waitForResponse(r=>r.url().endsWith('/api/mainline/recommend')&&r.request().method()==='POST',{timeout:180000});
        await page.locator('#recommend-button').click();
            await page.waitForSelector('#recommend-picker:not([hidden])');
            await page.locator('#recommend-culture-select').selectOption('brazil');
            await page.locator('#recommend-confirm-button').click();
        const rec=await response;assert.equal(rec.status(),200);
        assert(audio.includes(rec.request().postDataJSON().seed_track_id));
        assert.equal(rec.request().postDataJSON().target_culture,'brazil');
        assert.equal((await rec.json()).request.effective_target_culture,'brazil');
        await page.waitForSelector('.rec-card[data-rec-index]');
        assert(await page.locator('#library-picker-button').isEnabled());
        report.checks.push('live: chosen catalog song audio Range and real GPU recommendations');
      }
      if(!real) {
        await page.locator('#library-picker-button').click();
        await input.fill('browse');await page.locator('#library-search-form').evaluate(form=>form.requestSubmit());
        await page.waitForFunction(()=>document.querySelectorAll('.library-result').length>=24);
        await page.locator('#library-results').evaluate(n=>{n.scrollTop=n.scrollHeight;n.dispatchEvent(new Event('scroll'));});
        await page.waitForFunction(()=>document.querySelectorAll('.library-result').length>=48);
        const ids=await page.locator('.library-result-title').allTextContents();
        assert.equal(new Set(ids).size,ids.length,'Scroll pages must not repeat songs');
        report.checks.push(`${language}: original cover/metadata list and incremental pagination`);
        await page.locator('#library-picker-close').click();
        await page.locator('#library-picker-button').click();await input.fill('nothing');
        await page.locator('#library-search-form').evaluate(form=>form.requestSubmit());
        await page.waitForFunction(()=>document.querySelector('#library-status').textContent===EchoI18n.t('music.library.empty'));
        await input.fill('failure');await page.locator('#library-search-form').evaluate(form=>form.requestSubmit());
        await page.waitForFunction(()=>document.querySelector('#library-status').textContent===EchoI18n.t('music.library.failed'));
        await input.fill('retry');await page.locator('#library-search-form').evaluate(form=>form.requestSubmit());
        await page.waitForSelector('.library-result');
        await page.evaluate(()=>EchoI18n.setLanguage('en'));
        assert.equal(await page.locator('#library-picker-title').innerText(),'Choose From Library');
        await page.locator('#library-picker-close').click();
        report.checks.push(`${language}: empty search, failure recovery and live translation`);
      }
      await context.close();
    }
    assert.deepEqual(report.errors,[]);assert.deepEqual(report.unexpectedAI,[]);
  } finally {await browser.close();fs.writeFileSync(`${out}/report.json`,JSON.stringify(report,null,2));}
  console.log(`PASS ${report.checks.length} catalog/layout checks; zero browser errors or paid AI calls`);
}
main().catch(e=>{console.error(e);process.exitCode=1;});
