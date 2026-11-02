// Run against the local static web server; every API is mocked, so no paid API calls occur.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const base = process.env.ECHO_TEST_URL || 'http://127.0.0.1:18040';
const english = 'Gentle guitar and soft vocals create a relaxed atmosphere for quiet listening.';
const track = {track_id:'fixture_br',title:'A Test Song',artist:'Test Artist',culture:'brazil',country:'BR',
  source_dataset:'itunes',label:'流行',label_en:'Pop',description:'轻柔吉他与人声交织。',description_en:english,
  city:'João Pessoa',lat:-7.12,lng:-34.88,location_precision:'artist_city',country_source:'artist_location',
  city_labels:{en:'João Pessoa', 'zh-CN':'若昂佩索阿'},location_source_url:'https://www.wikidata.org/wiki/Qfixture'};
const old = {...track,trackId:track.track_id,country:'巴西',city:'巴西利亚',lat:-15.7939,lng:-47.8828,
  genre:'流行',descriptionEn:english,locationPrecision:'国家首都坐标',sourceDataset:'itunes'};
async function main() {
  const browser = await chromium.launch({headless:true,
    ...(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {})});
  try {
    for (const language of ['en','de','es','pt','ja','ko','zh-CN']) {
      const context = await browser.newContext({viewport:{width:1440,height:1000}});
      const errors = [], paidRequests = [];
      await context.addInitScript(({language,old}) => {
        if (!localStorage.getItem('echo-test-seeded')) {
          localStorage.setItem('echo-language',language);
          localStorage.setItem('echo-test-seeded','true');
        }
        localStorage.setItem('echo-music-station-state',JSON.stringify({version:1,
          activeRecommendations:[old],currentTrack:old,primaryTrack:old,
          mapStatus:'10条真实连接已激活',mapMode:'推荐网络已展开'}));
        window.flashFrames=[];
        const sample=()=>{
          if(document.body && getComputedStyle(document.body).visibility!=='hidden') {
            const nav=[...document.querySelectorAll('nav [data-i18n]')].map(n=>n.textContent).join(' ');
            if(language==='en' && /[\u3400-\u9fff]/.test(nav))window.flashFrames.push(nav);
          }
          requestAnimationFrame(sample);
        }; requestAnimationFrame(sample);
      },{language,old});
      await context.route('https://cdn.jsdelivr.net/**',route=>route.fulfill({body:'',contentType:'application/javascript'}));
      await context.route('**/api/**', async route => {
        const url=new URL(route.request().url());
        if(route.request().method()==='POST' && /ai/.test(url.pathname))paidRequests.push(url.pathname);
        let body={ok:true};
        if(url.pathname.includes('/mainline/status'))body={ok:true,n_tracks:30000,n_metadata_rows:30000,metadata_coverage:{description_rows:30000}};
        else if(url.pathname.includes('/catalog')||url.pathname.includes('/favorites'))body={ok:true,items:[track],count:1};
        else if(url.pathname.includes('/tracks/')||url.pathname.includes('/random'))body=track;
        else if(url.pathname.includes('/kimi/status'))body={has_local_key:false};
        await route.fulfill({json:body});
      });
      const page = await context.newPage();
      page.on('pageerror',error=>errors.push(error.message));
      for(const name of ['index','music','favorites','settings']) {
        await page.goto(`${base}/${name}.html`,{waitUntil:'domcontentloaded'});
        await page.waitForTimeout(name==='music'?900:200);
        assert.equal(await page.locator('html').getAttribute('lang'),language);
        assert.equal(await page.locator('body').evaluate(n=>getComputedStyle(n).visibility),'visible');
        assert.equal(await page.evaluate(()=>window.flashFrames.length),0,`${name}: untranslated first frame`);
        const labels=await page.locator('[data-i18n]').allTextContents();
        if(['en','de','es','pt'].includes(language))assert(!/[\u3400-\u9fff]/.test(labels.join(' ')),`${name}: Chinese UI label`);
        if(name==='music') {
          assert.equal(await page.locator('.rec-description').first().textContent(),language==='zh-CN'?track.description:english);
          if(['en','de','es','pt'].includes(language))assert(!/[\u3400-\u9fff]/.test(await page.locator('.rec-card').innerText()),'Cached geography must be localized');
          const location=await page.evaluate(()=>({nulls:EchoGeo.coordinates({lat:null,lng:null}),empty:EchoGeo.coordinates({lat:'',lng:''}),
            zero:EchoGeo.coordinates({lat:0,lng:0}),invalid:EchoGeo.coordinates({lat:91,lng:0}),
            iso:EchoGeo.countryCode('巴西'),point:EchoGeo.resolve({country:'DE',source_dataset:'itunes'},{lat:51.5,lng:-0.1})}));
          assert.equal(location.nulls,null); assert.equal(location.empty,null); assert.equal(location.invalid,null);
          assert.deepEqual(location.zero,{lat:0,lng:0});assert.equal(location.iso,'BR');
          assert.equal(location.point.countryCode,'');assert.equal(location.point.precision,'unknown');
          assert.equal(await page.evaluate(()=>EchoGeo.countryCode('XK')),'XK');
          const saved=await page.evaluate(item=>({known:EchoFavoriteStore.normalize(item),
            unknown:EchoFavoriteStore.normalize({trackId:'unknown',sourceDataset:'itunes',locationPrecision:'unknown',lat:0,lng:-160})}),track);
          assert.equal(saved.known.city,track.city);assert.equal(saved.known.lat,track.lat);
          assert.equal(saved.known.location_source_url,track.location_source_url);
          assert.equal(saved.unknown.lat,null);assert.equal(saved.unknown.lng,null);
          assert((await page.locator('#readout-origin').textContent()).includes('João Pessoa') || language==='zh-CN');
          await page.evaluate(()=>EchoI18n.setLanguage('de'));
          assert.equal(await page.locator('.rec-description').first().textContent(),english);
          await page.evaluate(()=>EchoI18n.setLanguage('en'));
          assert(!/[\u3400-\u9fff]/.test(await page.locator('#map-status').innerText()));
          await page.reload({waitUntil:'domcontentloaded'}); await page.waitForTimeout(300);
          assert(!/[\u3400-\u9fff]/.test(await page.locator('#map-status').innerText()));
          await page.evaluate(lang=>EchoI18n.setLanguage(lang),language);
          if(language==='en')await page.screenshot({path:'tmp/i18n-map-fix/music-en.png',fullPage:true});
        }
        // Restore the test language after any in-page language switches.
        await page.evaluate(lang=>EchoI18n.setLanguage(lang),language);
      }
      assert.deepEqual(errors,[],`Browser errors (${language})`);
      assert.deepEqual(paidRequests,[],`Unnecessary AI calls (${language})`);
      console.log(`PASS ${language}: four pages, first paint, cached geography, descriptions, zero paid calls`);
      await context.close();
    }
  } finally { await browser.close(); }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
