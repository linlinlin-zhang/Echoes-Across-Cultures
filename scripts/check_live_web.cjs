// Real deployed pages and APIs. Each run uses isolated anonymous browser sessions.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const base = process.env.ECHO_TEST_URL || 'https://resonance.website';
const out = process.env.ECHO_CHECK_OUTPUT || 'tmp/web-self-check';
fs.mkdirSync(out, { recursive: true });
const report = { base, checks: [], failures: [], pageErrors: [], failedInternalRequests: [], unexpectedAIRequests: [] };
const checked = name => { report.checks.push(name); console.log('PASS', name); };
async function main() {
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_PATH,
    args:['--enable-unsafe-swiftshader','--autoplay-policy=no-user-gesture-required','--mute-audio',
      '--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream']});
  try {
    for (const [mode, viewport] of [['desktop',{width:1440,height:1000}],['mobile',{width:390,height:844}]]) {
      const context = await browser.newContext({viewport,isMobile:mode==='mobile',hasTouch:mode==='mobile'});
      await context.addInitScript(() => {
        if (!localStorage.getItem('echo-language')) localStorage.setItem('echo-language','en');
        if (!localStorage.getItem('echo-app-config')) localStorage.setItem('echo-app-config',JSON.stringify({recommendCount:3,startupTrackMode:'none'}));
      });
      await context.route('**/api/ai/**',async route => {
        if (route.request().method()==='POST') {
          report.unexpectedAIRequests.push(route.request().url());
          await route.fulfill({status:409,json:{detail:'AI provider invocation disabled in automatic self-check'}});
        } else await route.continue();
      });
      const page = await context.newPage();
      page.on('pageerror',e=>report.pageErrors.push({mode,error:e.message}));
      page.on('response',r=>{
        if(r.url().startsWith(base) && r.status()>=400) report.failedInternalRequests.push({mode,status:r.status(),url:r.url()});
      });
      for (const name of ['index','music','favorites','settings']) {
        const url = `${base}/${name}.html${name==='music'?'?seed=itunes_1440838661':''}`;
        const response = await page.goto(url,{waitUntil:'domcontentloaded',timeout:60000});
        assert.equal(response.status(),200);
        await page.waitForFunction(()=>document.body && getComputedStyle(document.body).visibility==='visible');
        await page.waitForTimeout(name==='index'?1500:600);
        assert.equal(await page.locator('html').getAttribute('lang'),'en');
        const size = await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));
        if(size.scroll>size.width+2) report.failures.push(`${mode}/${name}: horizontal overflow ${size.scroll}/${size.width}`);
        if (name==='music') {
          await page.waitForFunction(()=>document.querySelector('#readout-origin')?.textContent.includes('Jacksonville'),{},{timeout:45000});
          await page.waitForFunction(()=>document.querySelectorAll('#map-svg path').length>100,{},{timeout:45000});
          assert.equal(await page.locator('.map-label.is-current').count(),1);
          const marker=await page.locator('.map-label.is-current').evaluate(n=>({x:parseFloat(n.style.left),y:parseFloat(n.style.top)}));
          const rect=await page.locator('#station-map').boundingBox();
          assert(Math.abs(marker.x-rect.width/2)<5 && Math.abs(marker.y-rect.height/2)<5,'Current marker must be centered on its verified point');
          checked(`${mode}: real geography and country boundaries`);
          const cornerOverlap=await page.locator('#map-ui').evaluate(ui=>{
            const overlaps=(a,b)=>{
              const x=ui.querySelector(a).getBoundingClientRect(),y=ui.querySelector(b).getBoundingClientRect();
              return x.left<y.right&&x.right>y.left&&x.top<y.bottom&&x.bottom>y.top;
            };
            return overlaps('.top-left','.top-right')||overlaps('.bottom-left','.bottom-right');
          });
          assert.equal(cornerOverlap,false,'Map captions must have separate space');
          assert.equal(await page.locator('.insights-empty-bar > span').count(),2,'Empty chart captions need separate lines');
          checked(`${mode}: map captions and empty chart layout`);
          if(mode==='mobile') {
            const before=await page.locator('#map-svg path').first().getAttribute('d');
            const client=await context.newCDPSession(page);
            const x=rect.x+rect.width/2,y=rect.y+rect.height/2;
            await client.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x,y}]});
            await client.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:x+45,y:y+25}]});
            await client.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
            await page.waitForTimeout(250);
            assert.notEqual(await page.locator('#map-svg path').first().getAttribute('d'),before);
            await client.detach();checked('mobile: touch map panning');
          }
          if(mode==='desktop') {
            const before=await page.locator('#map-svg path').first().getAttribute('d');
            await page.mouse.move(rect.x+rect.width/2,rect.y+rect.height/2);
            await page.mouse.wheel(0,-120); await page.waitForTimeout(250);
            assert.notEqual(await page.locator('#map-svg path').first().getAttribute('d'),before);
            await page.mouse.down(); await page.mouse.move(rect.x+rect.width/2+60,rect.y+rect.height/2+30,{steps:6});await page.mouse.up();
            checked('map zoom and drag');
            const audio=await page.request.get(`${base}/api/mainline/audio/itunes_1440838661`,{headers:{Range:'bytes=0-4095'},timeout:60000});
            assert.equal(audio.status(),206);assert.equal((await audio.body()).length,4096);
            await page.evaluate(()=>{document.querySelector('#audio-player').muted=true;});
            await page.locator('#play-button').click();
            await page.waitForFunction(()=>document.querySelector('#audio-player').currentTime>0,{},{timeout:30000});
            await page.locator('#play-button').click();
            checked('real audio range, play and pause');
            console.log('RUN real GPU recommendation');
            const recResponse=page.waitForResponse(r=>r.url().endsWith('/api/mainline/recommend')&&r.request().method()==='POST',{timeout:360000});
            await page.locator('#recommend-button').click();
            await page.waitForSelector('#recommend-picker:not([hidden])');
            await page.locator('#recommend-culture-select').selectOption('brazil');
            await page.locator('#recommend-confirm-button').click();
            const rec=await recResponse;assert.equal(rec.status(),200);
            assert.equal(rec.request().postDataJSON().prefer_cuda,true,'The page should prefer an available GPU by default');
            await page.waitForFunction(()=>document.querySelectorAll('.rec-card').length===3,{},{timeout:30000});
            const desc=await page.locator('.rec-description').allTextContents();
            assert(desc.every(s=>s.trim()&&!/[\u3400-\u9fff]/.test(s)),'English mode requires English song descriptions');
            checked('real recommendation and English descriptions');
            const responseJSON=await rec.json();
            assert.equal(rec.request().postDataJSON().target_culture,'brazil');
            assert.equal(responseJSON.request.effective_target_culture,'brazil');
            checked('selected target culture is applied by the recommendation engine');
            fs.writeFileSync(`${out}/recommendation.json`,JSON.stringify(responseJSON,null,2));
            const favoriteResponse=page.waitForResponse(r=>r.url().includes('/api/favorites')&&r.request().method()==='POST',{timeout:15000});
            await page.locator('[data-favorite-index="0"]').click();
            assert.equal((await favoriteResponse).status(),200);
            checked('favorite from a real recommendation');
          }
        }
        if(name==='favorites') {
          await page.waitForFunction(()=>/^\d+$/.test(document.querySelector('#stat-count')?.textContent||''),{},{timeout:60000});
          {
            assert(await page.locator('[data-track-id]').count()>0,'Favorites must load completely');
            const first=page.locator('[data-track-id]').first();
            const tid=await first.getAttribute('data-track-id');
            if(mode==='mobile')await first.locator('[data-remove-id]').tap();
            else await first.locator('[data-remove-id]').click();
            await page.waitForTimeout(500);
            assert.equal(await page.locator(`[data-track-id="${tid}"]`).count(),0);
            checked(`${mode}: favorites render and removal`);
          }
        }
        if(name==='settings') {
          await page.locator('#recommend-count').fill('4');await page.locator('#recommend-count').dispatchEvent('change');
          assert.equal(await page.evaluate(()=>JSON.parse(localStorage.getItem('echo-app-config')).recommendCount),4);
          await page.locator('#app-language').selectOption('de');
          assert.equal(await page.locator('html').getAttribute('lang'),'de');
          await page.reload({waitUntil:'domcontentloaded'});
          assert.equal(await page.locator('html').getAttribute('lang'),'de');
          assert.equal(await page.evaluate(()=>JSON.parse(localStorage.getItem('echo-app-config')).recommendCount),4);
          await page.evaluate(()=>EchoI18n.setLanguage('en'));
          checked(`${mode}: settings and language persistence`);
        }
        await page.screenshot({path:`${out}/${mode}-${name}.png`,fullPage:true});
        checked(`${mode}/${name}: page, layout and assets`);
      }
      await context.close();
    }
    for(const denied of [false,true]) {
      const context=await browser.newContext({permissions:['microphone']});
      await context.addInitScript(denied=>{
        localStorage.setItem('echo-language','en');
        localStorage.setItem('echo-app-config',JSON.stringify({startupTrackMode:'none'}));
        if(denied) navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException('Self-check denied microphone','NotAllowedError');};
      },denied);
      await context.route('**/api/ai/**',async route=>{
        if(route.request().method()==='POST') {
          report.unexpectedAIRequests.push(route.request().url());
          await route.fulfill({status:409,json:{detail:'Self-check disabled provider calls'}});
        }else await route.continue();
      });
      const page=await context.newPage();page.on('pageerror',e=>report.pageErrors.push({mode:'recording',error:e.message}));
      await page.goto(base+'/music.html',{waitUntil:'domcontentloaded'});
      await page.locator('#record-button').click();
      if(denied) {
        await page.waitForFunction(()=>document.querySelector('#map-status').textContent.includes('microphone permissions'));
        assert.equal(await page.locator('#record-button.is-recording').count(),0);
        checked('recording permission failure is visible and recoverable');
      }else {
        await page.waitForSelector('#record-button.is-recording');await page.waitForTimeout(1200);
        await page.locator('#record-button').click();
        await page.waitForFunction(()=>document.querySelector('#audio-player').src.startsWith('blob:'));
        assert.equal(await page.locator('.map-label.is-current').count(),0);
        checked('fake microphone recording, local playback and unlocated map state');
      }
      await context.close();
    }
  } catch(error) {report.failures.push(error.stack||String(error));}
  finally {
    await browser.close();
    fs.writeFileSync(`${out}/report.json`,JSON.stringify(report,null,2));
  }
  assert.deepEqual(report.pageErrors,[],'Runtime page errors');
  assert.deepEqual(report.failedInternalRequests,[],'Failed same-origin requests');
  assert.deepEqual(report.unexpectedAIRequests,[],'Unexpected automatic AI calls');
  assert.deepEqual(report.failures,[],'Live application self-check failures');
}
main().catch(e=>{console.error(e);process.exitCode=1;});
