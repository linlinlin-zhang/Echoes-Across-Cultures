// Real browser cache reuse; AI provider calls are disabled during this check.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const base = process.env.ECHO_TEST_URL || 'https://resonance.website';
const out = process.env.ECHO_CHECK_OUTPUT || 'tmp/performance-20261102';
async function main() {
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_PATH});
  const report = {base, visits:[], unexpectedAI:[]};
  try {
    const context = await browser.newContext();
    await context.addInitScript(()=>{
      localStorage.setItem('echo-language','en');
      localStorage.setItem('echo-app-config',JSON.stringify({startupTrackMode:'none'}));
      window.cacheCheckBlockedAI=[];
      const original=window.fetch.bind(window);
      window.fetch=(input,options={})=>{
        const url=new URL(typeof input==='string'?input:input.url,location.href);
        const method=options.method||input.method||'GET';
        if(url.pathname.startsWith('/api/ai/')&&method.toUpperCase()==='POST') {
          window.cacheCheckBlockedAI.push(url.href);
          return Promise.resolve(new Response(JSON.stringify({detail:'Paid AI disabled during cache check'}),{status:409,headers:{'Content-Type':'application/json'}}));
        }
        return original(input,options);
      };
    });
    const page=await context.newPage();
    const cdp=await context.newCDPSession(page);await cdp.send('Network.enable');
    const sameOrigin=new Set();let visit;
    cdp.on('Network.requestWillBeSent',p=>{if(p.request.url.startsWith(base))sameOrigin.add(p.requestId);});
    cdp.on('Network.loadingFinished',p=>{if(visit&&sameOrigin.has(p.requestId))visit.transferred_bytes+=p.encodedDataLength;});
    cdp.on('Network.requestServedFromCache',p=>{if(visit&&sameOrigin.has(p.requestId))visit.cache_hits++;});
    for(const name of ['cold','reload']) {
      visit={name,transferred_bytes:0,cache_hits:0};const start=Date.now();
      if(name==='cold')await page.goto(base+'/music.html?seed=itunes_1440838661',{waitUntil:'networkidle',timeout:60000});
      else await page.reload({waitUntil:'networkidle',timeout:60000});
      await page.waitForFunction(()=>document.querySelector('#readout-origin')?.textContent.includes('Jacksonville'));
      visit.elapsed_ms=Date.now()-start;report.visits.push({...visit});
      report.unexpectedAI.push(...await page.evaluate(()=>window.cacheCheckBlockedAI));
    }
    const [cold,reload]=report.visits;
    report.repeat_transfer_reduction=1-reload.transferred_bytes/cold.transferred_bytes;
    assert(reload.cache_hits>=5,JSON.stringify(report));
    assert(reload.transferred_bytes<cold.transferred_bytes,JSON.stringify(report));
    assert.deepEqual(report.unexpectedAI,[]);
  } finally {
    fs.mkdirSync(out,{recursive:true});fs.writeFileSync(`${out}/browser-cache.json`,JSON.stringify(report,null,2));
    await browser.close();
  }
  console.log(JSON.stringify(report));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
