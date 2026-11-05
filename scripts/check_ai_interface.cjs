// Exercise AI interaction on deployed HTML, replacing provider responses in the browser.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const base = process.env.ECHO_TEST_URL || 'https://resonance.website';
async function main() {
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_PATH});
  const checks = [], requests = [], errors = [];
  try {
    const context = await browser.newContext({viewport:{width:1280,height:900}});
    await context.addInitScript(()=>{
      localStorage.setItem('echo-language','en');
      localStorage.setItem('echo-app-config',JSON.stringify({startupTrackMode:'none'}));
    });
    let fail = false;
    await context.route('**/api/ai/**',async route=>{
      if(route.request().method() !== 'POST') return route.continue();
      assert(route.request().url().endsWith('/chat/stream'),'All provider requests must stay in this mock');
      const request = route.request().postDataJSON();requests.push(request);
      if(fail) return route.fulfill({status:503,json:{detail:'Self-check simulated provider outage'}});
      const event = (name,data)=>`event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
      const body = (request.thinking_mode==='thinking' ? event('reasoning',{delta:'Self-check reasoning.'}) : '') +
        event('content',{delta:'## Self-check music notes\n\nA calm rhythm connects these songs.\n\n<script>window.unsafeSelfCheck=true</script>'}) + event('done',{ok:true});
      return route.fulfill({status:200,contentType:'text/event-stream',body});
    });
    const page = await context.newPage();page.on('pageerror',e=>errors.push(e.message));
    await page.goto(base+'/music.html?seed=itunes_1440838661',{waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>document.querySelector('#readout-origin')?.textContent.includes('Jacksonville'));
    await page.locator('#ai-run-button').click();
    await page.waitForFunction(()=>!document.querySelector('#ai-run-button').disabled&&document.querySelector('.is-assistant .md-body')?.textContent.includes('Self-check music notes'));
    assert.equal(requests.at(-1).thinking_mode,'fast');
    assert.equal(await page.locator('#ai-messages script').count(),0);
    assert.equal(await page.evaluate(()=>Boolean(window.unsafeSelfCheck)),false);
    checks.push('fast streamed response and sanitized Markdown');
    assert.equal(await page.locator('[data-ai-mode],.ai-mode-toggle').count(),0);
    checks.push('mode selector removed; analysis uses fast mode');
    await page.locator('#ai-question').fill('Compare the rhythms.');
    await page.locator('#ai-send-button').click();
    await page.waitForFunction(()=>!document.querySelector('#ai-send-button').disabled&&document.querySelectorAll('.ai-message.is-assistant').length===2);
    assert(requests.at(-1).messages.some(m=>m.role==='assistant'));
    assert.equal(requests.at(-1).messages.at(-1).content,'Compare the rhythms.');
    checks.push('follow-up retains conversation');
    fail = true;
    await page.locator('#ai-question').fill('Retry after a simulated outage.');
    await page.locator('#ai-send-button').click();
    await page.waitForFunction(()=>!document.querySelector('#ai-send-button').disabled&&document.querySelector('.ai-message.is-system')?.textContent.includes('503'));
    checks.push('provider failure is visible and controls recover');
    assert(requests.every(request=>request.thinking_mode==='fast'),'Analysis and follow-up must always use fast mode');
    assert.deepEqual(errors,[]);
    const report = {base,checks,simulated_provider_requests:requests.length,actual_paid_provider_requests:0,page_errors:errors};
    fs.mkdirSync('tmp/web-self-check',{recursive:true});
    fs.writeFileSync('tmp/web-self-check/ai-interface.json',JSON.stringify(report,null,2));
    for(const check of checks) console.log('PASS',check);
    await context.close();
  } finally {await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
