const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict');
const base=process.env.ECHO_TEST_URL||'http://127.0.0.1:18040';
const description='Soft instrumental music with a steady pulse.';
const make=(id,lat,lng)=>({track_id:id,title:id,artist:'Map fixture',country:'FJ',city:'Fixture city',lat,lng,
  country_source:'artist_location',source_dataset:'itunes',location_precision:'artist_city',
  description_en:description,label_en:'Pop',location_source_url:'https://www.wikidata.org/wiki/Qfixture'});
const cached=t=>({...t,raw:t,trackId:t.track_id,descriptionEn:t.description_en,sourceDataset:t.source_dataset,
  locationPrecision:t.location_precision,locationSourceUrl:t.location_source_url});
async function main(){
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH});
 try{
  const east=make('east',-17,179.5),west=make('west',-17,-179.5);
  const cases=[
   {name:'date line and duplicate markers',current:east,recs:[west,make('duplicate',-17,179.5)],markers:2,close:true},
   {name:'unknown current preserves known recommendations',current:{...make('unknown',null,null),location_precision:'unknown'},recs:[east,west],markers:2},
   {name:'zero coordinates remain valid',current:make('zero',0,0),recs:[],markers:1},
   {name:'invalid and old placeholder coordinates stay unlocated',current:{...make('invalid',999,114),location_precision:'unknown'},
    recs:[{...make('placeholder',0,-160),location_precision:'unknown'}],markers:0}
  ];
  for(const fixture of cases){
   const context=await browser.newContext({viewport:{width:1440,height:1000}});const errors=[];
   await context.addInitScript(f=>{
    localStorage.setItem('echo-language','en');
    localStorage.setItem('echo-music-station-state',JSON.stringify({version:1,currentTrack:f.current,primaryTrack:f.current,activeRecommendations:f.recs}));
   },{current:cached(fixture.current),recs:fixture.recs.map(cached)});
   const tracks=Object.fromEntries([fixture.current,...fixture.recs].map(t=>[t.track_id,t]));
   await context.route('**/api/**',async route=>{
    const path=new URL(route.request().url()).pathname;let data={ok:true};
    assert(!path.startsWith('/api/ai/')||route.request().method()!=='POST','No AI calls in map tests');
    if(path.includes('/tracks/'))data=tracks[decodeURIComponent(path.split('/').pop())]||fixture.current;
    else if(path.includes('/random'))data=fixture.current;
    else if(path.includes('/favorites'))data={items:[]};
    else if(path.includes('/status'))data={ok:true,n_tracks:30000,n_metadata_rows:30000,has_local_key:false};
    await route.fulfill({json:data});
   });
   const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
   await page.goto(base+'/music.html',{waitUntil:'domcontentloaded'});await page.waitForTimeout(1200);
   assert.equal(await page.locator('.map-label').count(),fixture.markers,fixture.name);
   if(fixture.close){
    const points=await page.locator('.map-label').evaluateAll(ns=>ns.map(n=>({x:parseFloat(n.style.left),y:parseFloat(n.style.top)})));
    assert(Math.abs(points[0].x-points[1].x)<50,'Dateline neighbours must be drawn next to each other');
   }
   assert.deepEqual(errors,[],fixture.name);console.log('PASS',fixture.name);await context.close();
  }
 }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
