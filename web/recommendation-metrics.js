// Display measurements come only from real model scores or a versioned saved measurement.
(() => {
  const keys=['serendipity','styleNovelty','listeningRelevance','minorityExposure','score','relevance','unexpectedness'];
  function value(input) {
    if(input===null||input===undefined||input===''||typeof input==='boolean')return null;
    const number=Number(input);
    return Number.isFinite(number)&&number>=0&&number<=1?number:null;
  }
  function read(item={}) {
    const raw=item.raw||item;
    const components=item.score_components||raw.score_components||{};
    const measured=item.recommendation_metrics||raw.recommendation_metrics||{};
    const saved=item.metrics?.schemaVersion===2?item.metrics:{};
    const styleNovelty=value(measured.style_novelty??components.style_novelty??components.novelty??saved.styleNovelty);
    const listeningRelevance=value(measured.listening_relevance??components.listening_relevance??components.relevance??saved.listeningRelevance);
    return {schemaVersion:2,styleNovelty,listeningRelevance,
      serendipity:styleNovelty===null||listeningRelevance===null?null:listeningRelevance*(0.58+styleNovelty*0.42),
      minorityExposure:value(components.minority??saved.minorityExposure),
      score:value(item.score??raw.score??saved.score),relevance:listeningRelevance,unexpectedness:styleNovelty};
  }
  function average(items=[]) {
    const readings=items.map(item=>read(item));
    const output={schemaVersion:2};
    for(const key of keys){const values=readings.map(x=>x[key]).filter(x=>x!==null);output[key]=values.length?values.reduce((a,b)=>a+b,0)/values.length:null;}
    return output;
  }
  window.EchoRecommendationMetrics={value,read,average};
})();
