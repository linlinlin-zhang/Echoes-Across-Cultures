// Check every observed catalog label through the real browser helper, including stale cache forms.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const path=require('node:path');const root=path.resolve(__dirname,'..');
const context={window:{EchoI18n:{getLanguage:()=> 'zh-CN'}}};vm.createContext(context);
for(const file of ['genre-data.js','metadata-i18n.js'])vm.runInContext(fs.readFileSync(path.join(root,'web',file),'utf8'),context);
const {genre,keyword}=context.window.EchoMetadata;
for(const value of ['록','ロック','Rock','搖滾','摇滚'])assert.equal(genre({label:value,label_en:'華語流行樂'}),'摇滚');
assert.equal(keyword('록','en'),'Rock');assert.equal(keyword('Rock','ko'),'록');
assert.equal(genre({genre:'록',genre_en:'華語流行樂'},'en'),'Rock');
assert.equal(genre({raw:{label:'록'},genreEn:'華語流行樂'},'zh-CN'),'摇滚');
assert.equal(keyword('K-Pop','zh-CN'),'韩国流行');assert.equal(keyword('House','zh-CN'),'浩室');
assert.equal(keyword('Dub','zh-CN'),'Dub 雷鬼');assert.equal(keyword('Lounge','zh-CN'),'沙发音乐');
assert.notEqual(keyword('Afro-Beat','en'),keyword('Afrobeats','en'));
const dictionary=context.window.EchoGenreData;
for(const [alias,id] of Object.entries(dictionary.aliases)){
  for(const language of ['zh-CN','en','de','es','pt','ja','ko'])assert.equal(genre({label:alias},language),dictionary.genres[id].labels[language]);
}
console.log(`PASS ${Object.keys(dictionary.aliases).length} aliases × 7 languages, stale cached fields and literal Rock/House/Dub regressions`);
