# Europeana / MTG-Jamendo 瑗挎柟鏉ユ簮琛ュ厖鎺㈡祴锛?2026-04-13锛?

## 1. 鐩爣

鏈疆宸ヤ綔鐩爣鏄洖绛斾袱涓棶棰橈細

1. `Europeana Sounds / Europeana API` 鍜?`MTG-Jamendo` 鏄惁鑷甫鍙敤鐨?`country / location / provider` 绛夊湴鐞嗘垨鏉ユ簮淇℃伅锛?
2. 濡傛灉鍙互锛屾槸鍚﹁兘鍏堝疄闄呬笅杞戒竴鎵归煶棰戯紝浣滀负鍚庣画鈥滄浛鎹? / 琛ュ厖瑗挎柟鏉ユ簮鈥濈殑鍊欓€夋睜锛?

## 2. 鏈€閲嶈鐨勭粨璁?

### 2.1 MTG-Jamendo

`MTG-Jamendo` **涓嶉€傚悎浣滀负鈥滄湁鍦扮悊淇℃伅鐨勮タ鏂归煶涔愯ˉ鍏呬富婧愨€?*

鍘熷洜鏄細

- 瀹樻柟 README 鏄庣‘璇寸殑棰濆 metadata 鍙寘鎷細`artist`銆乤lbum`銆乼rack title`銆乺elease date`銆乼rack URL`
- 鏈湴澶嶆牳 [raw.meta.tsv](E:/Desktop/Echo/tmp/mtg-jamendo-dataset/data/raw.meta.tsv) 澶村嚑鍒楋紝涔熷彧鏈夛細
  - `TRACK_ID`
  - `ARTIST_ID`
  - `ALBUM_ID`
  - `TRACK_NAME`
  - `ARTIST_NAME`
  - `ALBUM_NAME`
  - `RELEASEDATE`
  - `URL`
- 娌℃湁绋冲畾鐨?`country / nationality / location / region` 瀛楁

鍥犳锛屾湰杞?**娌℃湁** 鍚姩 MTG-Jamendo 鐨勫ぇ瑙勬ā涓嬭浇锛屽洜涓哄畠褰撳墠涓嶈兘瑙ｅ喅鈥滆タ鏂规潵婧愭湁鍦扮悊璇佹嵁鈥濊繖涓牳蹇冮渶姹傘€?

### 2.2 Europeana

`Europeana Search API` **鍙互** 鏂规柟鍖栧湴鎻愪緵浠ヤ笅瀛楁锛?

- `country`
- `dataProvider`
- `provider`
- `rights`
- `edmIsShownBy`
- `guid`

杩欎簺瀛楁瀵逛簬鈥滀慨澶嶆潵婧愬崟涓€鈥濊繖浠朵簨鏄湁浠峰€肩殑銆?

浣嗚娉ㄦ剰锛?

> Europeana 涓殑 `country` 鏇村儚鏄?鈥滄彁渚涜繖鏉℃潯鐩殑鏈烘瀯鎵€鍦ㄥ浗瀹垛€? 鑰屼笉涓€瀹氭槸 鈥滆繖棣栭煶涔愮殑鏂囧寲褰掑睘鍥藉鈥濄€?

鎹㈠彞璇濊锛屽畠闈炲父閫傚悎鐢ㄦ潵琛?**鏉ユ簮澶氭牱鎬?**锛屼絾涓嶉€傚悎鐩存帴鎷挎潵褰撳仛鈥?`france / germany / italy` 绾噣鏂囧寲鍩熸浛鎹㈡簮鈥濄€?

## 3. 宸插仛鐨勮剼鏈敼閫?

鏈疆宸插 [import_europeana_audio_search.py](E:/Desktop/Echo/dcas/scripts/import_europeana_audio_search.py) 鍋氫簡涓€娆℃湁瀹為檯浠峰€肩殑鍔犲己锛?

1. 涓嶅啀鍙潬 URL 鍚庣紑鏄惁鏄?`.mp3` 鍒ゆ柇鏄惁涓洪煶棰?
2. 改涓衡€滅湅 HTTP `Content-Type` 鍜岄煶棰戝ご閮ㄥ瓧鑺傗€濇潵鍒ゆ柇鏄惁鐪熺殑鏄煶棰?
3. 鍔犱簡鍐椾笅鍒楀瓧娈电殑鑳藉姏锛?
   - `region`
   - `language`
   - `license`
   - `url`
   - `notes`
   - `country`
   - `data_provider`
   - `provider`
   - `europeana_collection`
   - `europeana_dataset_name`
   - `edm_is_shown_by`
   - `edm_is_shown_at`
4. 鍔犱簡鍙噸澶嶄娇鐢ㄧ殑 `--query_filter` 鍙傛暟锛屽彲浠ョ洿鎺ョ敤 API `qf` 杩囨护锛屼緥濡傦細
   - `TYPE:SOUND`
   - `COUNTRY:France`

杩欎釜鏀瑰姩鐨勬牳蹇冩剰涔夋槸锛?

- Europeana 鍊欓€夋睜涓嶅啀鏄?鈥滃彧涓嬩笅鏉ラ煶棰戔€?鑰屾槸鈥滃甫鏉ユ簮璇佹嵁涓€璧蜂笅鏉モ€?
- 鍚庨潰鍗充娇涓嶇珛鍗冲苟鍏?V4锛岃繖鎵规暟鎹篃鑳戒綔涓虹湡姝ｇ殑鍊欓€夋睜缁х画瀹℃牳

## 4. 瀹為檯涓嬭浇鐨勫€欓€夋睜

鏈疆瀹為檯鎵ц浜?9` 缁勬煡璇€?

- `france`
  - `folk music`
  - `traditional music`
  - `song`
- `germany`
  - `musik`
  - `lied`
  - `folk music`
- `italy`
  - `musica`
  - `folk music`
  - `classical music`

杩囨护鏉′欢涓€鑷存槸锛?

- `TYPE:SOUND`
- `COUNTRY:<target country>`

姣忕粍鍏堜笅 `8` 鏉★紝鍋氫竴杞?small-batch source probe銆?

## 5. 缁撴灉缁熻

涓嬭浇鍜屽幓閲嶅悗鐨勭粨鏋滃涓嬶細

- 鍘熷涓嬭浇鏉＄洰锛?`72`
- 鎸夌収 `track_id` / `url` 鍘婚噸鍚庯細`64`
- 鍘婚噸鎹熷け锛?`8`

鍘婚噸鍚庣殑鍊欓€夋睜 metadata 鍦ㄨ繖閲岋細

- [metadata.csv](E:/Desktop/Echo/storage/public/source_probe/europeana_western_candidates_merged/metadata.csv)
- [metadata.csv.merge_report.json](E:/Desktop/Echo/storage/public/source_probe/europeana_western_candidates_merged/metadata.csv.merge_report.json)

鍒嗗浗瀹剁殑鏉＄洰鏁帮紙鎸夊綋鍓?`culture` 鏍囩锛夛細

- `france`: `16`
- `germany`: `24`
- `italy`: `24`

渚涙柟 / 鑱氬悎鏂瑰垎甯冿細

- `France -> Europeana Sounds`
- `Germany -> German Digital Library`
- `Italy -> CulturaItalia`

杩欐剰鍛崇潃锛?

- 鎴戜滑宸茬粡鐪熷疄鎶婄幇鏈夎タ鏂规潵婧愪粠鈥滃熀鏈彧鏈?FMA鈥濇墿鍑轰簡涓€姝?
- 鑷冲皯鐜板湪鎵嬩笂鏈変簡涓€涓?**闈?FMA 鐨勮タ鏂逛緵搴旀柟鍊欓€夋睜**

## 6. 璐ㄩ噺涓庡眬闄?

### 6.1 濂藉

杩欐壒鏁版嵁鐨勫ソ澶勬槸锛?

- 鏈夋槑纭殑 `country/provider/dataProvider/rights/url`
- 鍙互鐢ㄦ潵淇鈥滆タ鏂规潵婧愯繃浜庡崟涓€鈥濈殑闂
- 闊抽鏄湡瀹炲凡涓嬭浇鍒版湰鍦扮殑锛岀敤浜庡悗缁?embedding / benchmark 涓嶉渶鍐嶉噸鍋氭悳绱?

### 6.2 闄愬埗

鏈€澶х殑闄愬埗鏄細

- `Europeana country` 鍊惧悜浜庘€滄彁渚涙満鏋勫浗瀹垛€?
- 瀹冧笉绛変簬鈥滈煶涔愬唴瀹圭殑鏂囧寲鍘熶骇鍦扳€?

渚嬪锛屽綋鍓?`france` 杩欐壒閲岋紝宸茬粡鍑虹幇浜嗘棩鏈拰甯屽笇鑵婄殑鍐呭绾跨储銆?

鍥犳锛?

- 瀹冨彲浠ユ嬁鏉ヤ慨鈥?**source diversity**鈥?
- 浣嗕笉鑳界洿鎺ユ嬁鏉ユ敼鍐欐垚鈥?**france culture domain**鈥濊€屼笉鍋氫汉宸ュ鏍?

## 7. 褰撳墠鏈€绋崇殑瀹氫綅

鎴戝缓璁綋鍓嶆妸杩欐壒 Europeana 闊抽瀹氫綅涓猴細

**鈥滈潪 FMA 瑗挎柟鏉ユ簮琛ュ厖鍊欓€夋睜鈥?**

鑰屼笉鏄細

**鈥滃凡缁忓彲浠ョ洿鎺ユ浛鎹?V4 france/germany/italy 鐨勭函鍑€鏂囧寲鍩熸簮鈥?**

## 8. 涓嬩竴姝ュ缓璁?

鏈€鍚堢悊鐨勪笅涓€姝ユ槸锛?

1. 鍏堝杩?`64` 鏉″€欓€夋牱鏈仛涓€杞?metadata + title` 浜哄伐澶嶆牳
2. 鎶婃槑鏄句笉鏄煶涔愭垨鏂囧寲褰掑睘鏄庢樉涓嶅尮閰嶇殑鏉＄洰鎺掓帀
3. 鍙妸澶嶆牳閫氳繃鐨勬潯鐩綔涓?`supplementary western source`
4. 鍦ㄨ鏂囬噷鏄庣‘鍖哄垎锛?
   - 鈥滄枃鍖栧煙鏍囨敞鈥?   - 鈥滄潵婧愬钩鍙板鏍锋€р€?
   杩欎袱浠朵簨涓嶆槸涓€鍥炰簨

## 9. 鐩稿叧鏈湴璺緞

鏀瑰ソ鐨?Europeana 瀵煎叆鑴氭湰锛?

- [import_europeana_audio_search.py](E:/Desktop/Echo/dcas/scripts/import_europeana_audio_search.py)

绗竴杞笅杞界殑鍊欓€夌洰褰曪細

- [europeana_western_candidates](E:/Desktop/Echo/storage/public/source_probe/europeana_western_candidates)

鍘婚噸鍚庣殑鍚堝苟 metadata锛?

- [metadata.csv](E:/Desktop/Echo/storage/public/source_probe/europeana_western_candidates_merged/metadata.csv)

MTG-Jamendo 鍏冩暟鎹弬鑰冿細

- [README.md](E:/Desktop/Echo/tmp/mtg-jamendo-dataset/README.md)
- [raw.meta.tsv](E:/Desktop/Echo/tmp/mtg-jamendo-dataset/data/raw.meta.tsv)
