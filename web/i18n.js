(function () {
  const STORAGE_KEY = "echo-language";
  const APP_CONFIG_KEY = "echo-app-config";
  const DEFAULT_LANGUAGE = "zh-CN";

  const languages = [
    { code: "zh-CN", htmlLang: "zh-CN", nativeName: "中文", englishName: "Chinese", promptName: "中文" },
    { code: "en", htmlLang: "en", nativeName: "English", englishName: "English", promptName: "English" },
    { code: "de", htmlLang: "de", nativeName: "Deutsch", englishName: "German", promptName: "Deutsch" },
    { code: "es", htmlLang: "es", nativeName: "Español", englishName: "Spanish", promptName: "Español" },
    { code: "pt", htmlLang: "pt", nativeName: "Português", englishName: "Portuguese", promptName: "Português" },
    { code: "ja", htmlLang: "ja", nativeName: "日本語", englishName: "Japanese", promptName: "日本語" },
    { code: "ko", htmlLang: "ko", nativeName: "한국어", englishName: "Korean", promptName: "한국어" }
  ];

  const aliases = {
    zh: "zh-CN",
    "zh-cn": "zh-CN",
    cn: "zh-CN",
    chinese: "zh-CN",
    en: "en",
    "en-us": "en",
    english: "en",
    de: "de",
    german: "de",
    deutsch: "de",
    es: "es",
    spanish: "es",
    español: "es",
    pt: "pt",
    "pt-br": "pt",
    portuguese: "pt",
    português: "pt",
    ja: "ja",
    japanese: "ja",
    "日本語": "ja",
    ko: "ko",
    korean: "ko",
    "한국어": "ko"
  };

  const translations = {
    "zh-CN": {
      "common.nav.music": "音乐台",
      "common.nav.favorites": "收藏夹",
      "common.nav.settings": "设置界面",
      "common.brand.homeAria": "Echo 主页",
      "index.title": "Echo | 跨文化音乐推荐",
      "index.brandSub": "Soundscape Without Borders",
      "index.heroSub": "一个把音乐文化、情绪与听觉结构放在同一张世界地图上的研究型推荐主页。",
      "index.action.method": "进入方法层 +",
      "index.action.signals": "查看信号图谱 +",
      "music.title": "Echo | 音乐台",
      "music.brandSub": "音乐台",
      "music.nowPlaying": "正在播放",
      "music.currentPlaying": "当前播放",
      "music.upload": "上传音乐",
      "music.ai.meta": "AI 分析",
      "music.ai.title": "推荐讲解",
      "music.ai.copy": "把当前上传曲目、跨文化推荐结果等内容发送给AI，由 AI 对这次推荐路线做分析。",
      "music.ai.generate": "生成分析",
      "music.ai.generating": "分析生成中...",
      "music.ai.followup": "继续追问",
      "music.ai.followupPlaceholder": "例如：为什么马里科拉琴会被推荐给这首歌？",
      "music.ai.send": "发送",
      "music.ai.sending": "发送中...",
      "music.ai.assistantLabel": "Kimi 分析",
      "music.ai.userLabel": "我的问题",
      "music.ai.displayPrompt": "请基于当前上传曲目、跨文化推荐结果等内容，分析这次推荐路线。",
      "music.ai.displayFollowup": "继续追问：{question}",
      "music.ai.noText": "Kimi 没有返回可用文本。",
      "music.ai.fetchFailed": "请求没有成功发出，可能是本地后端未启动、网络问题，或接口地址不可达。",
      "favorites.title": "Echo | 收藏夹",
      "favorites.brandSub": "收藏夹",
      "favorites.heroMeta": "曲库界面",
      "favorites.heroTitle": "声音<br>收藏夹",
      "favorites.heroCopy": "把来自 iTunes、Jamendo 与上传音频的曲目整理成可浏览、可筛选、可继续接入推荐反馈的音乐集合。",
      "favorites.stat.tracks": "曲目",
      "favorites.stat.cultures": "文化域",
      "favorites.stat.covers": "封面",
      "favorites.import": "点击导入歌曲",
      "favorites.searchLabel": "歌曲检索",
      "favorites.searchPlaceholder": "曲名 / 创作者 / 流派",
      "favorites.cultureFilter": "文化筛选",
      "favorites.allCultures": "全部文化域",
      "favorites.sourceFilter": "来源筛选",
      "favorites.source.favorite": "收藏",
      "favorites.allSources": "全部来源",
      "favorites.localUpload": "本地导入",
      "favorites.sortLabel": "歌曲排序",
      "favorites.sort.shuffle": "随机漫游",
      "favorites.sort.title": "曲名 A-Z",
      "favorites.sort.artist": "创作者 A-Z",
      "favorites.sort.culture": "文化域",
      "favorites.sort.source": "来源",
      "favorites.noCover": "无封面",
      "favorites.openPlatform": "打开平台 ↗",
      "favorites.localTrack": "本地曲目",
      "favorites.removeTrack": "移出收藏夹",
      "favorites.empty": "没有匹配曲目。",
      "favorites.loading": "正在加载收藏曲库...",
      "favorites.loadFailed": "曲库加载失败，请确认本地后端正在运行。",
      "favorites.unknownTrack": "未知曲目",
      "favorites.unknownArtist": "未知创作者",
      "favorites.unknownGenre": "未标注流派",
      "favorites.unknownCulture": "未标注文化",
      "favorites.pendingAnalysis": "待分析",
      "settings.title": "Echo | 设置界面",
      "settings.brandSub": "设置界面",
      "settings.heroMeta": "模型设置",
      "settings.heroTitle": "Kimi<br>接口",
      "settings.heroCopy": "这里配置音乐台 AI 分析所需的 Moonshot/Kimi 接口。密钥仍只保存在当前浏览器，本地后端负责代理请求。",
      "settings.languageLabel": "界面与 AI 语言",
      "settings.languageHelp": "选择后会同步影响页面文案和 AI 分析回复语言。",
      "settings.apiKeyLabel": "Kimi API 密钥",
      "settings.apiKeyPlaceholder": "填入 Moonshot/Kimi 密钥",
      "settings.modelLabel": "模型",
      "settings.endpointLabel": "接口地址",
      "settings.save": "保存设置",
      "settings.test": "测试连接",
      "settings.clear": "清空密钥",
      "settings.note": "可以把密钥保存在当前浏览器本地，也可以填入 configs/secrets/kimi.local.json；音乐台会通过 /api/ai/kimi/chat 让本地后端代理转发。",
      "settings.readFailed": "读取本地设置失败，请重新保存一次。",
      "settings.backendDetected": "已检测到后端本地 Kimi 配置：{source}",
      "settings.saved": "设置已保存。回到音乐台后即可生成 AI 分析。",
      "settings.languageSaved": "语言已保存。",
      "settings.testing": "正在通过本地后端测试 Kimi 连接...",
      "settings.testSuccess": "Kimi 连接成功：{content}",
      "settings.testFailed": "Kimi 连接失败：{error}",
      "settings.keyCleared": "密钥已从当前浏览器本地清空。",
      "culture.west": "西方",
      "culture.china": "中国",
      "culture.southeast_asia": "东南亚",
      "culture.eastern_europe": "东欧",
      "culture.japan": "日本",
      "culture.africa": "非洲",
      "culture.middle_east": "中东",
      "culture.celtic": "凯尔特",
      "culture.korea": "韩国",
      "culture.caribbean": "加勒比",
      "culture.brazil": "巴西",
      "culture.india": "印度",
      "culture.nordic": "北欧",
      "culture.latin": "拉丁",
      "culture.balkans": "巴尔干",
      "culture.central_asia": "中亚",
      "culture.andean": "安第斯"
    },
    en: {
      "common.nav.music": "Music Station",
      "common.nav.favorites": "Favorites",
      "common.nav.settings": "Settings",
      "common.brand.homeAria": "Echo home",
      "index.title": "Echo | Cross-Cultural Music Recommendation",
      "index.heroSub": "A research recommendation homepage that places music culture, emotion, and auditory structure on one world map.",
      "index.action.method": "Enter method layer +",
      "index.action.signals": "View signal atlas +",
      "music.title": "Echo | Music Station",
      "music.brandSub": "Music Station",
      "music.nowPlaying": "Now playing",
      "music.currentPlaying": "Current track",
      "music.upload": "Upload music",
      "music.ai.meta": "AI Analysis",
      "music.ai.title": "Recommendation Notes",
      "music.ai.copy": "Send the uploaded track and cross-cultural recommendations to AI, then let AI analyze this recommendation route.",
      "music.ai.generate": "Generate analysis",
      "music.ai.generating": "Generating...",
      "music.ai.followup": "Follow up",
      "music.ai.followupPlaceholder": "Example: why was this kora track recommended?",
      "music.ai.send": "Send",
      "music.ai.sending": "Sending...",
      "music.ai.assistantLabel": "Kimi analysis",
      "music.ai.userLabel": "My question",
      "music.ai.displayPrompt": "Analyze this recommendation route using the uploaded track and cross-cultural results.",
      "music.ai.displayFollowup": "Follow-up: {question}",
      "music.ai.noText": "Kimi did not return usable text.",
      "music.ai.fetchFailed": "The request was not sent successfully. The local backend, network, or endpoint may be unavailable.",
      "favorites.title": "Echo | Favorites",
      "favorites.brandSub": "Favorites",
      "favorites.heroTitle": "Sound<br>Favorites",
      "favorites.heroCopy": "Browse, filter, and reuse tracks from iTunes, Jamendo, and uploaded audio as a music collection for recommendation feedback.",
      "favorites.stat.tracks": "Tracks",
      "favorites.stat.cultures": "Cultures",
      "favorites.stat.covers": "Covers",
      "favorites.import": "Import songs",
      "favorites.searchLabel": "Search",
      "favorites.searchPlaceholder": "Title / creator / genre",
      "favorites.cultureFilter": "Culture",
      "favorites.allCultures": "All cultures",
      "favorites.sourceFilter": "Source",
      "favorites.source.favorite": "Favorites",
      "favorites.allSources": "All sources",
      "favorites.localUpload": "Local import",
      "favorites.sortLabel": "Sort",
      "favorites.sort.shuffle": "Shuffle",
      "favorites.sort.title": "Title A-Z",
      "favorites.sort.artist": "Creator A-Z",
      "favorites.sort.culture": "Culture",
      "favorites.sort.source": "Source",
      "favorites.noCover": "No Cover",
      "favorites.openPlatform": "Open platform ↗",
      "favorites.localTrack": "Local track",
      "favorites.removeTrack": "Remove from favorites",
      "favorites.empty": "No matching tracks.",
      "favorites.loading": "Loading favorite library...",
      "favorites.loadFailed": "Library failed to load. Check that the local backend is running.",
      "favorites.unknownTrack": "Unknown track",
      "favorites.unknownArtist": "Unknown creator",
      "favorites.unknownGenre": "Unlabeled genre",
      "favorites.unknownCulture": "Unlabeled culture",
      "favorites.pendingAnalysis": "Pending analysis",
      "settings.title": "Echo | Settings",
      "settings.brandSub": "Settings",
      "settings.heroMeta": "Model Settings",
      "settings.heroTitle": "Kimi<br>API",
      "settings.heroCopy": "Configure the Moonshot/Kimi API used by Music Station AI analysis. The key stays in this browser, while the local backend proxies requests.",
      "settings.languageLabel": "Interface and AI language",
      "settings.languageHelp": "This changes page copy and the language used for AI analysis replies.",
      "settings.apiKeyLabel": "Kimi API key",
      "settings.apiKeyPlaceholder": "Enter Moonshot/Kimi key",
      "settings.modelLabel": "Model",
      "settings.endpointLabel": "Endpoint",
      "settings.save": "Save settings",
      "settings.test": "Test connection",
      "settings.clear": "Clear key",
      "settings.note": "Save the key in this browser or configs/secrets/kimi.local.json; Music Station calls /api/ai/kimi/chat through the local backend proxy.",
      "settings.readFailed": "Failed to read local settings. Please save again.",
      "settings.backendDetected": "Detected local backend Kimi config: {source}",
      "settings.saved": "Settings saved. Return to Music Station to generate AI analysis.",
      "settings.languageSaved": "Language saved.",
      "settings.testing": "Testing Kimi connection through the local backend...",
      "settings.testSuccess": "Kimi connected: {content}",
      "settings.testFailed": "Kimi connection failed: {error}",
      "settings.keyCleared": "The key has been removed from this browser.",
      "culture.west": "Western",
      "culture.china": "China",
      "culture.southeast_asia": "Southeast Asia",
      "culture.eastern_europe": "Eastern Europe",
      "culture.japan": "Japan",
      "culture.africa": "Africa",
      "culture.middle_east": "Middle East",
      "culture.celtic": "Celtic",
      "culture.korea": "Korea",
      "culture.caribbean": "Caribbean",
      "culture.brazil": "Brazil",
      "culture.india": "India",
      "culture.nordic": "Nordic",
      "culture.latin": "Latin",
      "culture.balkans": "Balkans",
      "culture.central_asia": "Central Asia",
      "culture.andean": "Andean"
    },
    de: {
      "common.nav.music": "Musikstation",
      "common.nav.favorites": "Favoriten",
      "common.nav.settings": "Einstellungen",
      "common.brand.homeAria": "Echo Startseite",
      "index.title": "Echo | Interkulturelle Musikempfehlung",
      "index.heroSub": "Eine Forschungsoberfläche, die Musikkultur, Emotion und Hörstruktur auf einer Weltkarte verbindet.",
      "index.action.method": "Methodenebene öffnen +",
      "index.action.signals": "Signalatlas ansehen +",
      "music.title": "Echo | Musikstation",
      "music.brandSub": "Musikstation",
      "music.nowPlaying": "Aktuelle Wiedergabe",
      "music.currentPlaying": "Aktueller Titel",
      "music.upload": "Musik hochladen",
      "music.ai.meta": "AI Analyse",
      "music.ai.title": "Empfehlung erklären",
      "music.ai.copy": "Sende den aktuellen Titel und die interkulturellen Empfehlungen an die AI, damit sie diese Route analysiert.",
      "music.ai.generate": "Analyse erstellen",
      "music.ai.generating": "Analyse läuft...",
      "music.ai.followup": "Nachfrage",
      "music.ai.followupPlaceholder": "Beispiel: Warum wurde dieser Kora-Titel empfohlen?",
      "music.ai.send": "Senden",
      "music.ai.sending": "Wird gesendet...",
      "music.ai.assistantLabel": "Kimi Analyse",
      "music.ai.userLabel": "Meine Frage",
      "music.ai.displayPrompt": "Analysiere diese Empfehlungsroute anhand des hochgeladenen Titels und der interkulturellen Ergebnisse.",
      "music.ai.displayFollowup": "Nachfrage: {question}",
      "favorites.title": "Echo | Favoriten",
      "favorites.brandSub": "Favoriten",
      "favorites.heroTitle": "Klang<br>Favoriten",
      "favorites.heroCopy": "Durchsuche und filtere Titel aus iTunes, Jamendo und Uploads als Musiksammlung für Empfehlungsfeedback.",
      "favorites.import": "Songs importieren",
      "favorites.removeTrack": "Aus Favoriten entfernen",
      "favorites.searchLabel": "Suche",
      "favorites.searchPlaceholder": "Titel / Künstler / Genre",
      "settings.title": "Echo | Einstellungen",
      "settings.brandSub": "Einstellungen",
      "settings.heroTitle": "Kimi<br>API",
      "settings.languageLabel": "Sprache für Oberfläche und AI",
      "settings.languageHelp": "Ändert Seitentexte und die Antwortsprache der AI-Analyse.",
      "settings.apiKeyLabel": "Kimi API-Schlüssel",
      "settings.apiKeyPlaceholder": "Moonshot/Kimi-Schlüssel eintragen",
      "settings.save": "Speichern",
      "settings.test": "Verbindung testen",
      "settings.clear": "Schlüssel löschen"
    },
    es: {
      "common.nav.music": "Mesa musical",
      "common.nav.favorites": "Favoritos",
      "common.nav.settings": "Ajustes",
      "common.brand.homeAria": "Inicio de Echo",
      "index.title": "Echo | Recomendación musical intercultural",
      "index.heroSub": "Una portada de investigación que ubica cultura musical, emoción y estructura auditiva en un mismo mapa mundial.",
      "music.title": "Echo | Mesa musical",
      "music.brandSub": "Mesa musical",
      "music.upload": "Subir música",
      "music.ai.meta": "Análisis AI",
      "music.ai.title": "Explicación",
      "music.ai.copy": "Envía la pista actual y las recomendaciones interculturales a la AI para analizar esta ruta.",
      "music.ai.generate": "Generar análisis",
      "music.ai.followup": "Seguir preguntando",
      "music.ai.send": "Enviar",
      "music.ai.assistantLabel": "Análisis de Kimi",
      "music.ai.userLabel": "Mi pregunta",
      "music.ai.displayPrompt": "Analiza esta ruta de recomendación con la pista subida y los resultados interculturales.",
      "music.ai.displayFollowup": "Seguimiento: {question}",
      "favorites.title": "Echo | Favoritos",
      "favorites.brandSub": "Favoritos",
      "favorites.heroTitle": "Favoritos<br>sonoros",
      "favorites.import": "Importar canciones",
      "favorites.removeTrack": "Quitar de favoritos",
      "settings.title": "Echo | Ajustes",
      "settings.brandSub": "Ajustes",
      "settings.languageLabel": "Idioma de interfaz y AI",
      "settings.languageHelp": "Cambia los textos de la página y el idioma de respuesta del análisis AI.",
      "settings.save": "Guardar",
      "settings.test": "Probar conexión",
      "settings.clear": "Borrar clave"
    },
    pt: {
      "common.nav.music": "Estação musical",
      "common.nav.favorites": "Favoritos",
      "common.nav.settings": "Configurações",
      "common.brand.homeAria": "Início do Echo",
      "index.title": "Echo | Recomendação musical intercultural",
      "index.heroSub": "Uma página de pesquisa que coloca cultura musical, emoção e estrutura auditiva no mesmo mapa mundial.",
      "music.title": "Echo | Estação musical",
      "music.brandSub": "Estação musical",
      "music.upload": "Enviar música",
      "music.ai.meta": "Análise AI",
      "music.ai.title": "Explicação",
      "music.ai.copy": "Envie a faixa atual e as recomendações interculturais para a AI analisar esta rota.",
      "music.ai.generate": "Gerar análise",
      "music.ai.followup": "Perguntar mais",
      "music.ai.send": "Enviar",
      "music.ai.assistantLabel": "Análise Kimi",
      "music.ai.userLabel": "Minha pergunta",
      "music.ai.displayPrompt": "Analise esta rota de recomendação com a faixa enviada e os resultados interculturais.",
      "music.ai.displayFollowup": "Pergunta: {question}",
      "favorites.title": "Echo | Favoritos",
      "favorites.brandSub": "Favoritos",
      "favorites.heroTitle": "Favoritos<br>sonoros",
      "favorites.import": "Importar músicas",
      "favorites.removeTrack": "Remover dos favoritos",
      "settings.title": "Echo | Configurações",
      "settings.brandSub": "Configurações",
      "settings.languageLabel": "Idioma da interface e da AI",
      "settings.languageHelp": "Altera os textos da página e o idioma das respostas da análise AI.",
      "settings.save": "Salvar",
      "settings.test": "Testar conexão",
      "settings.clear": "Limpar chave"
    },
    ja: {
      "common.nav.music": "音楽台",
      "common.nav.favorites": "お気に入り",
      "common.nav.settings": "設定",
      "common.brand.homeAria": "Echo ホーム",
      "index.title": "Echo | 異文化音楽推薦",
      "index.heroSub": "音楽文化、感情、聴覚構造を同じ世界地図上に置く研究型推薦ホームです。",
      "music.title": "Echo | 音楽台",
      "music.brandSub": "音楽台",
      "music.upload": "音楽をアップロード",
      "music.ai.meta": "AI 分析",
      "music.ai.title": "推薦解説",
      "music.ai.copy": "現在の曲と異文化推薦結果を AI に送り、この推薦ルートを分析します。",
      "music.ai.generate": "分析を生成",
      "music.ai.followup": "続けて質問",
      "music.ai.send": "送信",
      "music.ai.assistantLabel": "Kimi 分析",
      "music.ai.userLabel": "自分の質問",
      "music.ai.displayPrompt": "アップロード曲と異文化推薦結果に基づいて、この推薦ルートを分析してください。",
      "music.ai.displayFollowup": "追加質問：{question}",
      "favorites.title": "Echo | お気に入り",
      "favorites.brandSub": "お気に入り",
      "favorites.heroTitle": "音の<br>お気に入り",
      "favorites.import": "曲を読み込む",
      "favorites.removeTrack": "お気に入りから削除",
      "settings.title": "Echo | 設定",
      "settings.brandSub": "設定",
      "settings.languageLabel": "画面と AI の言語",
      "settings.languageHelp": "ページ文言と AI 分析の返信言語を変更します。",
      "settings.save": "保存",
      "settings.test": "接続テスト",
      "settings.clear": "キーを消去"
    },
    ko: {
      "common.nav.music": "음악 스테이션",
      "common.nav.favorites": "즐겨찾기",
      "common.nav.settings": "설정",
      "common.brand.homeAria": "Echo 홈",
      "index.title": "Echo | 교차 문화 음악 추천",
      "index.heroSub": "음악 문화, 감정, 청각 구조를 하나의 세계 지도에 배치하는 연구형 추천 홈입니다.",
      "music.title": "Echo | 음악 스테이션",
      "music.brandSub": "음악 스테이션",
      "music.upload": "음악 업로드",
      "music.ai.meta": "AI 분석",
      "music.ai.title": "추천 해설",
      "music.ai.copy": "현재 곡과 교차 문화 추천 결과를 AI에 보내 추천 경로를 분석합니다.",
      "music.ai.generate": "분석 생성",
      "music.ai.followup": "추가 질문",
      "music.ai.send": "보내기",
      "music.ai.assistantLabel": "Kimi 분석",
      "music.ai.userLabel": "내 질문",
      "music.ai.displayPrompt": "업로드한 곡과 교차 문화 추천 결과를 바탕으로 이 추천 경로를 분석해 주세요.",
      "music.ai.displayFollowup": "추가 질문: {question}",
      "favorites.title": "Echo | 즐겨찾기",
      "favorites.brandSub": "즐겨찾기",
      "favorites.heroTitle": "사운드<br>즐겨찾기",
      "favorites.import": "곡 가져오기",
      "favorites.removeTrack": "즐겨찾기에서 제거",
      "settings.title": "Echo | 설정",
      "settings.brandSub": "설정",
      "settings.languageLabel": "인터페이스 및 AI 언어",
      "settings.languageHelp": "페이지 문구와 AI 분석 답변 언어를 변경합니다.",
      "settings.save": "저장",
      "settings.test": "연결 테스트",
      "settings.clear": "키 지우기"
    }
  };

  function normalizeLanguage(value) {
    const raw = String(value || "").trim();
    if (!raw) return DEFAULT_LANGUAGE;
    if (languages.some((item) => item.code === raw)) return raw;
    return aliases[raw.toLowerCase()] || DEFAULT_LANGUAGE;
  }

  function readJson(key) {
    try {
      return JSON.parse(localStorage.getItem(key) || "{}");
    } catch (error) {
      return {};
    }
  }

  function getLanguage() {
    const direct = localStorage.getItem(STORAGE_KEY);
    if (direct) return normalizeLanguage(direct);
    const appConfig = readJson(APP_CONFIG_KEY);
    if (appConfig.language) return normalizeLanguage(appConfig.language);
    const kimiConfig = readJson("echo-kimi-config");
    if (kimiConfig.language) return normalizeLanguage(kimiConfig.language);
    return DEFAULT_LANGUAGE;
  }

  function storeLanguage(language) {
    const code = normalizeLanguage(language);
    localStorage.setItem(STORAGE_KEY, code);
    const appConfig = readJson(APP_CONFIG_KEY);
    appConfig.language = code;
    localStorage.setItem(APP_CONFIG_KEY, JSON.stringify(appConfig));
    return code;
  }

  function format(template, params) {
    return String(template || "").replace(/\{(\w+)\}/g, (_, key) => {
      return params && Object.prototype.hasOwnProperty.call(params, key) ? String(params[key]) : "";
    });
  }

  function t(key, params, language) {
    const code = normalizeLanguage(language || getLanguage());
    const selected = translations[code] || {};
    const english = translations.en || {};
    const fallback = translations[DEFAULT_LANGUAGE] || {};
    return format(selected[key] || (code !== "en" ? english[key] : "") || fallback[key] || key, params);
  }

  function languageInfo(language) {
    const code = normalizeLanguage(language || getLanguage());
    return languages.find((item) => item.code === code) || languages[0];
  }

  function apply(root, language) {
    const code = normalizeLanguage(language || getLanguage());
    const target = root || document;
    const info = languageInfo(code);
    document.documentElement.lang = info.htmlLang;
    if (document.body?.dataset.i18nTitle) {
      document.title = t(document.body.dataset.i18nTitle, {}, code);
    }
    target.querySelectorAll("[data-i18n]").forEach((node) => {
      node.textContent = t(node.dataset.i18n, {}, code);
    });
    target.querySelectorAll("[data-i18n-html]").forEach((node) => {
      node.innerHTML = t(node.dataset.i18nHtml, {}, code);
    });
    target.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
      node.setAttribute("placeholder", t(node.dataset.i18nPlaceholder, {}, code));
    });
    target.querySelectorAll("[data-i18n-aria-label]").forEach((node) => {
      node.setAttribute("aria-label", t(node.dataset.i18nAriaLabel, {}, code));
    });
  }

  function setLanguage(language) {
    const code = storeLanguage(language);
    apply(document, code);
    window.dispatchEvent(new CustomEvent("echo-language-change", { detail: { language: code } }));
    return code;
  }

  window.EchoI18n = {
    languages,
    normalizeLanguage,
    getLanguage,
    setLanguage,
    apply,
    t,
    languageInfo,
    promptLanguageName(language) {
      return languageInfo(language).promptName;
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => apply(document));
  } else {
    apply(document);
  }
}());
