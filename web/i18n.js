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

  const homeIndexTranslations = {
    "zh-CN": {
      "index.heroLabel": "跨文化音乐发现",
      "index.stat.tracksLabel": "曲目",
      "index.stat.tracksCopy": "来自世界各地的真实音乐",
      "index.stat.culturesLabel": "文化域",
      "index.stat.culturesCopy": "个文化域，从东方到西方",
      "index.stat.readyLabel": "就绪",
      "index.stat.readyCopy": "每首都配有封面、介绍和标签",
      "index.feature.startLabel": "01 / 你的起点",
      "index.feature.startCopy": "上传一首歌，或者从 30,000 首里选一首，这就是你的音乐旅行起点。",
      "index.feature.aiLabel": "02 / AI 带路",
      "index.feature.aiCopy": "AI 会分析音乐的“DNA”，为你推荐风格相近但来自不同文化的声音。",
      "index.feature.memoryLabel": "03 / 收藏回忆",
      "index.feature.memoryCopy": "遇到喜欢的歌就收藏起来，你的收藏夹会慢慢变成一张属于你的世界音乐地图。",
      "index.method.label": "它如何工作",
      "index.method.copy": "从上传一首歌开始，AI 会分析它的旋律、节奏和情感，然后带你去听世界各地风格相近的声音。",
      "index.method.bigCopy": "每首歌都有自己的<span class=\"highlight\">声音 DNA</span>。上传你喜欢的音乐，AI 会理解它的特质，然后从地球另一端为你找到那些你从未听过、却可能一见如故的声音。",
      "index.signal.label": "音乐旅程",
      "index.signal.title": "声音<br>路线",
      "index.signal.copy": "上传一首歌，AI 会为你画出一条声音旅行路线。从你的起点出发，跨越语言和文化的边界，去发现那些你从未听过却可能一见如故的音乐。",
      "index.signal.orbit": "内容 - 风格 - 情感 - 反馈 - 路线 -",
      "index.signal.node.zc": "ZC<br>旋律 / 节奏",
      "index.signal.node.zs": "ZS<br>文化 / 风格",
      "index.signal.node.za": "ZA<br>情绪效价 / 唤醒度",
      "settings.recommendCountLabel": "每次推荐歌曲数量",
      "settings.recommendCountHelp": "控制每次跨文化推荐返回的歌曲数量，范围 1-30，默认 10。",
      "settings.recommendCountSaved": "推荐数量设置已保存。",
      "index.globe.surface": "全球音乐表面",
      "index.globe.autoRotate": "自动旋转",
      "index.globe.regionInitializing": "区域初始化中",
      "index.globe.genreSignal": "流派信号已激活",
      "index.globe.region.eastAsia.name": "东亚",
      "index.globe.region.eastAsia.style": "古琴 / 演歌 / 盘索里 / 城市流行",
      "index.globe.region.eastAsia.ring": "东亚 - 古琴 - 琵琶 - 二胡 - 笛子 - 演歌 - 日本流行 - 韩国流行 - 城市流行 - 盘索里 - 粤剧 - 能乐 - ",
      "index.globe.region.eastAsia.text": "东亚 音乐表面：古琴 泛音 指法。 琵琶 轮指 扫弦。 二胡 滑音 揉弦。 笛子 花舌 气息。 尺八 虚吹 共鸣。 扬琴 滚奏 点描。 箫 气振 长音。 琴歌 吟唱 古典。 雅乐 仪式 持续。 能乐 唻声 假面。 歌舞伎 长呗 三味线。 演歌 颤音 哀愁。 盘索里 叙事 鼓点 呼喊。 伽倻琴 散调 摇曳。 韩国流行 钩子 编舞 同步。 日本流行 旋律 密度 和声。 城市流行 铜管 合成器。 粤语流行 抒情 轮廓 填词。 台语歌谣 苦情 转音。 五声音阶 宫商角徵羽 动机。",
      "index.globe.region.southeastAsia.name": "东南亚",
      "index.globe.region.southeastAsia.style": "甘美兰 / 当杜特 / 库林唐",
      "index.globe.region.southeastAsia.ring": "东南亚 - 甘美兰 - 当杜特 - 库林唐 - 皮帕 - 莫兰 - 克隆钟 - 泰国古典 - 越南民乐 - ",
      "index.globe.region.southeastAsia.text": "东南亚 音乐表面：甘美兰 金属琴 交错 循环。 甘美兰 铜排琴 层叠。 当杜特 鼓点 脉冲 电子。 库林唐 锣列 齐奏。 皮帕 仪式 推进 弓弦。 莫兰 声腔 装饰 高亢。 克隆钟 弦拨 葡萄牙 混血。 影戏 音色 摇曳 皮影。 泰国古典 玛胡里 竖琴。 越南民乐 独弦琴 颤音。 印尼流行 dangdut 摇摆。 马来传统 诺巴 拉巴布。 缅甸弯琴 缅甸鼓 序列。 吕宋库利堂 锣阵 合奏。 巴厘岛 安格隆 竹筒 敲击。",
      "index.globe.region.southAsia.name": "南亚",
      "index.globe.region.southAsia.style": "拉格 / 卡瓦利 / 电影歌曲 / 班格拉",
      "index.globe.region.southAsia.ring": "南亚 - 拉格 - 塔布拉 - 西塔尔 - 卡瓦利 - 电影歌曲 - 班格拉 - 加扎勒 - 卡纳提克 - 辛杜尔 - ",
      "index.globe.region.southAsia.text": "南亚 音乐表面：印度斯坦 拉格 上行 展开。 卡纳提克 克尔提 结构 变奏。 塔布拉 音节 循环 即兴。 姆里丹甘 共鸣 节拍。 西塔尔 弦 滑音 共鸣。 萨罗德 拨弦 深沉。 萨朗吉 弓弦 哭腔。 班苏里 竹笛 气息。 卡瓦利 拍手 应答 重复。 电影歌曲 管弦 推进 混合。 班格拉 舞蹈 律动 打击。 加扎勒 诗性 旋律 爱情。 辛杜尔 宗教 鼓声 呼唤。 拉瓦尼 孟买 舞厅 旋律。 帕哈里 山歌 风笛。 南印度 维纳琴 滑音 泛音。 阿拉佩塔 快速 交替 音节。 达尔帕德 庄严 节拍 慢板。",
      "index.globe.region.mena.name": "中东与北非",
      "index.globe.region.mena.style": "玛卡姆 / 乌德琴 / 莱伊 / 格纳瓦",
      "index.globe.region.mena.ring": "中东北非 - 玛卡姆 - 乌德琴 - 卡农琴 - 奈笛 - 莱伊 - 格纳瓦 - 达布卡 - 塔克西姆 - 沙比 - 穆瓦沙赫 - ",
      "index.globe.region.mena.text": "中东北非 音乐表面：玛卡姆 微分音 路径 转调。 塔克西姆 即兴 自由 呼吸。 乌德琴 拨弦 共鸣 泛音。 卡农琴 滑奏 刮奏 装饰。 奈笛 气息 颤音 空灵。 卡曼恰 弓弦 旋转。 桑图尔 锤琴 泛音。 莱伊 电声 唱腔 叙事。 格纳瓦 古恩布里 恍惚 重复。 达布卡 队列舞 脉冲 重拍。 沙比 抒情 摇曳 突尼斯。 穆瓦沙赫 古典 合唱 安达卢西亚。 塔尔 北非 鼓 手掌。 本迪尔 框鼓 苏菲 旋转。 马卡姆希 摩洛哥 街头 节奏。 波斯 桑图尔 塔尔 序列。 伊拉克 马卡姆 鼻音 长音。 阿拉伯 弹拨 乌德 共鸣腔。",
      "index.globe.region.europe.name": "欧洲",
      "index.globe.region.europe.style": "法朵 / 弗拉门戈 / 香颂 / Techno",
      "index.globe.region.europe.ring": "欧洲 - 法朵 - 弗拉门戈 - 香颂 - 凯尔特 - Techno - 复调 - Krautrock - 巴洛克 - 斯拉夫 - ",
      "index.globe.region.europe.text": "欧洲 音乐表面：法朵 思念 旋律 吉他。 弗拉门戈 拍掌 扫弦 舞步 呐喊。 香颂 文本 前景 诗意。 凯尔特 里尔 舞步 风笛 提琴。 巴尔干 非对称 节拍 铜管。 斯拉夫 合唱 和声 忧郁。 阿尔卑斯 约德尔 假声 切换。 巴洛克 通奏低音 对位 羽管键琴。 古典 奏鸣曲 发展 再现。 浪漫 标题 音乐 诗意。 印象派 音色 和声 模糊。 Techno 四拍 地板 合成器。 Krautrock 机动 律动 重复。 Ambient 纹理 漂浮 空间。 后朋克 噪声 粗粝 反叛。 北欧 冰岛 氛围 空灵。 凯尔特 竖琴 泛音 古典。 民谣 叙事 手风琴 口琴。 歌剧 咏叹调 美声 戏剧。 复调 格里高利 经文歌 对位。",
      "index.globe.region.westAfrica.name": "西非",
      "index.globe.region.westAfrica.style": "非洲节拍 / Highlife / Mbalax / 科拉琴",
      "index.globe.region.westAfrica.ring": "西非 - 非洲节拍 - Highlife - Mbalax - 科拉琴 - Juju - 棕榈酒 - 格里奥 - 布基纳 - 曼丁卡 - ",
      "index.globe.region.westAfrica.text": "西非 音乐表面：非洲节拍 铜管 固定音型 费拉。 Highlife 吉他 网格 铜管 舞曲。 Mbalax 萨巴尔 复节奏 切分。 格里奥 科拉琴 琶音 叙事。 Juju 说话鼓 应答 赞美。 棕榈酒 吉他 轻拨 即兴。 富吉 赞美 人声 鼓声。 说话鼓 音调 变化 模仿。 金贝鼓 手掌 低音 高音。 巴拉风 木琴 泛音 合奏。 姆贝莱 铁琴 交错 装饰。 塞内加尔 萨巴尔 竞赛 舞蹈。 马里 曼丁卡 史诗 吟唱。 加纳 阿丁克拉 鼓语 信号。 尼日利亚 约鲁巴 鼓阵 仪式。 呼应 唱法 激活 对话。 复节奏 多层 交叉 位移。 非洲 吹奏 巴拉风 循环。",
      "index.globe.region.americas.name": "美洲",
      "index.globe.region.americas.style": "爵士 / 布鲁斯 / 萨尔萨 / 桑巴",
      "index.globe.region.americas.ring": "美洲 - 爵士 - 布鲁斯 - 福音 - 萨尔萨 - 桑巴 - 昆比亚 - 嘻哈 - 雷鬼顿 - 探戈 - 蓝草 - ",
      "index.globe.region.americas.text": "美洲 音乐表面：布鲁斯 弯音 十二小节 呼喊。 爵士 摇摆 镲片 即兴 和声。 福音 呼应 合唱 钢琴 管风琴。 萨尔萨 克拉维 铜管 钢琴。 桑巴 鼓阵 切分 踩踏。 桑巴 波萨诺瓦 吉他 摇摆。 波萨诺瓦 和声 行走 低音。 昆比亚 手风琴 线条 鼓点。 嘻哈 Breakbeat 采样 韵律。 雷鬼顿 Dembow 低音 循环。 乡村 小提琴 班卓 叙事。 蓝草 曼陀林 指弹 快速。 探戈 手风琴 悲剧 切分。 坎东布雷 仪式 鼓声 呼唤。 雷鬼 切分 反拍 社会。 斯卡 切分 牙买加 铜管。 索卡 节日 特立尼达 钢鼓。 安第斯 排箫 盖纳 悲歌。 马里亚奇 小号 铜管 合唱。 卡利普索 铁鼓 讽刺 叙事。 恰朗加 小提琴 墨西哥 舞曲。 芳丹戈 手鼓 巴西 东北。",
      "index.globe.region.oceania.name": "大洋洲 / 太平洋",
      "index.globe.region.oceania.style": "呼拉 / 哈卡 / 迪吉里杜 / 岛屿雷鬼",
      "index.globe.region.oceania.ring": "大洋洲 太平洋 - 呼拉 - 哈卡 - 迪吉里杜 - 岛屿雷鬼 - 合唱吟唱 - 拉卡拉卡 - 冲浪摇滚 - ",
      "index.globe.region.oceania.text": "大洋洲 太平洋 音乐表面：呼拉 旋律 钢吉他 草裙。 哈卡 群体 吟唱 力量 跺脚。 迪吉里杜 持续低音 循环呼吸。 岛屿雷鬼 切分 热带 和声。 波利尼西亚 合唱 堆叠 和声。 毛利 帕泰 吟唱 战舞。 拉卡拉卡 合唱 舞蹈 汤加。 萨摩亚 西瓦塔 舞蹈 歌唱。 澳洲 原住民 迪吉里杜 梦境。 澳洲 摇滚 朋克 独立。 冲浪摇滚 吉他 混响 夏威夷。 夏威夷 乌库勒勒 滑音 拨弦。 美拉尼西亚 排箫 合奏 低音。 密克罗尼西亚 竹竿 敲击 舞蹈。 太平洋 钢鼓 岛屿 节日。 仪式 节奏 锁定 集体。 太平洋 吉他 弹唱 叙事。 海洋 拍岸 风声 自然 采样。"
    },
    en: {
      "index.heroLabel": "Cross-Cultural Music Discovery",
      "index.stat.tracksLabel": "Tracks",
      "index.stat.tracksCopy": "Real music from around the world",
      "index.stat.culturesLabel": "Cultures",
      "index.stat.culturesCopy": "cultural domains from east to west",
      "index.stat.readyLabel": "Ready",
      "index.stat.readyCopy": "Covers, notes, and tags are attached",
      "index.feature.startLabel": "01 / Your starting point",
      "index.feature.startCopy": "Upload a song, or choose one from 30,000 tracks. That becomes the start of your music journey.",
      "index.feature.aiLabel": "02 / AI route",
      "index.feature.aiCopy": "AI reads the musical DNA and recommends related sounds from different cultures.",
      "index.feature.memoryLabel": "03 / Saved memories",
      "index.feature.memoryCopy": "Save the songs you like, and your favorites slowly become a personal world-music map.",
      "index.method.label": "How It Works",
      "index.method.copy": "Start with one song. AI reads melody, rhythm, and affect, then guides you toward related sounds across the world.",
      "index.method.bigCopy": "Every song carries its own <span class=\"highlight\">sound DNA</span>. Upload music you love, and AI will find unfamiliar sounds that may feel immediately close.",
      "index.signal.label": "Music Journey",
      "index.signal.title": "Sound<br>Route",
      "index.signal.copy": "Upload a song and AI draws a listening route, crossing language and cultural boundaries from your starting point.",
      "index.signal.orbit": "content - style - affect - feedback - route -",
      "index.signal.node.zc": "ZC<br>melody / rhythm",
      "index.signal.node.zs": "ZS<br>culture / style",
      "index.signal.node.za": "ZA<br>valence / arousal",
      "settings.recommendCountLabel": "Recommendations per request",
      "settings.recommendCountHelp": "Number of songs returned per cross-cultural recommendation (1-30, default 10).",
      "settings.recommendCountSaved": "Recommendation count saved.",
      "index.globe.surface": "Global Music Surface",
      "index.globe.autoRotate": "Auto Rotate",
      "index.globe.regionInitializing": "Region Initializing",
      "index.globe.genreSignal": "Genre Signal Active"
    }
  };

  const fallbackGlobeRegions = {
    eastAsia: {
      name: "EAST ASIA",
      style: "GUQIN / ENKA / PANSORI / CITY POP",
      ring: "EAST ASIA - GUQIN - PIPA - ERHU - DIZI - ENKA - J-POP - K-POP - CITY POP - PANSORI - CANTONOPERA - NOH - ",
      text: "EAST ASIA SURFACE: GUQIN HARMONICS FINGER NOISE. PIPA TREMOLO DOWNSTROKE. ERHU PORTAMENTO VIBRATO. DIZI FLOWER TONGUE BREATH. SHAKUHACHI BLOW RESONANCE. YANGQIN ROLL POINTILLISM. XIAO BREATH LONG TONE. QIN SONG CHANT CLASSICAL. GAGAKU CEREMONY DRONE. NOH UTTERANCE MASK. KABUKI NAGAUTA SHAMISEN. ENKA VIBRATO MELANCHOLY. PANSORI NARRATIVE DRUM SHOUT. GAYAGEUM SANJO SWAY. K-POP HOOK CHOREO SYNC. J-POP MELODY DENSITY HARMONY. CITY POP BRASS SYNTHESIZER. CANTONPOP LYRIC CONTOUR WORDPAINT. TAIWANESE BALLAD TURN PHRASE. PENTATONIC GONG SHANG JUE ZHI YU MOTIF. "
    },
    southeastAsia: {
      name: "SOUTHEAST ASIA",
      style: "GAMELAN / DANGDUT / KULINTANG",
      ring: "SOUTHEAST ASIA - GAMELAN - DANGDUT - KULINTANG - PINPEAT - MOR LAM - KERONCONG - THAI CLASSICAL - VIET FOLK - ",
      text: "SOUTHEAST ASIA SURFACE: GAMELAN INTERLOCKING METALLOPHONES CYCLE. GAMELAN GANGSA BRONZE LAYERING. DANGDUT DRUM PULSE ELECTRONIC. KULINTANG GONG ROW UNISON. PINPEAT CEREMONIAL BOW DRIVE. MOR LAM VOCAL ORNAMENT HIGH. KERONCONG STRUM PORTUGUESE HYBRID. SHADOW PLAY TIMBRE SWAY PUPPET. THAI CLASSICAL MAHORI HARP. VIETNAMESE DAN BAU TREMOLO. INDONESIAN POP DANGDUT GROOVE. MALAY TRADITIONAL REBAB NOBAT. BURMESE SAUNG GAUK HARPSERIES. LUZON KULINTANG GONG ENSEMBLE. BALI ANGKLUNG BAMBOO SHAKER. LUK THUNG MELODIC COUNTRY SWAY. "
    },
    southAsia: {
      name: "SOUTH ASIA",
      style: "RAGA / QAWWALI / FILMI / BHANGRA",
      ring: "SOUTH ASIA - RAGA - TABLA - SITAR - QAWWALI - FILMI - BHANGRA - GHAZAL - CARNATIC - SINDHI - ",
      text: "SOUTH ASIA SURFACE: HINDUSTANI RAGA ASCENT DESCENT UNFOLD. CARNATIC KRITI STRUCTURE VARIATION. TABLA BOL CYCLE IMPROVISATION. MRIDANGAM RESONANCE BEAT. SITAR STRING GLIDE RESONANCE. SAROD PLUCK DEEP SUSTAIN. SARANGI BOW CRY TONE. BANSURI BAMBOO FLUTE BREATH. QAWWALI CLAP RESPONSE REPETITION. FILMI ORCHESTRAL SWELL HYBRID. BHANGRA DANCE GROOVE DHOL HIT. GHAZAL POETIC MELODY LOVE. SINDHI DEVOTIONAL DRUM CALL. LAVANI MUMBAI DANCE HALL MELODY. PAHARI HILL SONG PIPE DRONE. SOUTH INDIA VEENA GLIDE HARMONICS. ALAPANA RAPID ALTERNATE SYLLABLE. DHRUPAD SOLEMN TINTAL SLOW. "
    },
    mena: {
      name: "MENA",
      style: "MAQAM / OUD / RAI / GNAWA",
      ring: "MENA - MAQAM - OUD - QANUN - NAY - RAI - GNAWA - DABKE - TAQSIM - SHABI - MUWASHSHAH - ",
      text: "MENA SURFACE: MAQAM MICROTONAL PATH MODULATION. TAQSIM IMPROVISATION FREE BREATH. OUD PLUCKED RESONANCE HARMONICS. QANUN GLISSANDO STRUM ORNAMENT. NAY BREATH TREMOLO ETHEREAL. KAMANCHA BOW ROTATE SPIRAL. SANTUR HAMMER DULCIMER HARMONICS. RAI ELECTRIC VOCAL EDGE NARRATIVE. GNAWA GUEMBRI TRANCE REPETITION. DABKE LINE DANCE PULSE HEAVY. SHABI LYRIC SWAY TUNISIAN. MUWASHSHAH CLASSIC CHORAL ANDALUSIAN. TAR NORTH AFRICAN DRUM PALM. BENDIR FRAME DRUM SUFI SPIN. MAKAMASHI MOROCCO STREET RHYTHM. PERSIAN SANTUR TAR SEQUENCE. IRAQI MAQAM NASAL LONG TONE. ARAB PLUCK OUD RESONANCE CHAMBER. "
    },
    europe: {
      name: "EUROPE",
      style: "FADO / FLAMENCO / CHANSON / TECHNO",
      ring: "EUROPE - FADO - FLAMENCO - CHANSON - CELTIC - TECHNO - POLYPHONY - KRAUTROCK - BAROQUE - SLAVIC - ",
      text: "EUROPE SURFACE: FADO SAUDADE LINE GUITAR. FLAMENCO PALMAS RASGUEADO DANCE CRIER. CHANSON TEXT FORWARD POETIC FORM. CELTIC REEL PIPE FIDDLE STEP. BALKAN ASYMMETRIC BRASS METER. SLAVIC CHORAL HARMONY MELANCHOLY. ALPINE YODEL FALSETTO SWITCH. BAROQUE BASSO CONTINUO HARPSICHORD COUNTERPOINT. CLASSICAL SONATA DEVELOPMENT RECAPTURE. ROMANTIC PROGRAM MUSIC POETIC. IMPRESSIONIST TIMBRE HARMONY BLUR. TECHNO FOUR ON FLOOR SYNTHESIZER. KRAUTROCK MOTORIK REPETITION FLOW. AMBIENT TEXTURE FLOAT SPACE. POST PUNK NOISE RAW REBEL. NORDIC ICELAND ATMOSPHERE ETHEREAL. CELTIC HARP HARMONICS CLASSICAL. BALLAD NARRATIVE ACCORDION HARMONICA. OPERA ARIA BEL CANTO DRAMA. POLYPHONY GREGORIAN MOTET COUNTERPOINT. "
    },
    westAfrica: {
      name: "WEST AFRICA",
      style: "AFROBEAT / HIGHLIFE / MBALAX / KORA",
      ring: "WEST AFRICA - AFROBEAT - HIGHLIFE - MBALAX - KORA - JUJU - PALM WINE - GRIOT - BURKINA - MANDING - ",
      text: "WEST AFRICA SURFACE: AFROBEAT HORN OSTINATO FELA DRIVE. HIGHLIFE GUITAR LATTICE BRASS DANCE. MBALAX SABAR POLYRHYTHM CUT. GRIOT KORA ARPEGGIO NARRATIVE EPIC. JUJU TALKING DRUM RESPONSE PRAISE. PALM WINE GUITAR LIGHT STRUM JAM. FUJI VOCAL PRAISE DRUM CALL. TALKING DRUM PITCH SHIFT IMITATION. DJEMBE PALM BASS HIGH SLAP. BALAFON WOOD XYLOPHONE HARMONICS. MBIRA IRON KEYS INTERLOCK ORNAMENT. SENEGAL SABAR CONTEST DANCE. MALI MANDING EPIC CHANT LONG. GHANA ADINKRA DRUM LANGUAGE SIGNAL. NIGERIA YORUBA DRUM RITUAL ENSEMBLE. CALL RESPONSE FIELD ACTIVE DIALOGUE. POLYRHYTHM MULTILAYER CROSS DISPLACEMENT. AFRICAN BLOW BALAFON LOOP CYCLE. "
    },
    americas: {
      name: "AMERICAS",
      style: "JAZZ / BLUES / SALSA / SAMBA",
      ring: "AMERICAS - JAZZ - BLUES - GOSPEL - SALSA - SAMBA - CUMBIA - HIP HOP - REGGAETON - TANGO - BLUEGRASS - ",
      text: "AMERICAS SURFACE: BLUES BENT NOTE TWELVE BAR CALL. JAZZ SWING RIDE CYMBAL IMPROVISE HARMONY. GOSPEL CALL RESPONSE CHOIR ORGAN PIANO. SALSA CLAVE BRASS PIANO MONTUNO. SAMBA BATUCADA CUT STOMP. BOSSA NOVA GUITAR SWAY WALK BASS. CUMBIA ACCORDION LINE DRUM PULSE. HIP HOP BREAKBEAT SAMPLE FLOW. REGGAETON DEMBOW BASS LOOP. COUNTRY FIDDLE BANJO NARRATIVE. BLUEGRASS MANDOLIN FLATPICK FAST. TANGO BANDONEON TRAGIC CUT. CANDELE RITUAL DRUM CALL. REGGAE CUT SOCIAL COMMENT SKA. SKA CUT JAMAICA BRASS HORN. SOCA FESTIVAL TRINIDAD PAN DRUM. ANDEAN PANPIPE QUENA LAMENT. MARIACHI TRUMPET BRASS CHORUS. CALYPSO IRON DRUM SATIRE NARRATIVE. CHARANGA FIDDLE MEXICAN DANCE. FORRO ACCORDION ZABUMBA TRIANGLE. "
    },
    oceania: {
      name: "OCEANIA / PACIFIC",
      style: "HULA / HAKA / YIDAKI / ISLAND REGGAE",
      ring: "OCEANIA PACIFIC - HULA - HAKA - YIDAKI - ISLAND REGGAE - CHORAL CHANT - LAKALAKA - SURF ROCK - ",
      text: "OCEANIA PACIFIC SURFACE: HULA MELODY STEEL GUITAR GRASS SKIRT. HAKA GROUP CHANT FORCE STOMP. YIDAKI DRONE CIRCULAR BREATH TEXTURE. ISLAND REGGAE SKANK TROPICAL HARMONY. POLYNESIAN CHORAL STACK HARMONY. MAORI PATTE CHANT WAR DANCE. LAKALAKA CHORAL DANCE TONGA. SAMOAN SIVA DANCE SONG. AUSTRALIA ABORIGINAL YIDAKI DREAMTIME. AUSTRALIA ROCK PUNK INDIE INDEPENDENT. SURF ROCK GUITAR REVERB HAWAIIAN. HAWAIIAN UKULELE GLIDE STRUM. MELANESIAN PANPIPE ENSEMBLE BASS. MICRONESIA BAMBOO STICK DANCE. PACIFIC STEEL DRUM ISLAND FESTIVAL. CEREMONY RHYTHM LOCKED COLLECTIVE. PACIFIC GUITAR STRUM BALLAD NARRATIVE. OCEAN WAVE WIND NATURE FIELD SAMPLE. "
    }
  };

  const localizedHomeOverrides = {
    de: {
      "index.heroLabel": "Interkulturelle Musikentdeckung",
      "index.stat.tracksLabel": "Titel",
      "index.stat.tracksCopy": "Echte Musik aus aller Welt",
      "index.stat.culturesLabel": "Kulturen",
      "index.stat.culturesCopy": "Kulturräume von Ost bis West",
      "index.stat.readyLabel": "Bereit",
      "index.stat.readyCopy": "Cover, Notizen und Tags sind vorhanden",
      "index.feature.startLabel": "01 / Dein Startpunkt",
      "index.feature.startCopy": "Lade einen Song hoch oder wähle einen aus 30.000 Titeln. Dort beginnt deine Musikreise.",
      "index.feature.aiLabel": "02 / AI führt",
      "index.feature.aiCopy": "AI liest die musikalische DNA und empfiehlt verwandte Klänge aus anderen Kulturen.",
      "index.feature.memoryLabel": "03 / Erinnerungen sammeln",
      "index.feature.memoryCopy": "Speichere Songs, die dir gefallen; daraus wächst deine persönliche Weltmusikkarte.",
      "index.method.label": "So funktioniert es",
      "index.method.copy": "Starte mit einem Song. AI liest Melodie, Rhythmus und Stimmung und führt dich zu verwandten Klängen weltweit.",
      "index.method.bigCopy": "Jeder Song trägt seine eigene <span class=\"highlight\">Klang-DNA</span>. Lade Musik hoch, die du liebst, und AI findet unbekannte Klänge, die sich sofort nah anfühlen können.",
      "index.signal.label": "Musikreise",
      "index.signal.title": "Klang<br>Route",
      "index.signal.copy": "Lade einen Song hoch; AI zeichnet daraus eine Hörroute über sprachliche und kulturelle Grenzen hinweg.",
      "index.signal.orbit": "Inhalt - Stil - Affekt - Feedback - Route -",
      "index.signal.node.zc": "ZC<br>Melodie / Rhythmus",
      "index.signal.node.zs": "ZS<br>Kultur / Stil",
      "index.signal.node.za": "ZA<br>Valenz / Erregung",
      "index.globe.surface": "Globale Musikoberfläche",
      "index.globe.autoRotate": "Automatische Drehung",
      "index.globe.regionInitializing": "Region wird geladen",
      "index.globe.genreSignal": "Genre-Signal aktiv"
    },
    es: {
      "index.heroLabel": "Descubrimiento musical intercultural",
      "index.stat.tracksLabel": "Pistas",
      "index.stat.tracksCopy": "Música real de todo el mundo",
      "index.stat.culturesLabel": "Culturas",
      "index.stat.culturesCopy": "dominios culturales de este a oeste",
      "index.stat.readyLabel": "Listo",
      "index.stat.readyCopy": "Con portadas, notas y etiquetas",
      "index.feature.startLabel": "01 / Tu punto de partida",
      "index.feature.startCopy": "Sube una canción o elige una entre 30.000 pistas. Ahí empieza tu viaje musical.",
      "index.feature.aiLabel": "02 / Ruta de AI",
      "index.feature.aiCopy": "La AI lee el ADN musical y recomienda sonidos cercanos de otras culturas.",
      "index.feature.memoryLabel": "03 / Recuerdos guardados",
      "index.feature.memoryCopy": "Guarda las canciones que te gusten y tus favoritos formarán un mapa musical personal.",
      "index.method.label": "Cómo funciona",
      "index.method.copy": "Empieza con una canción. La AI lee melodía, ritmo y emoción, y te guía hacia sonidos afines del mundo.",
      "index.method.bigCopy": "Cada canción tiene su propio <span class=\"highlight\">ADN sonoro</span>. Sube música que amas y la AI encontrará sonidos desconocidos que pueden sentirse cercanos.",
      "index.signal.label": "Viaje musical",
      "index.signal.title": "Ruta<br>sonora",
      "index.signal.copy": "Sube una canción y la AI dibuja una ruta de escucha que cruza lenguas y culturas desde tu punto de partida.",
      "index.signal.orbit": "contenido - estilo - afecto - feedback - ruta -",
      "index.signal.node.zc": "ZC<br>melodía / ritmo",
      "index.signal.node.zs": "ZS<br>cultura / estilo",
      "index.signal.node.za": "ZA<br>valencia / activación",
      "index.globe.surface": "Superficie musical global",
      "index.globe.autoRotate": "Rotación automática",
      "index.globe.regionInitializing": "Región iniciando",
      "index.globe.genreSignal": "Señal de género activa"
    },
    pt: {
      "index.heroLabel": "Descoberta musical intercultural",
      "index.stat.tracksLabel": "Faixas",
      "index.stat.tracksCopy": "Música real do mundo todo",
      "index.stat.culturesLabel": "Culturas",
      "index.stat.culturesCopy": "domínios culturais de leste a oeste",
      "index.stat.readyLabel": "Pronto",
      "index.stat.readyCopy": "Capas, notas e etiquetas incluídas",
      "index.feature.startLabel": "01 / Seu ponto de partida",
      "index.feature.startCopy": "Envie uma música ou escolha uma entre 30.000 faixas. A viagem começa ali.",
      "index.feature.aiLabel": "02 / Rota da AI",
      "index.feature.aiCopy": "A AI lê o DNA musical e recomenda sons relacionados de outras culturas.",
      "index.feature.memoryLabel": "03 / Memórias salvas",
      "index.feature.memoryCopy": "Salve as faixas que você gosta e seus favoritos viram um mapa musical pessoal.",
      "index.method.label": "Como funciona",
      "index.method.copy": "Comece com uma faixa. A AI lê melodia, ritmo e afeto, e guia você a sons próximos pelo mundo.",
      "index.method.bigCopy": "Cada música carrega seu próprio <span class=\"highlight\">DNA sonoro</span>. Envie uma faixa querida e a AI encontrará sons desconhecidos que podem parecer próximos.",
      "index.signal.label": "Viagem musical",
      "index.signal.title": "Rota<br>sonora",
      "index.signal.copy": "Envie uma faixa e a AI desenha uma rota de escuta que cruza línguas e culturas a partir do seu ponto inicial.",
      "index.signal.orbit": "conteúdo - estilo - afeto - feedback - rota -",
      "index.signal.node.zc": "ZC<br>melodia / ritmo",
      "index.signal.node.zs": "ZS<br>cultura / estilo",
      "index.signal.node.za": "ZA<br>valência / ativação",
      "index.globe.surface": "Superfície musical global",
      "index.globe.autoRotate": "Rotação automática",
      "index.globe.regionInitializing": "Região inicializando",
      "index.globe.genreSignal": "Sinal de gênero ativo"
    },
    ja: {
      "index.heroLabel": "越境する音楽発見",
      "index.stat.tracksLabel": "楽曲",
      "index.stat.tracksCopy": "世界各地の実際の音楽",
      "index.stat.culturesLabel": "文化圏",
      "index.stat.culturesCopy": "東から西までの文化圏",
      "index.stat.readyLabel": "準備完了",
      "index.stat.readyCopy": "カバー、解説、タグを収録",
      "index.feature.startLabel": "01 / あなたの出発点",
      "index.feature.startCopy": "曲をアップロードするか、30,000曲から選びます。そこが音楽の旅の始まりです。",
      "index.feature.aiLabel": "02 / AI が案内",
      "index.feature.aiCopy": "AI が音楽のDNAを読み、別の文化から近い響きを推薦します。",
      "index.feature.memoryLabel": "03 / 記憶を保存",
      "index.feature.memoryCopy": "好きな曲を保存すると、お気に入りが自分だけの世界音楽地図になります。",
      "index.method.label": "仕組み",
      "index.method.copy": "一曲から始めます。AI が旋律、リズム、感情を読み取り、世界の近い響きへ案内します。",
      "index.method.bigCopy": "すべての曲には固有の<span class=\"highlight\">音のDNA</span>があります。好きな音楽をアップロードすると、AI が未知なのに近しく感じる音を探します。",
      "index.signal.label": "音楽の旅",
      "index.signal.title": "音の<br>ルート",
      "index.signal.copy": "曲をアップロードすると、AI が言語と文化を越えるリスニングルートを描きます。",
      "index.signal.orbit": "内容 - スタイル - 感情 - フィードバック - ルート -",
      "index.signal.node.zc": "ZC<br>旋律 / リズム",
      "index.signal.node.zs": "ZS<br>文化 / スタイル",
      "index.signal.node.za": "ZA<br>快不快 / 覚醒度",
      "index.globe.surface": "グローバル音楽面",
      "index.globe.autoRotate": "自動回転",
      "index.globe.regionInitializing": "地域を初期化中",
      "index.globe.genreSignal": "ジャンル信号が有効"
    },
    ko: {
      "index.heroLabel": "교차 문화 음악 발견",
      "index.stat.tracksLabel": "트랙",
      "index.stat.tracksCopy": "세계 각지의 실제 음악",
      "index.stat.culturesLabel": "문화권",
      "index.stat.culturesCopy": "동쪽에서 서쪽까지의 문화권",
      "index.stat.readyLabel": "준비됨",
      "index.stat.readyCopy": "커버, 설명, 태그 포함",
      "index.feature.startLabel": "01 / 나의 출발점",
      "index.feature.startCopy": "곡을 업로드하거나 30,000곡 중 하나를 고르세요. 거기서 음악 여행이 시작됩니다.",
      "index.feature.aiLabel": "02 / AI 안내",
      "index.feature.aiCopy": "AI가 음악 DNA를 읽고 다른 문화의 가까운 소리를 추천합니다.",
      "index.feature.memoryLabel": "03 / 기억 저장",
      "index.feature.memoryCopy": "좋아하는 곡을 저장하면 즐겨찾기가 나만의 세계 음악 지도가 됩니다.",
      "index.method.label": "작동 방식",
      "index.method.copy": "한 곡에서 시작합니다. AI가 멜로디, 리듬, 정서를 읽고 세계의 가까운 소리로 안내합니다.",
      "index.method.bigCopy": "모든 곡은 고유한 <span class=\"highlight\">사운드 DNA</span>를 지닙니다. 좋아하는 음악을 올리면 AI가 낯설지만 가깝게 느껴질 소리를 찾아냅니다.",
      "index.signal.label": "음악 여행",
      "index.signal.title": "소리<br>경로",
      "index.signal.copy": "곡을 업로드하면 AI가 언어와 문화를 가로지르는 청취 경로를 그립니다.",
      "index.signal.orbit": "콘텐츠 - 스타일 - 정서 - 피드백 - 경로 -",
      "index.signal.node.zc": "ZC<br>멜로디 / 리듬",
      "index.signal.node.zs": "ZS<br>문화 / 스타일",
      "index.signal.node.za": "ZA<br>정서가 / 각성도",
      "index.globe.surface": "글로벌 음악 표면",
      "index.globe.autoRotate": "자동 회전",
      "index.globe.regionInitializing": "지역 초기화 중",
      "index.globe.genreSignal": "장르 신호 활성화"
    }
  };

  const localizedGlobeRegions = {
    de: {
      eastAsia: {
        name: "OSTASIEN",
        style: "GUQIN / ENKA / PANSORI / CITY POP",
        ring: "OSTASIEN - GUQIN - PIPA - ERHU - DIZI - ENKA - J-POP - K-POP - CITY POP - PANSORI - CANTONOPER - NOH - ",
        text: "OSTASIEN MUSIKOBERFLAECHE: GUQIN KLANGE FINGERGERAEUSCH. PIPA TREMOLO SCHLAGABWERTS. ERHU PORTAMENTO VIBRATO. DIZI BLUMENZUNGE ATEM. SHAKUHACHI BLASRESONANZ. YANGQIN ROLLE PUNKTILLISMUS. XIAO LANGER ATEM TON. QIN GESANG CHANT KLASSISCH. GAGAKU ZEREMONIE DRONE. NOH AUSSPRACHE MASKE. KABUKI NAGAUTA SHAMISEN. ENKA VIBRATO MELANCHOLIE. PANSORI ERZAEHLERISCHER TROMMELRUF. GAYAGEUM SANJO SCHWINGUNG. K-POP HOOK CHOREOGRAFIE SYNC. J-POP MELODIEDICHTE HARMONIE. CITY POP BRASS SYNTHESIZER. CANTONPOP BALLADE KONTUR WORTMALEREI. TAIWANISCHE BALLADE DREHRUNG PHRASE. PENTATONISCHE GONG SHANG JUE ZHI YU MOTIVE. "
      },
      southeastAsia: {
        name: "SUEDOSTASIEN",
        style: "GAMELAN / DANGDUT / KULINTANG",
        ring: "SUEDOSTASIEN - GAMELAN - DANGDUT - KULINTANG - PINPEAT - MOR LAM - KERONCONG - THAI KLASSISCH - VIET FOLK - ",
        text: "SUEDOSTASIEN MUSIKOBERFLAECHE: GAMELAN VERZAHNTE METALLOPHONE ZYKLUS. GAMELAN GANGSA BRONZESCHICHTUNG. DANGDUT TROMMELPULS ELEKTRONISCH. KULINTANG GONGREIHE EINKLANG. PINPEAT ZEREMONIELLER BOGENANTRIEB. MOR LAM VOKALORNAMENT HOCH. KERONCONG STRUM PORTUGIESISCHE HYBRIDE. SCHATTENSPIEL TIMBRE SCHWINGUNG PUPPE. THAI KLASSISCH MAHORI HARFE. VIETNAMESE DAN BAU TREMOLO. INDONESISCHER POP DANGDUT GROOVE. MALAY TRADITIONELL REBAB NOBAT. BURMESISCH SAUNG GAUK HARFENSERIE. LUZON KULINTANG GONG ENSEMBLE. BALI ANKLUNG BAMBUSCHUELLER. LUK THUNG MELODISCH LAND SCHWINGUNG. "
      },
      southAsia: {
        name: "SUEDASIEN",
        style: "RAGA / QAWWALI / FILMI / BHANGRA",
        ring: "SUEDASIEN - RAGA - TABLA - SITAR - QAWWALI - FILMI - BHANGRA - GHAZAL - KARNATIK - SINDHI - ",
        text: "SUEDASIEN MUSIKOBERFLAECHE: HINDUSTANI RAGA AUFSTIEG ABSTIEG ENTFALTUNG. KARNATISCHE KRITI STRUKTUR VARIATION. TABLA BOL-ZYKLUS IMPROVISATION. MRIDANGAM RESONANZ SCHLAG. SITAR SAITE GLEITRESONANZ. SAROD ANRIFF TIEF NACHKLANG. SARANGI BOGEN SCHREI TON. BANSURI BAMBUSFLOETE ATEM. QAWWALI KLATSCH-ANTWORT WIEDERHOLUNG. FILMI ORCHESTERWELLE HYBRIDE. BHANGRA TANZGROOVE DHOL SCHLAG. GHAZAL POETISCHE MELODIE LIEBE. SINDHI ANDAECHTIGER TROMMELRUF. LAVANI MUMBAI TANZHALLE MELODIE. PAHARI HUELLIED ROEHRCHEN DRONE. SUEDINDIEN VEENA GLEIT KLANGE. ALAPANA SCHNELLER WECHSEL SYLLABEN. DHRUPAD FEIERLICH TINTAL LANGSAM. "
      },
      mena: {
        name: "NAHOST UND NORDAFRIKA",
        style: "MAQAM / OUD / RAI / GNAWA",
        ring: "NAHOST NORDAFRIKA - MAQAM - OUD - QANUN - NAY - RAI - GNAWA - DABKE - TAQSIM - SHABI - MUWASHSHAH - ",
        text: "NAHOST NORDAFRIKA MUSIKOBERFLAECHE: MAQAM MIKROTONALER WEG MODULATION. TAQSIM IMPROVISATION FREIER ATEM. OUD ZUPFRESONANZ KLANGE. QANUN GLISSANDO STRUM ORNAMENT. NAY ATEM TREMOLO AETHERISCH. KAMANCHA BOGEN DREHUNG SPIRALE. SANTUR HAMMER DULCIMER KLANGE. RAI ELEKTRISCHE STIMME KANTE ERZAHLUNG. GNAWA GUEMBRI TRANCE WIEDERHOLUNG. DABKE REIHENTANZ PULS SCHWER. SHABI LYRIK SCHWINGUNG TUNESISCH. MUWASHSHAH KLASSISCH CHORAL ANDALUSISCH. TAR NORDAFRIKANISCHE TROMMEL HANDFLAECHE. BENDIR RAHMENTROMMEL SUFI DREHUNG. MAKAMASHI MAROKKO STRASSENRHYTHMUS. PERSISCH SANTUR TAR SEQUENZ. IRAKISCH MAQAM NASA LANGER TON. ARABISCH ANRIFF OUD RESONANZRAUM. "
      },
      europe: {
        name: "EUROPA",
        style: "FADO / FLAMENCO / CHANSON / TECHNO",
        ring: "EUROPA - FADO - FLAMENCO - CHANSON - KELTISCH - TECHNO - POLYPHONIE - KRAUTROCK - BAROCK - SLAWISCH - ",
        text: "EUROPA MUSIKOBERFLAECHE: FADO SAUDADE LINIE GITARRE. FLAMENCO PALMAS RASGUEADO TANZ RUFE. CHANSON TEXT VORDERGRUNDIG POETISCHE FORM. KELTISCHER REEL PIPE FIDDLE SCHRITT. BALKAN ASYMMETRISCH BRASS METRIK. SLAWISCHER CHORAL HARMONIE MELANCHOLIE. ALPEN JODEL FALSETTO WECHSEL. BAROCK BASSO CONTINUO CEMBALO KONTRAPUNKT. KLASSISCHE SONATE ENTWICKLUNG REKAPITULATION. ROMANTISCHE PROGRAMMMUSIK POETISCH. IMPRESSIONISTISCH TIMBRE HARMONIE VERSCHWIMMEN. TECHNO FOUR ON FLOOR SYNTHESIZER. KRAUTROCK MOTORIK WIEDERHOLUNG FLUSS. AMBIENT TEXTUR SCHWEBEN RAUM. POST PUNK RAW GERAEUSCH REBELL. NORDISCH ISLAND ATMOSPHAERE AETHERISCH. KELTISCHE HARFE KLANGE KLASSISCH. BALLADE ERZAEHLUNG AKKORDION HARMONIKA. OPER ARIE BEL CANTO DRAMA. POLYPHONIE GREGORIANISCH MOTET KONTRAPUNKT. "
      },
      westAfrica: {
        name: "WESTAFRIKA",
        style: "AFROBEAT / HIGHLIFE / MBALAX / KORA",
        ring: "WESTAFRIKA - AFROBEAT - HIGHLIFE - MBALAX - KORA - JUJU - PALM WINE - GRIOT - BURKINA - MANDING - ",
        text: "WESTAFRIKA MUSIKOBERFLAECHE: AFROBEAT BLAESER-OSTINATO FELA ANTRIEB. HIGHLIFE GITARRENGEFLECHT BRASS TANZ. MBALAX SABAR POLYRHYTHMUS SCHNITT. GRIOT KORA ARPEGGIO ERZAHLUNG EPOS. JUJU SPRECHTROMMEL ANTWORT LOB. PALM-WINE-GITARRE LICHTES STRUM JAM. FUJI VOKAL LOB TROMMELRUF. SPRECHTROMMEL TONHOEHE VERSCHIEBUNG NACHAHMUNG. DJEMBE HANDFLAECHE BASS HOCH SCHLAG. BALAFON HOLZXYLOPHON KLANGE. MBIRA EISEN TASTEN VERZAHNUNG ORNAMENT. SENEGAL SABAR WETTBEWERB TANZ. MALI MANDING EPOS CHANT LANG. GHANA ADINKRA TROMMELSPRACHE SIGNAL. NIGERIA YORUBA TROMMEL RITUAL ENSEMBLE. CALL RESPONSE FELD AKTIV DIALOG. POLYRHYTHMUS MEHRSCHICHTIG KREUZ VERSCHIEBUNG. AFRICANISCH BALAFON SCHLEIFE ZYKLUS. "
      },
      americas: {
        name: "AMERIKA",
        style: "JAZZ / BLUES / SALSA / SAMBA",
        ring: "AMERIKA - JAZZ - BLUES - GOSPEL - SALSA - SAMBA - CUMBIA - HIP HOP - REGGAETON - TANGO - BLUEGRASS - ",
        text: "AMERIKA MUSIKOBERFLAECHE: BLUES BENT NOTE ZWOELF TAKT ANRUF. JAZZ SWING RIDE BECKEN IMPROVISATION HARMONIE. GOSPEL CALL RESPONSE CHOR ORGEL KLAVIER. SALSA CLAVE BRASS PIANO MONTUNO. SAMBA BATUCADA SCHNITT TRITT. BOSSA NOVA GITARRE SCHWINGUNG WALKING BASS. CUMBIA AKKORDEONLINIE TROMMELPULS. HIP HOP BREAKBEAT SAMPLE FLUSS. REGGAETON DEMBOW BASS SCHLEIFE. COUNTRY FIDDLE BANJO ERZAHLUNG. BLUEGRASS MANDOLINE FLATPICK SCHNELL. TANGO BANDONEON TRAGISCH SCHNITT. REGGAE SCHNITT SOZIALER KOMMENTAR SKA. SKA SCHNITT JAMAICA BRASS HORN. SOCA FESTIVAL TRINIDAD PAN TROMMEL. ANDEAN PANPFEIFE QUENA KLAGE. MARIACHI TROMPETE BRASS CHOR. CALYPSO EISENTROMMEL SATIRE ERZAHLUNG. CHARANGA FIDDLE MEXIKANISCHER TANZ. FORRO AKKORDION ZABUMBA DREIECK. "
      },
      oceania: {
        name: "OZEANIEN / PAZIFIK",
        style: "HULA / HAKA / YIDAKI / ISLAND REGGAE",
        ring: "OZEANIEN PAZIFIK - HULA - HAKA - YIDAKI - ISLAND REGGAE - CHORCHANT - LAKALAKA - SURF ROCK - ",
        text: "OZEANIEN PAZIFIK MUSIKOBERFLAECHE: HULA MELODIE STEEL GUITAR GRASROCK. HAKA GRUPPENCHANT KRAFT TRITT. YIDAKI DRONE ZIRKULAERER ATEM TEXTUR. ISLAND REGGAE SKANK TROPISCHE HARMONIE. POLYNESISCHER CHORAL STAPEL HARMONIE. MAORI PATTE CHANT KRIEGSTANZ. LAKALAKA CHORALTANZ TONGA. SAMOANISCHER SIVA TANZLIED. AUSTRALIEN ABORIGINES YIDAKI TRAUMZEIT. AUSTRALIEN ROCK PUNK INDIE UNABHAENGIG. SURF ROCK GITARRE REVERB HAWAII. HAWAII UKULELE GLEIT STRUM. MELANESISCH PANPFEIFE ENSEMBLE BASS. MIKRONESIEN BAMBUSSTABTANZ. PAZIFISCHE STEEL DRUM INSELFESTIVAL. ZEREMONIE RHYTHMUS VERGEMEINSCHAFTET. PAZIFISCHE GITARRE STRUM BALLADE ERZAHLUNG. OZEAN WELLE WIND NATUR FELD SAMPLE. "
      }
    },
    es: {
      eastAsia: {
        name: "ASIA ORIENTAL",
        style: "GUQIN / ENKA / PANSORI / CITY POP",
        ring: "ASIA ORIENTAL - GUQIN - PIPA - ERHU - DIZI - ENKA - J-POP - K-POP - CITY POP - PANSORI - CANTONES - NOH - ",
        text: "ASIA ORIENTAL SUPERFICIE MUSICAL: GUQIN ARMONICOS RUIDO DE DEDOS. PIPA TREMOLO GOLPE ABAJO. ERHU PORTAMENTO VIBRATO. DIZI LENGUA FLOR ALIENTO. SHAKUHACHI RESONANCIA DE SOPLO. YANGQIN RODILLO PUNTILLISMO. XIAO TONO LARGO ALIENTO. QIN CANTO CLASICO. GAGAKU DRONE CEREMONIAL. NOH EMISION MASCARA. KABUKI NAGAUTA SHAMISEN. ENKA VIBRATO MELANCOLIA. PANSORI GOLPE NARRATIVO TAMBOR. GAYAGEUM SANJO VAIVEN. K-POP GANCHO COREOGRAFIA SINCRONIA. J-POP DENSIDAD MELODICA ARMONIA. CITY POP METALES SINTETIZADOR. CANTONPOP CONTOUR MUSICAL PINTURA VERBAL. BALADA TAIWANESA GIRO FRASE. MOTIVO PENTATONICO GONG SHANG JUE ZHI YU. "
      },
      southeastAsia: {
        name: "SUDESTE ASIATICO",
        style: "GAMELAN / DANGDUT / KULINTANG",
        ring: "SUDESTE ASIATICO - GAMELAN - DANGDUT - KULINTANG - PINPEAT - MOR LAM - KERONCONG - TAI CLASICO - VIET FOLK - ",
        text: "SUDESTE ASIATICO SUPERFICIE MUSICAL: GAMELAN METALOFONOS ENTRELAZADOS CICLO. GAMELAN GANGSA CAPA DE BRONCE. DANGDUT PULSO DE TAMBOR ELECTRONICO. KULINTANG FILAS DE GONGS UNISON. PINPEAT IMPULSO CEREMONIAL ARCO. MOR LAM ORNAMENTO VOCAL ALTO. KERONCONG RASGUEO HIBRIDO PORTUGUES. TIMBRE DE TEATRO DE SOMBRAS TITERE. TAI CLASICO MAHORI ARPA. VIETNAMESE DAN BAU TREMOLO. POP INDONESIO DANGDUT GROOVE. MALAY TRADICIONAL REBAB NOBAT. BIRMANO SAUNG GAUK SERIE DE ARPAS. LUZON KULINTANG ENSEMBLE DE GONGS. BALI ANKLUNG SACUDIDOR DE BAMBU. LUK THUNG VAIVEN MELODICO CAMPESINO. "
      },
      southAsia: {
        name: "ASIA DEL SUR",
        style: "RAGA / QAWWALI / FILMI / BHANGRA",
        ring: "ASIA DEL SUR - RAGA - TABLA - SITAR - QAWWALI - FILMI - BHANGRA - GHAZAL - CARNATICO - SINDHI - ",
        text: "ASIA DEL SUR SUPERFICIE MUSICAL: RAGA HINDUSTANI ASCENSO DESCENSO DESPLIEGUE. KRITI CARNATICA ESTRUCTURA VARIACION. TABLA CICLOS BOL IMPROVISACION. MRIDANGAM RESONANCIA GOLPE. SITAR CUERDA DESLIZ RESONANCIA. SAROD PULSO SOSTENIDO PROFUNDO. SARANGI ARCO TONO DE LLANTO. BANSURI BAMBU FLAUTA ALIENTO. QAWWALI PALMAS Y RESPUESTA REPETICION. FILMI OLEAJE ORQUESTAL HIBRIDO. BHANGRA GROOVE DE BAILE DHOL GOLPE. GHAZAL LINEA POETICA AMOR. SINDHI DEVOCIONAL LLAMADA DE TAMBOR. LAVANI MUMBAI SALON DE BAILE MELODIA. PAHARI CANCION DE COLINA DRONE TUBO. SUR DE INDIA VEENA DESLIZ ARMONICOS. ALAPANA ALTERNANCIA RAPIDA SILABA. DHRUPAD SOLEMNE TINTAL LENTO. "
      },
      mena: {
        name: "ORIENTE MEDIO Y NORTE DE AFRICA",
        style: "MAQAM / OUD / RAI / GNAWA",
        ring: "ORIENTE MEDIO NORTE DE AFRICA - MAQAM - OUD - QANUN - NAY - RAI - GNAWA - DABKE - TAQSIM - SHABI - MUWASHSHAH - ",
        text: "ORIENTE MEDIO NORTE DE AFRICA SUPERFICIE MUSICAL: MAQAM RUTA MICROTONAL MODULACION. TAQSIM IMPROVISACION ALIENTO LIBRE. OUD RESONANCIA PUNTEADA ARMONICOS. QANUN GLISSANDO RASGUEO ORNAMENTO. NAY ALIENTO TREMOLO ETEREO. KAMANCHA ARCO GIRO ESPIRAL. SANTUR MARTILLO DULCIMER ARMONICOS. RAI VOZ ELECTRICA BORDE NARRATIVO. GNAWA GUEMBRI TRANCE REPETICION. DABKE PULSO DE DANZA LINEAL PESADO. SHABI VAIVEN LIRICO TUNECINO. MUWASHSHAH CLASICO CORAL ANDALUZ. TAR TAMBOR NORTEAFRICANO PALMA. BENDIR MARCO TAMBOR SUFI GIRO. MAKAMASHI MARRUECOS RITMO CALLEJERO. PERSA SANTUR TAR SECUENCIA. IRAQUI MAQAM NASA TONO LARGO. ARABE PULSO OUD RESONANCIA CAMARA. "
      },
      europe: {
        name: "EUROPA",
        style: "FADO / FLAMENCO / CHANSON / TECHNO",
        ring: "EUROPA - FADO - FLAMENCO - CHANSON - CELTA - TECHNO - POLIFONIA - KRAUTROCK - BARROCO - ESLAVO - ",
        text: "EUROPA SUPERFICIE MUSICAL: FADO LINEA DE SAUDADE GUITARRA. FLAMENCO PALMAS RASGUEADO BAILE CANTAOR. CHANSON TEXTO AL FRENTE FORMA POETICA. CELTA MOVIMIENTO DE REEL GAITA VIOLIN. BALCANES METRO ASIMETRICO BRASS. ESLAVO CORAL ARMONIA MELANCOLIA. ALPES YODEL FALSETO CAMBIO. BARROCO BAJO CONTINUO CLAVE CONTRAPUNTO. CLASICA SONATA DESARROLLO RECAPITULACION. ROMANTICA MUSICA PROGRAMATICA POETICA. IMPRESIONISTA TIMBRE ARMONIA DIFUSA. TECHNO CUATRO AL PISO SINTETIZADOR. KRAUTROCK MOTORIK REPETICION FLUJO. AMBIENT TEXTURA FLOTACION ESPACIO. POST PUNK RUIDO CRUDO REBELDE. NORDICO ISLANDIA ATMOSFERA ETEREA. CELTA ARPA ARMONICOS CLASICA. BALADA NARRATIVA ACORDION ARMONICA. OPERA ARIA BEL CANTO DRAMA. POLIFONIA GREGORIANA MOTET CONTRAPUNTO. "
      },
      westAfrica: {
        name: "AFRICA OCCIDENTAL",
        style: "AFROBEAT / HIGHLIFE / MBALAX / KORA",
        ring: "AFRICA OCCIDENTAL - AFROBEAT - HIGHLIFE - MBALAX - KORA - JUJU - PALM WINE - GRIOT - BURKINA - MANDINGO - ",
        text: "AFRICA OCCIDENTAL SUPERFICIE MUSICAL: AFROBEAT OSTINATO DE METALES FELA IMPULSO. HIGHLIFE MALLA DE GUITARRAS BRASS BAILE. MBALAX POLIRRITMO SABAR CORTE. GRIOT KORA ARPEGIO NARRATIVO EPICO. JUJU TAMBOR PARLANTE RESPUESTA ALABANZA. GUITARRA PALM WINE RASGUEO LIGERO JAM. FUJI VOZ ALABANZA LLAMADA DE TAMBOR. TAMBOR PARLANTE CAMBIO TONO IMITACION. DJEMBE PALMA BASS ALTO GOLPE. BALAFON XILOFONO MADERA ARMONICOS. MBIRA TECLAS HIERRO ENTRELAZADO ORNAMENTO. SENEGAL SABAR CONCURSO BAILE. MALI MANDINGO EPICO CANTO LARGO. GHANA ADINKRA LENGUAJE DE TAMBORES SEÑAL. NIGERIA YORUBA TAMBOR RITUAL ENSEMBLE. LLAMADA RESPUESTA CAMPO ACTIVO DIALOGO. POLIRRITMO MULTICAPA CRUZ DESPLAZAMIENTO. AFRICANO BALAFON BUCLE CICLO. "
      },
      americas: {
        name: "AMERICAS",
        style: "JAZZ / BLUES / SALSA / SAMBA",
        ring: "AMERICAS - JAZZ - BLUES - GOSPEL - SALSA - SAMBA - CUMBIA - HIP HOP - REGGAETON - TANGO - BLUEGRASS - ",
        text: "AMERICAS SUPERFICIE MUSICAL: BLUES NOTA DOBLADA DOCE COMPASES LLAMADA. JAZZ SWING RIDE PLATILLO IMPROVISACION ARMONIA. GOSPEL LLAMADA RESPUESTA CORO ORGANO PIANO. SALSA CLAVE BRASS PIANO MONTUNO. SAMBA BATUCADA CORTE PISOTON. BOSSA NOVA GUITARRA VAIVEN BAJO WALKING. CUMBIA LINEA DE ACORDEON PULSO DE TAMBOR. HIP HOP BREAKBEAT SAMPLE FLUJO. REGGAETON DEMBOW LAZO DE BAJO. COUNTRY FIDDLE BANJO NARRATIVO. BLUEGRASS MANDOLINA FLATPICK RAPIDO. TANGO BANDONEON CORTE TRAGICO. REGGAE CORTE COMENTARIO SOCIAL SKA. SKA CORTE JAMAICA BRASS TROMPETA. SOCA FESTIVAL TRINIDAD PAN TAMBOR. ANDEAN PANPASTOR QUENA LAMENTO. MARIACHI TROMPETA BRASS CORO. CALYPSO TAMBOR HIERRO SATIRA NARRATIVA. CHARANGA VIOLIN BAILE MEXICANO. FORRO ACORDION ZABUMBA TRIANGULO. "
      },
      oceania: {
        name: "OCEANIA / PACIFICO",
        style: "HULA / HAKA / YIDAKI / REGGAE ISLENO",
        ring: "OCEANIA PACIFICO - HULA - HAKA - YIDAKI - REGGAE ISLENO - CANTO CORAL - LAKALAKA - SURF ROCK - ",
        text: "OCEANIA PACIFICO SUPERFICIE MUSICAL: HULA MELODIA STEEL GUITAR FALDA PAJA. HAKA CANTO GRUPAL FUERZA PISOTON. YIDAKI DRONE RESPIRACION CIRCULAR TEXTURA. REGGAE ISLENO SKANK ARMONIA TROPICAL. CORAL POLINESIO CAPAS ARMONIA. MAORI PATTE CANTO DANZA GUERRA. LAKALAKA CANTO DANZA TONGA. SAMOANO SIVA CANCION DANZA. AUSTRALIA ABORIGEN YIDAKI TIEMPO SUENO. AUSTRALIA ROCK PUNK INDIE INDEPENDIENTE. SURF ROCK GUITARRA REVERB HAWAIIANO. HAWAII UKULELE DESLIZ RASGUEO. MELANESIO PASTOR ENSEMBLE BAJO. MICRONESIA DANZA BASTON BAMBU. PACIFICO STEEL DRUM FIESTA ISLENA. CEREMONIA RITMO FIJO COLECTIVO. PACIFICO GUITARRA RASGUEO BALADA NARRATIVA. OCCEANO VIENTO NATURA CAMPO SAMPLE. "
      }
    },
    pt: {
      eastAsia: {
        name: "ASIA ORIENTAL",
        style: "GUQIN / ENKA / PANSORI / CITY POP",
        ring: "ASIA ORIENTAL - GUQIN - PIPA - ERHU - DIZI - ENKA - J-POP - K-POP - CITY POP - PANSORI - CANTONES - NOH - ",
        text: "ASIA ORIENTAL SUPERFICIE MUSICAL: GUQIN ARMONICOS RUIDO DE DEDOS. PIPA TREMOLO GOLPE BAIXO. ERHU PORTAMENTO VIBRATO. DIZI LINGUA FLOR SOPRO. SHAKUHACHI RESSONANCIA DE SOPRO. YANGQIN ROLAGEM PONTILHISMO. XIAO SOPRO TOM LONGO. QIN CANTO CLÁSSICO. GAGAKU DRONE CERIMONIAL. NOH EMISSAO MASCARA. KABUKI NAGAUTA SHAMISEN. ENKA VIBRATO MELANCOLIA. PANSORI GOLPE NARRATIVO TAMBOR. GAYAGEUM SANJO BALANCO. K-POP GANCHO COREOGRAFIA SINCRONIA. J-POP DENSIDADE MELODICA HARMONIA. CITY POP METAIS SINTETIZADOR. CANTONPOP CONTOUR MUSICAL PINTURA VERBAL. BALADA TAIWANESA GIRO FRASE. MOTIVO PENTATONICO GONG SHANG JUE ZHI YU. "
      },
      southeastAsia: {
        name: "SUDESTE ASIATICO",
        style: "GAMELAN / DANGDUT / KULINTANG",
        ring: "SUDESTE ASIATICO - GAMELAN - DANGDUT - KULINTANG - PINPEAT - MOR LAM - KERONCONG - TAI CLÁSSICO - VIET FOLK - ",
        text: "SUDESTE ASIATICO SUPERFICIE MUSICAL: GAMELAN METALOFONES ENTRELAÇADOS CICLO. GAMELAN GANGSA CAMADA DE BRONZE. DANGDUT PULSO DE TAMBOR ELETRONICO. KULINTANG FILEIRAS DE GONGS UNISSAO. PINPEAT IMPULSO CERIMONIAL ARCO. MOR LAM ORNAMENTO VOCAL ALTO. KERONCONG BATIDA DE CORDAS HIBRIDO PORTUGUES. TIMBRE DE TEATRO DE SOMBRAS TITERE. TAI CLÁSSICO MAHORI HARPA. VIETNAMESE DAN BAU TREMOLO. POP INDONESIO DANGDUT GROOVE. MALAI TRADICIONAL REBAB NOBAT. BIRMANO SAUNG GAUK SERIE DE HARPAS. LUZON KULINTANG ENSEMBLE DE GONGS. BALI ANKLUNG CHOCALHO DE BAMBU. LUK THUNG BALANCO MELODICO CAMPONES. "
      },
      southAsia: {
        name: "ASIA DO SUL",
        style: "RAGA / QAWWALI / FILMI / BHANGRA",
        ring: "ASIA DO SUL - RAGA - TABLA - SITAR - QAWWALI - FILMI - BHANGRA - GHAZAL - CARNATICO - SINDHI - ",
        text: "ASIA DO SUL SUPERFICIE MUSICAL: RAGA HINDUSTANI ASCENSAO DESCENSAO DESDOBRAMENTO. KRITI CARNATICA ESTRUTURA VARIACAO. TABLA CICLOS BOL IMPROVISACAO. MRIDANGAM RESSONANCIA GOLPE. SITAR CORDA DESLIZ RESSONANCIA. SAROD PULSO SUSTENTADO PROFUNDO. SARANGI ARCO TOM DE CHORO. BANSURI BAMBU FLAUTA SOPRO. QAWWALI PALMAS E RESPOSTA REPETICAO. FILMI ONDA ORQUESTRAL HIBRIDO. BHANGRA GROOVE DE DANCA DHOL GOLPE. GHAZAL LINHA POETICA AMOR. SINDHI DEVOCIONAL CHAMADA DE TAMBOR. LAVANI MUMBAI SALAO DE DANCA MELODIA. PAHARI CANCAO DE COLINA DRONE TUBO. SUL DA INDIA VEENA DESLIZ ARMONICOS. ALAPANA ALTERNANCIA RAPIDA SILABA. DHRUPAD SOLENE TINTAL LENTO. "
      },
      mena: {
        name: "ORIENTE MEDIO E NORTE DA AFRICA",
        style: "MAQAM / OUD / RAI / GNAWA",
        ring: "ORIENTE MEDIO NORTE DA AFRICA - MAQAM - OUD - QANUN - NAY - RAI - GNAWA - DABKE - TAQSIM - SHABI - MUWASHSHAH - ",
        text: "ORIENTE MEDIO NORTE DA AFRICA SUPERFICIE MUSICAL: MAQAM CAMINHO MICROTONAL MODULACAO. TAQSIM IMPROVISACAO SOPRO LIVRE. OUD RESSONANCIA DEDILHADA ARMONICOS. QANUN GLISSANDO RASGUEO ORNAMENTO. NAY SOPRO TREMOLO ETEREO. KAMANCHA ARCO GIRO ESPIRAL. SANTUR MARTELO DULCIMER ARMONICOS. RAI VOZ ELETRICA BORDA NARRATIVA. GNAWA GUEMBRI TRANCE REPETICAO. DABKE PULSO DE DANCA LINEAR PESADO. SHABI BALANCO LIRICO TUNISIANO. MUWASHSHAH CLÁSSICO CORAL ANDALUZ. TAR TAMBOR NORTE-AFRICANO PALMA. BENDIR TAMBORETE SUFI GIRO. MAKAMASHAI MARROCOS RITMO DE RUA. PERSA SANTUR TAR SEQUENCIA. IRAQUIANO MAQAM NASA TOM LONGO. ARABE PULSO OUD RESSONANCIA CAMARA. "
      },
      europe: {
        name: "EUROPA",
        style: "FADO / FLAMENCO / CHANSON / TECHNO",
        ring: "EUROPA - FADO - FLAMENCO - CHANSON - CELTA - TECHNO - POLIFONIA - KRAUTROCK - BARROCO - ESLAVO - ",
        text: "EUROPA SUPERFICIE MUSICAL: FADO LINHA DE SAUDADE GUITARRA. FLAMENCO PALMAS RASGUEADO DANCA CANTAOR. CHANSON TEXTO EM PRIMEIRO PLANO FORMA POETICA. CELTA MOVIMENTO DE REEL GAITA VIOLINO. BALCANES METRO ASSIMETRICO BRASS. ESLAVO CORAL HARMONIA MELANCOLIA. ALPES YODEL FALSETO MUDANCA. BARROCO BAIXO CONTINUO CLAVE CONTRAPUNTO. CLÁSSICA SONATA DESENVOLVIMENTO RECAPITULACAO. ROMANTICA MUSICA PROGRAMATICA POETICA. IMPRESSIONISTA TIMBRE HARMONIA DIFUSA. TECHNO QUATRO NO CHAO SINTETIZADOR. KRAUTROCK MOTORIK REPETICAO FLUXO. AMBIENT TEXTURA FLOTACAO ESPACO. POST PUNK RUIDO CRU REBELDE. NORDICO ISLANDIA ATMOSFERA ETEREA. CELTA HARPA ARMONICOS CLÁSSICA. BALADA NARRATIVA ACORDEAO HARMONICA. OPERA ARIA BEL CANTO DRAMA. POLIFONIA GREGORIANA MOTET CONTRAPUNTO. "
      },
      westAfrica: {
        name: "AFRICA OCIDENTAL",
        style: "AFROBEAT / HIGHLIFE / MBALAX / KORA",
        ring: "AFRICA OCIDENTAL - AFROBEAT - HIGHLIFE - MBALAX - KORA - JUJU - PALM WINE - GRIOT - BURKINA - MANDINGO - ",
        text: "AFRICA OCIDENTAL SUPERFICIE MUSICAL: AFROBEAT OSTINATO DE METAIS FELA IMPULSO. HIGHLIFE MALHA DE GUITARRAS BRASS DANCA. MBALAX POLIRRITMO SABAR CORTE. GRIOT KORA ARPEGIO NARRATIVO EPICO. JUJU TAMBOR FALANTE RESPOSTA LOUVOR. GUITARRA PALM WINE RASGUEO LEVE JAM. FUJI VOZ LOUVOR CHAMADA DE TAMBOR. TAMBOR FALANTE MUDANCA TOM IMITACAO. DJEMBE PALMA BASS ALTO GOLPE. BALAFON XILOFONO MADEIRA ARMONICOS. MBIRA TECLAS FERRO ENTRELAÇADO ORNAMENTO. SENEGAL SABAR CONCURSO DANCA. MALI MANDINGO EPICO CANTO LONGO. GHANA ADINKRA LINGUAGEM DE TAMBORES SINAL. NIGERIA YORUBA TAMBOR RITUAL ENSEMBLE. CHAMADA RESPOSTA CAMPO ATIVO DIALOGO. POLIRRITMO MULTICAMADA CRUZ DESLOCAMENTO. AFRICANO BALAFON LOOP CICLO. "
      },
      americas: {
        name: "AMERICAS",
        style: "JAZZ / BLUES / SALSA / SAMBA",
        ring: "AMERICAS - JAZZ - BLUES - GOSPEL - SALSA - SAMBA - CUMBIA - HIP HOP - REGGAETON - TANGO - BLUEGRASS - ",
        text: "AMERICAS SUPERFICIE MUSICAL: BLUES NOTA DOBRADA DOZE COMPASSOS CHAMADA. JAZZ SWING RIDE PRATO IMPROVISACAO HARMONIA. GOSPEL CHAMADA RESPOSTA CORO ORGAO PIANO. SALSA CLAVE BRASS PIANO MONTUNO. SAMBA BATUCADA CORTE PISAO. BOSSA NOVA GUITARRA BALANCO BAIXO WALKING. CUMBIA LINHA DE ACORDEAO PULSO DE TAMBOR. HIP HOP BREAKBEAT SAMPLE FLUXO. REGGAETON DEMBOW ALCA DE BAIXO. COUNTRY FIDDLE BANJO NARRATIVO. BLUEGRASS MANDOLINA FLATPICK RAPIDO. TANGO BANDONEAO CORTE TRAGICO. REGGAE CORTE COMENTARIO SOCIAL SKA. SKA CORTE JAMAICA BRASS TROMBETA. SOCA FESTIVAL TRINIDAD PAN TAMBOR. ANDEAN PIFE QUENA LAMENTO. MARIACHI TROMBETA BRASS CORO. CALYPSO TAMBOR FERRO SATIRA NARRATIVA. CHARANGA VIOLINO DANCA MEXICANA. FORRO ACORDEAO ZABUMBA TRIANGULO. "
      },
      oceania: {
        name: "OCEANIA / PACIFICO",
        style: "HULA / HAKA / YIDAKI / REGGAE INSULAR",
        ring: "OCEANIA PACIFICO - HULA - HAKA - YIDAKI - REGGAE INSULAR - CANTO CORAL - LAKALAKA - SURF ROCK - ",
        text: "OCEANIA PACIFICO SUPERFICIE MUSICAL: HULA MELODIA STEEL GUITAR SAIA DE PALHA. HAKA CANTO COLETIVO FORCA PISAO. YIDAKI DRONE RESPIRACAO CIRCULAR TEXTURA. REGGAE INSULAR SKANK HARMONIA TROPICAL. CORAL POLINESIO CAMADAS HARMONIA. MAORI PATTE CANTO DANCA GUERRA. LAKALAKA CANTO DANCA TONGA. SAMOANO SIVA CANCAO DANCA. AUSTRALIA ABORIGINE YIDAKI TEMPO SONHO. AUSTRALIA ROCK PUNK INDIE INDEPENDENTE. SURF ROCK GUITARRA REVERB HAWAIANO. HAWAI UKULELE DESLIZ RASGUEO. MELANESIO PIFE ENSEMBLE BAIXO. MICRONESIA DANCA BASTAO BAMBU. PACIFICO STEEL DRUM FESTA ILHA. CERIMONIA RITMO FIXO COLETIVO. PACIFICO GUITARRA RASGUEO BALADA NARRATIVA. OCEANO VIENTO NATURALEZA CAMPO SAMPLE. "
      }
    },
    ja: {
      eastAsia: {
        name: "東アジア",
        style: "古琴 / 演歌 / パンソリ / シティポップ",
        ring: "東アジア - 古琴 - 琵琶 - 二胡 - 笛子 - 演歌 - J-POP - K-POP - シティポップ - パンソリ - 粤劇 - 能 - ",
        text: "東アジア 音楽面：古琴 泛音 指のノイズ。琵琶 トレモロ 下弦。二胡 ポルタメント ビブラート。笛子 舌花 息。尺八 吹奏共鳴。揚琴 ロール ポアンティリズム。洞簫 長い息の音。琴 吟唱 古典。雅楽 儀式ドローン。能 発声 面。歌舞伎 長唄 三味線。演歌 ビブラート 憂愁。パンソリ 語り 太鼓の叫び。伽倻琴 散揺 揺らぎ。K-POP フック 振付 同期。J-POP メロディ密度 和声。シティポップ ブラス シンセサイザー。粤劇 歌詞輪郭 言葉描き。台湾歌謡 転回句。五声音階 宮商角徴羽 モチーフ 活性。 "
      },
      southeastAsia: {
        name: "東南アジア",
        style: "ガムラン / ダンドゥット / クリンタン",
        ring: "東南アジア - ガムラン - ダンドゥット - クリンタン - ピンピート - モーラム - クロンチョン - タイ古典 - ベトナム民謡 - ",
        text: "東南アジア 音楽面：ガムラン 交差する金属打楽器 サイクル。ガムラン ガンサ 青銅重層。ダンドゥット 太鼓パルス 電子。クリンタン ゴング列 斉奏。ピンピート 儀礼的弓推進。モーラム 声の装飾 高音。クロンチョン 弦ストラム ポルトガル混血。影絵の音色 操り人形。タイ古典 マホリ ハープ。ベトナム 独弦琴 トレモロ。インドネシアンポップ ダンドゥット グルーヴ。マレー伝統 ラバブ ノバット。ビルマ サウンゴー ハープ連鎖。ルソン クリンタン ゴングアンサンブル。バリ アングルン 竹シェーカー。ルクトゥン 田舎メロディ 揺らぎ。 "
      },
      southAsia: {
        name: "南アジア",
        style: "ラーガ / カッワーリー / 映画音楽 / バングラ",
        ring: "南アジア - ラーガ - タブラ - シタール - カッワーリー - 映画音楽 - バングラ - ガザル - カルナータカ - シンディ - ",
        text: "南アジア 音楽面：ヒンドゥスターニー ラーガ 上行下行 展開。カルナータカ クリティ 構造 変奏。タブラ ボル周期 即興。ムリダンガム 共鳴 打音。シタール 弦 滑走 共鳴。サロード 弾弦 深い持続。サランギ 弓 泣きの音。バンスリ 竹笛 息。カッワーリー 手拍子応答 繰り返し。映画音楽 オーケストラの高まり 混成。バングラ 舞踊グルーヴ ドール打音。ガザル 詩的旋律 愛。シンディ 奉献 太鼓の呼び声。ラヴァニ ムンバイ ダンスホール旋律。パハリ 丘の歌 パイプドローン。南インド ヴィーナ 滑走 泛音。アラピパ 音交替 速い音節。ドルパード 厳か ティンタール 緩やか。 "
      },
      mena: {
        name: "中東・北アフリカ",
        style: "マカーム / ウード / ライ / グナワ",
        ring: "中東 北アフリカ - マカーム - ウード - カーヌーン - ナーイ - ライ - グナワ - ダブケ - タクシーム - シャービ - ムワッシャハ - ",
        text: "中東 北アフリカ 音楽面：マカーム 微分音の道筋 転調。タクシーム 即興 自由な息。ウード 撥弦の共鳴 泛音。カーヌーン グリッサンド 弾き装飾。ナーイ 息 トレモロ 空霊的。カマンチャー 弓 回転 螺旋。サントゥル ハンマー ダルシマー 泛音。ライ 電子的な声の縁 語り。グナワ ゲンブリのトランス 繰り返し。ダブケ 列舞 パルス 重い。シャービ 歌詞の揺らぎ チュニジア的。ムワッシャハ 古典 コラル アンダルシア。タール 北アフリカ太鼓 掌。ベンディル 枥太鼓 スフィ 回転。マカマーシ モロッコ 街のリズム。ペルシア サントゥル タール 連鎖。イラク マカーム 鼻声 長い音。アラブ 弾き ウード 共鳴室。 "
      },
      europe: {
        name: "ヨーロッパ",
        style: "ファド / フラメンコ / シャンソン / テクノ",
        ring: "ヨーロッパ - ファド - フラメンコ - シャンソン - ケルト - テクノ - ポリフォニー - クラウトロック - バロック - スラヴ - ",
        text: "ヨーロッパ 音楽面：ファド サウダージの線 ギター。フラメンコ 手拍子 ラスゲアード 舞踊 歌い手。シャンソン テキスト前景 詩的形態。ケルト リール パイプ フィドル 歩み。バルカン 非対称 ブラス 拍子。スラヴ コラル 和声 憂愁。アルプス ヨーデル ファルセット 切り替え。バッソ・コンティヌオ チェンバロ 対位法。古典ソナタ 展開 再現。ロマン派 プログラム音楽 詩的。印象派 音色 和声 ぼかし。テクノ 四つ打ち シンセサイザー。クラウトロック モトリック 繰り返し 流動。アンビエント テクスチャー 浮遊 空間。ポストパンク 電撃 原始 反逆。北欧 アイスランド 雲海 空霊。ケルト ハープ 泛音 古典。バラード 語り アコーディオン ハーモニカ。オペラ アリア ベルカント 劇。ポリフォニー グレゴリオ モテット 対位法。 "
      },
      westAfrica: {
        name: "西アフリカ",
        style: "アフロビート / ハイライフ / ンバラ / コラ",
        ring: "西アフリカ - アフロビート - ハイライフ - ンバラ - コラ - ジュジュ - パームワイン - グリオ - ブルキナ - マンディング - ",
        text: "西アフリカ 音楽面：アフロビート ホーンのオスティナート フェラ推進。ハイライフ ギター格子 ブラス 舞踊。ンバラ サバールのポリリズム 切り。グリオ コラのアルペジオ 語り 叙事詩。ジュジュ トーキングドラム 応答 賛美。パームワインギター 軽い弾き ジャム。フジ 声 賛美 太鼓の呼び声。トーキングドラム 音高変位 模倣。ジャンベ 掌 低音 高音 スラップ。バラフォン 木琴 泛音。ムビラ 鉄鍵 互い違い 装飾。セネガル サバール 競演 舞踊。マリ マンディング 叙事詩 歌 長い。ガーナ アディンクラ 太鼓語 信号。ナイジェリア ヨルバ 太鼓 儀式 アンサンブル。コールレスポンス 野原 活性 対話。ポリリズム 多層 交差 変位。アフリカン バラフォン ループ サイクル。 "
      },
      americas: {
        name: "アメリカ大陸",
        style: "ジャズ / ブルース / サルサ / サンバ",
        ring: "アメリカ大陸 - ジャズ - ブルース - ゴスペル - サルサ - サンバ - クンビア - ヒップホップ - レゲトン - タンゴ - ブルーグラス - ",
        text: "アメリカ大陸 音楽面：ブルース ベントノート 十二小節 呼び声。ジャズ スウィング ライド シンバル 即興 和声。ゴスペル 呼びかけ応答 オルガン ピアノ。サルサ クラーベ ブラス ピアノ モントゥーノ。サンバ バトゥカーダ 切り 踏み。ボサノヴァ ギター 揺らぎ ウォーキングベース。クンビア アコーディオン線 太鼓パルス。ヒップホップ ブレイクビート サンプリング フロー。レゲトン デンボウ ベースループ。カントリー フィドル バンジョー 語り。ブルーグラス マンドリン フラットピック 速い。タンゴ バンドネオン 悲劇的 切り。レゲエ 切り 社会批評 スカ。スカ 切り ジャマイカ ブラス ホーン。ソカ 祭典 トリニダード パン太鼓。アンデス パンパイプ ケーナ 嘆き。マリアチ トランペット ブラス コーラス。カリプソ 鉄太鼓 風刺 語り。シャランガ フィドル メキシコ舞踊。フォロー アコーディオン ザブンバ 三角。 "
      },
      oceania: {
        name: "オセアニア / 太平洋",
        style: "フラ / ハカ / イダキ / アイランドレゲエ",
        ring: "オセアニア 太平洋 - フラ - ハカ - イダキ - アイランドレゲエ - 合唱チャント - ラカラカ - サーフロック - ",
        text: "オセアニア 太平洋 音楽面：フラ 旋律 スチールギター 草裙。ハカ 集団チャント 力 踏み。イダキ ドローン 循環息 テクスチャー。アイランドレゲエ スカンク 熱帯和声。ポリネシア合唱 積層 和声。マオリ ハカ 戦舞 詩。ラカラカ 合唱舞踊 トンガ。サモア シヴァ 舞踊歌。オーストラリア 先住民 イダキ 夢の時代。オーストラリア ロック パンク インディ 独立。サーフロック ギター リヴァーブ ハワイアン。ハワイ ウクレレ 滑走 ストラム。メラネシア パンパイプ アンサンブル 低音。ミクロネシア 竹棒舞踊。太平洋 スチールドラム 島祭り。儀礼リズム 固定 共同体。太平洋 ギター ストラム バラード 語り。海 波 風 自然 フィールド サンプル。 "
      }
    },
    ko: {
      eastAsia: {
        name: "동아시아",
        style: "구친 / 엔카 / 판소리 / 시티팝",
        ring: "동아시아 - 구친 - 비파 - 얼후 - 디쯔 - 엔카 - J-POP - K-POP - 시티팝 - 판소리 - 월극 - 노 - ",
        text: "동아시아 음악 표면: 구친 배음 손가락 소음. 비파 트레몰로 아래현. 얼후 포르타멘토 비브라토. 디쯔 혀꽃 숨결. 샤쿠하치 불어と共鳴. 양친 롤 포앵티리즘. 동소 긴 숨의 소리. 금 노래 고전. 가가쿠 의식 드론. 노 발성 가면. 가부키 나가우타 샤미센. 엔카 비브라토 우울. 판소리 서사적 북 외침. 가야금 산조 흔들림. K-POP 훅 안무 동기화. J-POP 멜로디 밀도 화성. 시티팝 브라스 신디사이저. 월극 가사 윤곽 말그림. 대만 발라드 전환 구절. 오음계 궁상각치徵 동기 활성. "
      },
      southeastAsia: {
        name: "동남아시아",
        style: "가믈란 / 당둣 / 쿨린탕",
        ring: "동남아시아 - 가믈란 - 당魆 - 쿨린탕 - 핀피트 - 모람 - 크론총 - 태국고전 - 베트남민요 - ",
        text: "동남아시아 음악 표면: 가믈란 교차 금속타악 사이클. 가믈란 강사 청동 겹겹이. 당魆 드럼 펄스 전자. 쿨린탕 공 배열 합주. 핀피트 의례적 활 추진. 모람 보컬 장식 고음. 크론총 현 스트럼 포르투갈 혼합. 그림자극 음색 인형. 태국고전 마호리 하프. 베트남 독현금 트레몰로. 인도네시아 팝 당魆 그루브. 말레이 전통 라바브 노바트. 버마 사웅가우 하프 연쇄. 루손 쿨린탕 공 앙상블. 발리 앙클룽 대나무 셰이커. 룽퉁 시골 멜로디 흔들림. "
      },
      southAsia: {
        name: "남아시아",
        style: "라가 / 카왈리 / 영화음악 / 방그라",
        ring: "남아시아 - 라가 - 타블라 - 시타르 - 카왈리 - 영화음악 - 방그라 - 가잘 - 카르나틱 - 신디 - ",
        text: "남아시아 음악 표면: 힌두스타니 라가 상승하강 전개. 카르나틱 크리티 구조 변주. 타블라 볼 주기 즉흥. 미리단감 공명 타격. 시타르 현 활주 공명. 사로드 발현 깊은 지속. 사란기 활 울음 소리. 반수리 대나무 플룻 숨결. 카왈리 박수 응답 반복. 영화음악 오케스트라 고조 혼합. 방그라 댄스 그루브 돌 타격. 가잘 시적 선율 사랑. 신디 헌신 북 호출. 라바니 뭄바이 댄스홀 멜로디. 파하리 언덕노래 파이프 드론. 남인도 비나 활주 배음. 알라피나 빠른 음절 교체. 드루파드 엄숙 틴탈 느리게. "
      },
      mena: {
        name: "중동과 북아프리카",
        style: "마캄 / 우드 / 라이 / 그나와",
        ring: "중동 북아프리카 - 마캄 - 우드 - 카눈 - 네이 - 라이 - 그나와 - 다브케 - 탁심 - 샤비 - 무와샤 - ",
        text: "중동 북아프리카 음악 표면: 마캄 미분음 경로 전조. 탁심 즉흥 자유로운 숨. 우드 발현 공명 배음. 카눈 글리산도 스트럼 장식. 네이 숨결 트레몰로 공중. 카만차 활 회전 나선. 산투르 해머 더시머 배음. 라이 전자적 보컬 가장자리 서사. 그나와 겜브리 트랜스 반복. 다브케 줄춤 펄스 무겁게. 샤비 가사 흔들림 튀니지. 무와샤 고전 합창 안달루시아. 타르 북아프리카 북 손바닥. 벤디르 테두리 북 수피 회전. 마카마시 모로코 거리 리듬. 페르시아 산투르 타르 연쇄. 이라크 마캄 비음 긴 소리. 아랍 발현 우드 공명실. "
      },
      europe: {
        name: "유럽",
        style: "파두 / 플라멩코 / 샹송 / 테크노",
        ring: "유럽 - 파두 - 플라멩코 - 샹송 - 켈틱 - 테크노 - 폴리포니 - 크라우트록 - 바로크 - 슬라브 - ",
        text: "유럽 음악 표면: 파두 사우다드 선율 기타. 플라멩코 팔마스 라스게아도 춤 노래꾼. 샹송 텍스트 전경 시적 형태. 켈틱 릴 파이프 피들 걸음. 발칸 비대칭 브라스 박자. 슬라브 합창 화성 우울. 알프스 요델 팔세토 전환. 바로크 통주저음 체발로 대위법. 고전 소나타 전개 재현. 낭만 프로그램음악 시적. 인상 음색 화성 흐림. 테크노 포 온 더 플로어 신디사이저. 크라우트록 모토릭 반복 흐름. 앰비언트 텍스처 부유 공간. 포스트펑크 소음 원시 반항. 북유럽 아이슬란드 분위기 공중. 켈틱 하프 배음 고전. 발라드 서사 아코디언 하모니카. 오페라 아리아 벨칸토 드라마. 폴리포니 그레고리안 모테트 대위법. "
      },
      westAfrica: {
        name: "서아프리카",
        style: "아프로비트 / 하이라이프 / 음발라 / 코라",
        ring: "서아프리카 - 아프로비트 - 하이라이프 - 음발라 - 코라 - 주주 - 팜와인 - 그리오 - 부르키나 - 만딩 - ",
        text: "서아프리카 음악 표면: 아프로비트 혼 오스티나토 페라 추진. 하이라이프 기타 격자 브라스 춤. 음발라 사바르 폴리리듬 절단. 그리오 코라 아르페지오 서사시. 주주 토킹드럼 응답 찬양. 팜와인 기타 가벼운 스트럼 잼. 후지 보컬 찬양 북 호출. 토킹드럼 음높이 이동 모방. 젠베 손바닥 저음 고음 슬래프. 발라폰 나무 실로폰 배음. 음비라 철 건반 맞물림 장식. 세네갈 사바르 경쟁 춤. 말리 만딩 서사시 노래 길다. 가나 아딘크라 북 언어 신호. 나이제리아 요루바 북 의식 앙상블. 콜 앤 리스폰스 현장 활성 대화. 폴리리듬 다층 교차 변위. 아프리칸 발라폰 루프 사이클. "
      },
      americas: {
        name: "아메리카",
        style: "재즈 / 블루스 / 살사 / 삼바",
        ring: "아메리카 - 재즈 - 블루스 - 가스펠 - 살사 - 삼바 - 쿰비아 - 힙합 - 레게톤 - 탱고 - 블루그래스 - ",
        text: "아메리카 음악 표면: 블루스 벤트 노트 열두 마절 호출. 재즈 스윙 라이드 심벌 즉흥 화성. 가스펠 콜 앤 리스폰스 오르간 피아노. 살사 클라베 브라스 피아노 몬투뇨. 삼바 바투카다 절단 발구름. 보사노바 기타 흔들림 워킹베이스. 쿰비아 아코디언 라인 북 펄스. 힙합 브레이크비트 샘플 흐름. 레게톤 뎀보우 베이스 루프. 컨트리 피들 밴조 서사. 블루그래스 만돌린 플랫피크 빠르게. 탱고 반도네온 비극적 절단. 레게 절단 사회비평 스카. 스카 절단 자메이카 브라스 혼. 소카 트리니다드 팬 북 축제. 안데스 팬파이프 케나 비명. 마리아치 트럼펫 브라스 코러스. 칼립소 철북 풍자 서사. 샤랑가 피들 멕시코 춤. 포로 아코디언 자붐바 삼각형. "
      },
      oceania: {
        name: "오세아니아 / 태평양",
        style: "훌라 / 하카 / 이다키 / 아일랜드 레게",
        ring: "오세아니아 태평양 - 훌라 - 하카 - 이다키 - 아일랜드 레게 - 합창 챈트 - 라칼라카 - 서프 록 - ",
        text: "오세아니아 태평양 음악 표면: 훌라 선율 스틸 기타 풀스커트. 하카 집단 챈트 힘 발구름. 이다키 드론 순환 호흡 텍스처. 아일랜드 레게 스캥크 열대 화성. 폴리네시아 합창 겹겹이 화성. 마오리 하카 전쟁춤 시. 라칼라카 합창춤 통가. 사모아 시바 춤노래. 오스트레일리아 원주민 이다키 꿈의 시간. 오스트레일리아 록 펑크 인디 독립. 서프 록 기타 리버브 하와이안. 하와이 우쿨렐레 활주 스트럼. 멜라네시아 팬파이프 앙상블 저음. 미크로네시아 대나무 막대춤. 태평양 스틸 드럼 섬축제. 의례 리듬 고정 공동체. 태평양 기타 스트럼 발라드 서사. 바다 파도 바람 자연 필드 샘플. "
      }
    }
  };

  function flattenGlobeRegions(regions = {}) {
    const result = {};
    Object.entries(regions).forEach(([key, region]) => {
      Object.entries(region).forEach(([field, value]) => {
        result[`index.globe.region.${key}.${field}`] = value;
      });
    });
    return result;
  }

  Object.entries(fallbackGlobeRegions).forEach(([key, region]) => {
    Object.entries(region).forEach(([field, value]) => {
      homeIndexTranslations.en[`index.globe.region.${key}.${field}`] = value;
    });
  });

  ["de", "es", "pt", "ja", "ko"].forEach((language) => {
    homeIndexTranslations[language] = {
      ...homeIndexTranslations.en,
      ...(localizedHomeOverrides[language] || {}),
      ...flattenGlobeRegions(localizedGlobeRegions[language])
    };
  });

  const translations = {
    "zh-CN": {
      ...homeIndexTranslations["zh-CN"],
      "common.nav.music": "音乐台",
      "common.nav.favorites": "收藏夹",
      "common.nav.settings": "设置界面",
      "common.brand.homeAria": "Echo 主页",
      "index.title": "Echo | 听见世界的声音",
      "index.brandSub": "听见世界的声音",
      "index.heroSub": "想听听地球另一边的人在听什么吗？\n\n上传一首你喜欢的歌，我们会带你去探索 17 个文化域的声音，从古琴到巴萨诺瓦，从演歌到非洲鼓。",
      "index.action.start": "开始探索",
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
      "music.ai.followup": "进行对话",
      "music.ai.followupPlaceholder": "介绍一下被推荐的音乐吧",
      "music.ai.send": "发送",
      "music.ai.sending": "发送中...",
      "music.ai.assistantLabel": "AI 分析",
      "music.ai.userLabel": "我的问题",
      "music.ai.displayPrompt": "请基于当前上传曲目、跨文化推荐结果等内容，分析这次推荐路线。",
      "music.ai.displayFollowup": "进行对话：{question}",
      "music.ai.noText": "AI 没有返回可用文本。",
      "music.ai.fetchFailed": "请求没有成功发出，可能是本地后端未启动、网络问题，或接口地址不可达。",
      "favorites.title": "Echo | 收藏夹",
      "favorites.brandSub": "收藏夹",
      "favorites.heroMeta": "曲库界面",
      "favorites.heroTitle": "音乐<br>收藏夹",
      "favorites.heroCopy": "把来自 iTunes、Jamendo 与上传音频的曲目整理成可浏览、可筛选、可继续接入推荐反馈的音乐集合。",
      "favorites.stat.tracks": "曲目",
      "favorites.stat.cultures": "文化域",
      "favorites.stat.artists": "音乐人",
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
      "favorites.playLocal": "点击播放",
      "favorites.localTrack": "本地曲目",
      "favorites.removeTrack": "移出收藏夹",
      "favorites.empty": "没有匹配曲目。",
      "favorites.loading": "正在加载收藏曲库...",
      "favorites.loadFailed": "曲库加载失败，请确认本地后端正在运行。",
      "favorites.unknownTrack": "未知曲目",
      "favorites.unknownArtist": "未知创作者",
      "favorites.unknownGenre": "未知",
      "favorites.unknownCulture": "神秘的地方",
      "favorites.mysteriousPlace": "神秘的地方",
      "favorites.pendingAnalysis": "待分析",
      "settings.title": "Echo | 设置界面",
      "settings.brandSub": "设置界面",
      "settings.generalSection": "通用功能",
      "settings.generalTitle": "全局<br>偏好",
      "settings.generalCopy": "这里调整 Echo 的界面语言、AI 回复语言、专辑封面视觉与音乐台初始加载方式。",
      "settings.kimiSection": "AI 配置",
      "settings.heroMeta": "模型设置",
      "settings.heroTitle": "AI<br>接口",
      "settings.heroCopy": "这里配置音乐台 AI 分析所需的模型接口。密钥仍只保存在当前浏览器，本地后端负责代理请求。",
      "settings.languageLabel": "界面与 AI 语言",
      "settings.languageHelp": "选择后会同步影响页面文案和 AI 分析回复语言。",
      "settings.coverModeLabel": "专辑封面显示",
      "settings.coverMode.color": "全部彩色",
      "settings.coverMode.mono": "全部黑白",
      "settings.coverMode.mixed": "部分彩色",
      "settings.coverModeHelp": "该设置会影响收藏夹里专辑封面的彩色/黑白呈现。",
      "settings.coverRatioLabel": "彩色比例",
      "settings.startupTrackLabel": "进入音乐台时加载",
      "settings.startupTrack.last": "上次关闭时的歌",
      "settings.startupTrack.favorite": "从收藏夹随机",
      "settings.startupTrack.library": "从音乐库随机",
      "settings.startupTrack.none": "不加载歌曲",
      "settings.startupTrackHelp": "控制打开音乐台时播放器预先载入哪一首歌；通过收藏夹或链接进入的指定曲目仍会优先生效。",
      "settings.apiKeyLabel": "AI API 密钥",
      "settings.apiKeyPlaceholder": "填入 AI API 密钥",
      "settings.modelLabel": "模型",
      "settings.endpointLabel": "接口地址",
      "settings.save": "保存设置",
      "settings.test": "测试连接",
      "settings.clear": "清空密钥",
      "settings.note": "可以把密钥保存在当前浏览器本地，也可以填入本地后端配置文件；音乐台会通过本地后端代理转发 AI 请求。",
      "settings.readFailed": "读取本地设置失败，请重新保存一次。",
      "settings.backendDetected": "已检测到后端本地 AI 配置：{source}",
      "settings.saved": "设置已保存。回到音乐台后即可生成 AI 分析。",
      "settings.languageSaved": "语言已保存。",
      "settings.testing": "正在通过本地后端测试 AI 连接...",
      "settings.testSuccess": "AI 连接成功：{content}",
      "settings.testFailed": "AI 连接失败：{error}",
      "settings.keyCleared": "密钥已从当前浏览器本地清空。",
      "settings.coverSaved": "专辑封面显示设置已保存。",
      "settings.startupTrackSaved": "音乐台初始加载设置已保存。",
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
      ...homeIndexTranslations.en,
      "common.nav.music": "Music Station",
      "common.nav.favorites": "Favorites",
      "common.nav.settings": "Settings",
      "common.brand.homeAria": "Echo home",
      "index.brandSub": "Hear the World",
      "index.title": "Echo | Hear the World",
      "index.heroSub": "Ever wondered what people on the other side of the world are listening to?\n\nUpload a song you love, and we'll take you on a journey through 17 cultural soundscapes — from guzheng to bossa nova, from enka to djembe.",
      "index.action.start": "Start Exploring",
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
      "music.ai.followup": "Start a dialogue",
      "music.ai.followupPlaceholder": "Introduce the recommended music.",
      "music.ai.send": "Send",
      "music.ai.sending": "Sending...",
      "music.ai.assistantLabel": "AI analysis",
      "music.ai.userLabel": "My question",
      "music.ai.displayPrompt": "Analyze this recommendation route using the uploaded track and cross-cultural results.",
      "music.ai.displayFollowup": "Dialogue: {question}",
      "music.ai.noText": "AI did not return usable text.",
      "music.ai.fetchFailed": "The request was not sent successfully. The local backend, network, or endpoint may be unavailable.",
      "favorites.title": "Echo | Favorites",
      "favorites.brandSub": "Favorites",
      "favorites.heroTitle": "Music<br>Favorites",
      "favorites.heroCopy": "Browse, filter, and reuse tracks from iTunes, Jamendo, and uploaded audio as a music collection for recommendation feedback.",
      "favorites.stat.tracks": "Tracks",
      "favorites.stat.cultures": "Cultures",
      "favorites.stat.artists": "Artists",
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
      "favorites.playLocal": "Play",
      "favorites.localTrack": "Local track",
      "favorites.removeTrack": "Remove from favorites",
      "favorites.empty": "No matching tracks.",
      "favorites.loading": "Loading favorite library...",
      "favorites.loadFailed": "Library failed to load. Check that the local backend is running.",
      "favorites.unknownTrack": "Unknown track",
      "favorites.unknownArtist": "Unknown creator",
      "favorites.unknownGenre": "Unknown",
      "favorites.unknownCulture": "Mystery place",
      "favorites.mysteriousPlace": "Mystery place",
      "favorites.pendingAnalysis": "Pending analysis",
      "settings.title": "Echo | Settings",
      "settings.brandSub": "Settings",
      "settings.generalSection": "General",
      "settings.generalTitle": "Global<br>Preferences",
      "settings.generalCopy": "Adjust Echo's interface language, AI reply language, album-cover appearance, and Music Station startup track.",
      "settings.kimiSection": "AI Config",
      "settings.heroMeta": "Model Settings",
      "settings.heroTitle": "AI<br>API",
      "settings.heroCopy": "Configure the model API used by Music Station AI analysis. The key stays in this browser, while the local backend proxies requests.",
      "settings.languageLabel": "Interface and AI language",
      "settings.languageHelp": "This changes page copy and the language used for AI analysis replies.",
      "settings.coverModeLabel": "Album cover display",
      "settings.coverMode.color": "All color",
      "settings.coverMode.mono": "All black and white",
      "settings.coverMode.mixed": "Mixed color",
      "settings.coverModeHelp": "This controls color or black-and-white album covers in Favorites.",
      "settings.coverRatioLabel": "Color ratio",
      "settings.startupTrackLabel": "Load when entering Music Station",
      "settings.startupTrack.last": "Last closed track",
      "settings.startupTrack.favorite": "Random favorite",
      "settings.startupTrack.library": "Random library track",
      "settings.startupTrack.none": "Do not load",
      "settings.startupTrackHelp": "Choose which track the player preloads when Music Station opens; explicit favorite or shared links still take priority.",
      "settings.apiKeyLabel": "AI API key",
      "settings.apiKeyPlaceholder": "Enter AI API key",
      "settings.modelLabel": "Model",
      "settings.endpointLabel": "Endpoint",
      "settings.save": "Save settings",
      "settings.test": "Test connection",
      "settings.clear": "Clear key",
      "settings.note": "Save the key in this browser or in a local backend config file; Music Station sends AI requests through the local backend proxy.",
      "settings.readFailed": "Failed to read local settings. Please save again.",
      "settings.backendDetected": "Detected local backend AI config: {source}",
      "settings.saved": "Settings saved. Return to Music Station to generate AI analysis.",
      "settings.languageSaved": "Language saved.",
      "settings.testing": "Testing AI connection through the local backend...",
      "settings.testSuccess": "AI connected: {content}",
      "settings.testFailed": "AI connection failed: {error}",
      "settings.keyCleared": "The key has been removed from this browser.",
      "settings.coverSaved": "Album cover display setting saved.",
      "settings.startupTrackSaved": "Music Station startup setting saved.",
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
      ...homeIndexTranslations.de,
      "common.nav.music": "Musikstation",
      "common.nav.favorites": "Favoriten",
      "common.nav.settings": "Einstellungen",
      "common.brand.homeAria": "Echo Startseite",
      "index.title": "Echo | Interkulturelle Musikempfehlung",
      "index.heroSub": "Ein interkultureller Musikarbeitsplatz mit 30.000 realen Titeln, CultureMERT-Embeddings und der DCAS-Empfehlungskette.",
      "index.action.start": "Starten",
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
      "music.ai.followup": "Dialog starten",
      "music.ai.followupPlaceholder": "Stelle die empfohlene Musik vor.",
      "music.ai.send": "Senden",
      "music.ai.sending": "Wird gesendet...",
      "music.ai.assistantLabel": "AI Analyse",
      "music.ai.userLabel": "Meine Frage",
      "music.ai.displayPrompt": "Analysiere diese Empfehlungsroute anhand des hochgeladenen Titels und der interkulturellen Ergebnisse.",
      "music.ai.displayFollowup": "Dialog: {question}",
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
      "settings.generalCopy": "Passe Sprache, AI-Antworten, Coverdarstellung und den Starttitel der Musikstation an.",
      "settings.heroTitle": "AI<br>API",
      "settings.languageLabel": "Sprache für Oberfläche und AI",
      "settings.languageHelp": "Ändert Seitentexte und die Antwortsprache der AI-Analyse.",
      "settings.startupTrackLabel": "Beim Öffnen der Musikstation laden",
      "settings.startupTrack.last": "Zuletzt gespielter Titel",
      "settings.startupTrack.favorite": "Zufälliger Favorit",
      "settings.startupTrack.library": "Zufälliger Bibliothekstitel",
      "settings.startupTrack.none": "Keinen Titel laden",
      "settings.startupTrackHelp": "Legt fest, welchen Titel der Player beim Öffnen vorlädt; direkte Favoriten- oder Linkaufrufe haben Vorrang.",
      "settings.apiKeyLabel": "AI API-Schlüssel",
      "settings.apiKeyPlaceholder": "AI API-Schlüssel eintragen",
      "settings.save": "Speichern",
      "settings.test": "Verbindung testen",
      "settings.clear": "Schlüssel löschen",
      "settings.startupTrackSaved": "Starttitel-Einstellung gespeichert.",
      "settings.recommendCountLabel": "Empfehlungen pro Anfrage",
      "settings.recommendCountHelp": "Anzahl der Titel pro interkultureller Empfehlung (1-30, Standard 10).",
      "settings.recommendCountSaved": "Empfehlungsanzahl gespeichert."
    },
    es: {
      ...homeIndexTranslations.es,
      "common.nav.music": "Mesa musical",
      "common.nav.favorites": "Favoritos",
      "common.nav.settings": "Ajustes",
      "common.brand.homeAria": "Inicio de Echo",
      "index.title": "Echo | Recomendación musical intercultural",
      "index.heroSub": "Un espacio musical intercultural conectado a 30.000 pistas reales, embeddings CultureMERT y la ruta de recomendación DCAS.",
      "index.action.start": "Empezar",
      "music.title": "Echo | Mesa musical",
      "music.brandSub": "Mesa musical",
      "music.upload": "Subir música",
      "music.ai.meta": "Análisis AI",
      "music.ai.title": "Explicación",
      "music.ai.copy": "Envía la pista actual y las recomendaciones interculturales a la AI para analizar esta ruta.",
      "music.ai.generate": "Generar análisis",
      "music.ai.followup": "Iniciar diálogo",
      "music.ai.followupPlaceholder": "Presenta la música recomendada.",
      "music.ai.send": "Enviar",
      "music.ai.assistantLabel": "Análisis de AI",
      "music.ai.userLabel": "Mi pregunta",
      "music.ai.displayPrompt": "Analiza esta ruta de recomendación con la pista subida y los resultados interculturales.",
      "music.ai.displayFollowup": "Diálogo: {question}",
      "favorites.title": "Echo | Favoritos",
      "favorites.brandSub": "Favoritos",
      "favorites.heroTitle": "Favoritos<br>sonoros",
      "favorites.import": "Importar canciones",
      "favorites.removeTrack": "Quitar de favoritos",
      "settings.title": "Echo | Ajustes",
      "settings.brandSub": "Ajustes",
      "settings.generalCopy": "Ajusta el idioma, las respuestas de AI, las portadas y la pista inicial de la mesa musical.",
      "settings.languageLabel": "Idioma de interfaz y AI",
      "settings.languageHelp": "Cambia los textos de la página y el idioma de respuesta del análisis AI.",
      "settings.startupTrackLabel": "Cargar al entrar en la mesa musical",
      "settings.startupTrack.last": "Última pista cerrada",
      "settings.startupTrack.favorite": "Favorita aleatoria",
      "settings.startupTrack.library": "Biblioteca aleatoria",
      "settings.startupTrack.none": "No cargar pista",
      "settings.startupTrackHelp": "Elige qué pista precarga el reproductor al abrirse; los enlaces o favoritos directos tienen prioridad.",
      "settings.save": "Guardar",
      "settings.test": "Probar conexión",
      "settings.clear": "Borrar clave",
      "settings.startupTrackSaved": "Ajuste de pista inicial guardado.",
      "settings.recommendCountLabel": "Recomendaciones por solicitud",
      "settings.recommendCountHelp": "Número de pistas por recomendación intercultural (1-30, predeterminado 10).",
      "settings.recommendCountSaved": "Cantidad de recomendaciones guardada."
    },
    pt: {
      ...homeIndexTranslations.pt,
      "common.nav.music": "Estação musical",
      "common.nav.favorites": "Favoritos",
      "common.nav.settings": "Configurações",
      "common.brand.homeAria": "Início do Echo",
      "index.title": "Echo | Recomendação musical intercultural",
      "index.heroSub": "Uma bancada musical intercultural conectada a 30.000 faixas reais, embeddings CultureMERT e ao fluxo de recomendação DCAS.",
      "index.action.start": "Começar",
      "music.title": "Echo | Estação musical",
      "music.brandSub": "Estação musical",
      "music.upload": "Enviar música",
      "music.ai.meta": "Análise AI",
      "music.ai.title": "Explicação",
      "music.ai.copy": "Envie a faixa atual e as recomendações interculturais para a AI analisar esta rota.",
      "music.ai.generate": "Gerar análise",
      "music.ai.followup": "Iniciar diálogo",
      "music.ai.followupPlaceholder": "Apresente a música recomendada.",
      "music.ai.send": "Enviar",
      "music.ai.assistantLabel": "Análise AI",
      "music.ai.userLabel": "Minha pergunta",
      "music.ai.displayPrompt": "Analise esta rota de recomendação com a faixa enviada e os resultados interculturais.",
      "music.ai.displayFollowup": "Diálogo: {question}",
      "favorites.title": "Echo | Favoritos",
      "favorites.brandSub": "Favoritos",
      "favorites.heroTitle": "Favoritos<br>sonoros",
      "favorites.import": "Importar músicas",
      "favorites.removeTrack": "Remover dos favoritos",
      "settings.title": "Echo | Configurações",
      "settings.brandSub": "Configurações",
      "settings.generalCopy": "Ajuste idioma, respostas da AI, capas e a faixa inicial da estação musical.",
      "settings.languageLabel": "Idioma da interface e da AI",
      "settings.languageHelp": "Altera os textos da página e o idioma das respostas da análise AI.",
      "settings.startupTrackLabel": "Carregar ao entrar na estação musical",
      "settings.startupTrack.last": "Última faixa fechada",
      "settings.startupTrack.favorite": "Favorita aleatória",
      "settings.startupTrack.library": "Biblioteca aleatória",
      "settings.startupTrack.none": "Não carregar faixa",
      "settings.startupTrackHelp": "Escolha qual faixa o player pré-carrega ao abrir; favoritos ou links diretos têm prioridade.",
      "settings.save": "Salvar",
      "settings.test": "Testar conexão",
      "settings.clear": "Limpar chave",
      "settings.startupTrackSaved": "Configuração de faixa inicial salva.",
      "settings.recommendCountLabel": "Recomendações por solicitação",
      "settings.recommendCountHelp": "Número de faixas por recomendação intercultural (1-30, padrão 10).",
      "settings.recommendCountSaved": "Quantidade de recomendações salva."
    },
    ja: {
      ...homeIndexTranslations.ja,
      "common.nav.music": "音楽台",
      "common.nav.favorites": "お気に入り",
      "common.nav.settings": "設定",
      "common.brand.homeAria": "Echo ホーム",
      "index.title": "Echo | 異文化音楽推薦",
      "index.heroSub": "30,000曲の実曲、CultureMERT embedding、DCAS推薦パイプラインにつながる異文化音楽ワークベンチです。",
      "index.action.start": "開始",
      "music.title": "Echo | 音楽台",
      "music.brandSub": "音楽台",
      "music.upload": "音楽をアップロード",
      "music.ai.meta": "AI 分析",
      "music.ai.title": "推薦解説",
      "music.ai.copy": "現在の曲と異文化推薦結果を AI に送り、この推薦ルートを分析します。",
      "music.ai.generate": "分析を生成",
      "music.ai.followup": "対話を始める",
      "music.ai.followupPlaceholder": "推薦された音楽を紹介してください。",
      "music.ai.send": "送信",
      "music.ai.assistantLabel": "AI 分析",
      "music.ai.userLabel": "自分の質問",
      "music.ai.displayPrompt": "アップロード曲と異文化推薦結果に基づいて、この推薦ルートを分析してください。",
      "music.ai.displayFollowup": "対話：{question}",
      "favorites.title": "Echo | お気に入り",
      "favorites.brandSub": "お気に入り",
      "favorites.heroTitle": "音の<br>お気に入り",
      "favorites.import": "曲を読み込む",
      "favorites.removeTrack": "お気に入りから削除",
      "settings.title": "Echo | 設定",
      "settings.brandSub": "設定",
      "settings.generalCopy": "画面言語、AI 返信、カバー表示、音楽台の初期曲を調整します。",
      "settings.languageLabel": "画面と AI の言語",
      "settings.languageHelp": "ページ文言と AI 分析の返信言語を変更します。",
      "settings.startupTrackLabel": "音楽台に入る時に読み込む曲",
      "settings.startupTrack.last": "前回終了時の曲",
      "settings.startupTrack.favorite": "お気に入りからランダム",
      "settings.startupTrack.library": "全ライブラリからランダム",
      "settings.startupTrack.none": "曲を読み込まない",
      "settings.startupTrackHelp": "音楽台を開く時にプレイヤーが事前読み込みする曲を選びます。お気に入りやリンク指定は優先されます。",
      "settings.save": "保存",
      "settings.test": "接続テスト",
      "settings.clear": "キーを消去",
      "settings.startupTrackSaved": "初期曲の設定を保存しました。",
      "settings.recommendCountLabel": "リクエストごとの推薦数",
      "settings.recommendCountHelp": "異文化推薦で返される曲数を設定します（1-30、デフォルト10）。",
      "settings.recommendCountSaved": "推薦数を保存しました。"
    },
    ko: {
      ...homeIndexTranslations.ko,
      "common.nav.music": "음악 스테이션",
      "common.nav.favorites": "즐겨찾기",
      "common.nav.settings": "설정",
      "common.brand.homeAria": "Echo 홈",
      "index.title": "Echo | 교차 문화 음악 추천",
      "index.heroSub": "30,000개의 실제 트랙, CultureMERT 임베딩, DCAS 추천 파이프라인에 연결된 교차 문화 음악 워크벤치입니다.",
      "index.action.start": "시작하기",
      "music.title": "Echo | 음악 스테이션",
      "music.brandSub": "음악 스테이션",
      "music.upload": "음악 업로드",
      "music.ai.meta": "AI 분석",
      "music.ai.title": "추천 해설",
      "music.ai.copy": "현재 곡과 교차 문화 추천 결과를 AI에 보내 추천 경로를 분석합니다.",
      "music.ai.generate": "분석 생성",
      "music.ai.followup": "대화 시작",
      "music.ai.followupPlaceholder": "추천된 음악을 소개해 주세요.",
      "music.ai.send": "보내기",
      "music.ai.assistantLabel": "AI 분석",
      "music.ai.userLabel": "내 질문",
      "music.ai.displayPrompt": "업로드한 곡과 교차 문화 추천 결과를 바탕으로 이 추천 경로를 분석해 주세요.",
      "music.ai.displayFollowup": "대화: {question}",
      "favorites.title": "Echo | 즐겨찾기",
      "favorites.brandSub": "즐겨찾기",
      "favorites.heroTitle": "사운드<br>즐겨찾기",
      "favorites.import": "곡 가져오기",
      "favorites.removeTrack": "즐겨찾기에서 제거",
      "settings.title": "Echo | 설정",
      "settings.brandSub": "설정",
      "settings.generalCopy": "화면 언어, AI 답변, 커버 표시, 음악 스테이션 시작 곡을 조정합니다.",
      "settings.languageLabel": "인터페이스 및 AI 언어",
      "settings.languageHelp": "페이지 문구와 AI 분석 답변 언어를 변경합니다.",
      "settings.startupTrackLabel": "음악 스테이션 진입 시 불러오기",
      "settings.startupTrack.last": "마지막으로 닫은 곡",
      "settings.startupTrack.favorite": "즐겨찾기에서 랜덤",
      "settings.startupTrack.library": "전체 라이브러리 랜덤",
      "settings.startupTrack.none": "곡 불러오지 않기",
      "settings.startupTrackHelp": "음악 스테이션을 열 때 플레이어가 미리 불러올 곡을 정합니다. 즐겨찾기나 링크로 지정된 곡은 우선합니다.",
      "settings.save": "저장",
      "settings.test": "연결 테스트",
      "settings.clear": "키 지우기",
      "settings.startupTrackSaved": "시작 곡 설정이 저장되었습니다.",
      "settings.recommendCountLabel": "요청당 추천 수",
      "settings.recommendCountHelp": "교차 문화 추천에서 반환되는 곡 수를 설정합니다 (1-30, 기본값 10).",
      "settings.recommendCountSaved": "추천 수가 저장되었습니다."
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
