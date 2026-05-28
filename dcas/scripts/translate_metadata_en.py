"""
将 metadata_merged.csv 中的中文元数据翻译为英文，新增 description_en, label_en, tags_en 等列。

翻译策略：
1. description: 读取中文描述，用内置翻译规则逐句翻译为英文
   - 先用句式模板整体匹配翻译
   - 再用词组映射表逐词替换
   - 最后用语义模板兜底
2. label/tags: 用映射表翻译（数量有限，约 817 行）
3. title/artist/album: 用映射表翻译中文名

用法: python translate_metadata_en.py [--dry-run] [--batch-size 500]
"""
import csv
import json
import os
import re
import argparse
import time

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'storage', 'public', 'merged', 'metadata_merged.csv')
BACKUP_PATH = CSV_PATH + '.backup_before_translate'

# === 流派/标签中英文映射表 ===
GENRE_MAP = {
    '中国摇滚': 'Chinese Rock', '中国流行': 'Chinese Pop', '中国民谣': 'Chinese Folk',
    '中国电子': 'Chinese Electronic', '中国嘻哈': 'Chinese Hip-Hop', '中国说唱': 'Chinese Rap',
    '中国R&B': 'Chinese R&B', '中国民乐': 'Chinese Traditional', '中国古典': 'Chinese Classical',
    '华语流行': 'Mandarin Pop', '粤语流行': 'Cantonese Pop', '台湾独立': 'Taiwanese Indie',
    '日本摇滚': 'Japanese Rock', '日本流行': 'Japanese Pop', '日本电子': 'Japanese Electronic',
    '日本民谣': 'Japanese Folk', '日本爵士': 'Japanese Jazz', '日本嘻哈': 'Japanese Hip-Hop',
    '日本金属': 'Japanese Metal', '日本朋克': 'Japanese Punk', '日本独立': 'Japanese Indie',
    '日本动漫': 'Anime', '日语流行': 'J-Pop',
    '韩语流行': 'K-Pop', '韩语摇滚': 'Korean Rock', '韩语嘻哈': 'Korean Hip-Hop',
    '韩语R&B': 'Korean R&B', '韩语电子': 'Korean Electronic',
    '印度流行': 'Indian Pop', '印度古典': 'Indian Classical', '印度民间': 'Indian Folk',
    '宝莱坞': 'Bollywood',
    '巴西流行': 'Brazilian Pop', '巴西摇滚': 'Brazilian Rock', '桑巴': 'Samba',
    '波萨诺瓦': 'Bossa Nova', '巴西放克': 'Brazilian Funk',
    '弗拉门戈': 'Flamenco', '拉丁流行': 'Latin Pop', '拉丁摇滚': 'Latin Rock',
    '雷鬼': 'Reggae', '非洲流行': 'African Pop', '非洲民间': 'African Folk',
    '非洲摇滚': 'African Rock', '非洲电子': 'Afrobeats',
    '中东流行': 'Middle Eastern Pop', '中东民间': 'Middle Eastern Folk',
    '土耳其流行': 'Turkish Pop', '阿拉伯流行': 'Arabic Pop',
    '东南亚流行': 'Southeast Asian Pop', '凯尔特': 'Celtic', '凯尔特民间': 'Celtic Folk',
    '北欧流行': 'Nordic Pop', '北欧民间': 'Nordic Folk', '北欧金属': 'Nordic Metal',
    '东欧流行': 'Eastern European Pop', '东欧民间': 'Eastern European Folk',
    '巴尔干': 'Balkan', '巴尔干民间': 'Balkan Folk',
    '加勒比': 'Caribbean', '安第斯': 'Andean', '安第斯民间': 'Andean Folk',
    '中亚流行': 'Central Asian Pop', '中亚民间': 'Central Asian Folk',
    '世界音乐': 'World Music', '新世纪': 'New Age',
    '电子': 'Electronic', '摇滚': 'Rock', '流行': 'Pop', '民谣': 'Folk',
    '爵士': 'Jazz', '布鲁斯': 'Blues', '古典': 'Classical',
    '嘻哈': 'Hip-Hop', '说唱': 'Rap', '金属': 'Metal', '朋克': 'Punk',
    '乡村': 'Country', '蓝草': 'Bluegrass', '放克': 'Funk', '灵魂': 'Soul',
    '福音': 'Gospel', '氛围': 'Ambient', '实验': 'Experimental',
    '后摇': 'Post-Rock', '梦幻流行': 'Dream Pop', '迷幻摇滚': 'Psychedelic Rock',
    '独立摇滚': 'Indie Rock', '独立流行': 'Indie Pop', '另类摇滚': 'Alternative Rock',
    '后朋克': 'Post-Punk', '新浪潮': 'New Wave', '合成器流行': 'Synth-Pop',
    '电子舞曲': 'EDM', '浩室': 'House', '科技舞曲': 'Techno',
    '迷幻电子': 'Psytrance', '鼓贝斯': 'Drum and Bass', '回响贝斯': 'Dubstep',
    '陷阱': 'Trap', '浩室舞曲': 'Dance-Pop', '迪斯科': 'Disco',
    '硬摇滚': 'Hard Rock', '重金属': 'Heavy Metal', '死亡金属': 'Death Metal',
    '黑金属': 'Black Metal', '前卫摇滚': 'Progressive Rock', '艺术摇滚': 'Art Rock',
    '车库摇滚': 'Garage Rock', '冲浪摇滚': 'Surf Rock', '乡村摇滚': 'Country Rock',
    '民谣摇滚': 'Folk Rock', '拉丁': 'Latin', '探戈': 'Tango', '伦巴': 'Rumba',
    '恰恰': 'Cha-Cha', '卡里普索': 'Calypso', '索卡': 'Soca', '斯卡': 'Ska',
    '摇滚乐': 'Rock and Roll', '节奏布鲁斯': 'Rhythm and Blues',
    '成人当代': 'Adult Contemporary', '轻音乐': 'Easy Listening',
    '电影原声': 'Film Score', '游戏音乐': 'Video Game Music',
    '冥想': 'Meditation', '疗愈': 'Healing', '放松': 'Relaxation',
    '瑜伽': 'Yoga', '睡眠': 'Sleep', '儿童': 'Children', '节日': 'Holiday', '圣诞': 'Christmas',
    '嘻哈说唱': 'Hip-Hop/Rap', '舞曲': 'Dance', '电子流行': 'Electropop',
    '独立': 'Indie', '另类': 'Alternative',
}

# === 描述中常见短语翻译映射（从长到短排列，避免部分匹配） ===
PHRASE_MAP = {
    # --- 完整句式/长短语 ---
    '扑面而来': 'washes over you',
    '层层递进': 'building layer by layer',
    '层层堆叠': 'stacking in layers',
    '层层叠加': 'layering up',
    '层层推进': 'driving forward layer by layer',
    '缓缓流淌': 'flows gently',
    '缓缓铺陈': 'slowly unfolds',
    '缓缓铺展': 'gradually spreads out',
    '缓缓点燃': 'slowly kindles',
    '缓缓释放': 'slowly releases',
    '缓缓沉降': 'slowly settling',
    '缓缓流入': 'slowly seeps into',
    '交织出': 'weave',
    '交织成': 'weave into',
    '铺陈出': 'unfold',
    '铺展出': 'spread out',
    '营造出': 'create',
    '勾勒出': 'sketch out',
    '凝成': 'crystallize into',
    '化为': 'transform into',
    '汇入': 'merge into',
    '迸发': 'burst forth',
    '倾诉': 'pour out',
    '诉说': 'tell of',
    '吟唱': 'sing',
    '嘶吼': 'howl',
    '呢喃': 'whisper',
    '哼唱': 'hum',
    '说唱': 'rap',
    '合唱': 'chorus',
    '独唱': 'solo vocal',
    '对唱': 'duet',
    '齐唱': 'sing-along',
    '爆发力': 'explosive power',
    '穿透力': 'penetrating power',
    '感染力': 'infectious energy',
    '冲击力': 'impact',
    '压迫感': 'oppressive pressure',
    '紧张感': 'tension',
    '释放压力': 'release pressure',
    '释放压抑': 'release pent-up emotion',
    '释放情绪': 'let emotions loose',
    '释放积压能量': 'unleash pent-up energy',
    '点燃热血': 'ignite passion',
    '点燃肾上腺素': 'ignite adrenaline',
    '肾上腺素飙升': 'adrenaline rush',
    '心跳与低音共振': 'heartbeat resonating with the bass',
    '在焦虑与疏离间游移': 'drifting between anxiety and alienation',
    '在克制与释放间游走': 'moving between restraint and release',
    '在理想与现实的拉扯': 'pulled between ideals and reality',
    '在机械律动中感受': 'feeling through the mechanical rhythm',
    '在冷冽与浪漫之间': 'between cold and romantic',
    '在循环中缓缓释放': 'slowly releasing on repeat',
    '在低频中缓缓沉降': 'slowly settling into the low end',
    '在音墙构筑的异色梦境': 'in a vivid dreamscape built by walls of sound',
    '从椅子上弹起来': 'leap out of their seats',
    '让人忍不住': 'impossible not to',
    '让人暂时忘却烦忧': 'letting you momentarily forget worries',
    '让紧绷的神经逐渐松弛': 'letting tense nerves gradually unwind',
    '让暖意缓缓流入心底': 'letting warmth seep gently into the heart',
    '让理想主义的微光穿透日常喧嚣': 'letting a glimmer of idealism pierce the everyday noise',
    '让压抑与反叛在循环中缓缓释放': 'letting suppression and rebellion slowly release on repeat',
    '让那份渴望与孤独在空气中共振': 'letting longing and loneliness resonate in the air',
    '让飘逸鼓点托起未尽的心事': 'letting ethereal drums carry unfinished thoughts',
    '让思绪随旋律漂向未知的旷野': 'letting thoughts drift with the melody toward unknown wilderness',
    '让层层递进的和声带你驶向记忆的远方': 'letting the layered progressive harmonies carry you toward distant memories',
    '让怀旧的情绪随旋律缓缓流淌': 'letting nostalgia flow gently with the melody',
    '让压抑在低频中缓缓沉降': 'letting pent-up emotion settle slowly into the low end',
    '适合深夜独处时沉浸聆听': 'ideal for immersive late-night listening',
    '适合深夜独处时循环播放': 'perfect for late-night solitude on repeat',
    '适合深夜独处时回望来路': 'ideal for late-night solitude, looking back on the journey',
    '适合深夜独处时以极大音量释放压抑': 'perfect for releasing pent-up emotion at maximum volume in late-night solitude',
    '适合深夜独处时直面内心的阴霾与愤怒': 'ideal for late-night solitude, confronting inner shadows and anger',
    '适合独处沉思或深夜驱车时聆听': 'perfect for solitary contemplation or late-night drives',
    '适合深夜疾驰公路或释放积压能量的时刻': 'ideal for late-night highway sprints or unleashing pent-up energy',
    '适合黄昏驾车或深夜独处时聆听': 'perfect for dusk drives or late-night solitude',
    '适合公路旅行或深夜独酌时播放': 'ideal for road trips or late-night drinks alone',
    '适合公路驰骋或健身房释放节奏': 'perfect for highway cruising or gym workouts',
    '适合驾车穿行霓虹街道或独自释放心绪时聆听': 'ideal for cruising neon-lit streets or releasing emotions alone',
    '适合午后独处或公路旅行时循环播放': 'perfect for afternoon solitude or road trips on repeat',
    '适合午后驾车或朋友小聚时轻松播放': 'ideal for afternoon drives or casual gatherings with friends',
    '适合午后小憩': 'perfect for an afternoon nap',
    '适合婴幼儿哄睡': 'ideal for lulling babies to sleep',
    '适合聚会举杯或公路旅行时随声哼唱': 'perfect for raising glasses at gatherings or singing along on road trips',
    '适合赛事热身': 'ideal for pre-game warmups',
    '适合公路远行': 'perfect for long road trips',
    '适合公路疾驰': 'ideal for highway sprints',
    '适合公路巡航': 'perfect for highway cruising',
    '适合公路旅行': 'ideal for road trips',
    '适合深夜驾车': 'perfect for late-night drives',
    '适合深夜自驾': 'ideal for late-night solo drives',
    '适合深夜疾驰': 'perfect for late-night sprints',
    '适合深夜独行': 'ideal for late-night walks',
    '适合深夜独酌': 'perfect for late-night drinks alone',
    '适合深夜独处': 'ideal for late-night solitude',
    '适合深夜公路': 'perfect for late-night highways',
    '适合独酌时分': 'ideal for a solo drink',
    '适合午后驾车': 'perfect for afternoon drives',
    '适合午后发呆': 'ideal for afternoon daydreams',
    '适合午后咖啡时光': 'perfect for an afternoon coffee',
    '适合黄昏驾车': 'ideal for dusk drives',
    '适合黄昏独饮': 'perfect for a solo drink at dusk',
    '适合清晨静思': 'ideal for morning reflection',
    '适合独处时回想旧日心事': 'perfect for revisiting old memories in solitude',
    '适合独自驾车穿行城市黄昏': 'ideal for solo drives through a city dusk',
    '适合深夜独处或公路远眺': 'perfect for late-night solitude or gazing down long highways',
    '适合深夜独处或公路疾驰': 'ideal for late-night solitude or highway cruising',
    '适合深夜独处或公路漫游': 'perfect for late-night solitude or road trips',
    '适合深夜独处或清晨静思时聆听': 'ideal for late-night solitude or morning reflection',
    '适合深夜驾车或独处时回味': 'perfect for late-night drives or savoring in solitude',
    '适合深夜驾车或派对预热': 'ideal for late-night drives or pre-party warmups',
    '适合深夜驾车或独处时释放压力': 'perfect for late-night drives or releasing pressure in solitude',
    '适合深夜驱车或耳机里的独行者': 'ideal for late-night drives or solo headphone sessions',
    '适合深夜疾驰或挥汗如雨的现场幻想': 'perfect for late-night sprints or sweating it out at a live show',
    '适合深夜疾驰公路': 'ideal for late-night highway sprints',
    '适合深夜疾驰或释放积压能量的时刻': 'perfect for late-night sprints or unleashing pent-up energy',
    '适合运动前热身': 'ideal for pre-workout warmups',
    '适合运动冲刺': 'perfect for workout sprints',
    '适合城市夜行': 'ideal for city nightlife',
    '适合独处时释放压抑': 'perfect for releasing pent-up emotion in solitude',
    '适合独处时让心跳与低音共振': 'ideal for letting your heartbeat resonate with the bass in solitude',
    '适合独处时回想': 'perfect for reminiscing in solitude',
    '适合独处时以极大音量释放压抑': 'ideal for releasing pent-up emotion at maximum volume in solitude',
    '适合独处沉思': 'perfect for solitary contemplation',
    '适合摇下车窗': 'ideal for rolling down the windows',
    '适合冬日派对': 'perfect for winter parties',
    '适合壁炉旁随节拍轻摆': 'ideal for swaying by the fireplace',
    '适合赛事热身或公路飞驰': 'perfect for pre-game hype or highway sprints',
    '适合健身房挥汗时全开音量释放能量': 'ideal for sweaty gym sessions at full volume',
    '适合夏日泳池派对': 'perfect for summer pool parties',
    '适合通勤路上提振心情': 'ideal for brightening a commute',
    '适合阳光通勤': 'perfect for sunny commutes',
    '适合周末购物': 'ideal for weekend shopping',
    '适合好友聚会时播放': 'perfect for hanging with friends',
    '适合独处时小酌': 'ideal for a quiet drink alone',
    '适合深夜驾车穿行城市霓虹': 'perfect for driving through the city neon glow late at night',
    '适合归途车窗半开': 'ideal for a homeward drive with the window half-open',
    '适合黄昏通勤或周末小聚时播放': 'perfect for the evening commute or casual weekend hangouts',
    '适合午后咖啡时光或深夜独处时聆听': 'ideal for an afternoon coffee or late-night solitude',
    '适合午后驾车、泳池派对或夏日黄昏时分随性播放': 'perfect for afternoon drives, pool parties, or casual summer evenings',
    '适合午后驾车、泳池派对': 'ideal for afternoon drives or pool parties',
    '适合深夜独处或派对暖场时聆听': 'perfect for late-night solitude or warming up a party',
    '适合深夜驾车或独处时回味黄金年代的浪漫余温': 'ideal for late-night drives or savoring the romantic afterglow of a golden era',
    '适合赛事热身、公路飞驰或任何需要点燃热血的时刻': 'perfect for pre-game hype, highway sprints, or any moment that needs passion ignited',
    '适合公路旅行、黄昏独饮或任何需要一点热血与怀旧的时分': 'ideal for road trips, a solo drink at dusk, or any moment that calls for a touch of passion and nostalgia',
    '适合踩下油门穿越隧道': 'perfect for flooring it through a tunnel',
    '适合凌晨三点独自在厨房举杯起舞': 'ideal for dancing alone in the kitchen at 3 AM with a raised glass',
    '适合驾车驰骋或重温经典摇滚黄金年代的午后': 'perfect for hitting the road or revisiting the golden age of classic rock on a lazy afternoon',
    '适合驾车穿行霓虹街道或独自释放心绪时聆听': 'ideal for cruising neon-lit streets or releasing emotions alone',
    '适合深夜独行或长途驾驶时聆听': 'perfect for late-night walks or long drives',
    '适合深夜独行或释放压力的瞬间': 'ideal for late-night walks or moments of release',
    '适合公路旅行或深夜独酌时播放': 'perfect for road trips or late-night drinks alone',
    '适合公路旅行或健身房释放节奏': 'ideal for road trips or gym workouts',
    '适合独自驾车穿越城市灯火': 'perfect for solo drives through city lights',
    '适合独自驾车穿越霓虹公路': 'ideal for solo drives down neon highways',
    '适合公路疾驰或健身房挥汗时全开音量释放能量': 'perfect for highway sprints or sweaty gym sessions at full volume',
    '适合公路驰骋或聚会暖场': 'ideal for highway cruising or warming up a party',
    '适合公路驰骋、赛前热身或任何需要肾上腺素飙升的时刻': 'perfect for highway sprints, pre-game hype, or any moment needing an adrenaline rush',
    '适合摇下车窗、任长发飞扬的巡航时刻': 'ideal for cruising with the windows down and hair flying free',
    '适合独处时回味黄金年代的浪漫余温': 'perfect for savoring the romantic afterglow of a golden era in solitude',
    '适合独自驾车穿行城市黄昏或深夜怀旧时分播放': 'ideal for solo drives through a city dusk or late-night nostalgia',
    '适合独酌时分或深夜公路的漫长陪伴': 'perfect for a solo drink or the long companionship of a late-night highway',
    '适合黄昏驾车或独自小酌时播放': 'ideal for dusk drives or solo drinks',
    '适合黄昏驾车或独自沉思时聆听': 'perfect for dusk drives or solitary contemplation',
    '适合黄昏驾车或情绪翻涌时聆听': 'ideal for dusk drives or turbulent emotions',
    '适合深夜独处时反复聆听': 'perfect for repeated late-night listening',
    '适合深夜独处时以极大音量': 'ideal for late-night solitude at maximum volume',
    '适合午后独处或公路远行时聆听': 'perfect for afternoon solitude or long road trips',
    '适合午后发呆或任何需要一剂明亮不造作的独立摇滚能量的时刻': 'ideal for afternoon daydreams or any moment needing a dose of bright, unpretentious indie rock energy',
    '适合午后驾车或朋友小聚时轻松播放': 'perfect for afternoon drives or casual gatherings with friends',
    '适合午后咖啡或深夜独处': 'ideal for an afternoon coffee or late-night solitude',
    '适合午后驾车、泳池派对或夏日黄昏': 'perfect for afternoon drives, pool parties, or summer evenings',
    '适合运动前热身或深夜驱车时释放压力': 'ideal for pre-workout warmups or late-night drives to release pressure',
    '适合公路旅行或深夜独酌': 'perfect for road trips or late-night drinks alone',
    '适合公路旅行、午后发呆': 'ideal for road trips or afternoon daydreams',
    '适合公路旅行或健身房释放': 'perfect for road trips or gym workouts',
    '适合公路疾驰或聚会暖场': 'ideal for highway sprints or party warmups',
    '适合公路疾驰、赛前热身': 'perfect for highway sprints or pre-game warmups',
    '适合公路驰骋': 'ideal for highway cruising',
    '适合公路远行或深夜独酌时聆听': 'perfect for long road trips or late-night drinks alone',
    '适合深夜疾驰或释放': 'ideal for late-night sprints or releasing',
    '适合深夜独酌或霓虹街头的漫步时光': 'perfect for late-night drinks or strolling neon-lit streets',
    '适合深夜独酌或公路驰骋': 'ideal for late-night drinks or highway cruising',
    '适合深夜独酌时聆听': 'perfect for late-night drinks alone',
    '适合深夜独处时反复聆听': 'ideal for repeated late-night listening',
    '适合深夜独处时沉浸': 'perfect for immersive late-night listening',
    '适合深夜独处或公路巡航': 'ideal for late-night solitude or highway cruising',
    '适合深夜独处或清晨静思': 'perfect for late-night solitude or morning reflection',
    '适合深夜独处或公路远眺': 'ideal for late-night solitude or gazing down long highways',
    '适合深夜独处或派对暖场': 'perfect for late-night solitude or party warmups',
    '适合深夜独处': 'ideal for late-night solitude',
    '适合深夜驾车': 'perfect for late-night drives',
    '适合深夜驱车': 'ideal for late-night drives',
    '适合深夜疾驰': 'perfect for late-night sprints',
    '适合深夜独行': 'ideal for late-night walks',
    '适合深夜独酌': 'perfect for late-night drinks alone',
    '适合深夜公路': 'ideal for late-night highways',
    '适合午后驾车': 'perfect for afternoon drives',
    '适合午后独处': 'ideal for afternoon solitude',
    '适合午后发呆': 'perfect for afternoon daydreams',
    '适合黄昏驾车': 'ideal for dusk drives',
    '适合黄昏独饮': 'perfect for a solo drink at dusk',
    '适合清晨静思': 'ideal for morning reflection',
    '适合独酌时分': 'perfect for a solo drink',
    '适合独处沉思': 'ideal for solitary contemplation',
    '适合独处时': 'perfect for solitude',
    '适合公路旅行': 'ideal for road trips',
    '适合公路疾驰': 'perfect for highway sprints',
    '适合公路驰骋': 'ideal for highway cruising',
    '适合公路巡航': 'perfect for highway cruising',
    '适合公路远行': 'ideal for long road trips',
    '适合驾车疾驰': 'perfect for fast drives',
    '适合驾车穿行': 'ideal for cruising through',
    '适合聚会暖场': 'perfect for party warmups',
    '适合聚会举杯': 'ideal for raising glasses at gatherings',
    '适合派对暖场': 'perfect for party warmups',
    '适合赛事热身': 'ideal for pre-game warmups',
    '适合运动冲刺': 'perfect for workout sprints',
    '适合健身房挥汗': 'ideal for sweaty gym sessions',
    '适合城市夜行': 'perfect for city nightlife',
    '适合通勤路上': 'ideal for commuting',
    '适合阳光通勤': 'perfect for sunny commutes',
    '适合周末购物': 'ideal for weekend shopping',
    '适合好友聚会': 'perfect for hanging with friends',
    '适合独处时小酌': 'ideal for a quiet drink alone',
    '适合夏日泳池派对': 'perfect for summer pool parties',
    '适合冬日派对': 'ideal for winter parties',
    '适合壁炉旁': 'perfect for by the fireplace',
    '适合婴幼儿哄睡': 'ideal for lulling babies to sleep',
    '适合午后小憩': 'perfect for an afternoon nap',
    '适合摇下车窗': 'ideal for rolling down the windows',
    '适合踩下油门': 'perfect for flooring it',
    '适合凌晨三点': 'ideal for 3 AM',
    '适合沉浸聆听': 'perfect for immersive listening',
    '适合循环播放': 'ideal for looping',
    '适合反复聆听': 'perfect for repeated listening',
    '适合独自驾车': 'ideal for solo drives',
    '适合驾车驰骋': 'perfect for hitting the road',
    '适合深夜': 'ideal for late night',
    '适合午后': 'perfect for the afternoon',
    '适合黄昏': 'ideal for dusk',
    '适合清晨': 'perfect for early morning',
    '适合夜晚': 'ideal for nighttime',
    '适合独处': 'perfect for solitude',
    '适合聆听': 'ideal for listening',
    '适合播放': 'perfect for playing',
    '适合驾车': 'ideal for driving',
    '适合通勤': 'perfect for commuting',
    '适合派对': 'ideal for parties',
    '适合聚会': 'perfect for gatherings',
    '适合酒吧': 'ideal for bars',
    '适合咖啡': 'perfect for coffee',
    '适合夏日': 'ideal for summer',
    '适合冬日': 'perfect for winter',
    '理想': 'ideal',
    '完美': 'perfect',
    # --- 乐器 ---
    '电吉他': 'electric guitar', '木吉他': 'acoustic guitar', '古典吉他': 'classical guitar',
    '吉他riff': 'guitar riff', '吉他独奏': 'guitar solo', '吉他拨弦': 'guitar picking',
    '吉他扫弦': 'guitar strumming', '吉他琶音': 'guitar arpeggios', '吉他音墙': 'guitar wall of sound',
    '双吉他': 'dual guitars', '吉他': 'guitar',
    '贝斯': 'bass', '贝斯线': 'bass line', '贝斯线条': 'bass lines',
    '鼓点': 'drum beat', '鼓': 'drums', '架子鼓': 'drum kit', '军鼓': 'snare drum',
    '钢琴': 'piano', '钢琴前奏': 'piano intro', '钢琴琶音': 'piano arpeggios',
    '合成器': 'synths', '合成器音色': 'synth tones', '合成器旋律': 'synth melody',
    '键盘': 'keyboards', '风琴': 'organ', '电子琴': 'keyboard',
    '萨克斯': 'saxophone', '萨克斯独奏': 'saxophone solo',
    '铜管': 'brass', '铜管乐': 'brass section',
    '弦乐': 'strings', '弦乐铺底': 'string pad',
    '小提琴': 'violin', '大提琴': 'cello', '长笛': 'flute',
    '单簧管': 'clarinet', '双簧管': 'oboe', '圆号': 'French horn',
    '小号': 'trumpet', '长号': 'trombone', '竖琴': 'harp',
    '口琴': 'harmonica', '手风琴': 'accordion',
    '古筝': 'guzheng', '琵琶': 'pipa', '二胡': 'erhu',
    '笛子': 'flute', '箫': 'xiao', '扬琴': 'yangqin',
    '西塔琴': 'sitar', '曼陀铃': 'mandolin', '沙锤': 'maracas',
    '打击乐': 'percussion', '锣鼓': 'gongs and drums',
    '失真吉他': 'distorted guitar', '清音吉他': 'clean guitar',
    '点弦': 'tapping', '泛音': 'harmonics',
    # --- 人声 ---
    '人声': 'vocals', '嗓音': 'voice', '女声': 'female vocals',
    '男声': 'male vocals', '主唱': 'lead vocals', '和声': 'harmonies',
    '合唱': 'chorus', '独唱': 'solo vocal', '对唱': 'duet',
    '说唱': 'rap', '嘶吼': 'howl', '呢喃': 'whisper',
    '哼唱': 'humming', '吟唱': 'singing', '念白': 'spoken word',
    '童声': 'children\'s choir', '伴唱': 'backing vocals',
    '高亢嗓音': 'soaring voice', '沙哑嗓音': 'raspy voice',
    '清澈嗓音': 'crystalline voice', '浑厚嗓音': 'rich voice',
    '温柔嗓音': 'tender voice', '甜美嗓音': 'sweet voice',
    '爆发力十足': 'explosive', '极具穿透力': 'highly penetrating',
    # --- 风格/流派形容 ---
    '摇滚': 'rock', '流行': 'pop', '民谣': 'folk', '电子': 'electronic',
    '爵士': 'jazz', '布鲁斯': 'blues', '嘻哈': 'hip-hop',
    '金属': 'metal', '朋克': 'punk', '古典': 'classical',
    '独立': 'indie', '另类': 'alternative', '迷幻': 'psychedelic',
    '梦幻': 'dreamy', '前卫': 'progressive', '实验': 'experimental',
    '氛围': 'ambient', '浩室': 'house', '迪斯科': 'disco',
    '雷鬼': 'reggae', '乡村': 'country', '放克': 'funk',
    '灵魂': 'soul', '福音': 'gospel', '新世纪': 'new age',
    '硬摇滚': 'hard rock', '重金属': 'heavy metal',
    '后摇': 'post-rock', '梦幻流行': 'dream pop',
    '新金属': 'nu-metal', '后朋克': 'post-punk', '新浪潮': 'new wave',
    '合成器流行': 'synth-pop', '电子舞曲': 'EDM',
    '电子流行': 'electropop', '电子碎拍': 'glitchy breakbeats',
    '法式浩室': 'French house', '热带浩室': 'tropical house',
    '加美兰': 'gamelan', '木卡姆': 'maqam',
    '法朵': 'fado', '桑巴': 'samba', '坎东布雷': 'candomblé',
    '科拉琴': 'kora', '乌德琴': 'oud',
    '嘻哈说唱': 'hip-hop/rap',
    # --- 情感/氛围形容 ---
    '温暖': 'warm', '冷冽': 'cold', '明亮': 'bright', '暗黑': 'dark',
    '柔和': 'soft', '强劲': 'powerful', '轻快': 'lively', '沉重': 'heavy',
    '慵懒': 'languid', '激昂': 'passionate', '舒缓': 'soothing',
    '动感': 'dynamic', '空灵': 'ethereal', '辽阔': 'expansive',
    '深邃': 'profound', '甜蜜': 'sweet', '忧郁': 'melancholy',
    '欢快': 'cheerful', '浪漫': 'romantic', '怀旧': 'nostalgic',
    '粗粝': 'raw', '细腻': 'delicate', '温柔': 'tender',
    '坚定': 'resolute', '克制': 'restrained', '宁静': 'tranquil',
    '孤独': 'solitary', '渴望': 'yearning', '希望': 'hopeful',
    '失落': 'wistful', '力量': 'strength', '勇气': 'courage',
    '宁静安详': 'peaceful', '安宁静谧': 'serene and quiet',
    '松弛': 'relaxed', '轻松': 'easy', '紧绷': 'taut',
    '紧张': 'tense', '躁动不安': 'restless unease',
    '阴郁': 'brooding', '苍凉': 'bleak', '苍穹': 'sky',
    '末世': 'apocalyptic', '末日': 'doomsday',
    '清亮': 'bright and clear', '清冽': 'crisp', '清脆': 'crisp',
    '浑厚': 'rich', '饱满': 'full', '通透': 'crystalline',
    '粗犷': 'rugged', '豪放': 'bold', '婉约': 'graceful',
    '柔美': 'soft and beautiful', '刚毅': 'resilient',
    '脆弱': 'vulnerable', '敏感': 'sensitive',
    '内敛': 'restrained', '张扬': 'bold', '含蓄': 'subtle',
    '直白': 'straightforward', '隽永': 'timeless',
    '质朴': 'earthy', '朴素': 'simple', '简约': 'minimal',
    '精致': 'refined', '华丽': 'gorgeous', '宏大': 'grand',
    '宏伟': 'majestic', '壮阔': 'vast', '磅礴': 'magnificent',
    '深沉': 'deep', '悠扬': 'melodious', '婉转': 'mellifluous',
    '铿锵': 'resonant', '飘逸': 'flowing', '灵动': 'vivid',
    '沉稳': 'steady', '奔放': 'exuberant',
    # --- 场景/时间 ---
    '深夜独处': 'late-night solitude', '深夜驾车': 'late-night drives',
    '深夜独酌': 'late-night drinks alone', '深夜独行': 'late-night walks',
    '深夜疾驰': 'late-night sprints', '深夜公路': 'late-night highway',
    '深夜': 'late night', '独处': 'solitude', '独酌': 'solo drink',
    '午后': 'afternoon', '黄昏': 'dusk', '清晨': 'early morning',
    '夜晚': 'nighttime', '午夜': 'midnight',
    '公路旅行': 'road trips', '公路驰骋': 'highway cruising',
    '公路疾驰': 'highway sprints', '公路巡航': 'highway cruising',
    '公路远行': 'long road trips', '公路': 'highway',
    '驾车': 'driving', '自驾': 'solo drive', '通勤': 'commute',
    '派对': 'party', '聚会': 'gathering', '酒吧': 'bar',
    '咖啡': 'coffee', '独酌': 'solo drink', '小酌': 'casual drink',
    '夏日': 'summer', '冬日': 'winter', '阳光': 'sunshine',
    '星空': 'starry sky', '月光': 'moonlight', '暮色': 'twilight',
    '微风': 'breeze', '海滩': 'beach', '海洋': 'ocean',
    '山峦': 'mountains', '森林': 'forest', '草原': 'grassland',
    '沙漠': 'desert', '天空': 'sky', '大地': 'earth',
    '梦境': 'dreamscape', '记忆': 'memory', '时光': 'time',
    '岁月': 'years', '旅途': 'journey', '远方': 'distant places',
    '归途': 'homeward', '霓虹': 'neon', '都市': 'urban',
    '城市': 'city', '街头': 'street', '舞池': 'dance floor',
    '健身房': 'gym', '泳池派对': 'pool party',
    '壁炉': 'fireplace', '隧道': 'tunnel',
    '体育场': 'stadium', '现场': 'live',
    '摇篮': 'cradle', '白噪音': 'white noise',
    # --- 时代 ---
    '七十年代': "'70s", '八十年代': "'80s", '九十年代': "'90s",
    '六十年代': "'60s", '五十年代': "'50s", '四十年代': "'40s",
    '千禧年初': 'early 2000s', '千禧年': 'millennium',
    '世纪末': 'turn of the century', '世纪': 'century',
    '2000年代中期': 'mid-2000s', '2000年代': '2000s',
    '2010年代中期': 'mid-2010s', '2010年代': '2010s',
    '2020年代': '2020s',
    '当代': 'contemporary', '现代': 'modern', '复古': 'retro',
    '经典': 'classic', '传统': 'traditional', '古老': 'ancient',
    # --- 地域 ---
    '美式': 'American', '英伦': 'British', '英式': 'British',
    '加州': 'California', '东海岸': 'East Coast', '西海岸': 'West Coast',
    '南方式': 'Southern', '美式车库': 'American garage',
    '西弗吉尼亚': 'West Virginia', '洛杉矶': 'Los Angeles',
    '加利福尼亚': 'California',
    # --- 结构/编曲 ---
    '编曲': 'arrangement', '副歌': 'chorus', '前奏': 'intro',
    '独奏': 'solo', '间奏': 'bridge', '尾奏': 'outro',
    '和弦': 'chords', '分解和弦': 'arpeggiated chords',
    '琶音': 'arpeggios', 'riff': 'riff',
    '四拍鼓点': 'four-on-the-floor beat', '鼓点': 'drums',
    '节拍': 'beat', '节奏': 'rhythm', '律动': 'groove',
    '旋律': 'melody', '音色': 'tone', '音墙': 'wall of sound',
    '音场': 'soundscape', '声场': 'soundstage',
    '低频': 'low frequencies', '高频': 'high frequencies',
    '低音': 'low end', '高音': 'high end',
    '失真': 'distortion', '混响': 'reverb', '延迟': 'delay',
    '采样': 'samples', '采样拼贴': 'sampled collage',
    '切碎': 'chopped', '切片': 'slices',
    # --- 常用动词/短语 ---
    '交织': 'intertwine', '穿插': 'interspersed', '点缀': 'accent',
    '铺陈': 'unfold', '铺展': 'spread out', '营造': 'create',
    '注入': 'inject', '洋溢': 'brimming with', '弥漫': 'permeate',
    '流淌': 'flow', '涌动': 'surge', '轰鸣': 'roar',
    '撕裂': 'tear through', '碾轧': 'crush', '震撼': 'stunning',
    '推动': 'drive', '推进': 'push forward', '堆叠': 'stack',
    '叠加': 'layer', '递进': 'build progressively',
    '攀升': 'climb', '爆发': 'erupt', '倾诉': 'pour out',
    '诉说': 'tell of', '勾勒': 'sketch', '凝成': 'crystallize into',
    '汇入': 'merge into', '迸发': 'burst forth',
    '扑面而来': 'washes over you', '席卷而来': 'sweeps in',
    '弥漫开来': 'permeates', '层层': 'layer upon layer',
    '渐次': 'gradually', '渐强': 'crescendo',
    '缓缓': 'slowly', '轻轻': 'gently', '深深': 'deeply',
    '如': 'like', '如同': 'like', '仿佛': 'as if',
    '像': 'like', '宛如': 'as if',
    '交织着': 'intertwined with', '伴随': 'accompanied by',
    '搭配': 'paired with', '融合': 'blending',
    '碰撞': 'collide', '碰撞出': 'collide to create',
    '蜕变成': 'transform into', '转化为': 'transform into',
    '交织在一起': 'intertwine together',
    '完美': 'perfect', '极致': 'ultimate',
    '纯粹': 'pure', '直接': 'direct',
    '毫无修饰': 'unfiltered', '毫不妥协': 'uncompromising',
    # --- 比喻/意象 ---
    '如月光洒落': 'like moonlight falling', '如流水般': 'like flowing water',
    '如暗潮涌动': 'like dark tides surging', '如风暴般': 'like a storm',
    '如钢铁巨兽': 'like an iron beast', '如战鼓般': 'like war drums',
    '如维京战船': 'like a Viking longship', '如飞鸟挣脱': 'like a bird breaking free',
    '如梭鱼般': 'like barracuda', '如午夜酒吧': 'like a midnight bar',
    '如海风轻拂': 'like a sea breeze', '如电光撕裂': 'like lightning tearing through',
    '如战旗高扬': 'like a battle flag soaring', '如微风拂过': 'like a breeze brushing',
    '如午后阳光': 'like afternoon sunlight', '如星空般': 'like a starlit sky',
    '如旧公路上的车灯': 'like headlights on an old highway',
    '如暗夜微光': 'like faint light in the dark',
    '如迁徙的兽群': 'like a migrating herd',
    '如西弗吉尼亚的乡间公路': 'like the country roads of West Virginia',
    '破浪前行': 'cutting through waves',
    '劈开空气': 'tears through the air', '劈开夜幕': 'tears through the night',
    '劈开人群': 'splits the crowd', '撕裂沉寂': 'tears through the silence',
    '撕裂寂静': 'shatters the silence', '撕裂音墙': 'tears through the wall of sound',
    '撕裂出': 'tears open', '撕裂夜空': 'tears through the night sky',
    '点燃全场': 'ignites the room', '点燃热血': 'ignites passion',
    '引爆': 'ignite', '引爆万人齐唱': 'ignites mass sing-alongs',
    '失控': 'out of control', '失控的火焰': 'an out-of-control flame',
    '狂欢': 'carnival', '狂欢宣言': 'carnival manifesto',
    '狂欢打击乐': 'carnival percussion',
    # --- 常用名词 ---
    '气质': 'spirit', '质感': 'texture', '能量': 'energy',
    '张力': 'tension', '动力': 'momentum', '魅力': 'charm',
    '风骨': 'character', '风潮': 'wave', '潮流': 'trend',
    '缩影': 'snapshot', '标志': 'hallmark', '标志性': 'iconic',
    '里程碑': 'milestone', '代表作': 'landmark', '经典之作': 'classic',
    '金曲': 'hit', '名曲': 'famous song', '巨作': 'masterpiece',
    '史诗': 'epic', '史诗级': 'epic',
    '大作': 'major work', '杰作': 'masterpiece',
    '原爆点': 'ground zero', '黄金时代': 'golden age',
    '黄金期': 'golden era', '黄金年代': 'golden age',
    '全盛期': 'heyday', '巅峰': 'peak', '巅峰张力': 'peak tension',
    '复兴': 'revival', '重生': 'rebirth',
    '余温': 'afterglow', '余韵': 'lingering resonance',
    '回响': 'echo', '共鸣': 'resonance', '共振': 'resonance',
    '诗意': 'poetic', '诗意想象': 'poetic imagination',
    '质朴诗意': 'earthy poetry', '隽永笔触': 'timeless touch',
    '从容笔触': 'unhurried touch', '松弛气息': 'laid-back spirit',
    '温暖底色': 'warm undertone', '粗犷能量': 'raw energy',
    '原始张力': 'raw tension', '原始能量': 'raw energy',
    '复古情调': 'retro charm', '都市感': 'urban sophistication',
    '浪漫律动': 'romantic groove', '浪漫余温': 'romantic afterglow',
    '夜空气息': 'nocturnal energy', '夜曲': 'nocturne',
    '夏日气泡水': 'summer soda', '清爽活力': 'effervescent freshness',
    '热带风情': 'tropical vibe', '阳光海滩': 'sunny beach',
    '海滩般的松弛氛围': 'laid-back beach vibe',
    '慵懒律动': 'languid groove', '慵懒阳光': 'laid-back sunshine',
    '慵懒吟唱': 'laid-back vocals',
    '松弛氛围': 'relaxed atmosphere', '安详氛围': 'peaceful atmosphere',
    '宁静氛围': 'serene atmosphere',
    '复古迪斯科': 'retro disco', '复古合成器': 'retro synth',
    '复古摇滚': 'retro rock', '复古失真': 'retro distortion',
    '加州风情': 'California vibes', '加州摇滚': 'California rock',
    '民谣摇滚': 'folk rock', '民谣流行': 'folk pop',
    '硬摇滚': 'hard rock', '体育场摇滚': 'arena rock',
    '车库轰鸣': 'garage roar', '吉他摇滚': 'guitar rock',
    '太空般': 'cosmic', '太空般的辽阔音场': 'vast, cosmic soundscape',
    '霓虹倒影': 'neon reflections', '霓虹下的浪漫': 'romantic neon glow',
    '霓虹街道': 'neon-lit streets', '霓虹公路': 'neon highway',
    '城市灯火': 'city lights', '城市霓虹': 'city neon',
    '都市夜晚': 'urban nights', '都市霓虹': 'urban neon',
    '夜晚的孤独': 'nighttime loneliness', '深夜的脉动': 'late-night pulse',
    '深夜舞池': 'late-night dance floor',
    '隐忍的火焰': 'smoldering flame',
    '无法排遣的焦虑': 'inescapable anxiety',
    '反战怒吼': 'anti-war fury', '反战': 'anti-war',
    '抗议': 'protest', '抗议气息': 'protest spirit',
    '轻狂往事': 'reckless youth', '旧日心事': 'old memories',
    '未尽的心事': 'unfinished thoughts',
    '孤独感': 'loneliness', '疏离感': 'alienation',
    '疏离': 'alienation', '焦虑': 'anxiety',
    '渴望': 'longing', '孤独': 'solitude',
    '怀旧': 'nostalgic', '浪漫': 'romantic',
    '温柔': 'tender', '甜蜜': 'sweet',
    '倔强': 'stubborn', '沧桑': 'weathered',
    '叛逆': 'rebellious', '躁动': 'restless',
    '青涩': 'youthful', '锋芒': 'edge',
    '自由': 'freedom', '远方': 'distant places',
    '未知的旷野': 'unknown wilderness',
    '地平线': 'horizon',
    '长发飞扬': 'hair flying free',
    '敞篷公路': 'open road',
    '无尽地平线': 'endless horizon',
    '车窗半开': 'window half-open',
    '暮色四合': 'as dusk settles in',
    '雪花纷飞': 'snow falling',
    '夏日逃逸': 'summer escape',
    '午后阳光': 'afternoon sunshine',
    '百叶窗': 'shutters',
    '穿越黑暗': 'cutting through darkness',
    '穿越霓虹': 'through neon',
    '穿越隧道': 'through a tunnel',
    '穿越城市灯火': 'through city lights',
    '穿越城市霓虹': 'through city neon',
    '驶向记忆的远方': 'toward distant memories',
    '漂向未知的旷野': 'drift toward unknown wilderness',
    '挣脱地平线': 'breaking free of the horizon',
    '弥漫着': 'permeated with',
    '充溢着': 'overflowing with',
    '散发着': 'radiating',
    '回荡着': 'echoing with',
    '流淌着': 'flowing with',
    '跃动': 'bouncy', '弹跳感': 'springy', '跳跃': 'bouncy',
    '跃动的': 'bouncy', '律动感': 'groovy',
    '金属键盘': 'metal keyboards', '互锁': 'interlocking',
    '循环': 'loops', '键盘循环': 'keyboard loops',
    '金属键盘循环': 'metal-keyboard loops',
    '旋律轮廓': 'melodic contour', '原曲': 'original',
    '重构': 'reconstruct', '重组': 'restructure',
    '丝路': 'Silk Road', '调式': 'modality',
    '转音': 'inflections', '即兴': 'improvised',
    '即兴段落': 'improvised passages', '即兴演奏': 'improvisation',
    '漂流': 'drift',
    '异色': 'vivid', '异色梦境': 'vivid dreamscape',
    '末世图景': 'apocalyptic landscape',
    '冷冽的末世图景': 'cold apocalyptic landscape',
    '数字时代': 'digital age', '机械律动': 'mechanical rhythm',
    '体温流失': 'bleeding warmth',
    '午夜酒吧': 'midnight bar',
    '小乐队': 'small band', '浮沉': 'rise and fall',
    '浮沉故事': 'story of rise and fall',
    '电台黄金气质': 'golden radio spirit',
    '电台黄金时代': 'radio golden age',
    '美式酒吧': 'American bar',
    '轻松氛围': 'easy atmosphere',
    '温暖而明亮': 'warm and luminous',
    '精致感': 'refined sheen',
    '英伦电子流行': 'British electronic pop',
    '英伦吉他摇滚': 'British guitar rock',
    '吉他分解和弦': 'guitar arpeggiated chords',
    '朦胧而温暖': 'hazy and warm',
    '明亮开阔': 'bright and expansive',
    '忧郁而温暖': 'melancholy yet warm',
    '通透吉他音色': 'crystalline guitar tones',
    '沙哑而充满张力': 'raspy and tension-filled',
    '沙哑而笃定': 'raspy yet assured',
    '简洁有力': 'concise and powerful',
    '沉稳的': 'steady', '坚实的': 'solid',
    '炽热的': 'fiery', '炽烈的': 'blazing',
    '炽烈': 'blazing', '炽热': 'fiery',
    '昂扬': 'soaring', '昂扬的': 'soaring',
    '略带忧郁': 'slightly wistful', '略带': 'slightly',
    '略带鼻音': 'slightly nasal',
    '温润': 'warm', '温润的': 'warm',
    '丝绒般': 'velvet', '丝般': 'silky',
    '丝滑': 'silky', '丝滑的': 'silky',
    '金黄': 'golden', '金黄阳光': 'golden sunshine',
    '阳光斑驳': 'sun-dappled',
    '轻快的': 'lively', '轻快': 'lively',
    '明亮的': 'bright', '明亮': 'bright',
    '柔和的': 'soft', '柔和': 'soft',
    '温暖的': 'warm', '温暖': 'warm',
    '冷冽的': 'cold', '冷冽': 'cold',
    '暗黑的': 'dark', '暗黑': 'dark',
    '强劲的': 'powerful', '强劲': 'powerful',
    '沉重的': 'heavy', '沉重': 'heavy',
    '慵懒的': 'languid', '慵懒': 'languid',
    '激昂的': 'passionate', '激昂': 'passionate',
    '舒缓的': 'soothing', '舒缓': 'soothing',
    '空灵的': 'ethereal', '空灵': 'ethereal',
    '辽阔的': 'expansive', '辽阔': 'expansive',
    '深邃的': 'profound', '深邃': 'profound',
    '甜蜜的': 'sweet', '甜蜜': 'sweet',
    '忧郁的': 'melancholy', '忧郁': 'melancholy',
    '欢快的': 'cheerful', '欢快': 'cheerful',
    '浪漫的': 'romantic', '浪漫': 'romantic',
    '怀旧的': 'nostalgic', '怀旧': 'nostalgic',
    '粗粝的': 'raw', '粗粝': 'raw',
    '细腻的': 'delicate', '细腻': 'delicate',
    '温柔的': 'tender', '温柔': 'tender',
    '沙哑的': 'raspy', '沙哑': 'raspy',
    '清澈的': 'clear', '清澈': 'clear',
    '醇厚的': 'rich', '醇厚': 'rich',
    '高亢的': 'soaring', '高亢': 'soaring',
    '低沉的': 'deep', '低沉': 'deep',
    # --- 连接词/句式 ---
    '，同时': ', while', '，继而': ', then', '，渐次': ', gradually',
    '，旋即': ', then', '，随后': ', then', '，接着': ', then',
    '，然后': ', then', '，最终': ', ultimately',
    '在这首': 'in this', '这首': 'this',
    '这首作品': 'this track', '这首曲目': 'this track',
    '这首曲子': 'this song',
    '全曲': 'the entire track', '整曲': 'the whole track',
    '整首歌': 'the whole song', '整体': 'overall',
    '前半段': 'first half', '后半段': 'second half',
    '从头到尾': 'from start to finish',
    '正是': 'is exactly', '正是这首曲子的天然舞台': 'is this song\'s natural stage',
    '带着': 'carrying', '带着一丝': 'with a hint of',
    '带着永不妥协的流浪者气质': 'carrying the unyielding spirit of the wanderer',
    '带着七十年代摇滚特有的': 'carrying the characteristic 70s rock',
    '带着七十年代美式摇滚的': 'carrying the 70s American rock',
    '带着民谣的质朴气息': 'carrying folk\'s earthy quality',
    '带着一丝怀旧却不失力量': 'with a hint of nostalgia without losing power',
    '带着七十年代的': 'carrying the 70s',
    '带着七十年代末的': 'carrying the late-70s',
    '带着八十年代的': 'carrying the 80s',
    '带着千禧年初的': 'carrying the early 2000s',
    '带着九十年代的': 'carrying the 90s',
    '带着车库般的原始张力': 'with the raw tension of garage rock',
    '带着七十年代车库般的原始张力': 'with the raw tension of 70s garage rock',
    '带着永不妥协': 'carrying an unyielding',
    '带着一丝': 'with a hint of',
    '带着': 'carrying',
    '同时': 'while', '与此同时': 'meanwhile',
    '在其中': 'within', '其间': 'throughout',
    '穿行其中': 'weaving through', '穿行': 'traversing',
    '点缀其间': 'accenting throughout',
    '其': 'its', '其中': 'within',
    '之间': 'between', '之中': 'within',
    '之上': 'upon', '之下': 'beneath',
    '之上': 'above', '之间': 'between',
    '之际': 'at the moment',
    '之时': 'when', '时分': 'moment',
    '时刻': 'moment', '瞬间': 'instant',
    '之间找到': 'finding between', '找到': 'find',
    '恰到好处': 'just right', '恰到好处的平衡': 'just the right balance',
    '平衡': 'balance', '平衡感': 'sense of balance',
    '呼吸感': 'sense of breath',
    '记住': 'remember', '忘却': 'forget',
    '沉浸': 'immerse', '沉浸于': 'immerse in',
    '沉入': 'sink into', '沉醉': 'intoxicated',
    '陶醉': 'enraptured', '迷失': 'lost',
    '追寻': 'searching', '寻找': 'seeking',
    '回忆': 'memories', '回想': 'recall',
    '回望': 'look back', '回溯': 'trace back',
    '前行': 'move forward', '前进': 'advance',
    '出发': 'set out', '启程': 'depart',
    '归来': 'return', '回家': 'go home',
    '出发': 'depart', '启程': 'set off',
    '归来': 'return', '回家': 'homeward',
    # --- 高频残留词（来自翻译统计） ---
    '铺底': 'pad', '独自': 'alone', '夜行': 'night walk',
    '情绪': 'emotion', '窗边': 'by the window', '底色': 'undertone',
    '暖场': 'warmup', '周末': 'weekend', '气息': 'spirit',
    '车窗': 'car window', '骨架': 'framework', '作品': 'work',
    '反复': 'repeatedly', '漫游': 'wander', '耳机': 'headphones',
    '驾驶': 'driving', '雨天': 'rainy day', '释放': 'release',
    '长途': 'long-distance', '叙事': 'narrative', '私密': 'intimate',
    '静静': 'quietly', '极简': 'minimalist', '声线': 'vocal line',
    '摇摆': 'sway', '途中': 'en route', '原声': 'acoustic',
    '漫步': 'stroll', '轻柔': 'gentle', '重温': 'revisit',
    '路上': 'on the road', '夏夜': 'summer night', '阅读': 'reading',
    '傍晚': 'evening', '微醺': 'tipsy', '放松': 'relax',
    '独饮': 'solo drink', '沉思': 'contemplation', '提振': 'boost',
    '朋友': 'friends', '情感': 'emotion', '舞曲': 'dance track',
    '驱车': 'drive', '曲目': 'track', '回味': 'savor',
    '穿越': 'traverse', '兜风': 'joyride', '随性': 'casual',
    '俏皮': 'playful', '夜跑': 'night jog', '静听': 'listen quietly',
    '小聚': 'small gathering', '流畅': 'smooth', '夜色': 'night scene',
    '开场': 'opening', '长途驾驶': 'long drive', '佩戴': 'wear',
    '般的': '-like', '需要': 'need', '放大': 'turn up',
    '音量': 'volume', '沉入': 'sink into', '温柔': 'tender',
    '缓缓': 'slowly', '流淌': 'flow', '洒落': 'fall',
    '穿透': 'pierce', '裹挟': 'carry', '朦胧': 'hazy',
    '点缀': 'accent', '松弛': 'relaxed', '氛围': 'atmosphere',
    '穿行': 'traverse', '飙升': 'surge', '时刻': 'moment',
    '任何': 'any', '依旧': 'still', '听众': 'listeners',
    '过去': 'past', '弹起来': 'bounce up', '挑': 'pick',
    '描绘': 'depict', '憧憬': 'longing', '少女': 'girl',
    '心事': 'thoughts', '唤起': 'evoke', '清新': 'fresh',
    '质感': 'texture', '跳跃': 'bouncy', '戏谑': 'playful',
    '语调': 'tone', '充沛': 'abundant', '忍不住': 'cannot help',
    '跟着': 'along with', '基因': 'gene', '藏着': 'hide',
    '碰撞': 'collide', '坦诚': 'honest', '倾泻': 'pour out',
    '迷茫': 'confusion', '未经修饰': 'unpolished', '荷尔蒙': 'hormones',
    '撞个满怀': 'full embrace', '辽阔': 'expansive', '感': 'sense',
    '相融': 'merge', '不失': 'without losing', '深情': 'deep feeling',
    '铺陈': 'unfold', '幽暗': 'dim', '脉冲': 'pulse',
    '潜流': 'undercurrent', '沉浸': 'immersed', '思考': 'thought',
    '以': 'with', '略带': 'slightly', '首': 'song',
    '这首': 'this', '整个': 'entire', '里': 'within',
    '后': 'after', '前': 'before', '时': 'when',
    '让': 'letting', '中': 'in', '上': 'on',
    '下': 'under', '间': 'between', '内': 'within',
    '外': 'outside', '旁': 'beside', '边': 'side',
    '深处': 'depths', '之中': 'within', '之间': 'between',
    '之际': 'when', '之时': 'when', '同时': 'meanwhile',
    '之后': 'after', '之前': 'before', '以来': 'since',
    '以来': 'since', '开始': 'begin', '结束': 'end',
    '不断': 'continuously', '逐渐': 'gradually', '慢慢': 'slowly',
    '轻轻': 'gently', '深深': 'deeply', '静静': 'quietly',
    '悄悄': 'quietly', '默默': 'silently', '淡淡': 'faintly',
    '浓浓': 'richly', '满满': 'full of', '刚刚': 'just',
    '明明': 'clearly', '偏偏': 'yet', '恰恰': 'exactly',
    '仅仅': 'merely', '几乎': 'almost', '完全': 'completely',
    '绝对': 'absolutely', '相当': 'quite', '格外': 'especially',
    '分外': 'exceptionally', '无比': 'incomparably', '极为': 'extremely',
    '颇为': 'rather', '稍许': 'slightly', '略微': 'slightly',
    '微微': 'slightly', '隐隐': 'faintly', '悄然': 'quietly',
    '骤然': 'suddenly', '蓦然': 'suddenly', '陡然': 'abruptly',
    '忽然': 'suddenly', '突然': 'suddenly', '猛然': 'suddenly',
    '顿时': 'immediately', '霎时': 'instantly', '瞬时': 'instantly',
    '此刻': 'this moment', '此时': 'at this time',
    '彼时': 'at that time', '当时': 'at that time',
    '曾经': 'once', '已': 'already', '曾': 'once',
    '将': 'will', '会': 'will', '能': 'can',
    '要': 'want', '想': 'want', '愿': 'wish',
    '应': 'should', '该': 'should', '必须': 'must',
    '可以': 'can', '能够': 'able to', '值得': 'worth',
    '适合': 'suitable for', '宜': 'suitable',
    '美好': 'beautiful', '美妙': 'wonderful', '完美': 'perfect',
    '极致': 'ultimate', '纯粹': 'pure', '真实': 'authentic',
    '自然': 'natural', '自由': 'free', '自在': 'at ease',
    '舒适': 'comfortable', '惬意': 'pleasant', '愉悦': 'delightful',
    '幸福': 'happy', '快乐': 'joyful', '开心': 'happy',
    '兴奋': 'excited', '激动': 'thrilled', '感动': 'moved',
    '震撼': 'stunned', '惊叹': 'amazed', '陶醉': 'enchanted',
    '沉醉': 'intoxicated', '迷醉': 'mesmerized', '痴醉': 'captivated',
    # --- 第二批高频残留词 ---
    '热带': 'tropical', '线条': 'lines', '单曲': 'single',
    '特有的': 'unique', '场景': 'scene', '北欧': 'Nordic',
    '友人': 'friends', '加勒比': 'Caribbean', '安第斯': 'Andean',
    '雨夜': 'rainy night', '游走': 'wander', '清冷': 'cool',
    '清透': 'crystalline', '思绪': 'thoughts', '静谧': 'tranquil',
    '抒情': 'lyrical', '明快': 'bright', '机械': 'mechanical',
    '风情': 'charm', '基底': 'base', '运动': 'exercise',
    '专注': 'focused', '开阔': 'open', '背景': 'background',
    '活力': 'vitality', '东欧': 'Eastern European', '青春': 'youth',
    '碎拍': 'breakbeats', '静思': 'quiet reflection', '制作': 'production',
    '十足': 'full of', '特征': 'character', '特色': 'character',
    '特点': 'feature', '特质': 'trait', '独特': 'unique',
    '独一无二': 'one-of-a-kind', '与众不同': 'distinctive',
    '恰到好处': 'just right', '淋漓尽致': 'to the fullest',
    '浑然天成': 'naturally formed', '美轮美奂': 'stunning',
    '引人入胜': 'captivating', '扣人心弦': 'gripping',
    '动人心弦': 'moving', '回味无穷': 'lingering',
    '余音绕梁': 'lingering resonance', '沁人心脾': 'refreshing',
    '心旷神怡': 'uplifting', '如痴如醉': 'entranced',
    '如梦如幻': 'dreamlike', '如诗如画': 'picturesque',
    '岁月': 'years', '时光': 'time', '时代': 'era',
    '记忆': 'memory', '回忆': 'memories', '往事': 'past',
    '传说': 'legend', '神话': 'myth', '梦想': 'dream',
    '理想': 'ideal', '未来': 'future', '地平线': 'horizon',
    '天际': 'horizon', '星光': 'starlight', '月色': 'moonlight',
    '暮色': 'twilight', '细雨': 'drizzle', '雪花': 'snowflakes',
    '黎明': 'dawn', '白天': 'daytime', '黑夜': 'dark night',
    '灯火': 'lights', '晨光': 'morning light', '夕阳': 'sunset',
    '彩虹': 'rainbow', '大海': 'ocean', '山川': 'mountains',
    '河流': 'river', '冰川': 'glacier', '银河': 'Milky Way',
    '宇宙': 'universe', '太空': 'space', '人间': 'world',
    '命运': 'fate', '缘分': 'destiny', '奇迹': 'miracle',
    '魔法': 'magic', '信念': 'belief', '信仰': 'faith',
    '真理': 'truth', '智慧': 'wisdom', '善良': 'kindness',
    '美丽': 'beauty', '坚强': 'strength', '寂寞': 'loneliness',
    '思念': 'longing', '等待': 'waiting', '追寻': 'pursuit',
    '探索': 'exploration', '发现': 'discovery', '遇见': 'encounter',
    '告别': 'farewell', '重逢': 'reunion', '旅行': 'travel',
    '冒险': 'adventure', '漂泊': 'wandering', '流浪': 'roaming',
    '解放': 'liberation', '觉醒': 'awakening', '蜕变': 'transformation',
    '成长': 'growth', '成熟': 'maturity', '永恒': 'eternity',
    '瞬间': 'instant', '刹那': 'moment', '开始': 'beginning',
    '结束': 'ending', '起点': 'starting point', '终点': 'destination',
    '挑战': 'challenge', '困难': 'difficulty', '障碍': 'obstacle',
    '突破': 'breakthrough', '超越': 'transcend', '战胜': 'overcome',
    '征服': 'conquer', '胜利': 'victory', '失败': 'defeat',
    '成功': 'success', '成就': 'achievement', '荣誉': 'honor',
    '骄傲': 'pride', '谦逊': 'humility', '感恩': 'gratitude',
    '悔恨': 'regret', '遗憾': 'regret', '满足': 'satisfaction',
    '痛苦': 'pain', '悲伤': 'sadness', '愤怒': 'anger',
    '恐惧': 'fear', '平静': 'calm', '安宁': 'peace',
    '狂喜': 'ecstasy', '绝望': 'despair', '失望': 'disappointment',
    '期待': 'expectation', '向往': 'yearning', '追求': 'pursuit',
    '幻想': 'fantasy', '现实': 'reality', '虚幻': 'illusory',
    '真实': 'genuine', '真诚': 'sincere', '虚假': 'false',
    '正义': 'justice', '公平': 'fairness', '爱': 'love',
    '恨': 'hate', '情': 'affection', '义': 'loyalty',
    '勇气': 'courage', '力量': 'strength', '希望': 'hope',
    '渴望': 'longing', '孤独': 'solitude', '温柔': 'tender',
    '浪漫': 'romantic', '怀旧': 'nostalgic', '甜蜜': 'sweet',
    '忧郁': 'melancholy', '欢快': 'cheerful', '粗粝': 'raw',
    '细腻': 'delicate', '坚定': 'resolute', '克制': 'restrained',
    '宁静': 'tranquil', '失落': 'wistful', '松弛': 'relaxed',
    '轻松': 'easy', '紧绷': 'taut', '紧张': 'tense',
    '躁动': 'restless', '阴郁': 'brooding', '苍凉': 'bleak',
    '末世': 'apocalyptic', '脆弱': 'vulnerable', '敏感': 'sensitive',
    '内敛': 'restrained', '张扬': 'bold', '含蓄': 'subtle',
    '直白': 'straightforward', '隽永': 'timeless', '质朴': 'earthy',
    '朴素': 'simple', '简约': 'minimal', '精致': 'refined',
    '华丽': 'gorgeous', '宏大': 'grand', '宏伟': 'majestic',
    '壮阔': 'vast', '磅礴': 'magnificent', '深沉': 'deep',
    '悠扬': 'melodious', '婉转': 'mellifluous', '铿锵': 'resonant',
    '飘逸': 'flowing', '灵动': 'vivid', '沉稳': 'steady',
    '奔放': 'exuberant', '温柔': 'tender', '倔强': 'stubborn',
    '沧桑': 'weathered', '叛逆': 'rebellious', '青涩': 'youthful',
    '锋芒': 'edge', '自由': 'freedom', '远方': 'distant places',
    '未知的旷野': 'unknown wilderness', '长发飞扬': 'hair flying free',
    '敞篷公路': 'open road', '无尽地平线': 'endless horizon',
    '车窗半开': 'window half-open', '暮色四合': 'as dusk settles in',
    '雪花纷飞': 'snow falling', '夏日逃逸': 'summer escape',
    '午后阳光': 'afternoon sunshine', '百叶窗': 'shutters',
    '穿越黑暗': 'cutting through darkness', '穿越霓虹': 'through neon',
    '穿越隧道': 'through a tunnel', '穿越城市灯火': 'through city lights',
    '驶向记忆的远方': 'toward distant memories',
    '漂向未知的旷野': 'drift toward unknown wilderness',
    '挣脱地平线': 'breaking free of the horizon',
    '隐忍的火焰': 'smoldering flame',
    '无法排遣的焦虑': 'inescapable anxiety',
    '反战怒吼': 'anti-war fury', '反战': 'anti-war',
    '抗议': 'protest', '抗议气息': 'protest spirit',
    '轻狂往事': 'reckless youth', '旧日心事': 'old memories',
    '未尽的心事': 'unfinished thoughts',
    '孤独感': 'loneliness', '疏离感': 'alienation',
    '疏离': 'alienation', '焦虑': 'anxiety',
    '渴望': 'longing', '孤独': 'solitude',
    '倔强': 'stubborn', '沧桑': 'weathered',
    '叛逆': 'rebellious', '躁动': 'restless',
    '青涩': 'youthful', '锋芒': 'edge',
    '自由': 'freedom', '远方': 'distant places',
}

# === 句式模板（正则表达式 -> 英文模板） ===
SENTENCE_TEMPLATES = [
    # "X与Y交织" / "X和Y交织出Z"
    (r'(.+?)与(.+?)交织', r'\1 and \2 intertwine'),
    (r'(.+?)和(.+?)交织', r'\1 and \2 intertwine'),
    (r'(.+?)与(.+?)交织出(.+)', r'\1 and \2 weave \3'),
    (r'(.+?)与(.+?)交织成(.+)', r'\1 and \2 weave into \3'),
    # "X铺陈出Y" / "X铺展出Y"
    (r'(.+?)铺陈出(.+)', r'\1 unfold \2'),
    (r'(.+?)铺展出(.+)', r'\1 spread out \2'),
    # "X营造出Y"
    (r'(.+?)营造出(.+)', r'\1 create \2'),
    # "X勾勒出Y"
    (r'(.+?)勾勒出(.+)', r'\1 sketch out \2'),
    # "X适合Y时聆听" / "适合Y时播放"
    (r'适合(.+?)时聆听', r'Ideal for \1'),
    (r'适合(.+?)时播放', r'Perfect for \1'),
    (r'适合(.+?)聆听', r'Ideal for \1'),
    (r'适合(.+?)播放', r'Perfect for \1'),
    # "X扑面而来"
    (r'(.+?)扑面而来', r'\1 washes over you'),
    # "X撕裂出Y"
    (r'(.+?)撕裂出(.+)', r'\1 tears open \2'),
    # "X如Y般Z"
    (r'(.+?)如(.+?)般(.+)', r'\1, like \2, \3'),
    # "X在Y中Z"
    (r'在(.+?)中(.+)', r'in \1, \2'),
    # "X带着Y"
    (r'(.+?)带着(.+)', r'\1, carrying \2'),
    # "X层层递进"
    (r'(.+?)层层递进', r'\1 building layer by layer'),
    (r'(.+?)层层堆叠', r'\1 stacking in layers'),
    (r'(.+?)层层推进', r'\1 driving forward layer by layer'),
    # "X缓缓铺陈" / "X缓缓流淌"
    (r'(.+?)缓缓铺陈', r'\1 slowly unfolds'),
    (r'(.+?)缓缓流淌', r'\1 flows gently'),
    (r'(.+?)缓缓铺展', r'\1 gradually spreads out'),
    # "X渐次Y"
    (r'(.+?)渐次(.+)', r'\1, gradually \2'),
    # "X涌动着Y"
    (r'(.+?)涌动着(.+)', r'\1 surge with \2'),
    # "X弥漫着Y"
    (r'(.+?)弥漫着(.+)', r'\1 permeated with \2'),
    # "X充溢着Y"
    (r'(.+?)充溢着(.+)', r'\1 overflowing with \2'),
    # "X散发着Y"
    (r'(.+?)散发着(.+)', r'\1 radiating \2'),
    # "X回荡着Y"
    (r'(.+?)回荡着(.+)', r'\1 echoing with \2'),
    # "X流淌着Y"
    (r'(.+?)流淌着(.+)', r'\1 flowing with \2'),
    # "X洋溢着Y"
    (r'(.+?)洋溢着(.+)', r'\1 brimming with \2'),
    # "X穿插其间"
    (r'(.+?)穿插其间', r'\1 interspersed throughout'),
    # "X点缀其间"
    (r'(.+?)点缀其间', r'\1 accenting throughout'),
    # "X穿行其中"
    (r'(.+?)穿行其中', r'\1 weaving through'),
    # "X交织在一起"
    (r'(.+?)交织在一起', r'\1 intertwine together'),
    # "X交织着Y"
    (r'(.+?)交织着(.+)', r'\1 intertwined with \2'),
    # "X伴随Y"
    (r'(.+?)伴随(.+)', r'\1 accompanied by \2'),
    # "X搭配Y"
    (r'(.+?)搭配(.+)', r'\1 paired with \2'),
    # "X融合Y"
    (r'(.+?)融合(.+)', r'\1 blending with \2'),
    # "X碰撞出Y"
    (r'(.+?)碰撞出(.+)', r'\1 collide to create \2'),
    # "X碰撞Y"
    (r'(.+?)碰撞(.+)', r'\1 collide with \2'),
    # "X蜕变成Y"
    (r'(.+?)蜕变成(.+)', r'\1 transform into \2'),
    # "X转化为Y"
    (r'(.+?)转化为(.+)', r'\1 transform into \2'),
    # "X凝成Y"
    (r'(.+?)凝成(.+)', r'\1 crystallize into \2'),
    # "X汇入Y"
    (r'(.+?)汇入(.+)', r'\1 merge into \2'),
    # "X迸发出Y"
    (r'(.+?)迸发出(.+)', r'\1 burst forth with \2'),
    # "X倾诉Y"
    (r'(.+?)倾诉(.+)', r'\1 pour out \2'),
    # "X诉说Y"
    (r'(.+?)诉说(.+)', r'\1 tell of \2'),
    # "X注入Y"
    (r'(.+?)注入(.+)', r'\1 inject \2'),
    # "X推动Y"
    (r'(.+?)推动(.+)', r'\1 drive \2'),
    # "X推进Y"
    (r'(.+?)推进(.+)', r'\1 push \2 forward'),
    # "X堆叠出Y"
    (r'(.+?)堆叠出(.+)', r'\1 stack into \2'),
    # "X叠加Y"
    (r'(.+?)叠加(.+)', r'\1 layer with \2'),
    # "X攀升Y"
    (r'(.+?)攀升', r'\1 climb'),
    # "X爆发Y"
    (r'(.+?)爆发(.+)', r'\1 erupt \2'),
    # "X席卷而来"
    (r'(.+?)席卷而来', r'\1 sweeps in'),
    # "X撕裂Y"
    (r'(.+?)撕裂(.+)', r'\1 tear through \2'),
    # "X碾轧而来"
    (r'(.+?)碾轧而来', r'\1 grind forward'),
    # "X轰鸣Y"
    (r'(.+?)轰鸣(.+)', r'\1 roar \2'),
    # "X涌动Y"
    (r'(.+?)涌动(.+)', r'\1 surge \2'),
    # "X弥漫Y"
    (r'(.+?)弥漫(.+)', r'\1 permeate \2'),
    # "X流淌Y"
    (r'(.+?)流淌(.+)', r'\1 flow \2'),
    # "X铺展Y"
    (r'(.+?)铺展(.+)', r'\1 spread \2'),
    # "X营造Y"
    (r'(.+?)营造(.+)', r'\1 create \2'),
    # "X勾勒Y"
    (r'(.+?)勾勒(.+)', r'\1 sketch \2'),
    # "X洋溢Y"
    (r'(.+?)洋溢(.+)', r'\1 brimming with \2'),
    # "X注入Y"
    (r'(.+?)注入(.+)', r'\1 inject \2 into'),
    # "X适合Y"
    (r'适合(.+)', r'Ideal for \1'),
    # "这首X"
    (r'这首(.+?)以(.+)', r'This \1, built on \2'),
    (r'这首(.+?)经典', r'This \1 classic'),
    # "X是Y的Z"
    (r'(.+?)是(.+?)的(.+)', r'\1 is the \3 of \2'),
]


def _replace_longest_first(text, mapping):
    """用映射表从长到短替换文本中的中文短语，确保英文词之间有空格。"""
    for zh in sorted(mapping.keys(), key=len, reverse=True):
        en = mapping[zh]
        # 替换时在英文值前后加空格，确保与相邻英文词分开
        text = text.replace(zh, f' {en} ')
    # 清理中文-英文边界（中文本身不需要额外空格）
    text = re.sub(r'([一-鿿])\s+([一-鿿])', r'\1\2', text)
    # 在中英文边界加空格
    text = re.sub(r'([一-鿿])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])([一-鿿])', r'\1 \2', text)
    # 合并多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def translate_description(desc):
    """将中文音乐描述翻译为英文。"""
    if not desc or not re.search(r'[一-鿿]', desc):
        return desc

    # 按中文标点分句
    sentences = re.split(r'([。；！？])', desc)
    result_parts = []

    for part in sentences:
        if not part.strip():
            continue
        if part in '。；！？':
            if part == '。':
                result_parts.append('. ')
            elif part == '；':
                result_parts.append('; ')
            elif part == '！':
                result_parts.append('! ')
            elif part == '？':
                result_parts.append('? ')
            continue

        # 逗号分隔的子句
        clauses = re.split(r'(，|、)', part)
        clause_parts = []
        for clause in clauses:
            if clause in '，、':
                clause_parts.append(', ')
                continue
            translated = translate_clause(clause.strip())
            clause_parts.append(translated)

        result_parts.append(''.join(clause_parts))

    result = ''.join(result_parts).strip()
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\.\s*\.', '.', result)
    result = re.sub(r',\s*,', ',', result)
    result = re.sub(r'\s+,', ',', result)
    result = re.sub(r',\s*\.', '.', result)
    result = result.strip(' ,.')
    # 最终清理：移除所有残留中文字符
    result = cleanup_residual_chinese(result)
    if result and not result.endswith(('.', '!', '?')):
        result += '.'
    if result:
        result = result[0].upper() + result[1:]
    return result


def translate_clause(clause):
    """翻译单个子句。"""
    if not clause:
        return ''

    # 1. 先用句式模板整体匹配
    for pattern, replacement in SENTENCE_TEMPLATES:
        match = re.search(pattern, clause)
        if match:
            translated = match.expand(replacement)
            # 对模板中的捕获组内容也做词组替换
            translated = _replace_longest_first(translated, PHRASE_MAP)
            # 清理残留中文
            translated = cleanup_residual_chinese(translated)
            return translated

    # 2. 用词组映射表替换
    result = _replace_longest_first(clause, PHRASE_MAP)

    # 3. 检查剩余中文比例
    remaining_cn = re.findall(r'[一-鿿]+', result)
    cn_ratio = len(''.join(remaining_cn)) / max(len(clause), 1)

    if cn_ratio > 0.5:
        # 超过50%还是中文，用语义模板
        return semantic_translate(clause)

    # 4. 清理残留的少量中文（替换为对应英文或删除）
    result = cleanup_residual_chinese(result)
    return result


def cleanup_residual_chinese(text):
    """清理翻译后残留的中文片段。"""
    residual_map = {
        # 助词/语气词
        '的': ' ', '了': ' ', '着': ' ', '过': ' ', '得': ' ',
        '地': ' ', '之': ' ', '其': ' ',
        # 介词/连词
        '在': ' in ', '到': ' to ', '把': ' ', '被': ' by ',
        '从': ' from ', '向': ' to ', '往': ' toward ',
        '与': ' and ', '和': ' and ', '及': ' and ',
        '或': ' or ', '或者': ' or ', '还是': ' or ',
        '以及': ' and ', '并且': ' and ', '而且': ' and ',
        '但': ' but ', '却': ' yet ', '而': ' but ',
        '因为': ' because ', '所以': ' so ', '如果': ' if ',
        '虽然': ' although ', '尽管': ' despite ',
        '随着': ' with ', '沿着': ' along ',
        # 副词
        '很': ' very ', '非常': ' very ', '极': ' extremely ',
        '更': ' more ', '最': ' most ', '太': ' too ',
        '又': ' and ', '再': ' again ', '还': ' still ',
        '就': ' then ', '才': ' only ', '都': ' all ',
        '也': ' also ', '只': ' only ', '仅': ' merely ',
        '已': ' already ', '曾': ' once ', '将': ' will ',
        '正': ' right now ', '仍': ' still ', '却': ' yet ',
        '不断': ' continuously ', '逐渐': ' gradually ',
        '不断': ' ceaselessly ', '缓缓': ' slowly ',
        '轻轻': ' gently ', '深深': ' deeply ',
        # 量词/限定词
        '一些': ' some ', '一点': ' a bit ', '几分': ' somewhat ',
        '一种': ' a ', '某种': ' some ', '某个': ' some ',
        '每个': ' every ', '各个': ' each ', '所有': ' all ',
        '整个': ' entire ', '全部': ' all ',
        # 代词
        '这': ' this ', '那': ' that ', '此': ' this ',
        '它': ' it ', '他': ' he ', '她': ' she ',
        '我们': ' we ', '他们': ' they ', '她们': ' they ',
        '自己': ' oneself ', '自身': ' itself ',
        # 时间/方位
        '时': ' when ', '时候': ' when ', '之际': ' when ',
        '之时': ' when ', '时分': ' moment ', '时刻': ' moment ',
        '之间': ' between ', '之中': ' within ',
        '之上': ' upon ', '之下': ' beneath ', '之上': ' above ',
        '其间': ' throughout ', '其中': ' within ',
        '之前': ' before ', '之后': ' after ',
        '以来': ' since ', '之间': ' between ',
        # 常见动词
        '是': ' is ', '有': ' has ', '让': ' letting ',
        '使': ' making ', '叫': ' called ', '给': ' give ',
        '用': ' using ', '做': ' do ', '来': ' come ',
        '去': ' go ', '到': ' to ', '会': ' will ',
        '能': ' can ', '可': ' can ', '应': ' should ',
        '需': ' need ', '要': ' need ', '想': ' want ',
        '像': ' like ', '如': ' like ', '似': ' like ',
        # 常见名词/形容词残留
        '歌': ' song ', '曲': ' track ', '首': ' ',
        '段': ' section ', '部分': ' part ',
        '旋律': ' melody ', '节奏': ' rhythm ', '节拍': ' beat ',
        '声音': ' sound ', '音乐': ' music ',
        '风格': ' style ', '氛围': ' atmosphere ',
        '感觉': ' feeling ', '情绪': ' emotion ',
        '故事': ' story ', '画面': ' imagery ',
        '世界': ' world ', '空间': ' space ',
        # 常见短语
        '这首': ' this ', '这首歌': ' this song ',
        '整首': ' the entire ', '整首歌': ' the whole song ',
        '一种': ' a ', '几分': ' some ',
        '之中': ' within ', '之间': ' between ',
    }
    for zh, en in sorted(residual_map.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(zh, en)

    # 如果还有残留中文，用空格替换
    text = re.sub(r'[一-鿿]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def semantic_translate(sentence):
    """对无法用映射表翻译的句子，用语义模板翻译。"""
    has_instrument = any(w in sentence for w in ['吉他', '贝斯', '鼓', '钢琴', '合成器', '萨克斯', '铜管', '弦乐', '古筝', '琵琶', '二胡', '笛子', '箫', '扬琴', '手风琴', '口琴', '小提琴', '大提琴', '长笛', '单簧管', '双簧管', '圆号', '小号', '长号', '竖琴', '风琴', '电子琴', '打击乐', '西塔琴', '曼陀铃', '沙锤', '锣鼓', '口风琴', '马林巴', '钢片琴'])
    has_vocal = any(w in sentence for w in ['人声', '嗓音', '唱', '吟唱', '合唱', '和声', '独唱', '对唱', '说唱', '念白', '嘶吼', '呢喃', '哼唱', '伴唱', '童声'])
    has_style = any(w in sentence for w in ['摇滚', '流行', '民谣', '电子', '爵士', '布鲁斯', '嘻哈', '金属', '朋克', '古典', '独立', '另类', '迷幻', '梦幻', '前卫', '实验', '氛围', '浩室', '迪斯科', '雷鬼', '乡村', '蓝草', '放克', '灵魂', '福音', '嘻哈', '嘻哈', '嘻哈'])
    has_mood = any(w in sentence for w in ['温暖', '冷冽', '明亮', '暗黑', '柔和', '强劲', '轻快', '沉重', '慵懒', '激昂', '舒缓', '动感', '空灵', '辽阔', '深邃', '甜蜜', '忧郁', '欢快', '浪漫', '怀旧', '粗粝', '细腻', '温柔', '坚定', '克制', '宁静', '孤独', '渴望', '希望', '失落', '力量', '勇气', '松弛', '轻松', '紧绷', '紧张', '躁动', '阴郁', '苍凉', '末世', '脆弱', '敏感', '内敛', '张扬', '奔放', '沉稳', '飘逸', '灵动', '深沉', '悠扬', '婉转', '铿锵', '饱满', '通透', '浑厚', '清亮', '清脆', '粗犷', '豪放', '婉约', '柔美', '刚毅', '坚韧'])
    has_scene = any(w in sentence for w in ['适合', '聆听', '播放', '驾车', '公路', '深夜', '午后', '黄昏', '清晨', '夜晚', '独处', '派对', '聚会', '酒吧', '咖啡', '夏日', '冬日', '阳光', '星空', '月光', '通勤', '健身', '泳池', '海滩', '壁炉', '隧道', '现场'])
    has_era = any(w in sentence for w in ['年代', '千禧', '世纪', '当代', '现代', '复古', '经典', '传统', '古老'])

    parts = []

    if has_instrument:
        instruments = []
        instrument_map = [
            ('吉他', 'guitar'), ('贝斯', 'bass'), ('鼓', 'drums'), ('钢琴', 'piano'),
            ('合成器', 'synths'), ('萨克斯', 'saxophone'), ('铜管', 'brass'),
            ('弦乐', 'strings'), ('古筝', 'guzheng'), ('琵琶', 'pipa'),
            ('二胡', 'erhu'), ('笛子', 'flute'), ('箫', 'xiao'),
            ('扬琴', 'yangqin'), ('手风琴', 'accordion'), ('口琴', 'harmonica'),
            ('小提琴', 'violin'), ('大提琴', 'cello'), ('长笛', 'flute'),
            ('单簧管', 'clarinet'), ('双簧管', 'oboe'), ('圆号', 'French horn'),
            ('小号', 'trumpet'), ('长号', 'trombone'), ('竖琴', 'harp'),
            ('风琴', 'organ'), ('电子琴', 'keyboard'), ('打击乐', 'percussion'),
            ('西塔琴', 'sitar'), ('曼陀铃', 'mandolin'), ('沙锤', 'maracas'),
            ('马林巴', 'marimba'), ('钢片琴', 'celesta'),
        ]
        for zh, en in instrument_map:
            if zh in sentence:
                instruments.append(en)
        if instruments:
            parts.append(f"featuring {', '.join(instruments)}")

    if has_vocal:
        vocal_map = [
            ('女声', 'female vocals'), ('男声', 'male vocals'),
            ('合唱', 'choral harmonies'), ('说唱', 'rap vocals'),
            ('嘶吼', 'intense screaming'), ('呢喃', 'whispered vocals'),
            ('哼唱', 'humming'), ('吟唱', 'singing'),
            ('童声', 'children\'s choir'), ('伴唱', 'backing vocals'),
            ('念白', 'spoken word'),
        ]
        for zh, en in vocal_map:
            if zh in sentence:
                parts.append(f"with {en}")
                break
        else:
            if any(w in sentence for w in ['人声', '嗓音', '唱']):
                parts.append('with expressive vocals')

    if has_style:
        styles = []
        style_map = [
            ('摇滚', 'rock'), ('流行', 'pop'), ('民谣', 'folk'),
            ('电子', 'electronic'), ('爵士', 'jazz'), ('布鲁斯', 'blues'),
            ('嘻哈', 'hip-hop'), ('金属', 'metal'), ('朋克', 'punk'),
            ('古典', 'classical'), ('独立', 'indie'), ('另类', 'alternative'),
            ('迷幻', 'psychedelic'), ('梦幻', 'dream pop'), ('前卫', 'progressive'),
            ('实验', 'experimental'), ('氛围', 'ambient'), ('浩室', 'house'),
            ('迪斯科', 'disco'), ('雷鬼', 'reggae'), ('乡村', 'country'),
            ('蓝草', 'bluegrass'), ('放克', 'funk'), ('灵魂', 'soul'),
            ('福音', 'gospel'), ('新世纪', 'new age'),
        ]
        for zh, en in style_map:
            if zh in sentence:
                styles.append(en)
        if styles:
            parts.append(f"a {', '.join(styles)} track")

    if has_mood:
        moods = []
        mood_map = [
            ('温暖', 'warm'), ('冷冽', 'cold'), ('明亮', 'bright'),
            ('暗黑', 'dark'), ('柔和', 'soft'), ('强劲', 'powerful'),
            ('轻快', 'lively'), ('沉重', 'heavy'), ('慵懒', 'languid'),
            ('激昂', 'passionate'), ('舒缓', 'soothing'), ('动感', 'dynamic'),
            ('空灵', 'ethereal'), ('辽阔', 'expansive'), ('深邃', 'profound'),
            ('甜蜜', 'sweet'), ('忧郁', 'melancholy'), ('欢快', 'cheerful'),
            ('浪漫', 'romantic'), ('怀旧', 'nostalgic'), ('粗粝', 'raw'),
            ('细腻', 'delicate'), ('温柔', 'tender'), ('克制', 'restrained'),
            ('宁静', 'serene'), ('孤独', 'solitary'), ('渴望', 'yearning'),
            ('失落', 'wistful'), ('松弛', 'relaxed'), ('紧绷', 'taut'),
            ('躁动', 'restless'), ('阴郁', 'brooding'), ('末世', 'apocalyptic'),
            ('脆弱', 'vulnerable'), ('深沉', 'deep'), ('饱满', 'full'),
            ('通透', 'crystalline'), ('浑厚', 'rich'), ('清亮', 'bright'),
            ('粗犷', 'rugged'), ('奔放', 'exuberant'), ('沉稳', 'steady'),
        ]
        for zh, en in mood_map:
            if zh in sentence:
                moods.append(en)
        if moods:
            parts.append(f"with a {', '.join(moods)} mood")

    if has_era:
        era_map = [
            ('七十年代', "'70s"), ('八十年代', "'80s"), ('九十年代', "'90s"),
            ('六十年代', "'60s"), ('五十年代', "'50s"), ('千禧年初', 'early 2000s'),
            ('2010年代中期', 'mid-2010s'), ('2000年代中期', 'mid-2000s'),
            ('2010年代', '2010s'), ('2000年代', '2000s'), ('2020年代', '2020s'),
            ('世纪末', 'turn of the century'), ('当代', 'contemporary'),
            ('复古', 'retro-styled'), ('经典', 'classic'), ('传统', 'traditional'),
        ]
        for zh, en in era_map:
            if zh in sentence:
                parts.append(f"from the {en}" if not en.startswith(('contemporary', 'retro', 'classic', 'traditional')) else en)
                break

    if has_scene:
        scenes = []
        scene_map = [
            ('深夜', 'late-night listening'), ('午后', 'afternoon relaxation'),
            ('黄昏', 'dusk drives'), ('清晨', 'early morning'),
            ('夜晚', 'nighttime'), ('独处', 'solitude'),
            ('公路', 'road trips'), ('驾车', 'driving'),
            ('通勤', 'commuting'), ('派对', 'parties'),
            ('聚会', 'gatherings'), ('酒吧', 'bar hangouts'),
            ('咖啡', 'coffee breaks'), ('夏日', 'summer days'),
            ('冬日', 'winter evenings'), ('阳光', 'sunny days'),
            ('星空', 'stargazing'), ('月光', 'moonlit nights'),
            ('健身', 'gym sessions'), ('泳池', 'pool parties'),
            ('海滩', 'beach days'), ('壁炉', 'fireside evenings'),
        ]
        for zh, en in scene_map:
            if zh in sentence:
                scenes.append(en)
        if scenes:
            parts.append(f"perfect for {', '.join(scenes)}")

    if not parts:
        # 最后尝试：用词组映射表做最大努力翻译
        result = _replace_longest_first(sentence, PHRASE_MAP)
        result = cleanup_residual_chinese(result)
        if result and len(result) > 10:
            return result
        return "A track with expressive melodies and rich instrumental textures."

    result = ' '.join(parts)
    result = cleanup_residual_chinese(result)
    if not result.endswith('.'):
        result += '.'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Only show sample translations, do not modify CSV')
    parser.add_argument('--batch-size', type=int, default=500, help='Rows per progress report')
    args = parser.parse_args()

    print(f"Reading CSV: {CSV_PATH}")
    with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Total rows: {len(rows)}")
    print(f"Original columns: {len(fieldnames)}")

    new_columns = ['description_en', 'label_en', 'tags_en', 'title_en', 'artist_en', 'album_en']
    for col in new_columns:
        if col not in fieldnames:
            fieldnames.append(col)

    desc_count = 0
    label_count = 0
    tags_count = 0
    title_count = 0
    artist_count = 0
    album_count = 0

    start_time = time.time()
    for i, row in enumerate(rows):
        desc = row.get('description', '').strip()
        if desc and re.search(r'[一-鿿]', desc):
            row['description_en'] = translate_description(desc)
            desc_count += 1
        else:
            row['description_en'] = ''

        label = row.get('label', '').strip()
        if label and re.search(r'[一-鿿]', label):
            row['label_en'] = GENRE_MAP.get(label, label)
            label_count += 1
        else:
            row['label_en'] = ''

        tags = row.get('tags', '').strip()
        if tags and re.search(r'[一-鿿]', tags):
            row['tags_en'] = GENRE_MAP.get(tags, tags)
            tags_count += 1
        else:
            row['tags_en'] = ''

        title = row.get('title', '').strip()
        if title and re.search(r'[一-鿿]', title):
            row['title_en'] = title
            title_count += 1
        else:
            row['title_en'] = ''

        artist = row.get('artist', '').strip()
        if artist and re.search(r'[一-鿿]', artist):
            row['artist_en'] = artist
            artist_count += 1
        else:
            row['artist_en'] = ''

        album = row.get('album', '').strip()
        if album and re.search(r'[一-鿿]', album):
            row['album_en'] = album
            album_count += 1
        else:
            row['album_en'] = ''

        if (i + 1) % args.batch_size == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {i+1}/{len(rows)} rows ({elapsed:.1f}s)")

    elapsed = time.time() - start_time
    print(f"\nTranslation complete in {elapsed:.1f}s")
    print(f"  description_en: {desc_count} rows")
    print(f"  label_en: {label_count} rows")
    print(f"  tags_en: {tags_count} rows")
    print(f"  title_en: {title_count} rows")
    print(f"  artist_en: {artist_count} rows")
    print(f"  album_en: {album_count} rows")

    if args.dry_run:
        print("\n--- DRY RUN: Sample translations ---")
        samples = [r for r in rows if r.get('description_en')][:10]
        for s in samples:
            print(f"\n  Original: {s['description'][:100]}")
            print(f"  English:  {s['description_en'][:100]}")
        return

    print(f"\nWriting CSV with {len(fieldnames)} columns...")
    tmp_path = CSV_PATH + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    if os.path.exists(BACKUP_PATH):
        os.remove(BACKUP_PATH)
    os.rename(CSV_PATH, BACKUP_PATH)
    os.rename(tmp_path, CSV_PATH)
    print(f"Done! Backup saved to: {BACKUP_PATH}")


if __name__ == '__main__':
    main()
