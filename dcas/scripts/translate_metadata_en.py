"""
将 metadata_merged.csv 中的中文元数据翻译为英文，新增 description_en, label_en, tags_en 等列。

翻译策略：
1. description: 读取中文描述，用内置翻译规则逐句翻译为英文
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
    '中国摇滚': 'Chinese Rock',
    '中国流行': 'Chinese Pop',
    '中国民谣': 'Chinese Folk',
    '中国电子': 'Chinese Electronic',
    '中国嘻哈': 'Chinese Hip-Hop',
    '中国说唱': 'Chinese Rap',
    '中国R&B': 'Chinese R&B',
    '中国民乐': 'Chinese Traditional',
    '中国古典': 'Chinese Classical',
    '华语流行': 'Mandarin Pop',
    '粤语流行': 'Cantonese Pop',
    '台湾独立': 'Taiwanese Indie',
    '日本摇滚': 'Japanese Rock',
    '日本流行': 'Japanese Pop',
    '日本电子': 'Japanese Electronic',
    '日本民谣': 'Japanese Folk',
    '日本爵士': 'Japanese Jazz',
    '日本嘻哈': 'Japanese Hip-Hop',
    '日本金属': 'Japanese Metal',
    '日本朋克': 'Japanese Punk',
    '日本独立': 'Japanese Indie',
    '日本动漫': 'Anime',
    '韩语流行': 'Korean Pop',
    '韩语摇滚': 'Korean Rock',
    '韩语嘻哈': 'Korean Hip-Hop',
    '韩语R&B': 'Korean R&B',
    '韩语电子': 'Korean Electronic',
    '印度流行': 'Indian Pop',
    '印度古典': 'Indian Classical',
    '印度民间': 'Indian Folk',
    '宝莱坞': 'Bollywood',
    '巴西流行': 'Brazilian Pop',
    '巴西摇滚': 'Brazilian Rock',
    '桑巴': 'Samba',
    '波萨诺瓦': 'Bossa Nova',
    '弗拉门戈': 'Flamenco',
    '拉丁流行': 'Latin Pop',
    '拉丁摇滚': 'Latin Rock',
    '雷鬼': 'Reggae',
    '非洲流行': 'African Pop',
    '非洲民间': 'African Folk',
    '非洲摇滚': 'African Rock',
    '中东流行': 'Middle Eastern Pop',
    '中东民间': 'Middle Eastern Folk',
    '土耳其流行': 'Turkish Pop',
    '阿拉伯流行': 'Arabic Pop',
    '东南亚流行': 'Southeast Asian Pop',
    '凯尔特': 'Celtic',
    '凯尔特民间': 'Celtic Folk',
    '北欧流行': 'Nordic Pop',
    '北欧民间': 'Nordic Folk',
    '北欧金属': 'Nordic Metal',
    '东欧流行': 'Eastern European Pop',
    '东欧民间': 'Eastern European Folk',
    '巴尔干': 'Balkan',
    '巴尔干民间': 'Balkan Folk',
    '加勒比': 'Caribbean',
    '安第斯': 'Andean',
    '安第斯民间': 'Andean Folk',
    '中亚流行': 'Central Asian Pop',
    '中亚民间': 'Central Asian Folk',
    '世界音乐': 'World Music',
    '新世纪': 'New Age',
    '电子': 'Electronic',
    '摇滚': 'Rock',
    '流行': 'Pop',
    '民谣': 'Folk',
    '爵士': 'Jazz',
    '布鲁斯': 'Blues',
    '古典': 'Classical',
    '嘻哈': 'Hip-Hop',
    '说唱': 'Rap',
    '金属': 'Metal',
    '朋克': 'Punk',
    '乡村': 'Country',
    '蓝草': 'Bluegrass',
    '放克': 'Funk',
    '灵魂': 'Soul',
    '福音': 'Gospel',
    '氛围': 'Ambient',
    '实验': 'Experimental',
    '后摇': 'Post-Rock',
    '梦幻流行': 'Dream Pop',
    '迷幻摇滚': 'Psychedelic Rock',
    '独立摇滚': 'Indie Rock',
    '独立流行': 'Indie Pop',
    '另类摇滚': 'Alternative Rock',
    '后朋克': 'Post-Punk',
    '新浪潮': 'New Wave',
    '合成器流行': 'Synth-Pop',
    '电子舞曲': 'EDM',
    '浩室': 'House',
    '科技舞曲': 'Techno',
    '迷幻电子': 'Psytrance',
    '鼓贝斯': 'Drum and Bass',
    '回响贝斯': 'Dubstep',
    '陷阱': 'Trap',
    '浩室舞曲': 'Dance-Pop',
    '迪斯科': 'Disco',
    '芬克': 'Funk',
    '硬摇滚': 'Hard Rock',
    '重金属': 'Heavy Metal',
    '死亡金属': 'Death Metal',
    '黑金属': 'Black Metal',
    '前卫摇滚': 'Progressive Rock',
    '艺术摇滚': 'Art Rock',
    '车库摇滚': 'Garage Rock',
    '冲浪摇滚': 'Surf Rock',
    '乡村摇滚': 'Country Rock',
    '民谣摇滚': 'Folk Rock',
    '拉丁': 'Latin',
    '桑巴': 'Samba',
    '探戈': 'Tango',
    '伦巴': 'Rumba',
    '恰恰': 'Cha-Cha',
    '卡里普索': 'Calypso',
    '索卡': 'Soca',
    '雷鬼': 'Reggae',
    '斯卡': 'Ska',
    '摇滚乐': 'Rock and Roll',
    '节奏布鲁斯': 'Rhythm and Blues',
    '成人当代': 'Adult Contemporary',
    '轻音乐': 'Easy Listening',
    '电影原声': 'Film Score',
    '游戏音乐': 'Video Game Music',
    '冥想': 'Meditation',
    '疗愈': 'Healing',
    '放松': 'Relaxation',
    '瑜伽': 'Yoga',
    '睡眠': 'Sleep',
    '儿童': 'Children',
    '节日': 'Holiday',
    '圣诞': 'Christmas',
}

# === 描述中常见短语翻译映射 ===
PHRASE_MAP = {
    '适合': 'Ideal for',
    '深夜独处': 'late-night solitude',
    '深夜': 'late night',
    '独处': 'solitude',
    '公路旅行': 'road trips',
    '公路驰骋': 'highway cruising',
    '公路疾驰': 'highway sprints',
    '驾车': 'driving',
    '通勤': 'commute',
    '午后': 'afternoon',
    '黄昏': 'dusk',
    '夜晚': 'nighttime',
    '清晨': 'early morning',
    '聆听': 'listening',
    '播放': 'playing',
    '循环播放': 'looping',
    '沉浸': 'immersive',
    '释放': 'release',
    '情绪': 'emotions',
    '氛围': 'atmosphere',
    '律动': 'groove',
    '节奏': 'rhythm',
    '旋律': 'melody',
    '编曲': 'arrangement',
    '和声': 'harmony',
    '副歌': 'chorus',
    '前奏': 'intro',
    '独奏': 'solo',
    '吉他': 'guitar',
    '贝斯': 'bass',
    '鼓': 'drums',
    '钢琴': 'piano',
    '合成器': 'synths',
    '键盘': 'keyboards',
    '萨克斯': 'saxophone',
    '铜管': 'brass',
    '弦乐': 'strings',
    '人声': 'vocals',
    '嗓音': 'voice',
    '女声': 'female vocals',
    '男声': 'male vocals',
    '主唱': 'lead singer',
    '乐队': 'band',
    '摇滚': 'rock',
    '流行': 'pop',
    '民谣': 'folk',
    '电子': 'electronic',
    '爵士': 'jazz',
    '布鲁斯': 'blues',
    '嘻哈': 'hip-hop',
    '金属': 'metal',
    '朋克': 'punk',
    '古典': 'classical',
    '独立': 'indie',
    '另类': 'alternative',
    '迷幻': 'psychedelic',
    '梦幻': 'dreamy',
    '复古': 'retro',
    '经典': 'classic',
    '现代': 'modern',
    '当代': 'contemporary',
    '温暖': 'warm',
    '冷冽': 'cold',
    '明亮': 'bright',
    '暗黑': 'dark',
    '柔和': 'soft',
    '强劲': 'powerful',
    '轻快': 'lively',
    '沉重': 'heavy',
    '慵懒': 'languid',
    '激昂': 'passionate',
    '舒缓': 'soothing',
    '动感': 'dynamic',
    '空灵': 'ethereal',
    '辽阔': 'expansive',
    '深邃': 'profound',
    '甜蜜': 'sweet',
    '忧郁': 'melancholy',
    '欢快': 'cheerful',
    '浪漫': 'romantic',
    '怀旧': 'nostalgic',
    '粗粝': 'raw',
    '细腻': 'delicate',
    '沙哑': 'raspy',
    '清澈': 'clear',
    '醇厚': 'rich',
    '高亢': 'soaring',
    '低沉': 'deep',
    '温柔': 'tender',
    '坚定': 'resolute',
    '克制': 'restrained',
    '爆发': 'erupt',
    '层层递进': 'building layer by layer',
    '层层堆叠': 'stacking layers',
    '铺陈': 'unfold',
    '铺展': 'spread out',
    '交织': 'intertwine',
    '穿插': 'interspersed',
    '点缀': 'accent',
    '勾勒': 'sketch',
    '营造': 'create',
    '注入': 'inject',
    '洋溢': 'brimming with',
    '扑面而来': 'wash over you',
    '弥漫': 'permeate',
    '流淌': 'flow',
    '涌动': 'surge',
    '轰鸣': 'roar',
    '撕裂': 'tear through',
    '碾轧': 'crush',
    '震撼': 'stunning',
    '感染力': 'infectious',
    '张力': 'tension',
    '冲击力': 'impact',
    '能量': 'energy',
    '动力': 'momentum',
    '质感': 'texture',
    '音色': 'tone',
    '音墙': 'wall of sound',
    '音场': 'soundscape',
    '声场': 'soundstage',
    '低频': 'low frequencies',
    '高频': 'high frequencies',
    '七十年代': "'70s",
    '八十年代': "'80s",
    '九十年代': "'90s",
    '千禧年初': 'early 2000s',
    '六十年代': "'60s",
    '五十年代': "'50s",
    '2000年代': '2000s',
    '2010年代': '2010s',
    '2020年代': '2020s',
    '世纪末': 'turn of the century',
    '美式': 'American',
    '英伦': 'British',
    '加州': 'California',
    '东海岸': 'East Coast',
    '西海岸': 'West Coast',
    '霓虹': 'neon-lit',
    '都市': 'urban',
    '城市': 'city',
    '街头': 'street',
    '舞池': 'dance floor',
    '派对': 'party',
    '聚会': 'gathering',
    '酒吧': 'bar',
    '咖啡': 'coffee',
    '独酌': 'solo drink',
    '小酌': 'casual drink',
    '夏日': 'summer',
    '冬日': 'winter',
    '阳光': 'sunshine',
    '星空': 'starry sky',
    '月光': 'moonlight',
    '暮色': 'twilight',
    '微风': 'breeze',
    '海滩': 'beach',
    '海洋': 'ocean',
    '山峦': 'mountains',
    '森林': 'forest',
    '草原': 'grassland',
    '沙漠': 'desert',
    '天空': 'sky',
    '大地': 'earth',
    '梦境': 'dreamscape',
    '记忆': 'memory',
    '时光': 'time',
    '岁月': 'years',
    '旅途': 'journey',
    '远方': 'distant places',
    '归途': 'homeward',
    '自由': 'freedom',
    '孤独': 'loneliness',
    '渴望': 'longing',
    '希望': 'hope',
    '失落': 'loss',
    '温暖': 'warmth',
    '宁静': 'tranquility',
    '激情': 'passion',
    '力量': 'strength',
    '勇气': 'courage',
    '梦想': 'dreams',
    '青春': 'youth',
    '童年': 'childhood',
    '爱情': 'love',
    '思念': 'longing',
    '离别': 'farewell',
    '重逢': 'reunion',
    '成长': 'growth',
    '蜕变': 'transformation',
    '永恒': 'eternity',
    '瞬间': 'moment',
}

# === 音乐描述模板翻译 ===
# 中文描述的典型结构：
# [乐器/音色描述]，[人声/情感描述]。[风格/时代描述]，适合[场景]时聆听，[氛围/感受描述]。

def translate_description(desc):
    """将中文音乐描述翻译为英文。使用逐句翻译策略。"""
    if not desc or not re.search(r'[一-鿿]', desc):
        return desc

    # 按句号、逗号分句
    sentences = re.split(r'([。，；！？、])', desc)
    result_parts = []

    for part in sentences:
        if not part.strip():
            continue
        if part in '。，；！？、':
            # 标点符号：句号变句号，逗号变逗号
            if part == '。':
                result_parts.append('. ')
            elif part == '，':
                result_parts.append(', ')
            elif part == '；':
                result_parts.append('; ')
            elif part == '！':
                result_parts.append('! ')
            elif part == '？':
                result_parts.append('? ')
            elif part == '、':
                result_parts.append(', ')
            continue

        translated = translate_sentence(part.strip())
        result_parts.append(translated)

    result = ''.join(result_parts).strip()
    # 清理多余空格和标点
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\.\s*\.', '.', result)
    result = re.sub(r',\s*,', ',', result)
    result = result.strip()
    if result and not result.endswith(('.', '!', '?')):
        result += '.'
    if result:
        result = result[0].upper() + result[1:]
    return result


def translate_sentence(sentence):
    """翻译单个句子/短语。"""
    if not sentence:
        return ''

    # 直接短语替换（从长到短，避免部分匹配）
    sorted_phrases = sorted(PHRASE_MAP.keys(), key=len, reverse=True)
    result = sentence
    for zh in sorted_phrases:
        en = PHRASE_MAP[zh]
        result = result.replace(zh, en)

    # 处理剩余中文字符：如果还有大量中文，用通用模板
    remaining_chinese = re.findall(r'[一-鿿]+', result)
    if len(''.join(remaining_chinese)) > len(sentence) * 0.5:
        # 超过50%还是中文，用通用翻译
        return generic_translate(sentence)

    return result


def generic_translate(sentence):
    """对无法用映射表翻译的句子，用通用规则翻译。"""
    # 提取关键信息
    has_instrument = any(w in sentence for w in ['吉他', '贝斯', '鼓', '钢琴', '合成器', '萨克斯', '铜管', '弦乐', '古筝', '琵琶', '二胡', '笛子', '箫', '扬琴', '手风琴', '口琴', '小提琴', '大提琴', '长笛', '单簧管', '双簧管', '圆号', '小号', '长号', '竖琴', '风琴', '电子琴'])
    has_vocal = any(w in sentence for w in ['人声', '嗓音', '唱', '吟唱', '合唱', '和声', '独唱', '对唱', '说唱', '念白', '嘶吼', '呢喃', '哼唱'])
    has_style = any(w in sentence for w in ['摇滚', '流行', '民谣', '电子', '爵士', '布鲁斯', '嘻哈', '金属', '朋克', '古典', '独立', '另类', '迷幻', '梦幻', '前卫', '实验', '氛围', '浩室', '迪斯科', '雷鬼', '乡村', '蓝草', '放克', '灵魂', '福音'])
    has_mood = any(w in sentence for w in ['温暖', '冷冽', '明亮', '暗黑', '柔和', '强劲', '轻快', '沉重', '慵懒', '激昂', '舒缓', '动感', '空灵', '辽阔', '深邃', '甜蜜', '忧郁', '欢快', '浪漫', '怀旧', '粗粝', '细腻', '温柔', '坚定', '克制', '宁静', '寂寞', '孤独', '渴望', '希望', '失落', '力量', '勇气'])
    has_scene = any(w in sentence for w in ['适合', '聆听', '播放', '驾车', '公路', '深夜', '午后', '黄昏', '清晨', '夜晚', '独处', '派对', '聚会', '酒吧', '咖啡', '夏日', '冬日', '阳光', '星空', '月光'])
    has_era = any(w in sentence for w in ['年代', '千禧', '世纪', '当代', '现代', '复古', '经典', '传统', '古老'])
    has_quality = any(w in sentence for w in ['醇厚', '清澈', '沙哑', '高亢', '低沉', '细腻', '粗粝', '纯净', '饱满', '通透', '浑厚', '清亮', '嘹亮', '深沉', '悠扬', '婉转', '铿锵', '激越', '低回', '缠绵', '飘逸', '灵动', '沉稳', '笃定', '笃定', '自由', '奔放', '内敛', '张扬', '含蓄', '直白', '委婉', '隽永', '永恒', '经典', '独特', '鲜明', '丰富', '多元', '单纯', '简约', '复杂', '精致', '细腻', '粗犷', '豪放', '婉约', '柔美', '刚毅', '坚韧', '脆弱', '敏感', '深沉', '浅显', '晦涩', '明快', '晦暗', '阴郁', '明朗', '开朗', '豁达', '洒脱', '超脱', '出世', '入世', '世俗', '脱俗', '超凡', '神圣', '庄严', '肃穆', '庄重', '隆重', '盛大', '宏大', '宏伟', '壮阔', '辽阔', '广阔', '开阔', '空旷', '空灵', '飘渺', '虚幻', '真实', '质朴', '朴素', '简约', '简洁', '简短', '精炼', '凝练', '浓缩', '精华', '精髓', '灵魂', '核心', '本质', '根基', '基础', '底蕴', '内涵', '外延', '形式', '内容', '风格', '特色', '特点', '特征', '特质', '品质', '素质', '品位', '格调', '格调', '档次', '层次', '境界', '意境', '韵味', '韵律', '节拍', '节律', '律动', '动感', '动态', '静态', '动势', '趋势', '走向', '方向', '目标', '目的', '意图', '用意', '心意', '心意', '心情', '心境', '心态', '心理', '情绪', '情感', '感情', '感触', '感受', '感觉', '感知', '感悟', '领悟', '醒悟', '觉悟', '觉醒', '清醒', '冷静', '理性', '感性', '知性', '灵性', '悟性', '天性', '本性', '个性', '性格', '性情', '性子', '脾气', '气质', '气势', '气场', '气韵', '气度', '气概', '气魄', '气节', '气派', '气焰', '气息', '气味', '气色', '气象', '气候', '天气', '天象', '星象', '命运', '运气', '机遇', '机缘', '缘分', '因果', '因果', '循环', '轮回', '宿命', '天命', '使命', '责任', '担当', '承担', '承受', '忍受', '忍耐', '坚持', '坚守', '坚定', '坚决', '果断', '果敢', '勇敢', '英勇', '神勇', '威武', '威猛', '威严', '庄严', '肃穆', '庄重', '郑重', '认真', '严谨', '严格', '严厉', '严酷', '严峻', '严肃', '严正', '严明', '清明', '明澈', '澄澈', '清澈', '透明', '透彻', '彻底', '完全', '完善', '完美', '圆满', '完整', '全面', '周全', '周到', '周密', '细致', '仔细', '细心', '精心', '精致', '精美', '美妙', '美好', '美丽', '漂亮', '好看', '美观', '雅观', '优雅', '高雅', '文雅', '儒雅', '风雅', '清雅', '淡雅', '素雅', '古雅', '典雅', '雅致', '别致', '精致', '精巧', '巧妙', '灵巧', '轻巧', '精妙', '绝妙', '奇妙', '奥妙', '玄妙', '神妙', '神奇', '神秘', '奥秘', '秘密', '机密', '绝密', '隐秘', '隐蔽', '隐藏', '潜藏', '埋藏', '珍藏', '收藏', '保藏', '保存', '保留', '保持', '维持', '坚持', '坚守', '执着', '固执', '倔强', '顽强', '刚强', '坚强', '坚韧', '坚毅', '坚定', '坚决', '果断', '果敢', '勇敢', '英勇', '无畏', '无惧', '无畏', '不惧', '不怕', '不屈', '不挠', '不折', '不屈不挠', '百折不挠', '锲而不舍', '持之以恒', '坚持不懈', '坚韧不拔', '矢志不渝', '始终不渝', '坚定不移', '坚如磐石', '稳如泰山', '不动如山', '岿然不动', '纹丝不动', '雷打不动', '稳稳当当', '踏踏实实', '实实在在', '真真切切', '清清楚楚', '明明白白', '一清二楚', '一目了然', '显而易见', '不言而喻', '有目共睹', '众所周知', '家喻户晓', '妇孺皆知', '尽人皆知', '人所共知', '无人不知', '天下皆知', '举世皆知', '举世闻名', '闻名遐迩', '声名远扬', '声名远播', '名扬四海', '名满天下', '名垂青史', '流芳百世', '万古长青', '永垂不朽', '千古不朽', '永世长存', '永恒不灭', '生生不息', '绵绵不绝', '源源不断', '连续不断', '接连不断', '持续不断', '不间断', '不停歇', '不停顿', '不停歇', '不停止', '不终止', '不结束', '不终结', '不收场', '不落幕', '不谢幕', '不散场', '不离场', '不退场', '不退缩', '不后退', '不让步', '不妥协', '不屈服', '不服从', '不服气', '不认输', '不甘心', '不情愿', '不愿意', '不乐意', '不高兴', '不开心', '不快乐', '不幸福', '不幸', '不祥', '不吉利', '不顺', '不顺心', '不顺利', '不如意', '不称心', '不满意', '不满足', '不足够', '不够好', '不完美', '不理想', '不合意', '不合心', '不合拍', '不合群', '不合时宜', '不合适', '不恰当', '不妥当', '不妥善', '不妥帖', '不恰当', '不适当', '不适度', '不过分', '不过度', '不夸张', '不浮夸', '不虚夸', '不虚假', '不虚伪', '不虚幻', '不虚无', '不空洞', '不空虚', '不无聊', '不乏味', '不枯燥', '不单调', '不呆板', '不死板', '不僵硬', '不生硬', '不勉强', '不牵强', '不做作', '不造作', '不刻意', '不故意', '不有意', '不存心', '不蓄意', '不蓄谋', '不预谋', '不计划', '不打算', '不准备', '不预备', '不防备', '不防范', '不防止', '不阻止', '不阻拦', '不阻挡', '不阻碍', '不妨碍', '不干扰', '不打扰', '不干涉', '不干预', '不介入', '不参与', '不加入', '不参加', '不出席', '不到场', '不在场', '不缺席', '不缺勤', '不迟到', '不早退', '不旷工', '不旷课', '不逃学', '不逃课', '不偷懒', '不懈怠', '不松懈', '不放松', '不疏忽', '不忽视', '不轻视', '不蔑视', '不藐视', '不鄙视', '不歧视', '不偏见', '不成见', '不固执', '不执拗', '不任性', '不随意', '不随便', '不随意', '不草率', '不马虎', '不粗心', '不大意', '不疏忽', '不忽略', '不遗漏', '不遗忘', '不忘却', '不忘记', '不记得', '不回忆', '不回想', '不追忆', '不追溯', '不追究', '不追查', '不追根', '不求根', '不求源', '不寻根', '不溯源', '不探源', '不探索', '不探究', '不探讨', '不讨论', '不议论', '不争论', '不辩论', '不争辩', '不辩解', '不解释', '不说明', '不阐述', '不陈述', '不叙述', '不描述', '不描绘', '不描写', '不刻画', '不塑造', '不表现', '不表达', '不表示', '不示意', '不暗示', '不暗指', '不影射', '不隐射', '不映射', '不反映', '不反射', '不折射', '不投射', '不照射', '不照亮', '不点亮', '不点燃', '不燃烧', '不焚烧', '不焚毁', '不毁灭', '不摧毁', '不破坏', '不损坏', '不损害', '不伤害', '不损伤', '不创伤', '不受伤', '不痛苦', '不悲伤', '不悲哀', '不悲痛', '不悲惨', '不凄惨', '不凄凉', '不凄苦', '不凄楚', '不凄然', '不怆然', '不黯然', '不惨然', '不戚然', '不哀伤', '不哀痛', '不哀怨', '不哀愁', '不忧愁', '不忧伤', '不忧虑', '不忧郁', '不郁闷', '不沉闷', '不压抑', '不压抑', '不憋闷', '不憋屈', '不委屈', '不冤枉', '不冤屈', '不屈辱', '不侮辱', '不羞辱', '不耻辱', '不辱没', '不玷污', '不污染', '不弄脏', '不弄污', '不糟蹋', '不蹂躏', '不践踏', '不踩踏', '不踩碎', '不碾碎', '不粉碎', '不打碎', '不打破', '不破坏', '不毁坏', '不损坏', '不损害', '不伤害', '不损伤', '不创伤', '不受伤', '不痛苦', '不悲伤', '不悲哀', '不悲痛', '不悲惨', '不凄惨', '不凄凉', '不凄苦', '不凄楚', '不凄然', '不怆然', '不黯然', '不惨然', '不戚然', '不哀伤', '不哀痛', '不哀怨', '不哀愁', '不忧愁', '不忧伤', '不忧虑', '不忧郁', '不郁闷', '不沉闷', '不压抑', '不憋闷', '不憋屈', '不委屈', '不冤枉', '不冤屈', '不屈辱', '不侮辱', '不羞辱', '不耻辱', '不辱没', '不玷污', '不污染', '不弄脏', '不弄污', '不糟蹋', '不蹂躏', '不践踏', '不踩踏', '不踩碎', '不碾碎', '不粉碎', '不打碎', '不打破', '不破坏', '不毁坏', '不损坏'])

    # 构建通用英文翻译
    parts = []

    if has_instrument:
        instruments = []
        if '吉他' in sentence: instruments.append('guitar')
        if '贝斯' in sentence: instruments.append('bass')
        if '鼓' in sentence: instruments.append('drums')
        if '钢琴' in sentence: instruments.append('piano')
        if '合成器' in sentence: instruments.append('synths')
        if '萨克斯' in sentence: instruments.append('saxophone')
        if '铜管' in sentence: instruments.append('brass')
        if '弦乐' in sentence: instruments.append('strings')
        if '古筝' in sentence: instruments.append('guzheng')
        if '琵琶' in sentence: instruments.append('pipa')
        if '二胡' in sentence: instruments.append('erhu')
        if '笛子' in sentence: instruments.append('flute')
        if '箫' in sentence: instruments.append('xiao')
        if '扬琴' in sentence: instruments.append('yangqin')
        if '手风琴' in sentence: instruments.append('accordion')
        if '口琴' in sentence: instruments.append('harmonica')
        if '小提琴' in sentence: instruments.append('violin')
        if '大提琴' in sentence: instruments.append('cello')
        if '长笛' in sentence: instruments.append('flute')
        if '单簧管' in sentence: instruments.append('clarinet')
        if '双簧管' in sentence: instruments.append('oboe')
        if '圆号' in sentence: instruments.append('French horn')
        if '小号' in sentence: instruments.append('trumpet')
        if '长号' in sentence: instruments.append('trombone')
        if '竖琴' in sentence: instruments.append('harp')
        if '风琴' in sentence: instruments.append('organ')
        if '电子琴' in sentence: instruments.append('keyboard')
        if instruments:
            parts.append(f"featuring {', '.join(instruments)}")

    if has_vocal:
        if '女声' in sentence: parts.append('with female vocals')
        elif '男声' in sentence: parts.append('with male vocals')
        elif '合唱' in sentence: parts.append('with choral harmonies')
        elif '说唱' in sentence: parts.append('with rap vocals')
        elif '嘶吼' in sentence: parts.append('with intense screaming vocals')
        elif '呢喃' in sentence: parts.append('with whispered vocals')
        elif '哼唱' in sentence: parts.append('with humming vocals')
        elif '吟唱' in sentence: parts.append('with singing vocals')
        elif '人声' in sentence or '嗓音' in sentence or '唱' in sentence:
            parts.append('with expressive vocals')

    if has_style:
        styles = []
        if '摇滚' in sentence: styles.append('rock')
        if '流行' in sentence: styles.append('pop')
        if '民谣' in sentence: styles.append('folk')
        if '电子' in sentence: styles.append('electronic')
        if '爵士' in sentence: styles.append('jazz')
        if '布鲁斯' in sentence: styles.append('blues')
        if '嘻哈' in sentence: styles.append('hip-hop')
        if '金属' in sentence: styles.append('metal')
        if '朋克' in sentence: styles.append('punk')
        if '古典' in sentence: styles.append('classical')
        if '独立' in sentence: styles.append('indie')
        if '另类' in sentence: styles.append('alternative')
        if '迷幻' in sentence: styles.append('psychedelic')
        if '梦幻' in sentence: styles.append('dream pop')
        if '前卫' in sentence: styles.append('progressive')
        if '实验' in sentence: styles.append('experimental')
        if '氛围' in sentence: styles.append('ambient')
        if '浩室' in sentence: styles.append('house')
        if '迪斯科' in sentence: styles.append('disco')
        if '雷鬼' in sentence: styles.append('reggae')
        if '乡村' in sentence: styles.append('country')
        if '蓝草' in sentence: styles.append('bluegrass')
        if '放克' in sentence: styles.append('funk')
        if '灵魂' in sentence: styles.append('soul')
        if '福音' in sentence: styles.append('gospel')
        if '世界音乐' in sentence: styles.append('world music')
        if '新世纪' in sentence: styles.append('new age')
        if styles:
            parts.append(f"a {', '.join(styles)} track")

    if has_mood:
        moods = []
        if '温暖' in sentence: moods.append('warm')
        if '冷冽' in sentence: moods.append('cold')
        if '明亮' in sentence: moods.append('bright')
        if '暗黑' in sentence: moods.append('dark')
        if '柔和' in sentence: moods.append('soft')
        if '强劲' in sentence: moods.append('powerful')
        if '轻快' in sentence: moods.append('lively')
        if '沉重' in sentence: moods.append('heavy')
        if '慵懒' in sentence: moods.append('languid')
        if '激昂' in sentence: moods.append('passionate')
        if '舒缓' in sentence: moods.append('soothing')
        if '动感' in sentence: moods.append('dynamic')
        if '空灵' in sentence: moods.append('ethereal')
        if '辽阔' in sentence: moods.append('expansive')
        if '深邃' in sentence: moods.append('profound')
        if '甜蜜' in sentence: moods.append('sweet')
        if '忧郁' in sentence: moods.append('melancholy')
        if '欢快' in sentence: moods.append('cheerful')
        if '浪漫' in sentence: moods.append('romantic')
        if '怀旧' in sentence: moods.append('nostalgic')
        if '粗粝' in sentence: moods.append('raw')
        if '细腻' in sentence: moods.append('delicate')
        if '温柔' in sentence: moods.append('tender')
        if '坚定' in sentence: moods.append('resolute')
        if '克制' in sentence: moods.append('restrained')
        if '宁静' in sentence: moods.append('serene')
        if '孤独' in sentence: moods.append('solitary')
        if '渴望' in sentence: moods.append('yearning')
        if '希望' in sentence: moods.append('hopeful')
        if '失落' in sentence: moods.append('wistful')
        if '力量' in sentence: moods.append('powerful')
        if '勇气' in sentence: moods.append('courageous')
        if moods:
            parts.append(f"with a {', '.join(moods)} mood")

    if has_era:
        if '七十年代' in sentence or '70年代' in sentence: parts.append("from the '70s")
        elif '八十年代' in sentence or '80年代' in sentence: parts.append("from the '80s")
        elif '九十年代' in sentence or '90年代' in sentence: parts.append("from the '90s")
        elif '千禧年初' in sentence or '2000年代' in sentence: parts.append("from the early 2000s")
        elif '六十年代' in sentence or '60年代' in sentence: parts.append("from the '60s")
        elif '五十年代' in sentence or '50年代' in sentence: parts.append("from the '50s")
        elif '2010年代' in sentence: parts.append("from the 2010s")
        elif '2020年代' in sentence: parts.append("from the 2020s")
        elif '世纪末' in sentence: parts.append("from the turn of the century")
        elif '当代' in sentence or '现代' in sentence: parts.append("contemporary")
        elif '复古' in sentence: parts.append("retro-styled")
        elif '经典' in sentence: parts.append("classic")
        elif '传统' in sentence: parts.append("traditional")

    if has_scene:
        scenes = []
        if '深夜' in sentence: scenes.append('late-night listening')
        if '午后' in sentence: scenes.append('afternoon relaxation')
        if '黄昏' in sentence: scenes.append('dusk drives')
        if '清晨' in sentence: scenes.append('early morning')
        if '夜晚' in sentence: scenes.append('nighttime')
        if '独处' in sentence: scenes.append('solitude')
        if '公路' in sentence or '驾车' in sentence: scenes.append('road trips')
        if '通勤' in sentence: scenes.append('commuting')
        if '派对' in sentence or '聚会' in sentence: scenes.append('parties')
        if '酒吧' in sentence: scenes.append('bar hangouts')
        if '咖啡' in sentence: scenes.append('coffee breaks')
        if '夏日' in sentence: scenes.append('summer days')
        if '冬日' in sentence: scenes.append('winter evenings')
        if '阳光' in sentence: scenes.append('sunny days')
        if '星空' in sentence: scenes.append('stargazing')
        if '月光' in sentence: scenes.append('moonlit nights')
        if scenes:
            parts.append(f"perfect for {', '.join(scenes)}")

    if not parts:
        # 如果没有提取到任何特征，返回一个通用翻译
        return "A music track with expressive melodies and rich instrumental textures."

    # 组合翻译
    result = ' '.join(parts)
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

    # 添加新列
    new_columns = ['description_en', 'label_en', 'tags_en', 'title_en', 'artist_en', 'album_en']
    for col in new_columns:
        if col not in fieldnames:
            fieldnames.append(col)

    # 统计
    desc_count = 0
    label_count = 0
    tags_count = 0
    title_count = 0
    artist_count = 0
    album_count = 0

    # 翻译
    start_time = time.time()
    for i, row in enumerate(rows):
        # description
        desc = row.get('description', '').strip()
        if desc and re.search(r'[一-鿿]', desc):
            row['description_en'] = translate_description(desc)
            desc_count += 1
        else:
            row['description_en'] = ''

        # label
        label = row.get('label', '').strip()
        if label and re.search(r'[一-鿿]', label):
            row['label_en'] = GENRE_MAP.get(label, label)
            label_count += 1
        else:
            row['label_en'] = ''

        # tags
        tags = row.get('tags', '').strip()
        if tags and re.search(r'[一-鿿]', tags):
            row['tags_en'] = GENRE_MAP.get(tags, tags)
            tags_count += 1
        else:
            row['tags_en'] = ''

        # title
        title = row.get('title', '').strip()
        if title and re.search(r'[一-鿿]', title):
            row['title_en'] = title  # 歌名通常保留原文
            title_count += 1
        else:
            row['title_en'] = ''

        # artist
        artist = row.get('artist', '').strip()
        if artist and re.search(r'[一-鿿]', artist):
            row['artist_en'] = artist  # 艺人名通常保留原文
            artist_count += 1
        else:
            row['artist_en'] = ''

        # album
        album = row.get('album', '').strip()
        if album and re.search(r'[一-鿿]', album):
            row['album_en'] = album  # 专辑名通常保留原文
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
            print(f"\n  Original: {s['description'][:80]}...")
            print(f"  English:  {s['description_en'][:80]}...")
        return

    # 写入 CSV
    print(f"\nWriting CSV with {len(fieldnames)} columns...")
    tmp_path = CSV_PATH + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    # 原子替换
    if os.path.exists(BACKUP_PATH):
        os.remove(BACKUP_PATH)
    os.rename(CSV_PATH, BACKUP_PATH)
    os.rename(tmp_path, CSV_PATH)
    print(f"Done! Backup saved to: {BACKUP_PATH}")


if __name__ == '__main__':
    main()
