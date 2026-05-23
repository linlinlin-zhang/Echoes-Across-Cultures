from __future__ import annotations

PAL_TASK_QUESTION_ZH = (
    "请判断这两首曲目是否适合放在同一个歌单或相近的听歌场景里。"
    "请根据整体氛围、情绪基调和节奏能量判断它们是否相似；"
    "不要根据国家、语言、乐器或标签直接下结论。"
    "若相似填 yes，不相似填 no，不确定可留空并在 notes 中说明。"
)

PAL_TASK_QUESTION_EN = (
    "Judge whether these two tracks would fit naturally in the same playlist or a similar listening context. "
    "Base your judgment on overall atmosphere, emotional tone, and energy rather than culture, language, "
    "instruments, or metadata labels. Fill yes for similar, no for dissimilar, and leave blank with a note if unsure."
)

PAL_WEB_PROMPT_MAIN_ZH = "请听完这两首曲目，判断它们是否适合放在同一个歌单或相近的听歌场景里。"

PAL_WEB_PROMPT_HINT_ZH = (
    "重点看整体氛围、情绪基调和节奏能量是否接近，比如是否都适合放松、专注、庆典或沉浸式聆听。"
    "不要比较哪首更好，也不要根据国家、语言、乐器或标签直接下结论。"
)

PAL_README_REMINDER_EN = (
    "Judge whether the pair would fit naturally in the same playlist or a similar listening context."
)
