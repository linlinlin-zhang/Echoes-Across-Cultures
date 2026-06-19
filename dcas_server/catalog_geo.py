from __future__ import annotations

import csv
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any


COUNTRY_NAMES = {
    "AR": "Argentina",
    "BR": "Brazil",
    "CN": "China",
    "CU": "Cuba",
    "ES": "Spain",
    "GH": "Ghana",
    "HK": "Hong Kong",
    "ID": "Indonesia",
    "IE": "Ireland",
    "IN": "India",
    "IR": "Iran",
    "JM": "Jamaica",
    "JP": "Japan",
    "KR": "South Korea",
    "ML": "Mali",
    "MO": "Macau",
    "MX": "Mexico",
    "NG": "Nigeria",
    "PT": "Portugal",
    "SN": "Senegal",
    "SG": "Singapore",
    "TH": "Thailand",
    "TR": "Turkey",
    "TT": "Trinidad and Tobago",
    "TW": "Taiwan",
    "US": "United States",
    "ZA": "South Africa",
    "AU": "Australia",
    "AT": "Austria",
    "BB": "Barbados",
    "CA": "Canada",
    "DE": "Germany",
    "DK": "Denmark",
    "FR": "France",
    "GB": "United Kingdom",
    "GR": "Greece",
    "IS": "Iceland",
    "IT": "Italy",
    "MY": "Malaysia",
    "NL": "Netherlands",
    "NO": "Norway",
    "NZ": "New Zealand",
    "PL": "Poland",
    "SE": "Sweden",
    "TZ": "Tanzania",
    "VG": "British Virgin Islands",
}

COUNTRY_SEARCH_TERMS = {
    "AU": ("australia", "australian", "澳大利亚", "澳洲"),
    "AT": ("austria", "austrian", "vienna", "奥地利", "奧地利"),
    "BB": ("barbados", "barbadian", "巴巴多斯"),
    "CA": ("canada", "canadian", "toronto", "vancouver", "加拿大"),
    "DE": ("germany", "german", "berlin", "deutschland", "德国", "德國"),
    "DK": ("denmark", "danish", "copenhagen", "丹麦", "丹麥"),
    "FR": ("france", "french", "paris", "法国", "法國"),
    "GB": ("united kingdom", "uk", "britain", "british", "england", "london", "英国", "英國", "英伦", "英倫"),
    "GR": ("greece", "greek", "athens", "希腊", "希臘"),
    "IS": ("iceland", "icelandic", "reykjavik", "冰岛", "冰島"),
    "IT": ("italy", "italian", "rome", "意大利", "義大利"),
    "MY": ("malaysia", "malaysian", "kuala lumpur", "马来西亚", "馬來西亞"),
    "NL": ("netherlands", "dutch", "amsterdam", "荷兰", "荷蘭"),
    "NO": ("norway", "norwegian", "oslo", "挪威"),
    "NZ": ("new zealand", "aotearoa", "kiwi", "新西兰", "紐西蘭"),
    "PL": ("poland", "polish", "warsaw", "波兰", "波蘭"),
    "SE": ("sweden", "swedish", "stockholm", "瑞典"),
    "SG": ("singapore", "singaporean", "新加坡", "南洋"),
    "TH": ("thailand", "thai", "bangkok", "泰国", "泰國", "泰语", "泰語"),
    "TZ": ("tanzania", "tanzanian", "dar es salaam", "bongo flava", "坦桑尼亚", "坦桑尼亞"),
    "US": ("united states", "usa", "america", "american", "new york", "los angeles", "美国", "美國", "美式"),
    "VG": ("british virgin islands", "virgin islands", "tortola", "英属维尔京群岛", "英屬維爾京群島"),
    "AR": ("argentina", "buenos aires", "tango", "阿根廷", "探戈"),
    "BR": (
        "brazil",
        "brasil",
        "bahia",
        "rio",
        "samba",
        "bossa nova",
        "bossa",
        "mpb",
        "forro",
        "forró",
        "巴西",
        "桑巴",
        "波萨",
        "波薩",
        "巴萨诺瓦",
        "巴薩諾瓦",
    ),
    "CN": (
        "china",
        "chinese",
        "mainland china",
        "beijing",
        "shanghai",
        "mandarin",
        "mandopop",
        "c-pop",
        "chinese pop",
        "chinese rock",
        "中国",
        "中國",
        "中文",
        "华语",
        "華語",
        "国语",
        "國語",
        "普通话",
        "普通話",
        "中国摇滚",
        "中國搖滾",
        "中国流行",
        "中國流行",
        "内地",
        "大陆",
    ),
    "CU": ("cuba", "cuban", "havana", "son cubano", "rumba", "古巴"),
    "ES": ("spain", "spanish", "flamenco", "sevilla", "madrid", "西班牙", "弗拉门戈"),
    "GH": ("ghana", "accra", "highlife", "azonto", "west africa", "加纳", "西非"),
    "HK": ("hong kong", "hongkong", "hong kong china", "cantopop", "cantonese pop", "cantonese", "香港", "中国香港", "粤语", "粵語"),
    "ID": ("indonesia", "jakarta", "bali", "gamelan", "dangdut", "印尼", "印度尼西亚", "甘美兰"),
    "IE": ("ireland", "irish", "celtic", "dublin", "爱尔兰", "凯尔特"),
    "IN": ("india", "indian", "bollywood", "bhangra", "punjabi", "hindustani", "carnatic", "印度", "宝莱坞", "旁遮普"),
    "IR": ("iran", "persian", "tehran", "伊朗", "波斯"),
    "JM": ("jamaica", "kingston", "reggae", "dancehall", "ska", "dub", "牙买加", "雷鬼"),
    "JP": (
        "japan",
        "japanese",
        "tokyo",
        "j-pop",
        "jpop",
        "city pop",
        "enka",
        "anime",
        "animation",
        "日本",
        "日本流行",
        "日本流行乐",
        "日本流行樂",
        "动画",
        "動畫",
    ),
    "KR": (
        "korea",
        "korean",
        "south korea",
        "seoul",
        "k-pop",
        "kpop",
        "韩国",
        "韓國",
        "韩国流行",
        "韩国流行乐",
        "韓國流行",
        "韓國流行樂",
        "韩语",
        "韓語",
    ),
    "ML": ("mali", "bamako", "manding", "mandinka", "kora", "griot", "west africa", "马里", "西非"),
    "MO": ("macau", "macao", "macau china", "macao china", "澳门", "澳門", "中国澳门"),
    "MX": ("mexico", "mexican", "mariachi", "ranchera", "corrido", "墨西哥"),
    "NG": ("nigeria", "nigerian", "lagos", "abuja", "afrobeats", "afrobeat", "afro-pop", "afropop", "naija", "yoruba", "west africa", "尼日利亚", "西非", "非洲节奏"),
    "PT": ("portugal", "portuguese", "lisbon", "fado", "葡萄牙", "法朵"),
    "SN": ("senegal", "dakar", "mbalax", "west africa", "塞内加尔", "西非"),
    "TR": ("turkey", "turkish", "istanbul", "ankara", "makam", "土耳其"),
    "TT": ("trinidad", "tobago", "soca", "calypso", "特立尼达", "卡利普索"),
    "TW": (
        "taiwan",
        "taipei",
        "chinese taipei",
        "taiwan pop",
        "taiwanese pop",
        "mandopop",
        "台北",
        "台湾",
        "臺灣",
        "台式",
        "臺式",
        "台语",
        "臺語",
        "中国台北",
        "中华台北",
        "中国台湾",
    ),
    "ZA": ("south africa", "south african", "johannesburg", "amapiano", "kwaito", "南非"),
}

KNOWN_ARTIST_ORIGINS = {
    "ayra starr": "NG",
    "朴树": "CN",
    "许巍": "CN",
    "郑钧": "CN",
    "胡兵": "CN",
    "伍佰": "TW",
    "张雨生": "TW",
}
CHINA_REVIEW_ISOS = {"CN", "HK", "MO", "TW"}
DEFAULT_ARTIST_OVERRIDES_PATH = Path("configs/catalog_origin_artist_overrides.csv")

def _clean(value: Any) -> str:
    return str(value or "").strip()


def norm_text(value: Any) -> str:
    raw = _clean(value).casefold()
    chars: list[str] = []
    last_base_is_latin = False
    for char in unicodedata.normalize("NFKD", raw):
        if unicodedata.combining(char):
            if last_base_is_latin:
                continue
            chars.append(char)
            continue
        chars.append(char)
        last_base_is_latin = "LATIN" in unicodedata.name(char, "")
    deaccented = "".join(chars)
    return " ".join(re.sub(r"[^\w\s&'+.-]+", " ", deaccented, flags=re.UNICODE).split())


QUERY_TERM_TO_ISOS: dict[str, set[str]] = {}
for _iso, _terms in COUNTRY_SEARCH_TERMS.items():
    for _term in (COUNTRY_NAMES.get(_iso, ""), _iso, *_terms):
        _key = norm_text(_term)
        if _key:
            QUERY_TERM_TO_ISOS.setdefault(_key, set()).add(_iso)


def _contains_phrase(haystack: str, phrase: str) -> bool:
    normalized_phrase = norm_text(phrase)
    if not normalized_phrase:
        return False
    if re.search(r"[\u3400-\u9fff]", normalized_phrase):
        return normalized_phrase in haystack
    padded = f" {haystack} "
    if f" {normalized_phrase} " in padded:
        return True
    separator_padded = f" {re.sub(r'[-_/]+', ' ', haystack)} "
    return f" {normalized_phrase} " in separator_padded


@lru_cache(maxsize=1)
def _known_artist_origins() -> dict[str, str]:
    origins = {norm_text(artist): iso.upper() for artist, iso in KNOWN_ARTIST_ORIGINS.items()}
    path = Path(os.environ.get("ECHO_CATALOG_ARTIST_ORIGINS_PATH") or DEFAULT_ARTIST_OVERRIDES_PATH)
    if not path.exists():
        return origins
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                artist = norm_text(row.get("artist") or row.get("artist_name"))
                iso = _clean(row.get("country_iso") or row.get("iso")).upper()
                if artist and re.fullmatch(r"[A-Z]{2}", iso):
                    origins[artist] = iso
    except Exception:
        return origins
    return origins


def _row_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    return norm_text(" ".join(_clean(row.get(key)) for key in keys))


def is_itunes_storefront_country(row: dict[str, Any]) -> bool:
    source = f"{row.get('source_dataset', '')} {row.get('platform', '')} {row.get('platform_track_url', '')}".casefold()
    return "itunes" in source or "music.apple.com" in source


def _country_iso_from_value(value: Any) -> str:
    text = _clean(value)
    if not text:
        return ""
    upper = text.upper()
    if re.fullmatch(r"[A-Z]{2}", upper):
        return upper
    normalized = norm_text(text)
    for iso, name in COUNTRY_NAMES.items():
        if normalized == norm_text(name):
            return iso
    for iso, terms in COUNTRY_SEARCH_TERMS.items():
        if any(normalized == norm_text(term) for term in terms):
            return iso
    return ""


def _hint_iso(row: dict[str, Any]) -> tuple[str, str]:
    artist_text = norm_text(_clean(row.get("artist")))
    for artist, iso in _known_artist_origins().items():
        if _contains_phrase(artist_text, artist):
            return iso, "artist_origin_hint"

    descriptive_text = _row_text(
        row,
        (
            "title",
            "name",
            "album",
            "album_name",
            "artist",
            "label",
            "label_en",
            "tags",
            "tags_en",
            "description",
            "description_en",
            "album_description",
        ),
    )
    for iso, terms in COUNTRY_SEARCH_TERMS.items():
        if any(_contains_phrase(descriptive_text, term) for term in terms):
            return iso, "genre_or_metadata_hint"
    return "", ""


def infer_catalog_origin(row: dict[str, Any]) -> dict[str, Any]:
    raw_country = _clean(row.get("country"))
    storefront = is_itunes_storefront_country(row)
    iso, source = _hint_iso(row)
    raw_iso = _country_iso_from_value(raw_country)

    if iso:
        return {
            "country": COUNTRY_NAMES.get(iso, iso),
            "country_iso": iso,
            "country_source": source,
            "country_original": raw_country,
            "storefront_country": raw_country if storefront else "",
            "catalog_country_is_storefront": bool(storefront),
        }

    if storefront and raw_country:
        if raw_iso in CHINA_REVIEW_ISOS:
            return {
                "country": "",
                "country_iso": "",
                "country_source": "itunes_storefront_china_review",
                "country_original": raw_country,
                "storefront_country": raw_country,
                "catalog_country_is_storefront": True,
            }
        return {
            "country": COUNTRY_NAMES.get(raw_iso, raw_country),
            "country_iso": raw_iso,
            "country_source": "itunes_storefront",
            "country_original": raw_country,
            "storefront_country": raw_country,
            "catalog_country_is_storefront": True,
        }

    return {
        "country": raw_country,
        "country_iso": raw_iso,
        "country_source": "metadata_country" if raw_country else "",
        "country_original": raw_country,
        "storefront_country": "",
        "catalog_country_is_storefront": False,
    }


def catalog_origin_search_text(row: dict[str, Any]) -> str:
    origin = infer_catalog_origin(row)
    iso = str(origin.get("country_iso") or "").upper()
    parts = [
        origin.get("country", ""),
        iso,
        origin.get("country_source", ""),
        origin.get("country_original", "") if not origin.get("catalog_country_is_storefront") else "",
        *COUNTRY_SEARCH_TERMS.get(iso, ()),
    ]
    return norm_text(" ".join(_clean(part) for part in parts))


def catalog_origin_search_text_for_query(row: dict[str, Any], query_terms: list[str]) -> str:
    requested_isos: set[str] = set()
    for term in query_terms:
        requested_isos.update(QUERY_TERM_TO_ISOS.get(norm_text(term), set()))
    if not requested_isos:
        return ""

    artist_text = norm_text(row.get("artist"))
    descriptive_text = _row_text(
        row,
        (
            "title",
            "name",
            "album",
            "album_name",
            "artist",
            "label",
            "label_en",
            "tags",
            "tags_en",
            "description",
            "description_en",
            "album_description",
        ),
    )
    matched: list[str] = []
    for artist, iso in _known_artist_origins().items():
        if iso in requested_isos and _contains_phrase(artist_text, artist):
            matched.extend([iso, COUNTRY_NAMES.get(iso, ""), *COUNTRY_SEARCH_TERMS.get(iso, ())])
    for iso in sorted(requested_isos):
        terms = COUNTRY_SEARCH_TERMS.get(iso, ())
        if any(_contains_phrase(descriptive_text, term) for term in terms):
            matched.extend([iso, COUNTRY_NAMES.get(iso, ""), *terms])
    return norm_text(" ".join(_clean(part) for part in matched))
