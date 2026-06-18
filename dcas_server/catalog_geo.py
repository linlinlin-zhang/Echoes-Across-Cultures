from __future__ import annotations

import re
import unicodedata
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
    "TR": "Turkey",
    "TT": "Trinidad and Tobago",
    "TW": "Taiwan",
    "ZA": "South Africa",
}

COUNTRY_SEARCH_TERMS = {
    "AR": ("argentina", "buenos aires", "tango", "阿根廷", "探戈"),
    "BR": ("brazil", "brasil", "bahia", "rio", "samba", "bossa nova", "mpb", "forro", "forró", "巴西", "桑巴", "波萨"),
    "CN": ("china", "chinese", "beijing", "shanghai", "中国", "中文", "华语", "普通话"),
    "CU": ("cuba", "cuban", "havana", "son cubano", "rumba", "古巴"),
    "ES": ("spain", "spanish", "flamenco", "sevilla", "madrid", "西班牙", "弗拉门戈"),
    "GH": ("ghana", "accra", "highlife", "azonto", "west africa", "加纳", "西非"),
    "HK": ("hong kong", "hongkong", "hong kong china", "cantopop", "cantonese pop", "cantonese", "香港", "中国香港", "粤语", "粵語"),
    "ID": ("indonesia", "jakarta", "bali", "gamelan", "dangdut", "印尼", "印度尼西亚", "甘美兰"),
    "IE": ("ireland", "irish", "celtic", "dublin", "爱尔兰", "凯尔特"),
    "IN": ("india", "indian", "bollywood", "bhangra", "punjabi", "hindustani", "carnatic", "印度", "宝莱坞", "旁遮普"),
    "IR": ("iran", "persian", "tehran", "伊朗", "波斯"),
    "JM": ("jamaica", "kingston", "reggae", "dancehall", "ska", "dub", "牙买加", "雷鬼"),
    "JP": ("japan", "japanese", "tokyo", "j-pop", "jpop", "city pop", "enka", "anime", "日本"),
    "KR": ("korea", "korean", "seoul", "k-pop", "kpop", "韩国", "韩语"),
    "ML": ("mali", "bamako", "manding", "mandinka", "kora", "griot", "west africa", "马里", "西非"),
    "MO": ("macau", "macao", "macau china", "macao china", "澳门", "澳門", "中国澳门"),
    "MX": ("mexico", "mexican", "mariachi", "ranchera", "corrido", "墨西哥"),
    "NG": ("nigeria", "nigerian", "lagos", "abuja", "afrobeats", "afrobeat", "afro-pop", "afropop", "naija", "yoruba", "west africa", "尼日利亚", "西非", "非洲节奏"),
    "PT": ("portugal", "portuguese", "lisbon", "fado", "葡萄牙", "法朵"),
    "SN": ("senegal", "dakar", "mbalax", "west africa", "塞内加尔", "西非"),
    "TR": ("turkey", "turkish", "istanbul", "ankara", "makam", "土耳其"),
    "TT": ("trinidad", "tobago", "soca", "calypso", "特立尼达", "卡利普索"),
    "TW": ("taiwan", "taipei", "chinese taipei", "台北", "台湾", "臺灣", "中国台北", "中华台北", "中国台湾"),
    "ZA": ("south africa", "south african", "johannesburg", "amapiano", "kwaito", "南非"),
}

KNOWN_ARTIST_ORIGINS = {
    "ayra starr": "NG",
}

def _clean(value: Any) -> str:
    return str(value or "").strip()


def norm_text(value: Any) -> str:
    raw = _clean(value).casefold()
    deaccented = "".join(
        char for char in unicodedata.normalize("NFKD", raw) if not unicodedata.combining(char)
    )
    return " ".join(re.sub(r"[^0-9a-z\u3400-\u9fff]+", " ", deaccented).split())


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
    return f" {normalized_phrase} " in f" {haystack} "


def _row_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    return norm_text(" ".join(_clean(row.get(key)) for key in keys))


def is_itunes_storefront_country(row: dict[str, Any]) -> bool:
    source = f"{row.get('source_dataset', '')} {row.get('platform', '')} {row.get('platform_track_url', '')}".casefold()
    return "itunes" in source or "music.apple.com" in source


def _hint_iso(row: dict[str, Any]) -> tuple[str, str]:
    artist_text = norm_text(_clean(row.get("artist")))
    for artist, iso in KNOWN_ARTIST_ORIGINS.items():
        if _contains_phrase(artist_text, artist):
            return iso, "artist_origin_hint"

    descriptive_text = _row_text(
        row,
        (
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
        return {
            "country": "",
            "country_iso": "",
            "country_source": "itunes_storefront_ignored",
            "country_original": raw_country,
            "storefront_country": raw_country,
            "catalog_country_is_storefront": True,
        }

    return {
        "country": raw_country,
        "country_iso": "",
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
    for artist, iso in KNOWN_ARTIST_ORIGINS.items():
        if iso in requested_isos and _contains_phrase(artist_text, artist):
            matched.extend([iso, COUNTRY_NAMES.get(iso, ""), *COUNTRY_SEARCH_TERMS.get(iso, ())])
    for iso in sorted(requested_isos):
        terms = COUNTRY_SEARCH_TERMS.get(iso, ())
        if any(_contains_phrase(descriptive_text, term) for term in terms):
            matched.extend([iso, COUNTRY_NAMES.get(iso, ""), *terms])
    return norm_text(" ".join(_clean(part) for part in matched))
