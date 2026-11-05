"""Check source-preserving genre normalization, explicit filtering and localized search."""

from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcas_server.genre_metadata import apply_genre_metadata, genre_for_label, is_non_music
from dcas_server.lightweight_catalog import LightweightMainlineCatalog
from dcas_server.paths import Storage


def main():
    corrected = apply_genre_metadata(
        {"label": "록", "label_en": "華語流行樂", "tags": "록", "duration_ms": "0", "country": "GB"}
    )
    assert corrected["label"] == corrected["label_en"] == "Rock"
    assert corrected["label_original"] == "록" and corrected["label_en_original"] == "華語流行樂"
    assert corrected["genre_labels"]["zh-CN"] == "摇滚"
    assert corrected["country"] == "GB"
    assert corrected["duration_ms"] is None and not corrected["duration_known"]
    assert apply_genre_metadata(corrected.copy()) == corrected
    assert apply_genre_metadata({"label": "unknown new label", "label_en": "未翻译"})["label_en"] == ""
    assert genre_for_label("House")[1]["labels"]["zh-CN"] == "浩室"
    assert genre_for_label("Dub")[1]["labels"]["zh-CN"] == "Dub 雷鬼"
    assert genre_for_label("K-Pop")[1]["labels"]["zh-CN"] == "韩国流行"
    assert genre_for_label("Afrobeats")[0] != genre_for_label("Afro-Beat")[0]
    assert is_non_music({"label": "Hörspiele"}) and not is_non_music({"label": "Spoken Word"})
    assert genre_for_label("persian")[1]["labels"]["zh-CN"] == "波斯（文化标签）"
    assert apply_genre_metadata({"label": "Rock", "duration_ms": "123456"})["duration_ms"] == "123456"
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        path = root / "public/merged/metadata_merged.csv"
        path.parent.mkdir(parents=True)
        rows = [
            {
                "track_id": "rock",
                "label": "록",
                "title": "Example",
                "artist": "A",
                "culture": "west",
                "source_dataset": "itunes",
            },
            {
                "track_id": "pop",
                "label": "Pop",
                "title": "Example 2",
                "artist": "B",
                "culture": "west",
                "source_dataset": "itunes",
            },
            {
                "track_id": "drama",
                "label": "Hörspiele",
                "title": "Chapter 2",
                "artist": "C",
                "culture": "west",
                "source_dataset": "itunes",
            },
        ]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        catalog = LightweightMainlineCatalog(Storage(root))
        assert catalog.catalog()["total_available"] == 2
        assert catalog.catalog(exclude_low_signal=False)["total_available"] == 3
        assert catalog.catalog(q="摇滚")["items"][0]["track_id"] == "rock"
        assert catalog.catalog(q="Rock")["items"][0]["track_id"] == "rock"
        assert catalog.track("drama")["content_kind"] == "spoken_drama"
    # No model loading: exercise the real GPU recommendation eligibility method.
    try:
        from dcas_server.mainline_platform import MainlineRecommendationPlatform
    except ModuleNotFoundError as error:
        if error.name not in {"torch", "numpy"}:
            raise
        print("SKIP GPU module filter check: model dependencies not available")
    else:
        from types import SimpleNamespace

        platform = object.__new__(MainlineRecommendationPlatform)
        platform._low_signal_cache = {}
        platform.metadata_by_id = {"drama": {"label": "Hörspiele"}, "rock": {"label": "록"}}
        platform.tracks = SimpleNamespace(track_id=["drama", "rock"])
        assert platform._is_low_signal(0) and not platform._is_low_signal(1)
        print("PASS recommendation filter excludes explicit radio drama and preserves music")
    data = json.loads((ROOT / "web/genre-data.json").read_text(encoding="utf-8"))
    source = (ROOT / "web/genre-data.js").read_text(encoding="utf-8")
    assert json.loads(source.removeprefix("window.EchoGenreData = ").removesuffix(";\n")) == data
    print("PASS genre identity, provenance, locale ambiguity, unknown durations, filtering and multilingual search")


if __name__ == "__main__":
    main()
