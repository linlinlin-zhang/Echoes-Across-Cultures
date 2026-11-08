"""Exercise real recommendation math on small synthetic latent vectors, without model weights."""

from __future__ import annotations
from collections import Counter
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcas_server.mainline_platform import MainlineRecommendationPlatform
from dcas_server.paths import Storage


class FakeTracks(SimpleNamespace):
    def __len__(self):
        return len(self.track_id)


def verify(result, platform, seed_zs, seed_za, excluded):
    candidates = [i for i in range(len(platform.tracks)) if i not in excluded]
    distances = np.linalg.norm(platform.zs_all.numpy()[candidates] - seed_zs, axis=1)
    acoustic = -np.sum((platform.za_all.numpy()[candidates] - seed_za) ** 2, axis=1)
    expected_n = (distances - distances.min()) / (distances.max() - distances.min())
    expected_r = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min())
    for row in result["recommendations"]:
        i = candidates.index(platform.track_id_to_idx[row["track_id"]])
        metrics = row["recommendation_metrics"]
        np.testing.assert_allclose(metrics["style_novelty"], expected_n[i], atol=1e-6)
        np.testing.assert_allclose(metrics["listening_relevance"], expected_r[i], atol=1e-6)
        np.testing.assert_allclose(metrics["style_distance_raw"], distances[i], atol=1e-6)
        np.testing.assert_allclose(metrics["audio_transport_cost_raw"], -acoustic[i], atol=1e-6)
    for name in ["style_novelty", "listening_relevance"]:
        np.testing.assert_allclose(
            result["metrics"][name + "_mean"],
            np.mean([r["recommendation_metrics"][name] for r in result["recommendations"]]),
            atol=1e-7,
        )


def main():
    with tempfile.TemporaryDirectory() as folder:
        p = object.__new__(MainlineRecommendationPlatform)
        p.storage = Storage(Path(folder))
        p.device = torch.device("cpu")
        p._refresh_metadata_if_needed = lambda: None
        p.tracks = FakeTracks(
            track_id=np.array(["seed", "a", "b", "c", "d", "seed2"]),
            culture=np.array(["west", "japan", "brazil", "celtic", "japan", "west"]),
            source_dataset=np.array(["itunes"] * 6),
        )
        p.track_id_to_idx = {str(t): i for i, t in enumerate(p.tracks.track_id)}
        p.metadata_by_id = {
            str(t): {"track_id": str(t), "title": str(t), "artist": str(t), "label": "Rock"} for t in p.tracks.track_id
        }
        p._low_signal_cache = {}
        p.culture_counts = Counter(p.tracks.culture)
        p.source_counts = Counter(p.tracks.source_dataset)
        p.zs_all = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [2.0, 2.0], [0.5, 0.5]])
        p.za_all = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.0, 0.2], [0.5, 0.5], [1.0, 1.0], [0.05, 0.05]])
        p.culture_names, p.culture_centroids = p._build_culture_centroids()
        result = p.recommend(seed_track_ids=["seed"], k=3, recall_k=100, random_seed=11)
        assert result["request"]["target_selection"] == "random_cross_culture"
        assert result["request"]["target_culture"] is None
        assert result["request"]["effective_target_culture"] != "west"
        verify(result, p, np.array([0.0, 0.0]), np.array([0.0, 0.0]), [0])
        targets = {
            p._choose_target_culture(target_culture=None, seed_cultures=["west"], random_seed=i) for i in range(30)
        }
        assert targets == {"japan", "brazil", "celtic"}
        assert (
            p._choose_target_culture(target_culture=None, seed_cultures=["west"], random_seed=11)
            == result["request"]["effective_target_culture"]
        )
        assert (
            p.recommend(seed_track_ids=["seed"], target_culture="brazil", k=2)["request"]["effective_target_culture"]
            == "brazil"
        )
        try:
            p._choose_target_culture(target_culture="invalid", seed_cultures=[], random_seed=1)
        except ValueError:
            pass
        else:
            raise AssertionError("Unknown explicit target accepted")
        multi = p.recommend(seed_track_ids=["seed", "seed2"], k=3)
        assert all(np.isfinite(r["recommendation_metrics"]["listening_relevance"]) for r in multi["recommendations"])
        p._encode_external_embedding = lambda _: (np.array([0.0, 0.0]), p.zs_all[:1], p.za_all[:1])
        uploaded = p.recommend_embedding(
            embedding=np.array([0.0, 0.0]), upload_info={"title": "Upload"}, seed_culture="west", k=3, recall_k=100
        )
        assert uploaded["request"]["target_selection"] == "random_cross_culture"
        assert uploaded["request"]["effective_target_culture"] != "west"
        verify(uploaded, p, np.array([0.0, 0.0]), np.array([0.0, 0.0]), [])
        assert "recommendation_metrics" not in p.track("seed")
        assert p._result_metrics(recommendations=[])["style_novelty_mean"] is None
        p.culture_counts = Counter({"west": 1})
        assert p._choose_target_culture(target_culture=None, seed_cultures=["west"], random_seed=1) == "west"
    print(
        "PASS independent latent-distance and audio-cost scores, summary means, uploads, OT multi-seed path and server-side random targets"
    )


if __name__ == "__main__":
    main()
