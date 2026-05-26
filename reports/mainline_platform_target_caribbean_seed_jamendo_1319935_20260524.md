# Mainline Platform Recommendation Smoke Test

## Algorithm
- name: `dcas_mainline_seed_recommender`
- mode: `target`
- model: `dcas_full_v4_main_culturemert_stage3`
- reranker: `OT relevance + calibrated cultural reranking`

## Seed
- track_id: `jamendo_1319935`
- title: Ragga: cheerful, Caribbean, summer, pool, beach, sexy (0:42)
- artist: GarsuMene
- culture: `caribbean`
- source: `jamendo`
- platform_url: https://www.jamendo.com/track/1319935

## Recommendations
| rank | track_id | title | artist | culture | source | score | platform_url |
|---:|---|---|---|---|---|---:|---|
| 1 | `jamendo_1783148` | Dj Dzverbass-Body Shake Moombahton | Dj Dzverbass | `caribbean` | `jamendo` | 0.760028 | https://www.jamendo.com/track/1783148 |
| 2 | `jamendo_1815576` | Cocktail Mambo (1:00) | pinegroove | `caribbean` | `jamendo` | 0.743093 | https://www.jamendo.com/track/1815576 |
| 3 | `jamendo_1652159` | Buena - Monstrumental | Kali D En La Biitrola | `caribbean` | `jamendo` | 0.726654 | https://www.jamendo.com/track/1652159 |
| 4 | `jamendo_178471` | Fred - Hands Up Riddim | MADRAS FAMILY | `caribbean` | `jamendo` | 0.725789 | https://www.jamendo.com/track/178471 |
| 5 | `jamendo_916904` | The Squadron (Kain Remix) | Raziel Jamaerah | `caribbean` | `jamendo` | 0.722926 | https://www.jamendo.com/track/916904 |
| 6 | `jamendo_1314801` | 03 Yen Ki Dancehall | FOLI | `caribbean` | `jamendo` | 0.718630 | https://www.jamendo.com/track/1314801 |
| 7 | `jamendo_895270` | Dez A Sound | Meltin' Kolcha | `caribbean` | `jamendo` | 0.717605 | https://www.jamendo.com/track/895270 |
| 8 | `jamendo_1884932` | Stylish Percussion and Brass | Joel Loopez | `caribbean` | `jamendo` | 0.715240 | https://www.jamendo.com/track/1884932 |
| 9 | `jamendo_215564` | no no no | Missin Red | `caribbean` | `jamendo` | 0.713659 | https://www.jamendo.com/track/215564 |
| 10 | `jamendo_1588912` | That Fake Ish | Schitzo | `caribbean` | `jamendo` | 0.711760 | https://www.jamendo.com/track/1588912 |

## Metrics
```json
{
  "n": 10,
  "culture_counts": {
    "caribbean": 10
  },
  "source_counts": {
    "jamendo": 10
  },
  "mean_score": 0.7255385160446167,
  "with_cover_art": 10,
  "with_platform_link": 10
}
```

## Warnings
- 30k catalog does not yet have real user interaction logs; minority uses a catalog-balance proxy.
