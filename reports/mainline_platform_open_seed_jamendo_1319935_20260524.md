# Mainline Platform Recommendation Smoke Test

## Algorithm
- name: `dcas_mainline_seed_recommender`
- mode: `open`
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
| 1 | `jamendo_1783148` | Dj Dzverbass-Body Shake Moombahton | Dj Dzverbass | `caribbean` | `jamendo` | 0.918840 | https://www.jamendo.com/track/1783148 |
| 2 | `jamendo_2314235` | mambo reggaeton 15 sec | nikproteus | `latin` | `jamendo` | 0.906235 | https://www.jamendo.com/track/2314235 |
| 3 | `jamendo_2004044` | Indian Festive Celebration Beats | Gudappan | `india` | `jamendo` | 0.903725 | https://www.jamendo.com/track/2004044 |
| 4 | `jamendo_1815576` | Cocktail Mambo (1:00) | pinegroove | `caribbean` | `jamendo` | 0.898313 | https://www.jamendo.com/track/1815576 |
| 5 | `jamendo_1156905` | I Dont Care About ( Radio Edit ) | Hasenchat | `celtic` | `jamendo` | 0.898541 | https://www.jamendo.com/track/1156905 |
| 6 | `jamendo_1792132` | Indian Dance (v4) | Nargo | `india` | `jamendo` | 0.878103 | https://www.jamendo.com/track/1792132 |
| 7 | `jamendo_2240106` | Queen of Sheba - 30 sec edit | Abydos Music | `middle_east` | `jamendo` | 0.876060 | https://www.jamendo.com/track/2240106 |
| 8 | `jamendo_916904` | The Squadron (Kain Remix) | Raziel Jamaerah | `caribbean` | `jamendo` | 0.874077 | https://www.jamendo.com/track/916904 |
| 9 | `jamendo_1558296` | Localisa | Thebest kingmalandro | `caribbean` | `jamendo` | 0.870558 | https://www.jamendo.com/track/1558296 |
| 10 | `jamendo_1143803` | Goaaaaal ! | Grégoire Lourme | `brazil` | `jamendo` | 0.869155 | https://www.jamendo.com/track/1143803 |

## Metrics
```json
{
  "n": 10,
  "culture_counts": {
    "brazil": 1,
    "caribbean": 4,
    "celtic": 1,
    "india": 2,
    "latin": 1,
    "middle_east": 1
  },
  "source_counts": {
    "jamendo": 10
  },
  "mean_score": 0.8893606781959533,
  "with_cover_art": 10,
  "with_platform_link": 10
}
```

## Warnings
- 30k catalog does not yet have real user interaction logs; minority uses a catalog-balance proxy.
- open mode is the product seed-track adaptation of the mainline; strict benchmark mode is target.
