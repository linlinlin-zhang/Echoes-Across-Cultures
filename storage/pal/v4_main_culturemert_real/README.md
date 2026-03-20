# V4 Real PAL Bundle

This folder is ready for human PAL collection.

Recommended order:
1. Start with the pilot sheet and verify annotator understanding.
2. Revise instructions if the pilot reveals ambiguity.
3. Move to the round-1 sheet for the main annotation batch.
4. Save the completed CSV as tasks_round1_200_annotation_filled.csv in this folder.
5. Run run_pal_platform with pal_v4_main_culturemert_real.run.json.

Generated files:
- candidates: E:\Desktop\Echo\storage\pal\v4_main_culturemert_real\candidates_1000.jsonl
- candidate annotation sheet: E:\Desktop\Echo\storage\pal\v4_main_culturemert_real\candidates_1000_annotation.csv
- pilot tasks: E:\Desktop\Echo\storage\pal\v4_main_culturemert_real\pilot_tasks_20.jsonl
- pilot annotation sheet: E:\Desktop\Echo\storage\pal\v4_main_culturemert_real\pilot_tasks_20_annotation.csv
- round-1 tasks: E:\Desktop\Echo\storage\pal\v4_main_culturemert_real\tasks_round1_200.jsonl
- round-1 annotation sheet: E:\Desktop\Echo\storage\pal\v4_main_culturemert_real\tasks_round1_200_annotation.csv
- manifest: E:\Desktop\Echo\storage\pal\v4_main_culturemert_real\bundle_manifest.json

Annotation reminder:
- Judge whether the pair feels similar in affective function or listening intent.
- Do not decide directly from culture labels, language names, or source names.
- Fill `similar` with yes/no (or 1/0) and add one short rationale.
- Leave difficult cases blank and explain the reason in `notes`.
