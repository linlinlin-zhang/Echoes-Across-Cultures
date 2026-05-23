"""Generate detailed Chinese draft paper and figure generation scripts."""
import json
from pathlib import Path

BASE = Path(__file__).parent.parent

def load_json(path):
    p = BASE / path
    if p.exists():
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# Load all benchmark summaries
bench_paths = {
    'v4_main_cm': 'reports/benchmarks/v4_main_culturemert_stage3_lambdamart/benchmark_summary.json',
    'v4_main_gm': 'reports/benchmarks/v4_main_gemini_stage3_lambdamart/benchmark_summary.json',
    'routeA_cm': 'reports/benchmarks/v4_routeA_small_culturemert_stage3_lambdamart/benchmark_summary.json',
    'routeA_gm': 'reports/benchmarks/v4_routeA_small_gemini_stage3_lambdamart/benchmark_summary.json',
    'public_cn': 'reports/benchmarks/public_routeA_phase2_cn_lambdamart/benchmark_summary.json',
    'yambda': 'reports/benchmarks/yambda_5b_subset_global_log_benchmark/benchmark_summary.json',
}
bench_data = {k: load_json(v) for k, v in bench_paths.items()}

# Load ablation summaries
ablation_cm = load_json('reports/audits/ablation_v4_main_2026-04-05/v4_main_culturemert/summary.json')
ablation_gm = load_json('reports/audits/ablation_v4_main_2026-04-05/v4_main_gemini/summary.json')

# Load comparison files for ablation
def load_ablation_comp(base_dir, comp_name):
    return load_json(f'reports/audits/ablation_v4_main_2026-04-05/{base_dir}/{comp_name}')

# Load hparam sweeps
hparam_cm = load_json('reports/hparam/v4_main_culturemert_stage3_calibration_sweep/sweep_summary.json')

# Extract key numbers from benchmarks
def extract_table(bench, setting_name):
    """Extract comparison table from benchmark summary."""
    if not bench:
        return []
    comps = bench.get('comparisons', [])
    for comp in comps:
        if comp.get('setting', '') == setting_name:
            return comp.get('methods', [])
    return bench.get('comparisons', [])

print("All data loaded successfully")
print(f"Benchmarks: {len(bench_data)} loaded")
print(f"Ablation CM: {'ok' if ablation_cm else 'missing'}")
print(f"Ablation GM: {'ok' if ablation_gm else 'missing'}")
print(f"Hparam CM: {'ok' if hparam_cm else 'missing'}")
