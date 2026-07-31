"""Quantify how much a metric (approximate latency or filter time) varies across
selectivity levels, at a fixed recall target, for a given filter approach.

For example: for filter approach "mixed" at recall 0.7, how much does the
(interpolated) latency differ between the different a0_selectivity levels in a
given results file?

This script does not modify visualiser.py; it duplicates the small amount of
file-selection logic it needs to stay independent.
"""
import json
import re
import statistics
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Filter approach(es) to analyze, e.g. "mixed", "hybrid_avx", "postfilter", "indexing_avx".
target_approaches = ["mixed", "indexing_avx", "hybrid_avx", "postfilter"]
# Recall level(s) at which to interpolate the metric for comparison.
target_recalls = [.5, .6, .7, .8, .9, .95, .99]
# Metric to analyze: "approximate_latencies" or "filter_times".
metric_key = "filter_times"
# The corresponding "exact" baseline key for each metric_key. Exact search
# latencies/filter times have no recall axis (exact search always has recall
# 1.0), so they are compared directly per selectivity rather than by
# interpolating at a target recall.
exact_metric_key_by_metric_key = {
    "approximate_latencies": "exact_latencies_by_approach",
    "filter_times": "exact_filter_times_by_approach",
}
# Set this to a filename inside results/ to override the automatic latest-file selection.
results_filename_override = "deep-image-96-angular-9990000-2026-07-03 18:50:15-noneuclidean.json"

repo_root = Path(__file__).resolve().parents[2]
results_dir = repo_root / "results"


def extract_timestamp_from_filename(path):
    match = re.search(r"-(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.json$", path.name)
    if not match:
        return None
    return match.group(1)


def select_results_file(results_directory, override_filename=None):
    if override_filename:
        override_path = Path(override_filename)
        if not override_path.is_absolute():
            override_path = results_directory / override_path
        return override_path

    candidate_paths = [path for path in results_directory.iterdir() if path.is_file() and path.suffix == ".json"]
    if not candidate_paths:
        raise SystemExit(f"No results JSON files found in {results_directory}")

    timestamped_candidates = []
    for path in candidate_paths:
        timestamp = extract_timestamp_from_filename(path)
        if timestamp:
            timestamped_candidates.append((timestamp, path))

    if timestamped_candidates:
        return max(timestamped_candidates, key=lambda item: item[0])[1]

    return max(candidate_paths, key=lambda path: path.stat().st_mtime)


def load_experiment_data(results_directory, override_filename=None):
    results_file_path = select_results_file(results_directory, override_filename)
    print(f"Opened: {results_file_path}")
    with open(results_file_path, "r", encoding="utf-8") as f:
        return json.load(f), results_file_path


def dedupe_sort_by_recall(recalls, values):
    """Sort (recall, value) pairs by ascending recall, averaging values that
    share the same recall so the recall array is strictly increasing (a
    requirement for np.interp, which needs monotonic x)."""
    grouped = {}
    for recall, value in zip(recalls, values):
        grouped.setdefault(recall, []).append(value)

    sorted_recalls = sorted(grouped.keys())
    averaged_values = [statistics.fmean(grouped[recall]) for recall in sorted_recalls]
    return sorted_recalls, averaged_values


def interpolate_metric_at_recall(recalls, values, target_recall):
    """Return the metric value interpolated at target_recall, or None if
    target_recall falls outside the observed recall range for this series
    (extrapolation would be misleading, so we refuse instead of silently
    clamping)."""
    if len(recalls) < 2:
        return None
    if target_recall < recalls[0] or target_recall > recalls[-1]:
        return None
    return float(np.interp(target_recall, recalls, values))


def collect_selectivity_series(experiment_data, filter_approach, metric_key):
    """Returns a dict mapping a0_selectivity -> (sorted_recalls, averaged_values)
    for every top-level index-parameter key that contains the given filter
    approach."""
    series_by_selectivity = {}
    for index_params_json, per_approach_data in experiment_data.items():
        if filter_approach not in per_approach_data:
            continue
        index_params = json.loads(index_params_json)
        selectivity = index_params.get("a0_selectivity")
        if selectivity is None:
            continue

        recalls = per_approach_data[filter_approach].get("recalls", [])
        values = per_approach_data[filter_approach].get(metric_key, [])
        if not recalls or not values or len(recalls) != len(values):
            continue

        sorted_recalls, averaged_values = dedupe_sort_by_recall(recalls, values)
        # Multiple index-parameter keys could share the same selectivity
        # (e.g. differing in other params); merge their raw points before
        # interpolating so each selectivity contributes one series.
        if selectivity in series_by_selectivity:
            existing_recalls, existing_values = series_by_selectivity[selectivity]
            merged_recalls = list(existing_recalls) + list(sorted_recalls)
            merged_values = list(existing_values) + list(averaged_values)
            sorted_recalls, averaged_values = dedupe_sort_by_recall(merged_recalls, merged_values)
        series_by_selectivity[selectivity] = (sorted_recalls, averaged_values)

    return series_by_selectivity


def collect_exact_series_by_selectivity(experiment_data, exact_data_key):
    """Returns a dict mapping exact_approach_name -> {selectivity -> [raw values]}.

    Exact search baselines are stored per filter_approach (e.g. under
    "exact_latencies_by_approach"), but they describe the same underlying
    exact search, so values are merged across all filter_approaches, and
    across all index-parameter keys sharing a selectivity.
    """
    values_by_exact_approach = {}
    for index_params_json, per_approach_data in experiment_data.items():
        index_params = json.loads(index_params_json)
        selectivity = index_params.get("a0_selectivity")
        if selectivity is None:
            continue

        for filter_approach, approach_data in per_approach_data.items():
            exact_data = approach_data.get(exact_data_key)
            if not isinstance(exact_data, dict):
                continue
            for exact_approach_name, exact_values in exact_data.items():
                if not exact_values:
                    continue
                selectivity_map = values_by_exact_approach.setdefault(exact_approach_name, {})
                selectivity_map.setdefault(selectivity, []).extend(exact_values)

    return values_by_exact_approach


def analyze_exact(experiment_data, exact_data_key):
    values_by_exact_approach = collect_exact_series_by_selectivity(experiment_data, exact_data_key)
    if not values_by_exact_approach:
        print(f"  No exact-search data found for '{exact_data_key}'.")
        return

    for exact_approach_name in sorted(values_by_exact_approach):
        selectivity_to_values = values_by_exact_approach[exact_approach_name]
        selectivity_to_value = {
            selectivity: statistics.fmean(values)
            for selectivity, values in selectivity_to_values.items()
        }

        print(f"  Exact approach: {exact_approach_name}, metric: {exact_data_key}")
        for selectivity in sorted(selectivity_to_value):
            print(f"    selectivity={selectivity:.4g} -> {selectivity_to_value[selectivity]:.6g}")

        stats = summarize_variance(selectivity_to_value)
        if stats is None:
            print("    Not enough selectivity levels with coverage to compute variance.")
            continue

        print(
            "    stats: "
            f"n={stats['n']} mean={stats['mean']:.6g} stdev={stats['stdev']:.6g} "
            f"variance={stats['variance']:.6g} min={stats['min']:.6g} max={stats['max']:.6g} "
            f"range={stats['range']:.6g} cv={stats['coefficient_of_variation']:.4g}"
        )


def summarize_variance(selectivity_to_value):
    """selectivity_to_value: dict of selectivity -> interpolated metric value.
    Returns a dict of summary statistics, or None if fewer than 2 points."""
    values = list(selectivity_to_value.values())
    if len(values) < 2:
        return None

    mean = statistics.fmean(values)
    stdev = statistics.stdev(values)  # sample stdev (n-1 denominator)
    variance = statistics.variance(values)
    minimum = min(values)
    maximum = max(values)
    coefficient_of_variation = stdev / mean if mean != 0 else float("nan")

    return {
        "n": len(values),
        "mean": mean,
        "stdev": stdev,
        "variance": variance,
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
        "coefficient_of_variation": coefficient_of_variation,
    }


def analyze(experiment_data, filter_approach, target_recall, metric_key):
    series_by_selectivity = collect_selectivity_series(experiment_data, filter_approach, metric_key)
    if not series_by_selectivity:
        print(f"  No data found for filter approach '{filter_approach}'.")
        return

    selectivity_to_value = {}
    skipped = []
    for selectivity in sorted(series_by_selectivity):
        recalls, values = series_by_selectivity[selectivity]
        interpolated = interpolate_metric_at_recall(recalls, values, target_recall)
        if interpolated is None:
            skipped.append((selectivity, recalls[0] if recalls else None, recalls[-1] if recalls else None))
        else:
            selectivity_to_value[selectivity] = interpolated

    print(f"  Approach: {filter_approach}, target recall: {target_recall}, metric: {metric_key}")
    for selectivity in sorted(selectivity_to_value):
        print(f"    selectivity={selectivity:.4g} -> {selectivity_to_value[selectivity]:.6g}")
    if skipped:
        print("    Skipped (target recall out of observed range):")
        for selectivity, lo, hi in skipped:
            print(f"      selectivity={selectivity:.4g} (observed recall range [{lo}, {hi}])")

    stats = summarize_variance(selectivity_to_value)
    if stats is None:
        print("    Not enough selectivity levels with coverage to compute variance.")
        return

    print(
        "    stats: "
        f"n={stats['n']} mean={stats['mean']:.6g} stdev={stats['stdev']:.6g} "
        f"variance={stats['variance']:.6g} min={stats['min']:.6g} max={stats['max']:.6g} "
        f"range={stats['range']:.6g} cv={stats['coefficient_of_variation']:.4g}"
    )


def main():
    experiment_data, results_file_path = load_experiment_data(results_dir, results_filename_override)
    dataset_first_key = next(iter(experiment_data))
    dataset_label = Path(json.loads(dataset_first_key).get("dataset_file", results_file_path.stem)).stem
    print(f"Dataset: {dataset_label}\n")

    for filter_approach in target_approaches:
        print(f"=== {filter_approach} ===")
        for target_recall in target_recalls:
            analyze(experiment_data, filter_approach, target_recall, metric_key)
        print()

    exact_data_key = exact_metric_key_by_metric_key.get(metric_key)
    if exact_data_key:
        print("=== exact search baselines ===")
        analyze_exact(experiment_data, exact_data_key)
        print()


if __name__ == "__main__":
    main()
