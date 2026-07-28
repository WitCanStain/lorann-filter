import matplotlib.pyplot as plt
from labellines import labelLines
import json
import math
import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator, LogFormatterSciNotation, SymmetricalLogLocator, FuncFormatter

subplots_horizontal = 2
subplots_vertical = 3

# A4 portrait in inches (width x height).
# Increase this scale slightly if labels are still too small in exported figures.
a4_scale = 1.15
A4_WIDTH_IN = 8.27 * a4_scale
A4_HEIGHT_IN = 11.69 * a4_scale

# Toggle log scaling for latency plots: set True to use log scale
use_log_scale = True
# Choose axis for log scaling: 'y', 'x', or 'both'
log_axis = 'y'
# Toggle drawing exact postfilter/prefilter overlay lines in the graphs.
show_exact_lines = True
save_figures = False
# Set this to a filename inside results/ to override the automatic latest-file selection.
results_filename_override = "nytimes-256-angular-290000-2026-07-02 17:28:38.json"

repo_root = Path(__file__).resolve().parents[2]
results_dir = repo_root / "results"
# prefix = "deep-image-96-angular"  # the string to match



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

results_file_path = select_results_file(results_dir, results_filename_override)
print(f"Opened: {results_file_path}")
with open(results_file_path, 'r', encoding='utf-8') as f:
    experiment_data = json.load(f)

print(len(experiment_data.keys()))

filtered_keys = []
for key in list(experiment_data.keys()):
    filtered_keys.append(key)

if not filtered_keys:
    raise SystemExit("No experiment data matched the selected prefix and filters.")

first = filtered_keys[0]
n_input_vecs = json.loads(first)["n_input_vecs"]

subplot_rows = max(1, math.ceil(len(filtered_keys) / subplots_horizontal))


def collect_shared_log_bounds(data, metric_key, exact_data_key=None):
    all_y_values = []
    for filter_approach in data.keys():
        all_y_values.extend(data[filter_approach].get(metric_key, []))
        if exact_data_key:
            exact_data = data[filter_approach].get(exact_data_key)
            if isinstance(exact_data, dict):
                for exact_values in exact_data.values():
                    all_y_values.extend(exact_values)
            else:
                all_y_values.extend(data[filter_approach].get("exact_latencies", []))

    positive_y_values = [value for value in all_y_values if value and value > 0]
    if not positive_y_values:
        return None
    y_min = min(positive_y_values)
    y_max = max(positive_y_values)
    y_min_decade = 10 ** math.floor(math.log10(y_min))
    y_max_decade = 10 ** math.ceil(math.log10(y_max))
    return (y_min_decade, y_max_decade)


def plot_metric_grid(metric_key, fig_title, file_suffix, exact_data_key=None, exact_label_suffix="exact"):
    fig, axes = plt.subplots(
        subplot_rows,
        subplots_horizontal,
        figsize=(A4_WIDTH_IN, A4_HEIGHT_IN),
        constrained_layout=True,
    )
    plt.suptitle(fig_title, fontsize=16)
    axes = axes.flatten()

    for i, key in enumerate(sorted(filtered_keys, key=lambda k: json.loads(k)["a0_selectivity"])):
        index_data = json.loads(key)
        data = experiment_data[key]
        filter_approaches = data.keys()
        print("i:", i, "filter_approaches:", filter_approaches)

        for filter_approach in filter_approaches:
            recalls = data[filter_approach]["recalls"]
            y_values = data[filter_approach].get(metric_key, [])
            axes[i].plot(recalls, y_values, label=f"{filter_approach}")

        subplot_log_bounds = None
        if show_exact_lines and exact_data_key:
            exact_latencies_by_approach = {}
            for filter_approach in filter_approaches:
                exact_data = data[filter_approach].get(exact_data_key)
                if isinstance(exact_data, dict):
                    for exact_approach, exact_values in exact_data.items():
                        exact_latencies_by_approach.setdefault(exact_approach, []).extend(exact_values)
                else:
                    exact_latencies = data[filter_approach].get("exact_latencies")
                    exact_approach = data[filter_approach].get("exact_approach")
                    if exact_latencies and exact_approach:
                        exact_latencies_by_approach.setdefault(exact_approach, []).extend(exact_latencies)

            for exact_approach, exact_latencies in exact_latencies_by_approach.items():
                try:
                    avg_exact = sum(exact_latencies) / len(exact_latencies)
                    axes[i].axhline(y=avg_exact, linestyle='--', color='k', label=f"{exact_approach} ({exact_label_suffix})")
                except Exception:
                    pass

            subplot_log_bounds = collect_shared_log_bounds(data, metric_key, exact_data_key=exact_data_key)

        axes[i].set_title(f"{index_data['a0_selectivity']:.2f} Selectivity")
        axes[i].tick_params(axis='y', labelrotation=45)
        axes[i].legend(loc='best', fontsize='x-small', handlelength=2, borderpad=0.2, labelspacing=0.2, handletextpad=0.4, framealpha=0.7)

        # Values below this threshold are shown on a linear scale so the curves
        # remain visible near zero instead of being squashed against the axis.
        linthresh = subplot_log_bounds[0] if subplot_log_bounds else 1e-2

        if use_log_scale:
            if log_axis in ('y', 'both'):
                # axes[i].set_yscale('log')
                axes[i].set_yscale('symlog', linthresh=linthresh, linscale=1.0)
                # axes[i].set_ylim(bottom=1e-3)  # Set a minimum y-limit to avoid issues with log scale
                axes[i].margins(y=0.5)
            if log_axis in ('x', 'both'):
                axes[i].set_xscale('log')
        try:
            if use_log_scale and log_axis in ('y', 'both'):
                top_bound = subplot_log_bounds[1] if subplot_log_bounds else None
                axes[i].set_ylim(bottom=0, top=top_bound)
                axes[i].yaxis.set_major_locator(SymmetricalLogLocator(base=10.0, linthresh=linthresh))
                base_formatter = LogFormatterSciNotation(base=10.0, labelOnlyBase=True)
                axes[i].yaxis.set_major_formatter(
                    FuncFormatter(lambda y, pos, bf=base_formatter: "0" if y == 0 else bf(y, pos))
                )
            else:
                axes[i].set_ylim(bottom=0)
                axes[i].yaxis.set_major_locator(MaxNLocator(nbins=6))
        except Exception:
            pass
        
        labelLines(axes[i].get_lines(), align=False)
        

    if save_figures:
        fig.savefig(f"../../figures/{subplots_vertical}x{subplots_horizontal}-{dataset_label}-{file_suffix}.png")
    return fig


def get_dataset_label(experiment_payload):
    first_index_key = next(iter(experiment_payload))
    index_params = json.loads(first_index_key)
    dataset = index_params.get("dataset_file")
    if dataset:
        return Path(dataset).stem
    return prefix


dataset_label = get_dataset_label(experiment_data)

approx_fig = plot_metric_grid(
    metric_key="approximate_latencies",
    fig_title=f"Search Latency ({n_input_vecs} points), {dataset_label}",
    file_suffix="approximate-recall-latency_matrix",
    exact_data_key="exact_latencies_by_approach",
    exact_label_suffix="exact",
)

filter_fig = plot_metric_grid(
    metric_key="filter_times",
    fig_title=f"Filter Latency ({n_input_vecs} points), {dataset_label}",
    file_suffix="filtertime-recall-latency_matrix",
    exact_data_key="exact_filter_times_by_approach",
    exact_label_suffix="exact filter",
)
# plt.tight_layout()
plt.show()