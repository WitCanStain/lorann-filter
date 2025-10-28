import matplotlib.pyplot as plt
from labellines import labelLines
import json
from pathlib import Path
import matplotlib.pyplot as plt



prefix = "1M-mixedincluded-experimental"  # the string to match
experiment_data = {}
for path in Path('.').iterdir():
    if path.is_file() and path.name.startswith(prefix):
        with open(path, 'r') as f:
            print(f"Opened: {path}")
            data = json.load(f)
            for key in data.keys():
                experiment_data[key] = data[key]

# print(experiment_data)
print(len(experiment_data.keys()))


first = list(experiment_data.keys())[0]
n_input_vecs = json.loads(first)["n_input_vecs"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
plt.suptitle(f"Latency, Recall, and Selectivity ({n_input_vecs} points) with Mixed approach included", fontsize=16)
axes = axes.flatten()
for key in list(experiment_data.keys()):
    index_data = json.loads(key)
    if index_data["a0_selectivity"] == 0.1 or index_data["a0_selectivity"] == 0.9:
        del experiment_data[key]




for i, key in enumerate(sorted(experiment_data.keys(), key=lambda k: json.loads(k)["a0_selectivity"])):
    index_data = json.loads(key)

    data = experiment_data[key]
    filter_approaches = data.keys()
    for filter_approach in filter_approaches:
        recalls = data[filter_approach]["recalls"]
        latencies = data[filter_approach]["approximate_latencies"]
        axes[i].plot(recalls, latencies, label=filter_approach)
    # axes[i].set_ylabel("Latency (μs)")
    # axes[i].set_xlabel("Recall")
    axes[i].set_title(f"{index_data["a0_selectivity"]} Selectivity")
    axes[i].tick_params(axis='y', labelrotation=45)
    labelLines(axes[i].get_lines(), align=False)
fig.savefig(f"../../figures/3x-1M-mixedincluded-recall-latency_matrix.png")
plt.show()