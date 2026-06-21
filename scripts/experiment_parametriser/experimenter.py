# ctypes_test.py
import ctypes
import pathlib
from numpy.ctypeslib import ndpointer
import numpy as np
import time
import math
import random
import matplotlib.pyplot as plt
from labellines import labelLines
import json
from datetime import datetime

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

todays_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
repo_root = pathlib.Path(__file__).resolve().parents[2]
results_dir = repo_root / "results"
results_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Load the shared library into ctypes
    random.seed(42)
    script_dir = pathlib.Path(__file__).resolve().parent
    libname = script_dir.parent / "cpp" / "libfilter.so"
    c_lib = ctypes.CDLL(libname)
    c_lib.build_index.restype = ctypes.c_bool
    c_lib.build_index.argtypes = (
        ctypes.POINTER(ctypes.c_int), # filter_attribute_list
        ctypes.c_int, # n_attributes
        ctypes.c_int, # n_attributes_per_datapoint
        ctypes.c_int, # n_attr_idx_partitions
        ctypes.c_float, # selectivity
        ctypes.c_int, # n_input_vecs
        ctypes.c_int, # n_clusters
        ctypes.c_int, # global_dim
        ctypes.c_int, # rank
        ctypes.c_int, # train_size
        ctypes.c_bool, # euclidean
        ctypes.c_bool, # use_hdf5
        ctypes.c_char_p # dataset_file_path
        )
    
    c_lib.fast_filter_wrapper_profiled.restype = ctypes.c_float
    c_lib.fast_filter_wrapper_profiled.argtypes = (
        ctypes.POINTER(ctypes.c_int), # idxs
        ctypes.c_int, # n_idxs
        ctypes.c_int, # k
        ctypes.c_int, # M
        ctypes.c_int, # cluster_to_search
        ctypes.c_int, # points_to_rerank
        ctypes.POINTER(ctypes.c_int), # int_filter_attributes
        ctypes.c_int, # n_filter_attributes
        ctypes.c_char_p, # filter_approach
        ctypes.c_char_p, # exact_search_approach
        ctypes.POINTER(ctypes.c_float), # recall
        ctypes.POINTER(ctypes.c_int), # approx_latency
        ctypes.POINTER(ctypes.c_int), # exact_latency
        ctypes.POINTER(ctypes.c_int), # exact_filter_time
        ctypes.POINTER(ctypes.c_int), # avg_duration_cluster
        ctypes.c_bool) # verbose
    
    dataset_file = "gist-960-euclidean.hdf5" #"gist-960-euclidean.hdf5" #"fashion-mnist-784-euclidean.hdf5" nytimes-256-angular.hdf5 deep-image-96-angular.hdf5
    dataset_label = pathlib.Path(dataset_file).stem
    dataset_filter_attribute_range = [i for i in range(32)]
    n_input_vecs = 500_000 #999994 # 9990000 deep # 60k mnist
    results_file_name = results_dir / (f"{dataset_label}-{n_input_vecs}-{todays_time}.json")
    index_param_sets = [
        {
        "dataset_filter_attributes": np.array(dataset_filter_attribute_range, dtype=np.int32),
        "n_attributes_per_datapoint": 10,
        "n_attr_idx_partitions": 32,
        "n_input_vecs": n_input_vecs,
        "n_clusters": int(math.sqrt(n_input_vecs)),
        "global_dim": 256,
        "rank": 32,
        "train_size": 5,
        "a0_selectivity": 0.01,
        "euclidean": True,
        "dataset_file": dataset_file,
        },
        {
        "dataset_filter_attributes": np.array(dataset_filter_attribute_range, dtype=np.int32),
        "n_attributes_per_datapoint": 10,
        "n_attr_idx_partitions": 32,
        "n_input_vecs": n_input_vecs,
        "n_clusters": int(math.sqrt(n_input_vecs)),
        "global_dim": 256,
        "rank": 32,
        "train_size": 5,
        "a0_selectivity": 0.5,
        "euclidean": True,
        "dataset_file": dataset_file,
        },
        {
        "dataset_filter_attributes": np.array(dataset_filter_attribute_range, dtype=np.int32),
        "n_attributes_per_datapoint": 10,
        "n_attr_idx_partitions": 32,
        "n_input_vecs": n_input_vecs,
        "n_clusters": int(math.sqrt(n_input_vecs)),
        "global_dim": 256,
        "rank": 32,
        "train_size": 5,
        "a0_selectivity": 0.99,
        "euclidean": True,
        "dataset_file": dataset_file,
        },
    ]
    
    index_param_sets = []
    n_index_param_sets = 6
    n_clusters = int(math.sqrt(n_input_vecs))
    for i in range(0, n_index_param_sets):
        if i == 0:
            selectivity = 0.01
        elif i == n_index_param_sets - 1:
            selectivity = 0.99
        else:
            selectivity = (1 / (n_index_param_sets - 1)) * i
        index_param_set = {
            "dataset_filter_attributes": np.array(dataset_filter_attribute_range, dtype=np.int32),
            "n_attributes_per_datapoint": 32,
            "n_attr_idx_partitions": 10,
            "n_input_vecs": n_input_vecs,
            "n_clusters": n_clusters,
            "global_dim": 256,
            "rank": 32,
            "train_size": 5,
            "a0_selectivity": selectivity,
            "euclidean": True,
            "dataset_file": dataset_file,
            "log_on": True
        }
        index_param_sets.append(index_param_set)
    
    query_indices = [random.randint(0, n_input_vecs) for i in range(20)]
    # search_param_sets = []
    # for filter_approach in ["indexing_avx", "postfilter", "hybrid_avx", "mixed"]:#, "indexing", "mixed", "postfilter" "hybrid_avx", "indexing_avx",
    #     initial_M = 50
    #     M_increment = 200
    #     initial_clusters_to_search = 5
    #     clusters_to_search_increment = 5
    #     for i in range(10):
    #         search_params = {
    #             "clusters_to_search": 0,
    #             "points_to_rerank": 2000,
    #             "k": 10,
    #             "M": initial_M + (i * M_increment) if filter_approach != "postfilter" else -1,
    #             "filter_attributes": [0],
    #             "filter_approach": filter_approach,
    #             "exact_search_approach": "prefilter_avx",
    #             "n_repeat_runs": 1,
    #             "query_indices": query_indices,
    #             "label": filter_approach
    #         }
    #         if (filter_approach == "postfilter"):
    #             search_params["clusters_to_search"] = initial_clusters_to_search + (i * clusters_to_search_increment)
    #         search_param_sets.append(search_params)
        
    
    # experimenter parameters
    n_repeat_runs = 1
    verbose = False
    # exact_search_approach = "postfilter"

    try:
        with open(results_file_name, "r", encoding="utf-8") as f:
            experiment_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # print("File not found.")
        experiment_data = {}
    for key in experiment_data.keys():
        print(json.loads(key), ": ", experiment_data[key])

    for (idx, index_param_set) in enumerate(index_param_sets):
        
        # building the index
        print(f"Index parameters:\nn_attr_idx_partitions = {index_param_set["n_attr_idx_partitions"]}\na0_selectivity = {index_param_set["a0_selectivity"]}\nn_input_vecs = {n_input_vecs}\nn_clusters = {index_param_set["n_clusters"]}\nglobal_dim = {index_param_set["global_dim"]}\nrank = {index_param_set["rank"]}\ntrain_size = {index_param_set["train_size"]}\neuclidean = {index_param_set["euclidean"]}\n\n")
        start_time = time.process_time()
        index_res = c_lib.build_index(
            index_param_set["dataset_filter_attributes"].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            len(index_param_set["dataset_filter_attributes"]),
            index_param_set["n_attributes_per_datapoint"],
            index_param_set["n_attr_idx_partitions"],
            index_param_set["a0_selectivity"],
            index_param_set["n_input_vecs"], 
            index_param_set["n_clusters"], 
            index_param_set["global_dim"], 
            index_param_set["rank"], 
            index_param_set["train_size"], 
            index_param_set["euclidean"],
            "hdf5" in index_param_set["dataset_file"],
            ctypes.c_char_p(index_param_set["dataset_file"].encode('utf-8'))

        )
        end_time = time.process_time()
        elapsed_time = end_time - start_time
        print("Time taken to build index: ", elapsed_time)
        index_param_set["dataset_filter_attributes"] = index_param_set["dataset_filter_attributes"].tolist()
        index_param_dump = json.dumps(index_param_set, sort_keys=True, ensure_ascii=False)
        this_results_dict = {}
        outputs = {}
        fig, axs = plt.subplots(1, 2)
        exact_latency = 0
        
        initial_M = 10
        M_increment = 20
        initial_clusters_to_search = 10
        clusters_to_search_increment = 20
        filter_approaches = ["hybrid_avx", "postfilter", "indexing_avx", "mixed"] #, "indexing", "mixed", "postfilter" "hybrid_avx", "indexing_avx",
        
        for filter_approach in filter_approaches:
            i = 0
            recall = 0.0
            best_recall = 0.0
            best_recall_latency = None
            rounds_non_improving_recall = 0
            while recall < 0.999:
                if rounds_non_improving_recall >= 5:
                    print(f"Breaking out of loop for filter approach {filter_approach} after {rounds_non_improving_recall} rounds of same recall.")
                    break
                param_set = {
                    "clusters_to_search": min(index_param_set["n_clusters"], initial_clusters_to_search + (i * clusters_to_search_increment)) if filter_approach == "postfilter" else 0,
                    "points_to_rerank": 2000,
                    "k": 10,
                    "M": initial_M + math.ceil(i * M_increment * 100 * index_param_set["a0_selectivity"]) if filter_approach != "postfilter" else -1, # * 100 * index_param_set["a0_selectivity"]
                    "filter_attributes": [0],
                    "filter_approach": filter_approach,
                    "exact_search_approach": "postfilter" if i == 0 else "prefilter_avx",
                    "n_repeat_runs": 1,
                    "query_indices": query_indices,
                    "label": filter_approach
                }
                print(f"exact_search_approach: {param_set['exact_search_approach']}")
                print(f"Using {n_input_vecs} inputs and {param_set["filter_approach"]} filter method and {param_set["exact_search_approach"]} exact search approach.")
                print(f"Running experimenter with search parameters:\n\
                clusters_to_search = {param_set["clusters_to_search"]}\npoints_to_rerank = {param_set["points_to_rerank"]}\nk = {param_set["k"]}\nM = {param_set["M"]}\nfilter_attribute = {param_set["filter_attributes"]}\nfilter_approach = {param_set["filter_approach"]}\nexact_search_approach = {param_set["exact_search_approach"]}\n\n\
                experiment parameters:\nn_repeat_runs = {param_set["n_repeat_runs"]}\nn_query_indices = {len(param_set["query_indices"])}\n") #\nquery_indices = {param_set["query_indices"]}

                filter_approach_b_string = param_set["filter_approach"].encode('utf-8')
                exact_search_approach_b_string = param_set["exact_search_approach"].encode('utf-8')
                query_index_arr = np.array(param_set["query_indices"], dtype=np.int32)
                filter_attributes_arr = np.array(param_set["filter_attributes"], dtype=np.int32)

                # run the experiment
                start_time = time.process_time()
                param_recall = ctypes.c_float(0.)
                param_approx_latency = ctypes.c_int(0)
                param_exact_latency = ctypes.c_int(0)
                param_exact_filter_time = ctypes.c_int(0)
                param_filter_time = ctypes.c_int(0)
                c_lib.fast_filter_wrapper_profiled(
                    query_index_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    len(query_index_arr),
                    param_set["k"],
                    param_set["M"],
                    param_set["clusters_to_search"],
                    param_set["points_to_rerank"],
                    filter_attributes_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    len(filter_attributes_arr),
                    ctypes.c_char_p(filter_approach_b_string),
                    ctypes.c_char_p(exact_search_approach_b_string),
                    ctypes.byref(param_recall),
                    ctypes.byref(param_approx_latency),
                    ctypes.byref(param_exact_latency),
                    ctypes.byref(param_exact_filter_time),
                    ctypes.byref(param_filter_time),
                    verbose
                )
                if (abs(param_recall.value - recall) < 0.0001 or param_recall.value < best_recall):
                    rounds_non_improving_recall += 1
                else:
                    rounds_non_improving_recall = 0
                recall = param_recall.value
                if recall > best_recall:
                    best_recall = recall
                    best_recall_latency = param_approx_latency.value
                
                # if recall < best_recall and best_recall_latency is not None and param_approx_latency.value > best_recall_latency:
                #     print(f"Breaking out of loop for filter approach {filter_approach} as latency {param_approx_latency.value} is more than best recall latency {best_recall_latency}.")
                #     continue
                approx_latency = param_approx_latency.value
                exact_latency = param_exact_latency.value
                exact_filter_time = param_exact_filter_time.value
                filter_time = param_filter_time.value
                end_time = time.process_time()
                elapsed_time = end_time - start_time
                
                avg_approximate_search_latency = approx_latency
                avg_exact_search_latency = exact_latency
                avg_recall = recall
                avg_filter_time = filter_time
                avg_exact_filter_time = exact_filter_time
                print(bcolors.WARNING + "Average recall: " + str(avg_recall) + bcolors.ENDC)
                print(bcolors.OKBLUE + "Average exact (", param_set["exact_search_approach"], ") search latency: ", avg_exact_search_latency, " microseconds" + bcolors.ENDC)
                print(bcolors.OKBLUE + "Average exact (", param_set["exact_search_approach"], ") filter time: ", avg_exact_filter_time, " microseconds" + bcolors.ENDC)
                print(bcolors.OKCYAN + "Average approximate (", param_set["filter_approach"], ") search latency: ", avg_approximate_search_latency, " microseconds" + bcolors.ENDC)
                print("Average filter time: ", avg_filter_time, " microseconds")
                if param_set["filter_approach"] not in outputs:
                    outputs[param_set["filter_approach"]] = [{**param_set, "recall": avg_recall, "approx_latency": avg_approximate_search_latency, "exact_latency": avg_exact_search_latency, "exact_filter_time": avg_exact_filter_time, "exact_approach": param_set["exact_search_approach"], "filter_time": avg_filter_time}]
                else:
                    outputs[param_set["filter_approach"]].append({**param_set, "recall": avg_recall, "approx_latency": avg_approximate_search_latency, "exact_latency": avg_exact_search_latency, "exact_filter_time": avg_exact_filter_time, "exact_approach": param_set["exact_search_approach"], "filter_time": avg_filter_time})
                prev_filter_approach = param_set["filter_approach"]
                i += 1
                if param_set["clusters_to_search"] >= index_param_set["n_clusters"]:
                    break
            print(f"\nRan {i} experiments for filter approach {filter_approach}.\n")        
                
        for filter_approach in filter_approaches:
            all_recalls = [o["recall"] for o in outputs[filter_approach]]
            all_approximate_latencies = [o["approx_latency"] for o in outputs[filter_approach]]
            all_filter_times = [o["filter_time"] for o in outputs[filter_approach]]
            best_recall = max(all_recalls)
            best_recall_idx = all_recalls.index(best_recall)
            # print("recalls: ", (all_recalls))
            # print("all_approximate_latencies: ", (all_approximate_latencies))
            # print("all_filter_times: ", (all_filter_times))
            delete_idxs = []
            for i in range(len(all_recalls)):
                if (i != best_recall_idx):
                    regress = all_approximate_latencies[i] > all_approximate_latencies[best_recall_idx]
                    if (regress):
                        delete_idxs.append(i)
            for index in sorted(delete_idxs, reverse=True):
                del all_recalls[index]
                del all_approximate_latencies[index]
                del all_filter_times[index]

            exact_latencies_by_approach = {}
            exact_filter_times_by_approach = {}
            for output in outputs[filter_approach]:
                exact_approach_name = output.get("exact_approach", "exact")
                exact_latencies_by_approach.setdefault(exact_approach_name, []).append(output["exact_latency"])
                exact_filter_times_by_approach.setdefault(exact_approach_name, []).append(output["exact_filter_time"])

            
            axs[0].plot(all_recalls, all_filter_times, label=f"{filter_approach}-filteronly") # index_param_set["label"] if "label" in index_param_set else f"selectivity={index_param_set["a0_selectivity"]}"
            axs[1].plot(all_recalls, all_approximate_latencies, label=filter_approach) # index_param_set["label"] if "label" in index_param_set else f"selectivity={index_param_set["a0_selectivity"]}"
            for exact_approach_name, exact_latencies in exact_latencies_by_approach.items():
                exact_avg_latency = sum(exact_latencies) / len(exact_latencies)
                exact_avg_filter_time = sum(exact_filter_times_by_approach[exact_approach_name]) / len(exact_filter_times_by_approach[exact_approach_name])
                axs[0].axhline(y=exact_avg_filter_time, linestyle='--', label=exact_approach_name)
                axs[1].axhline(y=exact_avg_latency, linestyle='--', label=exact_approach_name)
            axs[0].set_yscale('log')
            axs[1].set_yscale('log')
            # print("all_recalls: ", all_recalls)
            # print("all_approximate_latencies: ", all_approximate_latencies)
            this_results_dict[filter_approach]= {"approximate_latencies": all_approximate_latencies, "exact_latencies_by_approach": exact_latencies_by_approach, "exact_filter_times_by_approach": exact_filter_times_by_approach, "recalls": all_recalls, "filter_times": all_filter_times}
        experiment_data[index_param_dump] = this_results_dict
        with open(results_file_name, 'w', encoding="utf-8") as f:
            json.dump(experiment_data, f)
        for ax in axs:
            labelLines(ax.get_lines(), align=False)
            ax.set_title(f"Latency, Recall, and {index_param_set["a0_selectivity"]:.3} Selectivity ({n_input_vecs} points)")
            ax.set_ylabel("Latency (μs)")
            ax.set_xlabel("Recall")
        todays_date = datetime.today().strftime('%Y-%m-%d')
        fig.savefig(f"../../figures/direct_figs/{todays_date}-{index_param_set["log_on"]}-{index_param_set["dataset_file"].split('.', 1)[0]}-{index_param_set["n_input_vecs"]}-recall-latency_a0{index_param_set["a0_selectivity"]}.png")
    manager = plt.get_current_fig_manager()
    manager.window.attributes('-zoomed', True)
    plt.show()


    
