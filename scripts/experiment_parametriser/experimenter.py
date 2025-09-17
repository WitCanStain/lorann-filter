# ctypes_test.py
import ctypes
import pathlib
from numpy.ctypeslib import ndpointer
import numpy as np
import time
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load the shared library into ctypes
    random.seed(42)
    script_dir = pathlib.Path(__file__).resolve().parent
    libname = script_dir.parent / "cpp" / "libfilter.so"
    c_lib = ctypes.CDLL(libname)
    c_lib.build_index.restype = ctypes.c_bool
    c_lib.build_index.argtypes = (
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int, 
        ctypes.c_int, 
        ctypes.c_int, 
        ctypes.c_int, 
        ctypes.c_int, 
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_char_p
        )
    
    # c_lib.filter.restype = ctypes.c_float
    # c_lib.filter.argtypes = ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p

    # c_lib.filter_proc.restype = ctypes.c_int
    # c_lib.filter_proc.argtypes = ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p

    # c_lib.filter_wrapper.restype = ctypes.c_float
    # c_lib.filter_wrapper.argtypes = ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p

    c_lib.fast_filter_wrapper_profiled.restype = ctypes.c_float
    c_lib.fast_filter_wrapper_profiled.argtypes = (
        ctypes.POINTER(ctypes.c_int), # idxs
        ctypes.c_int, # n_idxs
        ctypes.c_int, # k
        ctypes.c_int, # cluster_to_search
        ctypes.c_int, # points_to_rerank
        ctypes.POINTER(ctypes.c_int), # int_filter_attributes
        ctypes.c_int, # n_filter_attributes
        ctypes.c_char_p, # filter_approach
        ctypes.c_char_p, # exact_search_approach
        ctypes.POINTER(ctypes.c_float), # recall
        ctypes.POINTER(ctypes.c_int), # approx_latency
        ctypes.POINTER(ctypes.c_int), # exact_latency
        ctypes.c_bool) # verbose
        


    # index parameters
    dataset_filter_attribute_range = [i for i in range(30)]
    dataset_filter_attributes = np.array(dataset_filter_attribute_range, dtype=np.int32)
    n_attributes_per_datapoint = 5
    n_attr_idx_partitions = 10
    n_input_vecs = 60000 #999994
    n_clusters = 1024 # 1024 for full set
    global_dim = 256
    rank = 32
    train_size = 5
    euclidean = True
    
    query_indices = [random.randint(0, n_input_vecs) for i in range(1)]#[399529, 241926, 958223, 402175, 893348, 9781, 819157, 880067, 460738, 758298, 334374, 270022, 650928, 612145, 125639, 453611, 881900, 226359, 76249, 498268, 131075, 702495, 19438, 129779, 722313, 944585, 279510, 333237, 650012, 190935, 930905, 316057, 418856, 111895, 98062, 695562, 517225, 241595, 22717, 81649, 763585]
    #[random.randint(0, n_input_vecs) for i in range(100)]

    search_param_sets = [
        {
            "clusters_to_search": 64,
            "points_to_rerank": 2000,
            "k": 10,
            "filter_attributes": [1],
            "filter_approach": "indexing",
            "exact_search_approach": "prefilter",
            "n_repeat_runs": 1,
            "query_indices": query_indices,
        },
        {
            "clusters_to_search": 64,
            "points_to_rerank": 2000,
            "k": 10,
            "filter_attributes": [1, 2],
            "filter_approach": "indexing",
            "exact_search_approach": "prefilter",
            "n_repeat_runs": 1,
            "query_indices": query_indices,
        },
        {
            "clusters_to_search": 64,
            "points_to_rerank": 2000,
            "k": 10,
            "filter_attributes": [1, 2, 3],
            "filter_approach": "indexing",
            "exact_search_approach": "prefilter",
            "n_repeat_runs": 1,
            "query_indices": query_indices,
        }
    ]
    
    
    # experimenter parameters
    n_repeat_runs = 1
    
    # building the index
    start_time = time.process_time()
    index_res = c_lib.build_index(
        dataset_filter_attributes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        len(dataset_filter_attributes),
        n_attr_idx_partitions,
        n_attributes_per_datapoint,
        n_input_vecs, 
        n_clusters, 
        global_dim, 
        rank, 
        train_size, 
        euclidean,
        True,
        ctypes.c_char_p("fashion-mnist-784-euclidean.hdf5".encode('utf-8'))

    )
    end_time = time.process_time()
    elapsed_time = end_time - start_time
    print("Time taken to build index: ", elapsed_time)
    outputs = []

    print(f"Index parameters:\nn_attr_idx_partitions = {n_attr_idx_partitions}\nn_input_vecs = {n_input_vecs}\nn_clusters = {n_clusters}\nglobal_dim = {global_dim}\nrank = {rank}\ntrain_size = {train_size}\neuclidean = {euclidean}\n\n")
    for param_set in search_param_sets:
        print(f"Using {n_input_vecs} inputs and {param_set["filter_approach"]} filter method and {param_set["exact_search_approach"]} exact search approach.")
        print(f"Running experimenter with search parameters:\n\
    clusters_to_search = {param_set["clusters_to_search"]}\npoints_to_rerank = {param_set["points_to_rerank"]}\nk = {param_set["k"]}\nfilter_attribute = {param_set["filter_attributes"]}\nfilter_approach = {param_set["filter_approach"]}\nexact_search_approach = {param_set["exact_search_approach"]}\n\n\
    experiment parameters:\nn_repeat_runs = {param_set["n_repeat_runs"]}\nn_query_indices = {len(param_set["query_indices"])}\n") #\nquery_indices = {param_set["query_indices"]}

        # filter_attributes_b = [s.encode("utf-8") for s in param_set["filter_attributes"]]
        # FilterArrayType = ctypes.c_char_p * len(filter_attributes_b)
        # filter_attributes_c = FilterArrayType(*filter_attributes_b)
        filter_approach_b_string = param_set["filter_approach"].encode('utf-8')
        exact_search_approach_b_string = param_set["exact_search_approach"].encode('utf-8')
        query_index_arr = np.array(param_set["query_indices"], dtype=np.int32)
        filter_attributes_arr = np.array(param_set["filter_attributes"], dtype=np.int32)

        # run the experiment
        
        start_time = time.process_time()
        recalls = []
        approx_latencies = []
        exact_latencies = []
        for i in range(n_repeat_runs):
            recall = ctypes.c_float(0.)
            approx_latency = ctypes.c_int(0)
            exact_latency = ctypes.c_int(0)
            c_lib.fast_filter_wrapper_profiled(
                query_index_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                len(query_index_arr),
                param_set["k"],
                param_set["clusters_to_search"],
                param_set["points_to_rerank"],
                filter_attributes_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                len(filter_attributes_arr),
                ctypes.c_char_p(filter_approach_b_string),
                ctypes.c_char_p(exact_search_approach_b_string),
                ctypes.byref(recall),
                ctypes.byref(approx_latency),
                ctypes.byref(exact_latency),
                True
            )
            recalls.append(recall.value)
            approx_latencies.append(approx_latency.value)
            exact_latencies.append(exact_latency.value)
        end_time = time.process_time()
        elapsed_time = end_time - start_time
        total_approx_latency = sum(approx_latencies)
        total_exact_latency = sum(exact_latencies)
        total_recall = sum(recalls)
        avg_approximate_search_latency = int(total_approx_latency) / n_repeat_runs
        avg_exact_search_latency = int(total_exact_latency) / n_repeat_runs
        avg_recall = total_recall / n_repeat_runs
        print("Average recall: ", avg_recall)
        print("Average exact search latency: ", avg_exact_search_latency, " microseconds")
        print("Average approximate search latency: ", avg_approximate_search_latency, " microseconds")
        outputs.append({**param_set, "recall": avg_recall, "approx_latency": avg_approximate_search_latency, "exact_latency": avg_exact_search_latency})


    all_recalls = [o["recall"] for o in outputs]
    all_approximate_latencies = [o["approx_latency"] for o in outputs]
    all_exact_latencies = [o["exact_latency"] for o in outputs]

    

    plt.plot(all_recalls, all_approximate_latencies)
    plt.ylabel("Latency")
    plt.xlabel("Recall")
    plt.savefig('../../figures/recall-latency.png')
    # plt.show()


    
