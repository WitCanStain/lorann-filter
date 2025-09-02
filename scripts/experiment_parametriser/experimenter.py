# ctypes_test.py
import ctypes
import pathlib
from numpy.ctypeslib import ndpointer
import numpy as np
import time
import random
if __name__ == "__main__":
    # Load the shared library into ctypes
    random.seed(41)
    script_dir = pathlib.Path(__file__).resolve().parent
    libname = script_dir.parent / "cpp" / "libfilter.so"
    c_lib = ctypes.CDLL(libname)
    c_lib.build_index.restype = ctypes.c_bool
    c_lib.build_index.argtypes = ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool
    
    c_lib.filter.restype = ctypes.c_float
    c_lib.filter.argtypes = ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p

    c_lib.filter_proc.restype = ctypes.c_int
    c_lib.filter_proc.argtypes = ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p

    c_lib.filter_wrapper.restype = ctypes.c_float
    c_lib.filter_wrapper.argtypes = ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p

    c_lib.fast_filter_wrapper_profiled.restype = ctypes.c_float
    c_lib.fast_filter_wrapper_profiled.argtypes = ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p

    # index parameters
    n_attr_partitions = 10
    n_input_vecs = 999994 #999994
    n_clusters = 1024 # 1024 for full set
    global_dim = 256
    rank = 32
    train_size = 5
    euclidean = True
    
    query_indices = [random.randint(0, n_input_vecs) for i in range(200)]#[399529, 241926, 958223, 402175, 893348, 9781, 819157, 880067, 460738, 758298, 334374, 270022, 650928, 612145, 125639, 453611, 881900, 226359, 76249, 498268, 131075, 702495, 19438, 129779, 722313, 944585, 279510, 333237, 650012, 190935, 930905, 316057, 418856, 111895, 98062, 695562, 517225, 241595, 22717, 81649, 763585]
    #[random.randint(0, n_input_vecs) for i in range(100)]

    search_param_sets = [
        {
            "clusters_to_search": 64,
            "points_to_rerank": 2000,
            "k": 10,
            "filter_attributes": ["one", "two", "three"],
            "filter_approach": "prefilter",
            "exact_search_approach": "prefilter",
            "n_repeat_runs": 1,
            "query_indices": query_indices,
        },
        {
            "clusters_to_search": 64,
            "points_to_rerank": 2000,
            "k": 10,
            "filter_attributes": ["one", "two", "three"],
            "filter_approach": "indexing",
            "exact_search_approach": "prefilter",
            "n_repeat_runs": 1,
            "query_indices": query_indices,
        },
        # {
        #     "clusters_to_search": 64,
        #     "points_to_rerank": 2000,
        #     "k": 10,
        #     "filter_attributes": ["one"],
        #     "filter_approach": "indexing",
        #     "exact_search_approach": "postfilter",
        #     "n_repeat_runs": 1,
        #     "query_indices": query_indices,
        # }
    ]
    
    
    # experimenter parameters
    n_repeat_runs = 1
    
    # building the index
    start_time = time.process_time()
    index_res = c_lib.build_index(
        n_attr_partitions, 
        n_input_vecs, 
        n_clusters, 
        global_dim, 
        rank, 
        train_size, 
        euclidean
    )
    end_time = time.process_time()
    elapsed_time = end_time - start_time
    print("Time taken to build index: ", elapsed_time)


    print(f"Index parameters:\nn_attr_partitions = {n_attr_partitions}\nn_input_vecs = {n_input_vecs}\nn_clusters = {n_clusters}\nglobal_dim = {global_dim}\nrank = {rank}\ntrain_size = {train_size}\neuclidean = {euclidean}\n\n")
    for param_set in search_param_sets:
        print(f"Using {n_input_vecs} inputs and {param_set["filter_approach"]} filter method and {param_set["exact_search_approach"]} exact search approach.")
        print(f"Running experimenter with search parameters:\n\
    clusters_to_search = {param_set["clusters_to_search"]}\npoints_to_rerank = {param_set["points_to_rerank"]}\nk = {param_set["k"]}\nfilter_attribute = {param_set["filter_attributes"]}\nfilter_approach = {param_set["filter_approach"]}\nexact_search_approach = {param_set["exact_search_approach"]}\n\n\
    experiment parameters:\nn_repeat_runs = {param_set["n_repeat_runs"]}\nn_query_indices = {len(param_set["query_indices"])}\nquery_indices = {param_set["query_indices"]}\n\n")

        filter_attributes_b = [s.encode("utf-8") for s in param_set["filter_attributes"]]
        FilterArrayType = ctypes.c_char_p * len(filter_attributes_b)
        filter_attributes_c = FilterArrayType(*filter_attributes_b)
        filter_approach_b_string = param_set["filter_approach"].encode('utf-8')
        exact_search_approach_b_string = param_set["exact_search_approach"].encode('utf-8')
        arr = np.array(param_set["query_indices"], dtype=np.int32)

        # run the experiment
        
        start_time = time.process_time()
        for i in range(n_repeat_runs):
            avg_recall = c_lib.fast_filter_wrapper_profiled(
                arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), 
                len(arr), 
                param_set["k"], 
                param_set["clusters_to_search"], 
                param_set["points_to_rerank"], 
                filter_attributes_c,
                len(param_set["filter_attributes"]),
                ctypes.c_char_p(filter_approach_b_string), 
                ctypes.c_char_p(exact_search_approach_b_string)
            )

        end_time = time.process_time()
        elapsed_time = end_time - start_time
        print("Average recall: ", avg_recall)
        avg_elapsed_time_run = elapsed_time / (n_repeat_runs)
        avg_elapsed_time = elapsed_time / (n_repeat_runs * len(param_set["query_indices"]))
        # print("avg_elapsed_time per run: ", avg_elapsed_time_run)
    
