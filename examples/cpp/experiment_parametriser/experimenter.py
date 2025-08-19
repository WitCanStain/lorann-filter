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
    libname = pathlib.Path().absolute() / "../libfilter.so"
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
    n_attr_partitions = 30
    n_input_vecs=999994#999994
    n_clusters = 1024 # 1024 for full set
    global_dim = 256
    rank = 32
    train_size = 5
    euclidean = True

    # search parameters
    clusters_to_search = 64
    points_to_rerank = 2000
    k = 10
    filter_attributes = ["fourteen", "one", "three"]
    filter_attributes_b = [s.encode("utf-8") for s in filter_attributes]
    FilterArrayType = ctypes.c_char_p * len(filter_attributes_b)
    filter_attributes_c = FilterArrayType(*filter_attributes_b)
    filter_approach = "indexing"
    exact_search_approach = "prefilter"
    # filter_attribute_b_string = filter_attributes.encode('utf-8')
    filter_approach_b_string = filter_approach.encode('utf-8')
    exact_search_approach_b_string = exact_search_approach.encode('utf-8')
    
    # experimenter parameters
    n_repeat_runs = 1
    n_query_indices = 100
    query_indices = [random.randint(0, n_input_vecs) for i in range(n_query_indices)]#[i for i in range(400,500)]
    arr = np.array(query_indices, dtype=np.int32)

    print(f"Running experimenter with index parameters:\nn_attr_partitions = {n_attr_partitions}\nn_input_vecs = {n_input_vecs}\nn_clusters = {n_clusters}\nglobal_dim = {global_dim}\nrank = {rank}\ntrain_size = {train_size}\neuclidean = {euclidean}\n\nSearch parameters:\n\
clusters_to_search = {clusters_to_search}\npoints_to_rerank = {points_to_rerank}\nk = {k}\nfilter_attribute = {filter_attributes}\nfilter_approach = {filter_approach}\nexact_search_approach = {exact_search_approach}\n\n\
experiment parameters:\nn_repeat_runs = {n_repeat_runs}\nn_query_indices = {n_query_indices}\nquery_indices = {query_indices}\n\n")

    # run the experiment
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
    print(f"Using {n_input_vecs} inputs and {filter_approach} filter method and {exact_search_approach} exact search approach.")
    end_time = time.process_time()
    elapsed_time = end_time - start_time
    print("Time taken to build index: ", elapsed_time)

    start_time = time.process_time()
    for i in range(n_repeat_runs):
        avg_recall = c_lib.fast_filter_wrapper_profiled(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), 
            len(arr), 
            k, 
            clusters_to_search, 
            points_to_rerank, 
            filter_attributes_c,
            len(filter_attributes),
            ctypes.c_char_p(filter_approach_b_string), 
            ctypes.c_char_p(exact_search_approach_b_string)
        )

    end_time = time.process_time()
    elapsed_time = end_time - start_time
    print("Average recall: ", avg_recall)
    avg_elapsed_time_run = elapsed_time / (n_repeat_runs)
    avg_elapsed_time = elapsed_time / (n_repeat_runs * n_query_indices)
    # print("avg_elapsed_time: ", avg_elapsed_time)
    print("avg_elapsed_time per run: ", avg_elapsed_time_run)
    
