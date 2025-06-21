# ctypes_test.py
import ctypes
import pathlib
from numpy.ctypeslib import ndpointer
import time
if __name__ == "__main__":
    # Load the shared library into ctypes
    libname = pathlib.Path().absolute() / "../libfilter.so"
    c_lib = ctypes.CDLL(libname)
    k = 10
    n_attr_partitions = 10
    n_clusters = 1024
    global_dim = 256
    rank = 32
    train_size = 5
    euclidean = True
    clusters_to_search = 64
    points_to_rerank = 2000
    exact_search = False

    c_lib.build_index.restype = ctypes.c_bool
    c_lib.build_index.argtypes = ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool
    
    c_lib.filter.restype = ndpointer(dtype=ctypes.c_int, shape=(k,))
    c_lib.filter.argtypes = ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p

    
    
    filter_attribute = "yellow"
    filter_approach = "postfilter"
    filter_attribute_b_string = filter_attribute.encode('utf-8')
    filter_approach_b_string = filter_approach.encode('utf-8')
    
    n_repeat_runs = 1
    data_dim = 300
    query_indices = [30000]#[i for i in range(400,500)]

    start_time = time.process_time()
    index_res = c_lib.build_index(n_attr_partitions, n_clusters, global_dim, rank, train_size, euclidean)
    print("Index building result: ", index_res)
    end_time = time.process_time()
    elapsed_time = end_time - start_time
    print("Time taken to build index: ", elapsed_time)


    start_time = time.process_time()
    for i in range(n_repeat_runs):
        for query_index in query_indices:
            answer = c_lib.filter(query_index, exact_search, k, clusters_to_search, points_to_rerank, ctypes.c_char_p(filter_attribute_b_string), ctypes.c_char_p(filter_approach_b_string))
            # print("response: ", answer)

    end_time = time.process_time()
    elapsed_time = end_time - start_time
    avg_elapsed_time = elapsed_time / (n_repeat_runs * len(query_indices))
    print("avg_elapsed_time: ", avg_elapsed_time)
    
