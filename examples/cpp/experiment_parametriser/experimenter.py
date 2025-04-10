# ctypes_test.py
import ctypes
import pathlib
from numpy.ctypeslib import ndpointer
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
    filter_attribute = "brown"
    filter_approach = "postfilter"
    filter_attribute_b_string = filter_attribute.encode('utf-8')
    filter_approach_b_string = filter_approach.encode('utf-8')
    c_lib.filter.restype = ndpointer(dtype=ctypes.c_int, shape=(k,))
    c_lib.filter.argtypes = ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p
    answer = c_lib.filter(k, n_attr_partitions, n_clusters, global_dim, rank, train_size, ctypes.c_bool(euclidean), clusters_to_search, points_to_rerank, ctypes.c_char_p(filter_attribute_b_string), ctypes.c_char_p(filter_approach_b_string))
    print("response: ", answer)
