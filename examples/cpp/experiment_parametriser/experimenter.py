# ctypes_test.py
import ctypes
import pathlib

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
    c_lib.filter.restype = ctypes.c_int
    answer = c_lib.filter(k, n_attr_partitions, n_clusters, global_dim, rank, train_size, ctypes.c_bool(euclidean), clusters_to_search, point_to_rerank, ctypes.c_wchar_p(filter_attribute))
    print("response: ", answer)
