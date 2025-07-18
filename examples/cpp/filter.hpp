#include <string>
#ifdef _MSC_VER
    #define EXPORT_SYMBOL __declspec(dllexport)
#else
    #define EXPORT_SYMBOL
#endif

#ifdef __cplusplus
extern "C" {
#endif

EXPORT_SYMBOL int build_index(int n_attr_partitions, int n_clusters, int global_dim, int rank, int train_size, bool euclidean);

EXPORT_SYMBOL float filter(int q_idx, bool exact_search, int k,  int clusters_to_search, int points_to_rerank, std::string filter_attribute, std::string filter_approach);

EXPORT_SYMBOL int filter_proc(int n_attr_partitions, int n_clusters, int global_dim, int rank, int train_size, bool euclidean, int q_idx, bool exact_search, int k,  int clusters_to_search, int points_to_rerank, std::string filter_attribute, std::string filter_approach);

EXPORT_SYMBOL int filter_wrapper(int* idxs, int n_idxs, bool exact_search, int k,  int clusters_to_search, int points_to_rerank, std::string filter_attribute, std::string filter_approach);

EXPORT_SYMBOL int fast_filter_wrapper_profiled(int* idxs, int n_idxs, int k,  int clusters_to_search, int points_to_rerank, std::string filter_attribute, std::string filter_approach, std::string exact_search_approach);

#ifdef __cplusplus
}
#endif