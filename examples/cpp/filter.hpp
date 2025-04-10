#include <string>
#ifdef _MSC_VER
    #define EXPORT_SYMBOL __declspec(dllexport)
#else
    #define EXPORT_SYMBOL
#endif

#ifdef __cplusplus
extern "C" {
#endif

EXPORT_SYMBOL int filter(int k, int n_attr_partitions, int n_clusters, int global_dim, int rank, int train_size, bool euclidean, int clusters_to_search, int points_to_rerank, std::string filter_attribute);

#ifdef __cplusplus
}
#endif