#pragma once

#include <stdexcept>
#include <vector>
#include <avx_bitset.h>
#include "clustering.h"
#include "serialization.h"
#include "utils.h"
#include <boost/container/flat_set.hpp>
#include <boost/dynamic_bitset.hpp>
#include <bitset_matrix.h>
#include <cstdint>
#include "absl/container/flat_hash_map.h"

#define KMEANS_ITERATIONS 10
#define KMEANS_MAX_BALANCE_DIFF 16
#define SAMPLED_POINTS_PER_CLUSTER 256
#define GLOBAL_DIM_REDUCTION_SAMPLES 16384

namespace Lorann {


// typedef BitsetMatrix attribute_set;
typedef uint32_t attribute_set;
//typedef boost::container::flat_set<int> attribute_set;
typedef absl::flat_hash_map<uint32_t, std::vector<int>> attribute_data_map;
typedef absl::flat_hash_map<uint32_t, std::vector<uint32_t>> int32_attribute_data_map;
// typedef std::unordered_map<BitsetMatrix::BitsetView, std::vector<int>, BitsetMatrix::BitsetViewHash> attribute_data_map;
// typedef std::unordered_map<BitsetMatrix::BitsetView, std::vector<uint32_t>, BitsetMatrix::BitsetViewHash> int32_attribute_data_map;

class LorannBase {
 public:
  LorannBase(float *data, int m, int d, int n_clusters, int global_dim, BitsetMatrix& attributes, std::vector<int>& attribute_idxs, std::vector<std::uint32_t>& attribute_ints,   int rank, int train_size,
            bool euclidean, bool balanced)
      : _data(data),
        _n_samples(m),
        _dim(d),
        _n_clusters(n_clusters),
        _global_dim(global_dim <= 0 ? d : std::min(global_dim, d)),
        _attributes(attributes),
        _attribute_idxs(attribute_idxs),
        _attribute_ints(attribute_ints),
        _max_rank(std::min(rank, d)),
        _train_size(train_size),
        _euclidean(euclidean),
        _balanced(balanced),
        _n_attributes(attribute_idxs.size()) {
    if (d < 64) {
      throw std::invalid_argument(
          "LoRANN is meant for high-dimensional data: the dimensionality should be at least 64.");
    }

    if (m != attributes.size()) {
      throw std::invalid_argument(
          "Attributes have a different number of elements than the number of samples!");
    }

    LORANN_ENSURE_POSITIVE(m);
    LORANN_ENSURE_POSITIVE(n_clusters);
    LORANN_ENSURE_POSITIVE(rank);
    LORANN_ENSURE_POSITIVE(train_size);
  }

  /**
   * @brief Get the number of samples in the index.
   *
   * @return int
   */
  inline int get_n_samples() const { return _n_samples; }

  /**
   * @brief Get the dimensionality of the vectors in the index.
   *
   * @return int
   */
  inline int get_dim() const { return _dim; }

  /**
   * @brief Get the number of clusters.
   *
   * @return int
   */
  inline int get_n_clusters() const { return _n_clusters; }

  /**
   * @brief Get whether the index uses the Euclidean distance as the dissimilarity measure.
   *
   * @return bool
   */
  inline bool get_euclidean() const { return _euclidean; }

  /**
   * @brief Get a pointer to a vector in the index.
   *
   * @param idx The index to the vector.
   * @param out The output buffer.
   */
  inline void get_vector(const int idx, float *out) {
    if (idx < 0 || idx >= _n_samples) {
      throw std::invalid_argument("Invalid index");
    }

    std::memcpy(out, _data + idx * _dim, _dim);
  }

  /**
   * @brief Compute the dissimilarity between two vectors.
   *
   * @param u First vector
   * @param v Second vector

   * @return float The dissimilarity
   */
  inline float get_dissimilarity(const float *u, const float *v) {
    if (_euclidean) {
      return squared_euclidean(u, v, _dim);
    } else {
      return -dot_product(u, v, _dim);
    }
  }

  /**
   * @brief Build the index.
   *
   * @param approximate Whether to turn on various approximations during index construction.
   * Defaults to true. Setting approximate to false slows down the index construction but can
   * slightly increase the recall, especially if no exact re-ranking is used in the query phase.
   * @param num_threads Number of CPU threads to use (set to -1 to use all cores)
   */
  void build(const bool approximate = true, int num_threads = -1, int n_attribute_partitions=10) {
    build(_data, _n_samples, n_attribute_partitions, approximate, num_threads);
  }

  virtual void build(const float *query_data, const int query_n, const int n_attribute_partitions, const bool approximate,
                     int num_threads) {}

  virtual void search(const float *data, const int k, const int M, const int clusters_to_search,
                      const int points_to_rerank, int *idx_out, attribute_set filter_attributes, uint32_t filter_attributes_int, std::string filter_approach, std::chrono::microseconds* duration, float *dist_out = nullptr, bool verbose=false) const {}

  virtual ~LorannBase() {}

  /**
   * @brief Perform exact k-nn search using the index.
   *
   * @param q The query vector (dimension must match the index data dimension)
   * @param k The number of nearest neighbors
   * @param out The index output array of length k
   * @param dist_out The (optional) distance output array of length k
   */
  void exact_search(const float *q, int k, int *out, const attribute_set& filter_attributes, uint32_t filter_attributes_int,std::string filter_approach, float *dist_out = nullptr, std::chrono::microseconds *filter_duration = nullptr) const {
    float *data_ptr = _data;
    int n_datapoints;
    std::vector<int> attribute_data_idxs;
    std::chrono::microseconds duration_filter = std::chrono::microseconds(0);
    if (filter_approach == "indexing") {
      auto start_filter = std::chrono::high_resolution_clock::now();
      attribute_set smallest_idx;
      int smallest_idx_size = _n_samples;
      for (int attr = 0; attr < _n_attributes; ++attr) {
        if (is_bit_set(filter_attributes_int, attr)) {
          attribute_set attr_set = _attribute_index_map[attr];
          int attr_idx_size = _attribute_data_map[attr_set].size();
          if (attr_idx_size <= smallest_idx_size) {
            smallest_idx = attr_set;
            smallest_idx_size = _attribute_data_map[smallest_idx].size();
          }
        }
      }
      std::vector<int>& attribute_idx = _attribute_data_map[smallest_idx];
      attribute_data_idxs.reserve(attribute_idx.size());
      for (int i = 0; i < attribute_idx.size(); ++i) { // for each data point in the smallest index which the datapoints belong to, check if the data point has the other filter attributes as well, if yes then add to filtered list.
        int true_idx = attribute_idx[i];
        bool filters_match = _attributes.matches_int(true_idx, filter_attributes);
        // bool filters_match = (_attributes[true_idx] & filter_attributes) == filter_attributes;
        // for (const auto& attr: filter_attributes) {
        //   if (!_attributes[true_idx].count(attr)) {
        //     filters_match = false;
        //     break;
        //   }
        // }
        if (filters_match) {
          attribute_data_idxs.push_back(true_idx);
        }
      }
      // attribute_data_idxs = _attribute_data_map[filter_attributes];
      n_datapoints = attribute_data_idxs.size();
      auto stop_filter = std::chrono::high_resolution_clock::now();
      duration_filter = std::chrono::duration_cast<std::chrono::microseconds>(stop_filter - start_filter);
    } else if (filter_approach == "prefilter") {
      auto start_filter = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < _n_samples; i++) {
        bool filters_match = _attributes.matches_int(i, filter_attributes_int);
        if (filters_match) attribute_data_idxs.push_back(i);
      }
      n_datapoints = attribute_data_idxs.size();
      auto stop_filter = std::chrono::high_resolution_clock::now();
      duration_filter = std::chrono::duration_cast<std::chrono::microseconds>(stop_filter - start_filter);
    } else if (filter_approach == "prefilter_avx") {
      auto start_filter = std::chrono::high_resolution_clock::now();
      std::vector<uint16_t> masks(_n_samples);
      int num_blocks = build_subset_masks_avx512(_attribute_ints.data(), _n_samples, filter_attributes_int, masks.data());
      iterate_hits_from_masks(masks.data(), num_blocks, attribute_data_idxs);
      n_datapoints = attribute_data_idxs.size();
      auto stop_filter = std::chrono::high_resolution_clock::now();
      duration_filter = std::chrono::duration_cast<std::chrono::microseconds>(stop_filter - start_filter);
    } else if (filter_approach == "postfilter") {
      n_datapoints = _n_samples;
    } else {
      throw std::invalid_argument("filter_approach must be one of 'indexing', 'prefilter', or 'postfilter'");
    }
    if (filter_approach != "postfilter" && attribute_data_idxs.size() == 0) {
        throw std::runtime_error("No matches found for filter attributes!");
    }
    Vector dist(n_datapoints); // used to be above conditional
    if (_euclidean) {
      for (int i = 0; i < n_datapoints; ++i) {
        dist[i] = squared_euclidean(q, data_ptr + ((filter_approach == "postfilter") ? i : attribute_data_idxs[i]) * _dim, _dim);
      }
    } else {
      for (int i = 0; i < n_datapoints; ++i) {
        dist[i] = -dot_product(q, data_ptr + ((filter_approach == "postfilter") ? i : attribute_data_idxs[i]) * _dim, _dim);
      }
    }
    /* optimization for the special case k = 1 */
    if (k == 1) {
      Eigen::MatrixXf::Index index;
      dist.minCoeff(&index);
      out[0] = index;
      if (dist_out) dist_out[0] = dist[index];
      return;
    }
    const int knn_buffer = filter_approach == "postfilter" ? 8 : 1;
    const int final_k = k;
    if (k > n_datapoints) {
      k = n_datapoints;
    }
    Eigen::VectorXi shuffled_out(k * knn_buffer);
    select_k(k * knn_buffer, shuffled_out.data(), n_datapoints, NULL, dist.data(), dist_out, true);
    if (filter_approach == "postfilter") {
      auto start_filter = std::chrono::high_resolution_clock::now();
      std::vector<int> matched_idxs;
      matched_idxs.reserve(k);
      for (int i = 0; i < k * knn_buffer; ++i) {
        bool filters_match = _attributes.matches_int(shuffled_out[i], filter_attributes_int);
        if (filters_match) {
          matched_idxs.push_back(shuffled_out[i]);
        }
      }
      int matched_k = matched_idxs.size();
      int new_k = k;
      std::vector<std::vector<int>*> ptr_vec;
      while (matched_k < k) { // if not enough datapoints are found in k results, double it and search again
        new_k = new_k * 2 > _n_samples ? _n_samples : new_k * 2;
        Eigen::VectorXi new_out(new_k);
        matched_idxs.clear();
        select_k(new_k, new_out.data(), n_datapoints, NULL, dist.data(), dist_out, true);
        for (int i = 0; i < new_k; ++i) {
          bool filters_match = _attributes.matches_int(new_out[i], filter_attributes_int);
          if (filters_match) {
            matched_idxs.push_back(new_out[i]);
            if (matched_idxs.size() >= k) break;
          }
        }
        matched_k = matched_idxs.size();
        if (matched_k < k && new_k == _n_samples) {
          std::cout << "could not find enough samples (exact search, found " << matched_k << ")" << std::endl;
          break;
        }
      }
      auto stop_filter = std::chrono::high_resolution_clock::now();
      duration_filter = std::chrono::duration_cast<std::chrono::microseconds>(stop_filter - start_filter);
      if (matched_k >= k) {
        for (int i = 0; i < k; ++i) {
          out[i] = matched_idxs[i];
        }
      }
    } else {
      for (int i = 0; i < k; i++) {
        out[i] = attribute_data_idxs[shuffled_out[i]];
      }
    }
    for (int i = k; i < final_k; ++i) {
      out[i] = -1;
      if (dist_out) dist_out[i] = std::numeric_limits<float>::infinity();
    }

    if (filter_duration) {
      *filter_duration = duration_filter;
    }
    
    
  }

 protected:
  /* default constructor should only be used for serialization */
  LorannBase() = default;

  void select_final(const float *query_x, const int k, const int points_to_rerank, const int size,
                    const int *all_idxs, const float *all_distances, int *idx_out,
                    float *dist_out) const {
    const int n_selected = std::min(std::max(k, points_to_rerank), size);
    if (points_to_rerank == 0) {
      select_k(n_selected, idx_out, size, all_idxs, all_distances, dist_out, true);

      if (dist_out && _euclidean) {
        float query_norm = 0;
        for (int i = 0; i < _dim; ++i) {
          query_norm += query_x[i] * query_x[i];
        }
        for (int i = 0; i < n_selected; ++i) {
          dist_out[i] += query_norm;
        }
      }

      for (int i = n_selected; i < k; ++i) {
        idx_out[i] = -1;
        if (dist_out) dist_out[i] = std::numeric_limits<float>::infinity();
      }

      return;
    }
    std::vector<int> final_select(n_selected);
    select_k(n_selected, final_select.data(), size, all_idxs, all_distances); // by the end of this function call final_select has the actual indexes of the datapoints in attributes vector
    reorder_exact(query_x, k, final_select, idx_out, dist_out);
  }

  void reorder_exact(const float *q, int k, const std::vector<int> &in, int *out,
                     float *dist_out = nullptr) const {
    const int n = in.size();
    Vector dist(n);

    const float *data_ptr = _data;
    if (_euclidean) {
      for (int i = 0; i < n; ++i) {
        dist[i] = squared_euclidean(q, data_ptr + in[i] * _dim, _dim);
      }
    } else {
      for (int i = 0; i < n; ++i) {
        dist[i] = dot_product(q, data_ptr + in[i] * _dim, _dim);
      }
    }

    /* optimization for the special case k = 1 */
    if (k == 1) {
      Eigen::MatrixXf::Index index;
      dist.minCoeff(&index);
      out[0] = in[index];
      if (dist_out) dist_out[0] = dist[index];
      return;
    }
    // std::cout << "reorder_exact n: " << n << std::endl;
    const int final_k = k;
    if (k > n) {
      k = n;
    }
    // for (int i = 0; i < in.size(); i++) {
    //   std::cout << "r["<<i<<"]: " << in[i] << " ";
    // }
    select_k(k, out, in.size(), in.data(), dist.data(), dist_out, true);
    for (int i = k; i < final_k; ++i) {
      out[i] = -1;
      if (dist_out) dist_out[i] = std::numeric_limits<float>::infinity();
    }
  }

  std::vector<std::vector<int>> clustering(KMeans &global_clustering, const float *data,
                                           const int n, const float *train_data, const int train_n,
                                           const bool approximate, int num_threads, std::vector<uint32_t> attribute_partition_sets) {
    const int to_sample = SAMPLED_POINTS_PER_CLUSTER * _n_clusters;
    if (!_balanced && approximate && to_sample < 0.5f * n) {
      /* sample points for k-means */
      const RowMatrix sampled =
          sample_rows(Eigen::Map<const RowMatrix>(data, n, _global_dim), to_sample);
      (void)global_clustering.train(sampled.data(), sampled.rows(), sampled.cols(), num_threads);
      _cluster_map = global_clustering.assign(data, n, 1);
    } else {
      _cluster_map = global_clustering.train(data, n, _global_dim, num_threads);
    }

    /* Create filter attribute index maps for clusters for approximate search */
    // _attribute_partition_sets = std::move(attribute_partition_sets);
    for (int i = 0; i < _cluster_map.size(); i++) {
      attribute_data_map this_cluster_attribute_data_map;
      int32_attribute_data_map this_cluster_attribute_int_data_map;
      attribute_data_map this_cluster_reverse_index_map;
      std::vector<uint32_t> this_cluster_attributes_int;
      
      std::vector<int> cluster = _cluster_map[i];
      for (int j = 0; j < cluster.size(); j++) {
        // this_cluster_attributes.push_back(_attributes[cluster[j]]);
        this_cluster_attributes_int.push_back(_attribute_ints[cluster[j]]);
      }
      for (int k = 0; k < cluster.size(); k++) {
        if (this_cluster_attributes_int[k] != _attributes.get_attribute_int(cluster[k])) {
          std::cout << "cluster " << i << ", k " << k << " cluster[" << k << "] " << cluster[k] << " - mismatch" << std::endl;
        }
      }
      // std::cout << "Clustering cluster " << i << " with size " << cluster.size() << std::endl;
      for (uint32_t attr_set_int : attribute_partition_sets) {
        std::vector<int> attribute_data_idx_vec; // vector of indexes of datapoints that have at least one of the attributes in attribute_subvec_set
        std::vector<uint32_t> attribute_int_data_idx_vec;
        attribute_data_idx_vec.reserve(cluster.size());
        int non_applicable_points = 0;
        std::vector<int> this_cluster_reverse_index;
        for (int i = 0; i<cluster.size(); i++) {
          int idx = cluster[i];
          bool any_match = _attributes.any_match_int(idx, attr_set_int);
          if (any_match) {
            attribute_data_idx_vec.push_back(idx);
            attribute_int_data_idx_vec.push_back(_attribute_ints[idx]);
            this_cluster_reverse_index.push_back(i);
          } else {
            non_applicable_points++;
          }
          // for (const auto& attr: attr_bitset) { // for each attribute in the attribute set
          //   if (_attributes[idx].count(attr)) { // check if that datapoint has any of the current filter attributes
          //     attribute_data_idx_vec.push_back(idx); // if yes, add it to the map for the corresponding attribute set
          //     break; // break so we do not add the same datapoint multiple times for different attributes
          //   } else {
          //     non_applicable_points++;
          //   }
          // }
        }
        this_cluster_attribute_data_map.insert({attr_set_int, attribute_data_idx_vec});
        this_cluster_attribute_int_data_map.insert({attr_set_int, attribute_int_data_idx_vec});
        this_cluster_reverse_index_map.insert({attr_set_int, this_cluster_reverse_index});
      }
      _cluster_attribute_data_maps.push_back(this_cluster_attribute_data_map); // add cluster attribute data map to vector of all cluster attribute data maps
      _cluster_attribute_int_data_maps.push_back(this_cluster_attribute_int_data_map);
      _cluster_reverse_index_maps.push_back(this_cluster_reverse_index_map);
      _cluster_attribute_int_map.push_back(this_cluster_attributes_int);
    }

    int n_total_index_size = 0;
    for (const auto& attribute_data_map: _cluster_attribute_data_maps) {
      for (const auto& idx_vec: attribute_data_map) {
        n_total_index_size += idx_vec.second.size();
      }
    }
    std::cout << "Total index size: " << n_total_index_size << std::endl;

    /* printouts to help see if indexes were built correctly */
    // std::cout << "cluster map size: " << _cluster_map.size() << std::endl;
    // std::vector<int> cluster = _cluster_map[0];
    // attribute_data_map this_cluster_attribute_data_map = _cluster_attribute_data_maps[0];
    // std::vector<std::string> attribute_string_vec = {"brown"};
    // std::set<std::string> attribute_key(attribute_string_vec.begin(), attribute_string_vec.end());
    // std::vector<int> colour_partition_data_idxs = this_cluster_attribute_data_map[attribute_key];
    // std::cout << "colour_partition_data_idxs size: " << colour_partition_data_idxs.size() << std::endl;
    // for (int i = 0; i < colour_partition_data_idxs.size(); i++) {
    //   std::cout << colour_partition_data_idxs[i] << " - ";
    //   std::cout << "corresponding attribute: " << _attributes[colour_partition_data_idxs[i]] << "|";
    // }

    return global_clustering.assign(train_data, train_n, _train_size);
  }

  std::vector<std::vector<int>> clustering(KMeans &global_clustering, const float *data,
                                          const int n, const float *train_data, const int train_n,
                                          const bool approximate, int num_threads) {
    const int to_sample = SAMPLED_POINTS_PER_CLUSTER * _n_clusters;
    if (!_balanced && approximate && to_sample < 0.5f * n) {
      /* sample points for k-means */
      const RowMatrix sampled =
      sample_rows(Eigen::Map<const RowMatrix>(data, n, _global_dim), to_sample);
      (void)global_clustering.train(sampled.data(), sampled.rows(), sampled.cols(), num_threads);
      _cluster_map = global_clustering.assign(data, n, 1);
    } else {
      _cluster_map = global_clustering.train(data, n, _global_dim, num_threads);
    }
    return global_clustering.assign(train_data, train_n, _train_size);
  }

  friend class cereal::access;

  template <class Archive>
  void save(Archive &ar) const {
    ar(_n_samples);
    ar(_dim);
    ar(cereal::binary_data(_data, sizeof(float) * _n_samples * _dim), _n_clusters, _global_dim,
       _max_rank, _train_size, _euclidean, _balanced, _cluster_map, _global_centroid_norms);
  }

  template <class Archive>
  void load(Archive &ar) {
    ar(_n_samples);
    ar(_dim);

    _owned_data = Vector(_n_samples * _dim);
    _data = _owned_data.data();

    ar(cereal::binary_data(_data, sizeof(float) * _n_samples * _dim), _n_clusters, _global_dim,
       _max_rank, _train_size, _euclidean, _balanced, _cluster_map, _global_centroid_norms);

    _cluster_sizes = Eigen::VectorXi(_n_clusters);

    for (int i = 0; i < _n_clusters; ++i) {
      _cluster_sizes(i) = static_cast<int>(_cluster_map[i].size());
    }

    if (_euclidean) {
      Eigen::Map<RowMatrix> train_mat(_data, _n_samples, _dim);
      _data_norms = train_mat.rowwise().squaredNorm();
    }
  }

  float *_data;
  Vector _owned_data;

  int _n_samples;
  int _dim;
  int _n_clusters;
  int _global_dim;
  int _max_rank; /* max rank (r) for the RRR parameter matrices */
  int _train_size;
  bool _euclidean;
  bool _balanced;
  size_t _n_attributes;

  BitsetMatrix _attributes;
  std::vector<int> _attribute_idxs;
  std::vector<uint32_t> _attribute_ints;
  mutable attribute_data_map _attribute_data_map;
  mutable int32_attribute_data_map _attribute_int_data_map;
  mutable std::unordered_map<int, attribute_set> _attribute_index_map;
  mutable std::vector<attribute_data_map> _cluster_attribute_data_maps;
  mutable std::vector<attribute_data_map> _cluster_reverse_index_maps;
  mutable std::vector<int32_attribute_data_map> _cluster_attribute_int_data_maps;
  mutable std::vector<BitsetMatrix> _attribute_partition_sets;
  /* vector of points assigned to a cluster, for each cluster */
  std::vector<std::vector<int>> _cluster_map;
  std::vector<std::vector<uint32_t>> _cluster_attribute_int_map;

  Eigen::VectorXf _global_centroid_norms;
  Eigen::VectorXi _cluster_sizes;
  Vector _data_norms;
};

}  // namespace Lorann