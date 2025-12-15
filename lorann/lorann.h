#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Dense>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <bitset>
#include "lorann_base.h"
#include "quant.h"
#include "utils.h"
#include <bitset_matrix.h>
#include <avx_bitset.h>
#if defined(LORANN_USE_MKL)
#include "mkl.h"
#elif defined(LORANN_USE_OPENBLAS)
#include <cblas.h>
#endif

namespace Lorann {

template <typename DataQuantizer = SQ8Quantizer, typename QueryQuantizer = SQ8Quantizer>
class Lorann : public LorannBase {
 public:
  /**
   * @brief Construct a new Lorann object
   *
   * NOTE: The constructor does not build the actual index.
   *
   * @param data The data matrix as a float array of size $m \\times d$
   * @param m Number of points (rows) in the data matrix
   * @param d Number of dimensions (cols) in the data matrix
   * @param n_clusters Number of clusters. In general, for $m$ index points, a good starting point
   * is to set n_clusters as around $\\sqrt{m}$.
   * @param global_dim Globally reduced dimension ($s$). Must be either -1 or an integer that is a
   * multiple of 32. If global_dim = -1, no dimensionality reduction is used, but the original
   * dimensionality must be a multiple of 32 in this case. Higher values increase recall but also
   * increase the query latency. In general, a good starting point is to set global_dim = -1 if
   * $d < 200$, global_dim = 128 if $200 \\leq d \\leq 1000$, and global_dim = 256 if $d > 1000$.
   * @param rank Rank ($r$) of the parameter matrices. Must be 16, 32, or 64. Defaults to 32. Rank =
   * 64 is mainly useful if no exact re-ranking is performed in the query phase.
   * @param train_size Number of nearby clusters ($w$) used for training the reduced-rank regression
   * models. Defaults to 5, but lower values can be used if $m \\gtrsim 500 000$ to speed up the
   * index construction.
   * @param euclidean Whether to use Euclidean distance instead of (negative) inner product as the
   * dissimilarity measure. Defaults to false.
   * @param balanced Whether to use balanced clustering. Defaults to false.
   */
  explicit Lorann(float *data, int m, int d, int n_clusters, int global_dim, BitsetMatrix& attributes, std::vector<int>& attribute_idxs, std::vector<std::uint32_t>& attribute_ints, int rank = 32,
                  int train_size = 5, bool euclidean = false, bool balanced = false) //, std::vector<std::string>* attributes, std::vector<std::string>* attribute_idxs
      : LorannBase(data, m, d, n_clusters, global_dim, attributes, attribute_idxs, attribute_ints, rank + 1, train_size, euclidean, balanced) {
    if (!(rank == 16 || rank == 32 || rank == 64)) {
      throw std::invalid_argument("rank must be 16, 32, or 64");
    }

    if (_global_dim % 32) {
      throw std::invalid_argument("global_dim must be a multiple of 32");
    }
  }

  /**
   * @brief Query the index.
   *
   * @param data The query vector (dimensionality must match that of the index)
   * @param k The number of approximate nearest neighbors retrived
   * @param clusters_to_search Number of clusters to search
   * @param points_to_rerank Number of points for final (exact) re-ranking. If points_to_rerank is
   * set to 0, no re-ranking is performed and the original data does not need to be kept in memory.
   * In this case the final returned distances are approximate distances.
   * @param idx_out The index output array of length k
   * @param dist_out The (optional) distance output array of length k
   */
  void search(const float *data, const int k, const int M, const int clusters_to_search,
              const int points_to_rerank, int *idx_out, attribute_set& filter_attributes, uint32_t filter_attributes_int, 
              std::string filter_approach, std::chrono::microseconds* duration, float *dist_out = nullptr, bool verbose=false) const override {
    auto start_prework = std::chrono::high_resolution_clock::now();
    ColVector scaled_query;
    ColVector transformed_query;
    Eigen::Map<const Eigen::VectorXf> data_vec(data, _dim);
    bool use_attr_indexing = (filter_approach != "postfilter");
            
    if (use_attr_indexing && M == -1) {
      throw std::invalid_argument("Parameter M must be set when using attribute indexing.");
    }
    if (!use_attr_indexing && M != -1) {
      std::cout << "Parameter M has no effect with postfilter approach." << std::endl;
    }
    if (_euclidean) {
      scaled_query = -2. * data_vec;
    } else {
      scaled_query = -data_vec;
    }
    /* apply dimensionality reduction to the query */
#if defined(LORANN_USE_MKL) || defined(LORANN_USE_OPENBLAS)
    transformed_query = Vector(_global_dim);
    cblas_sgemv(CblasRowMajor, CblasTrans, _global_transform.rows(), _global_transform.cols(), 1,
                _global_transform.data(), _global_transform.cols(), scaled_query.data(), 1, 0,
                transformed_query.data(), 1);
#else
    transformed_query = _global_transform.transpose() * scaled_query;
#endif
    const float principal_axis = transformed_query[0];
    transformed_query[0] = 0; /* the first component is treated separately in fp32 precision */

    /* quantize the transformed query vector */
    VectorInt8 quantized_query(_global_dim);
    VectorInt8 quantized_query_doubled(_max_rank - 1);
    const float quantization_factor =
        quant_query.quantize_vector(transformed_query.data(), _global_dim, quantized_query.data());

    const float compensation = quantized_query.cast<float>().sum();
    const float compensation_data = compensation * quant_data.compensation_factor;
    const float compensation_query = compensation * quant_query.compensation_factor;
    // std::vector<int> I(clusters_to_search);

    int n_clusters = _centroids_quantized.cols();
    ColVectorInt cluster_labels(n_clusters);
    ColVector cluster_dists(n_clusters);

    compute_cluster_distances_sorted(quantized_query, quantization_factor, principal_axis,
                                     compensation_query, cluster_labels.data(), cluster_dists.data());

    // compute safe allocation size for result buffers
    const int total_pts_all = _cluster_sizes.sum();
    const int required_points = use_attr_indexing ? (M * k) : 0;
    int allocate_pts = total_pts_all;
    if (use_attr_indexing) {
      allocate_pts = std::max(total_pts_all, 2 * required_points);
    }
    if (allocate_pts <= 0) allocate_pts = 1;
    ColVector all_distances(allocate_pts);
    ColVectorInt all_idxs(allocate_pts); // all_idxs contains the original indexes of all resultant datapoints from the query
    ColVector tmp(_max_rank);

    

    int current_cumulative_size = 0;
    int total_smallest_idx_sizes = 0; // temporary, remove
    bool matching_results_found = false;
    double found_ratio_avg;
    int cumulative_cluster_size = 0;
    auto stop_prework = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds total_indexing_duration = (std::chrono::microseconds) 0;
    std::chrono::microseconds total_filter_preloop_duration = (std::chrono::microseconds) 0;
    std::chrono::microseconds total_hybrid_avx_duration = (std::chrono::microseconds) 0;
    std::chrono::microseconds total_hybrid_duration = (std::chrono::microseconds) 0;
    std::chrono::microseconds total_mixed_duration = (std::chrono::microseconds) 0;
    std::chrono::microseconds total_mixed_loop_duration = (std::chrono::microseconds) 0;
    std::chrono::microseconds total_duration_matvec = (std::chrono::microseconds) 0;
    std::chrono::microseconds total_duration_filterapproach = (std::chrono::microseconds) 0;
    auto start_clusters = std::chrono::high_resolution_clock::now();

    int cumulative_found_points = 0;
    int i = 0;
    auto cond = [&]() {
      if (use_attr_indexing) {
        return cumulative_found_points < required_points && i < n_clusters;
      } else {
        return i < clusters_to_search;
      }
    };
    while (cond()) {
      const int cluster = cluster_labels[i];
      // std::cout << "cluster " << i << std::endl;
      i++;
      if (cluster < 0 || cluster >= _n_clusters) {
        throw std::runtime_error("Invalid cluster index encountered in search()");
      }
      const int sz = _cluster_sizes[cluster];
      if (sz == 0) continue;
      cumulative_cluster_size += sz;
      std::vector<int> attribute_data_idxs;
      std::vector<int> cluster_attribute_data_idxs;
      std::vector<int>* attribute_data_idxs_ptr;
      std::vector<int>* cluster_attribute_data_idxs_ptr;
      int n_filtered_cluster_datapoints = 0;
      auto start_filter = std::chrono::high_resolution_clock::now();
      if (filter_approach == "indexing") {
        auto start_preloop = std::chrono::high_resolution_clock::now();
        attribute_data_map& this_cluster_attribute_data_map = _cluster_attribute_data_maps[cluster];
        attribute_data_map& this_cluster_reverse_index_map = _cluster_reverse_index_maps[cluster];
        attribute_set smallest_idx;
        smallest_idx.init(1, _n_attributes);
        int smallest_idx_size = _n_samples;
        for (int attr = 0; attr < _n_attributes; ++attr) {
          if (filter_attributes.is_set(0, attr)) {
            attribute_set& attr_set = _attribute_index_map[attr];
            int attr_idx_size = this_cluster_attribute_data_map[attr_set.key(0)].size();
            if (attr_idx_size <= smallest_idx_size) {
              smallest_idx = attr_set;
              smallest_idx_size = this_cluster_attribute_data_map[attr_set.key(0)].size();
            }
          }
        }
        std::vector<int>& attribute_idx = this_cluster_attribute_data_map[smallest_idx.key(0)];
        std::vector<int>& reverse_index = this_cluster_reverse_index_map[smallest_idx.key(0)];
        
        attribute_data_idxs.reserve(attribute_idx.size());
        cluster_attribute_data_idxs.reserve(attribute_idx.size());
        attribute_data_idxs_ptr = &attribute_data_idxs;
        cluster_attribute_data_idxs_ptr = &cluster_attribute_data_idxs;
        total_smallest_idx_sizes += attribute_idx.size();
        auto stop_preloop = std::chrono::high_resolution_clock::now();
        auto start_indexing = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < attribute_idx.size(); ++i) { // for each data point in the smallest index which the datapoints belong to, check if the data point has the other filter attributes as well, if yes then add to filtered list.
          bool filters_match = _attributes.matches(attribute_idx[i], filter_attributes);
          if (filters_match) {
            attribute_data_idxs.push_back(attribute_idx[i]);
            cluster_attribute_data_idxs.push_back(reverse_index[i]); // Need a REVERSE INDEX - mapping points of attribute_idx to the cluster point indices.
          }
        }
        n_filtered_cluster_datapoints = attribute_data_idxs_ptr->size();
        auto stop_indexing = std::chrono::high_resolution_clock::now();
        auto duration_indexing = std::chrono::duration_cast<std::chrono::microseconds>(stop_indexing - start_indexing);
        total_indexing_duration += duration_indexing;
        auto duration_preloop = std::chrono::duration_cast<std::chrono::microseconds>(stop_preloop - start_preloop);
        total_filter_preloop_duration += duration_preloop;
      } if (filter_approach == "indexing_avx") {
        auto start_indexing = std::chrono::high_resolution_clock::now();
        auto start_preloop = std::chrono::high_resolution_clock::now();
        int32_attribute_data_map& this_cluster_attribute_int_data_map = _cluster_attribute_int_data_maps[cluster];
        attribute_data_map& this_cluster_reverse_index_map = _cluster_reverse_index_maps[cluster];
        const std::vector<uint32_t>* smallest_idx_ptr = nullptr;
        // smallest_idx.init(1, _n_attributes);
        int smallest_idx_size = _n_samples;
        attribute_set* best_attr_set = nullptr;
        for (int attr = 0; attr < _n_attributes; ++attr) {
          if (filter_attributes.is_set(0, attr)) {
              attribute_set& attr_set = _attribute_index_map[attr];

              auto& candidate = this_cluster_attribute_int_data_map[attr_set.key(0)];
              int attr_idx_size = candidate.size();

              if (attr_idx_size <= smallest_idx_size) {
                  best_attr_set = &attr_set;
                  smallest_idx_ptr = &candidate;          // store pointer instead of copy
                  smallest_idx_size = attr_idx_size;
              }
          }
        }
        std::vector<int>& reverse_index = this_cluster_reverse_index_map[best_attr_set->key(0)];
        attribute_data_idxs.reserve(smallest_idx_size);
        cluster_attribute_data_idxs.reserve(smallest_idx_size);
        attribute_data_idxs_ptr = &attribute_data_idxs;
        cluster_attribute_data_idxs_ptr = &cluster_attribute_data_idxs;
        total_smallest_idx_sizes += smallest_idx_size;
        auto stop_preloop = std::chrono::high_resolution_clock::now();
        

        std::vector<uint16_t> masks((smallest_idx_size + 15) / 16);  // properly size for AVX512 blocks
        int num_blocks = build_subset_masks_avx512(smallest_idx_ptr->data(), smallest_idx_size, filter_attributes_int, masks.data());
        std::vector<int> set_bit_indexes;
        set_bit_indexes.reserve(smallest_idx_size);
        iterate_hits_from_masks(masks.data(), num_blocks, set_bit_indexes);
        const std::vector<int>& this_cluster = _cluster_map[cluster];
        for (int i = 0; i < set_bit_indexes.size(); ++i) {
          int cluster_point_idx = reverse_index[set_bit_indexes[i]];
          attribute_data_idxs.push_back(this_cluster[cluster_point_idx]);
          cluster_attribute_data_idxs.push_back(cluster_point_idx);
        }

        // for (int i = 0; i < attribute_idx.size(); ++i) { // for each data point in the smallest index which the datapoints belong to, check if the data point has the other filter attributes as well, if yes then add to filtered list.
        //   bool filters_match = _attributes.matches(attribute_idx[i], filter_attributes);
        //   if (filters_match) {
        //     attribute_data_idxs_ptr->push_back(attribute_idx[i]);
        //     cluster_attribute_data_idxs_ptr->push_back(reverse_index[i]);
        //   }
        // }
        n_filtered_cluster_datapoints = attribute_data_idxs_ptr->size();
        // attribute_idx < sz so the indices for intra-cluster points will be off. b_filter expects indexes for the cluster, whereas it is getting indexes for a sub-cluster.
        // Need a REVERSE INDEX - mapping points of attribute_idx to the cluster point indices. Due to subcluster partitioning, cluster indices do not match with partition indices.
        auto stop_indexing = std::chrono::high_resolution_clock::now();
        auto duration_indexing = std::chrono::duration_cast<std::chrono::microseconds>(stop_indexing - start_indexing);
        total_indexing_duration += duration_indexing;
        
        auto duration_preloop = std::chrono::duration_cast<std::chrono::microseconds>(stop_preloop - start_preloop);
        total_filter_preloop_duration += duration_preloop;
      } 
      else if (filter_approach == "mixed") {
        auto start_mixed = std::chrono::high_resolution_clock::now();
        attribute_data_map& this_cluster_attribute_data_map = _cluster_attribute_data_maps[cluster];
        attribute_data_map& this_cluster_reverse_index_map = _cluster_reverse_index_maps[cluster];
        uint32_t mask = filter_attributes_int; // must contain bits corresponding to attributes checked here
        int smallest_idx_size = _n_samples;
        attribute_set* best_attr_set = nullptr;
        auto start_mixed_loop = std::chrono::high_resolution_clock::now();
        while (mask) {
          unsigned bit = __builtin_ctz(mask);        // index of least-significant set bit
          int attr = static_cast<int>(bit);
          attribute_set& attr_set = _attribute_index_map[attr];
          const auto& candidate = this_cluster_attribute_data_map[attr_set.key(0)];
          int attr_idx_size = static_cast<int>(candidate.size());

          if (attr_idx_size < smallest_idx_size) {
            best_attr_set = &attr_set;
            smallest_idx_size = attr_idx_size;
          }
          mask &= mask - 1; // clear LSB
        }
        auto stop_mixed_loop = std::chrono::high_resolution_clock::now();
        attribute_data_idxs_ptr = &this_cluster_attribute_data_map[best_attr_set->key(0)];
        cluster_attribute_data_idxs_ptr = &this_cluster_reverse_index_map[best_attr_set->key(0)];
        
        // const std::vector<int>& this_cluster = _cluster_map[cluster];
        // int filter_matches = 0;
        // for (int i = 0; i < sz; ++i) {
        //   bool filters_match = _attributes.matches(this_cluster[i], filter_attributes);
        //   if (filters_match) {
        //     filter_matches++;
        //   }
        // }
        // std::vector<int> temp_attribute_data_idxs = this_cluster_attribute_data_map[smallest_idx.key(0)];
        // int idx_filter_matches = 0;
        // for (int i = 0; i < temp_attribute_data_idxs.size(); ++i) {
        //   bool filters_match = _attributes.matches(temp_attribute_data_idxs[i], filter_attributes);
        //   if (filters_match) {
        //     idx_filter_matches++;
        //   }
        // }
        // std::cout << "cluster " << cluster << " filter_matches: " << filter_matches << "/" << sz << ", idx_filter_matches: " << idx_filter_matches << "/" << temp_attribute_data_idxs.size() << std::endl;
        //
        n_filtered_cluster_datapoints = smallest_idx_size;
        auto stop_mixed = std::chrono::high_resolution_clock::now();
        auto duration_mixed = std::chrono::duration_cast<std::chrono::microseconds>(stop_mixed - start_mixed);
        auto duration_mixed_loop = std::chrono::duration_cast<std::chrono::microseconds>(stop_mixed_loop - start_mixed_loop);
        total_mixed_duration += duration_mixed;
        total_mixed_loop_duration += duration_mixed_loop;
      } else if (filter_approach == "hybrid_avx") {
        auto start_hybrid_avx = std::chrono::high_resolution_clock::now();
        cluster_attribute_data_idxs.reserve(sz);
        attribute_data_idxs.reserve(sz);
        std::vector<uint16_t> masks(sz);
        std::vector<uint32_t> this_cluster_attribute_ints = _cluster_attribute_int_map[cluster];
        int num_blocks = build_subset_masks_avx512(this_cluster_attribute_ints.data(), sz, filter_attributes_int, masks.data());
        iterate_hits_from_masks(masks.data(), num_blocks, cluster_attribute_data_idxs);
        const std::vector<int>& this_cluster = _cluster_map[cluster];
        for (int i: cluster_attribute_data_idxs) {
          attribute_data_idxs.push_back(this_cluster[i]);
        }
        attribute_data_idxs_ptr = &attribute_data_idxs;
        cluster_attribute_data_idxs_ptr = &cluster_attribute_data_idxs;
        n_filtered_cluster_datapoints = attribute_data_idxs_ptr->size();
        auto stop_hybrid_avx = std::chrono::high_resolution_clock::now();
        auto duration_hybrid_avx = std::chrono::duration_cast<std::chrono::microseconds>(stop_hybrid_avx - start_hybrid_avx);
        total_hybrid_avx_duration += duration_hybrid_avx;
      } else if (filter_approach == "hybrid") {
        auto start_hybrid = std::chrono::high_resolution_clock::now();
        cluster_attribute_data_idxs.reserve(sz);
        attribute_data_idxs.reserve(sz);
        attribute_data_idxs_ptr = &attribute_data_idxs;
        cluster_attribute_data_idxs_ptr = &cluster_attribute_data_idxs;
        const std::vector<int>& this_cluster = _cluster_map[cluster];
        for (int i = 0; i < sz; ++i) {
          bool filters_match = _attributes.matches(this_cluster[i], filter_attributes);
          if (filters_match) {
            attribute_data_idxs_ptr->push_back(this_cluster[i]);
            cluster_attribute_data_idxs_ptr->push_back(i);
          }
        }
        n_filtered_cluster_datapoints = attribute_data_idxs_ptr->size();
        auto stop_hybrid = std::chrono::high_resolution_clock::now();
        auto duration_hybrid = std::chrono::duration_cast<std::chrono::microseconds>(stop_hybrid - start_hybrid);
        total_hybrid_duration += duration_hybrid;
      }
      auto stop_filter = std::chrono::high_resolution_clock::now();
      cumulative_found_points += n_filtered_cluster_datapoints;
      // std::cout << "cumulative_found_points: " << cumulative_found_points << std::endl;
      if ((use_attr_indexing && n_filtered_cluster_datapoints == 0)) continue;
      const ColMatrixUInt8 &A = _A[cluster];
      const ColMatrixUInt8 &B = _B[cluster];
      const Vector &A_correction = _A_corrections[cluster];
      const Vector &B_correction = _B_corrections[cluster];
      auto start_matvec = std::chrono::high_resolution_clock::now();
      /* compute s = q^T A */
      quant_data.quantized_matvec_product_A(A, quantized_query, A_correction, quantization_factor,
                                            principal_axis, compensation_data, tmp.data());
      
      const float principal_axis_tmp = tmp[0];

      const float tmpfact = quant_query.quantize_vector(tmp.data() + 1, _max_rank - 1,
                                                        quantized_query_doubled.data());
      const float compensation_tmp =
          quantized_query_doubled.cast<float>().sum() * quant_data.compensation_factor;
      
      /* compute r = s^T B */
      if (use_attr_indexing) {
        quant_data.quantized_matvec_product_B_filter(B, quantized_query_doubled, cluster_attribute_data_idxs_ptr, B_correction, tmpfact,
                                                    principal_axis_tmp, compensation_tmp,
                                                    &all_distances[current_cumulative_size], verbose);
      } else {
        quant_data.quantized_matvec_product_B(B, quantized_query_doubled, B_correction, tmpfact,
                                                    principal_axis_tmp, compensation_tmp,
                                                    &all_distances[current_cumulative_size]);
      }
      auto stop_matvec = std::chrono::high_resolution_clock::now();
      auto duration_matvec = std::chrono::duration_cast<std::chrono::microseconds>(stop_matvec - start_matvec);
      total_duration_matvec += duration_matvec;
      auto duration_filter = std::chrono::duration_cast<std::chrono::microseconds>(stop_filter - start_filter);
      total_duration_filterapproach += duration_filter;
      if (_euclidean)
        add_inplace(_cluster_norms[cluster].data(), &all_distances[current_cumulative_size],
                    _cluster_norms[cluster].size());
      int to_copy = use_attr_indexing ? n_filtered_cluster_datapoints : sz;
      int needed_size = current_cumulative_size + to_copy;
      if (use_attr_indexing) { // when we use indexing, we process fewer results than the full size of the cluster due to filtering them beforehand.
        if (n_filtered_cluster_datapoints > 0)
          std::memcpy(&all_idxs[current_cumulative_size], attribute_data_idxs_ptr->data(), n_filtered_cluster_datapoints * sizeof(int));
        current_cumulative_size += n_filtered_cluster_datapoints;
      } else {
        if (sz > 0)
          std::memcpy(&all_idxs[current_cumulative_size], _cluster_map[cluster].data(), sz * sizeof(int));
        current_cumulative_size += sz;
      }
      // std::cout << "after memcpy" << std::endl;
    }
    auto stop_clusters = std::chrono::high_resolution_clock::now();
    auto duration_filter = std::chrono::duration_cast<std::chrono::microseconds>(stop_clusters - start_clusters);
    std::cout << "cumulative_cluster_size ratio: " << ((double) current_cumulative_size) / cumulative_cluster_size << std::endl;
    std::cout << "cluster search stopped after " << i << " clusters searched" << std::endl;
    // std::cout << "duration_filter: " << duration_filter.count() << " microseconds for " << filter_approach << std::endl;
    auto duration_prework = std::chrono::duration_cast<std::chrono::microseconds>(stop_prework - start_prework);
    matching_results_found = cumulative_found_points > 0;
    if (filter_approach != "postfilter" && filter_approach != "mixed" && !matching_results_found) {
      throw std::runtime_error("No matches found for filter attributes!");
    }
    if (verbose) {
      if (filter_approach == "indexing" || filter_approach == "indexing_avx") std::cout << "total_indexing_duration: " << total_indexing_duration.count() << " microseconds" << std::endl;
      // if (filter_approach == "indexing") std::cout << "total_filter_preloop_duration: " << total_filter_preloop_duration.count() << " microseconds" << std::endl;
      // if (filter_approach == "indexing") std::cout << "duration_matvec: " << total_duration_matvec.count() << " microseconds" << std::endl;
      if (filter_approach == "mixed") std::cout << "total_mixed_duration: " << total_mixed_duration.count() << " microseconds" << std::endl;
      if (filter_approach == "mixed") std::cout << "total_mixed_loop_duration: " << total_mixed_loop_duration.count() << " microseconds" << std::endl;
      if (filter_approach == "hybrid_avx") std::cout << "total_hybrid_avx_duration: " << total_hybrid_avx_duration.count() << " microseconds" << std::endl;
      if (filter_approach == "hybrid") std::cout << "total_hybrid_duration: " << total_hybrid_duration.count() << " microseconds" << std::endl;
      std::cout <<"total_duration_matvec: " << total_duration_matvec.count() << " microseconds" << std::endl;
      std::cout <<"total_duration_filterapproach: " << total_duration_filterapproach.count() << " microseconds" << std::endl;
      std::cout << "!! Average ratio of satisfactory points to cluster size: " << ((double) cumulative_found_points) / cumulative_cluster_size << std::endl;
      std::cout << "current_cumulative_size: " << current_cumulative_size << std::endl;
      std::cout << "duration_filter: " << duration_filter.count() << " microseconds" << std::endl;
      // std::cout << "duration_prework: " << duration_prework.count() << " microseconds" << std::endl;
      // std::cout << "total_smallest_idx_sizes: " << total_smallest_idx_sizes << std::endl;
    }
    auto start_postwork = std::chrono::high_resolution_clock::now();
    // ColVector filtered_distances(current_cumulative_size);
    // for (int i = 0; i < current_cumulative_size; ++i) { // this is needed because all_distances when using indexing will have more reserved memory than there are filtered datapoints so we need to filter it to include only the number of datapoints we want.
    //   filtered_distances[i] = all_distances[i];
    // }
    Eigen::VectorXi shuffled_out(k*8); // why is this needed?
    // std::cout << "current_cumulative_size for " << filter_approach << ": " << current_cumulative_size << std::endl;
    select_final(_euclidean ? data : scaled_query.data(), k*8, points_to_rerank, current_cumulative_size,
                 all_idxs.data(), all_distances.data(), shuffled_out.data(), dist_out);
    auto stop_postwork = std::chrono::high_resolution_clock::now();
    auto start_postfilter = std::chrono::high_resolution_clock::now();
    if (filter_approach == "postfilter" || filter_approach == "mixed") {
      std::vector<int> matched_idxs;
      matched_idxs.reserve(k);
      for (int i = 0; i < k*8; ++i) {
        bool filters_match = _attributes.matches(shuffled_out[i], filter_attributes);
        if (filters_match) {
          matched_idxs.push_back(shuffled_out[i]);
        }
      }
      int matched_k = matched_idxs.size();
      int new_k = k*8;
      int i = 0;
      while (matched_k < k) { // if not enough datapoints are found in k results, double it and search again
        i++;
        new_k = new_k * 2 > current_cumulative_size ? current_cumulative_size : new_k * 2;
        matched_idxs.clear();
        Eigen::VectorXi new_out(new_k);
        select_final(_euclidean ? data : scaled_query.data(), new_k, points_to_rerank, current_cumulative_size,
                 all_idxs.data(), all_distances.data(), new_out.data(), dist_out);
        for (int i = 0; i < new_k; ++i) {
          bool filters_match = _attributes.matches(new_out[i], filter_attributes);
          if (filters_match) {
            matched_idxs.push_back(new_out[i]);
            if (matched_idxs.size() >= k) break;
          }
        }
        matched_k = matched_idxs.size();
        if (matched_k < k && new_k == current_cumulative_size) {
          std::cout << "could not find enough samples (found " << matched_k << ")" << std::endl;
          break;
        }
      }
      std::cout << "Repeated postfilter search " << i << " times" << std::endl;
      if (verbose) std::cout << "final k: " << new_k << std::endl;
      if (matched_k >= k) {
        for (int i = 0; i < k; ++i) {
          idx_out[i] = matched_idxs[i];
        }
      }
    } else {
      for (int i = 0; i < k; ++i) {
        idx_out[i] = shuffled_out[i];
      }
    }
    auto stop_postfilter = std::chrono::high_resolution_clock::now();
    auto duration_postwork = std::chrono::duration_cast<std::chrono::microseconds>(stop_postwork - start_postwork);
    // std::cout << "duration_postwork: " << duration_postwork.count() << " microseconds for " << filter_approach << std::endl;
    auto duration_postfilter = std::chrono::duration_cast<std::chrono::microseconds>(stop_postfilter - start_postfilter);
    
    if (filter_approach == "postfilter" || filter_approach == "mixed") {
      std::cout << "duration_filter: " << duration_filter.count() << " microseconds for " << filter_approach << std::endl;
      std::cout << "duration_postfilter: " << duration_postfilter.count() << " microseconds for " << filter_approach << std::endl;
      duration_filter += duration_postfilter;
    }
    if (duration) *duration = duration_filter;
  }
  
  using LorannBase::build;

  /**
   * @brief Build the index.
   *
   * @param query_data A float array of training queries of size $n \\times d$ used to build the
   * index. Can be useful in the out-of-distribution setting where the training and query
   * distributions differ. Ideally there should be at least as many training query points as there
   * are index points.
   * @param query_n The number of training queries
   * @param approximate Whether to turn on various approximations during index construction.
   * Defaults to true. Setting approximate to false slows down the index construction but can
   * slightly increase the recall, especially if no exact re-ranking is used in the query phase.
   * @param num_threads Number of CPU threads to use (set to -1 to use all cores)
   */
  void build(const float *query_data, const int query_n, int n_attribute_partitions=-1, const bool approximate = true,
             int num_threads = -1) override {
    LORANN_ENSURE_POSITIVE(query_n);

#ifdef _OPENMP
    if (num_threads <= 0) {
      num_threads = omp_get_max_threads();
    }
#endif
    
    /* Construct index for exact search */
    std::vector<BitsetMatrix> attribute_partition_sets;
    if (n_attribute_partitions >= 0) {
      std::vector<std::vector<int>> attr_subvecs = split_vector(_attribute_idxs, n_attribute_partitions); // partition the attributes into groups of attributes
      // int attribute_integers_per_point = (_n_attributes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
      // int points_per_avx512_vector = 512 / (attribute_integers_per_point * sizeof(uint32_t));
      for (const auto& attr_subvec : attr_subvecs) {
        BitsetMatrix attribute_subvec_bitset;
        attribute_subvec_bitset.init(1, _n_attributes);
        for (int attribute_idx: attr_subvec) {
          attribute_subvec_bitset.set(0, attribute_idx);
        }
        attribute_partition_sets.push_back(attribute_subvec_bitset);
        std::vector<int> attribute_data_idx_vec; // vector of indexes of datapoints that have at least one of the attributes in attribute_subvec_bitset
        std::vector<uint32_t> attribute_data_attr_idx_vec; // vector of attribute integers of datapoints that have at least one of the attributes in attribute_subvec_bitset
        for (int i = 0; i<_n_samples; ++i) { // for each datapoint
          if (_attributes.any_match(i, attribute_subvec_bitset)) {
            attribute_data_idx_vec.push_back(i);
            attribute_data_attr_idx_vec.push_back(_attribute_ints[i]);
          }
        }
        for (int i = 0; i < _n_attributes; ++i) {
          if (attribute_subvec_bitset.is_set(0, i)) _attribute_index_map.insert({i, attribute_subvec_bitset});
        }
        _attribute_data_map.insert({attribute_subvec_bitset.key(0), attribute_data_idx_vec});
        _attribute_int_data_map.insert({attribute_subvec_bitset.key(0), attribute_data_attr_idx_vec});
      }
    }

    /* Some printouts to make sure data indexes were stored correctly */
  //   for (int i = 0; i < attribute_partition_sets.size(); ++i) {
  //     std::set<std::string> partition = attribute_partition_sets[i];
  //     std::cout << "partition " << i << ": ";
  //     for (auto& attribute: partition) {
  //       std::cout << attribute << ' ';
  //     }
  //     std::cout << std::endl;
  //   }
  //   std::vector<std::string> attribute_string_vec = {"brown"};
  //   std::set<std::string> attribute_key(attribute_string_vec.begin(), attribute_string_vec.end());
  //   std::vector<int> colour_partition_data_idxs = _attribute_data_map[attribute_key];
  //   std::cout << "colour_partition_data_idxs size: " << colour_partition_data_idxs.size() << std::endl;
  //   std::cout << "colour partition indexes:" << std::endl;
  //   for(int i=0; i < 10; ++i){
  //     std::cout << colour_partition_data_idxs[i] << " - ";
  //     std::cout << "corresponding attribute: " << _attributes[colour_partition_data_idxs[i]] << "|";
  //  }


    Eigen::Map<RowMatrix> train_mat(_data, _n_samples, _dim);
    Eigen::Map<const RowMatrix> query_mat(query_data, query_n, _dim);

    /* compute dimensionality reduction matrix */
    RowMatrix query_sample = sample_rows(query_mat, GLOBAL_DIM_REDUCTION_SAMPLES);
    Eigen::MatrixXf global_dim_reduction =
        compute_principal_components(query_sample.transpose() * query_sample, _global_dim);

    /* rotate the dimensionality reduction matrix beforehand so that we do not need to rotate
     * queries at query time */
    Eigen::MatrixXf sub_rotation = generate_rotation_matrix(_global_dim - 1);
    Eigen::MatrixXf rotation = Eigen::MatrixXf::Zero(_global_dim, _global_dim);
    rotation(0, 0) = 1;
    rotation.block(1, 1, _global_dim - 1, _global_dim - 1) = sub_rotation;
    _global_transform = global_dim_reduction * rotation;

    RowMatrix reduced_train_mat = train_mat * global_dim_reduction;

    /* clustering */
    KMeans global_clustering(_n_clusters, KMEANS_ITERATIONS, _euclidean, _balanced,
                             KMEANS_MAX_BALANCE_DIFF, 0);

    std::vector<std::vector<int>> cluster_train_map;
    if (query_mat.data() != train_mat.data()) {
      RowMatrix reduced_query_mat = query_mat * global_dim_reduction;
      cluster_train_map =
          clustering(global_clustering, reduced_train_mat.data(), reduced_train_mat.rows(),
                     reduced_query_mat.data(), reduced_query_mat.rows(), approximate, num_threads, attribute_partition_sets);
    } else {
      cluster_train_map =
          clustering(global_clustering, reduced_train_mat.data(), reduced_train_mat.rows(),
                     reduced_train_mat.data(), reduced_train_mat.rows(), approximate, num_threads, attribute_partition_sets);
    }

    /* rotate the cluster centroid matrix */
    RowMatrix centroid_mat = global_clustering.get_centroids();
    ColMatrix centroid_mat_rotated = (centroid_mat * rotation).transpose();
    Vector centroid_fix = centroid_mat_rotated.row(0);
    centroid_mat_rotated.row(0).array() *= 0;

    if (_euclidean) {
      _global_centroid_norms = centroid_mat.rowwise().squaredNorm();
      _data_norms = train_mat.rowwise().squaredNorm();
    }

    /* quantize the cluster centroids */
    _centroids_quantized = ColMatrixUInt8(centroid_mat_rotated.rows(), centroid_mat_rotated.cols());
    _centroid_correction = Vector(_centroids_quantized.cols() * 2);
    quant_query.quantize_matrix_A_unsigned(centroid_mat_rotated, _centroids_quantized.data(),
                                           _centroid_correction.data());

    _centroid_correction(Eigen::seqN(_n_clusters, _n_clusters)) = centroid_fix;

    _A.resize(_n_clusters);
    _B.resize(_n_clusters);
    _A_corrections.resize(_n_clusters);
    _B_corrections.resize(_n_clusters);

    if (_euclidean) {
      _cluster_norms.resize(_n_clusters);
    }

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (int i = 0; i < _n_clusters; ++i) {
      if (_cluster_map[i].size() == 0) continue;

      if (_euclidean) {
        _cluster_norms[i] = _data_norms(_cluster_map[i]);
      }

      RowMatrix pts = train_mat(_cluster_map[i], Eigen::placeholders::all);
      RowMatrix Q;

      if (cluster_train_map[i].size() >= _cluster_map[i].size()) {
        Q = query_mat(cluster_train_map[i], Eigen::placeholders::all);
      } else {
        Q = pts;
      }

      /* compute reduced-rank regression solution */
      Eigen::MatrixXf beta_hat, Y_hat;
      if (approximate) {
        beta_hat = (pts * _global_transform).transpose();
        Y_hat = (Q * _global_transform) * beta_hat;
      } else {
        Eigen::MatrixXf X = Q * _global_transform;
        beta_hat = X.colPivHouseholderQr().solve(Q * pts.transpose());
        Y_hat = X * beta_hat;
      }
      Eigen::MatrixXf V = compute_V(Y_hat, _max_rank, approximate);

      /* randomly rotate the matrix V */
      Eigen::MatrixXf sub_rot_mat = generate_rotation_matrix(V.cols() - 1);
      Eigen::MatrixXf rot_mat = Eigen::MatrixXf::Zero(V.cols(), V.cols());
      rot_mat(0, 0) = 1;
      rot_mat.block(1, 1, V.cols() - 1, V.cols() - 1) = sub_rot_mat;
      Eigen::MatrixXf V_rotated = V * rot_mat;

      ColMatrix A = beta_hat * V_rotated;
      ColMatrix B = V_rotated.transpose();

      /* quantize the A and B matrices */
      ColMatrixUInt8 A_quantized(A.rows() / quant_data.div_factor, A.cols());
      ColMatrixUInt8 B_quantized((B.rows() - 1) / quant_data.div_factor, B.cols());
      Vector A_correction(A.cols() * 2);
      Vector B_correction(B.cols() * 2);

      Vector A_fix = A.row(0);
      A.row(0).array() *= 0;
      Vector B_fix = B.row(0);

      A_correction(Eigen::seqN(A.cols(), A.cols())) = A_fix;
      B_correction(Eigen::seqN(B.cols(), B.cols())) = B_fix;

      quant_data.quantize_matrix_A_unsigned(A, A_quantized.data(), A_correction.data());
      quant_data.quantize_matrix_B_unsigned(B, B_quantized.data(), B_correction.data());

      _A[i] = A_quantized;
      _B[i] = B_quantized;

      _A_corrections[i] = A_correction;
      _B_corrections[i] = B_correction;
    }

    _cluster_sizes = Eigen::VectorXi(_n_clusters);
    for (int i = 0; i < _n_clusters; ++i) {
      _cluster_sizes(i) = static_cast<int>(_cluster_map[i].size());
    }
  }

 private:
  Lorann() = default; /* default constructor should only be used for serialization */

  void select_nearest_clusters(const VectorInt8 &query_quantized, const float quantization_factor,
                               const float correction, const float compensation, int k,
                               int *out) const {
    ColVector dists(_centroids_quantized.cols());
    quant_query.quantized_matvec_product_A(_centroids_quantized, query_quantized,
                                           _centroid_correction, quantization_factor, correction,
                                           compensation, dists.data());
    if (_euclidean)
      add_inplace(_global_centroid_norms.data(), dists.data(), _global_centroid_norms.size());
    select_k(k, out, _centroids_quantized.cols(), NULL, dists.data());
  }



  void compute_cluster_distances_sorted(const VectorInt8 &query_quantized, const float quantization_factor,
                               const float correction, const float compensation, int* labels,
                               float* dists) const {
    quant_query.quantized_matvec_product_A(_centroids_quantized, query_quantized,
                                           _centroid_correction, quantization_factor, correction,
                                           compensation, dists);
    if (_euclidean)
      add_inplace(_global_centroid_norms.data(), dists, _global_centroid_norms.size());
    

    int k_base = _centroids_quantized.cols();
    std::vector<int> perm(k_base);
    for (int i = 0; i < k_base; ++i) {
      perm[i] = i;
    }
    ArgsortComparator comp = {dists};
    miniselect::pdqpartial_sort_branchless(perm.begin(), perm.begin() + k_base, perm.end(), comp);
    for (int i = 0; i < k_base; ++i) {
      labels[i] = perm[i];
    }
  }



  friend class cereal::access;

  template <class Archive>
  void save(Archive &ar) const {
    ar(cereal::base_class<LorannBase>(this), _global_transform, _centroids_quantized,
       _centroid_correction, _A, _B, _A_corrections, _B_corrections, _cluster_norms);
  }

  template <class Archive>
  void load(Archive &ar) {
    ar(cereal::base_class<LorannBase>(this), _global_transform, _centroids_quantized,
       _centroid_correction, _A, _B, _A_corrections, _B_corrections, _cluster_norms);
  }

  DataQuantizer quant_data;
  QueryQuantizer quant_query;

  RowMatrix _global_transform;
  ColMatrixUInt8 _centroids_quantized;
  Vector _centroid_correction;

  std::vector<ColMatrixUInt8> _A;
  std::vector<ColMatrixUInt8> _B;
  std::vector<Vector> _A_corrections;
  std::vector<Vector> _B_corrections;
  std::vector<Vector> _cluster_norms;
};

}  // namespace Lorann

typedef Lorann::Lorann<Lorann::SQ4Quantizer, Lorann::SQ4Quantizer> lorann_sq4sq4;
typedef Lorann::Lorann<Lorann::SQ4Quantizer, Lorann::SQ8Quantizer> lorann_sq4sq8;
typedef Lorann::Lorann<Lorann::SQ8Quantizer, Lorann::SQ4Quantizer> lorann_sq8sq4;
typedef Lorann::Lorann<Lorann::SQ8Quantizer, Lorann::SQ8Quantizer> lorann_sq8sq8;

CEREAL_REGISTER_TYPE(lorann_sq4sq4)
CEREAL_REGISTER_POLYMORPHIC_RELATION(Lorann::LorannBase, lorann_sq4sq4)

CEREAL_REGISTER_TYPE(lorann_sq4sq8)
CEREAL_REGISTER_POLYMORPHIC_RELATION(Lorann::LorannBase, lorann_sq4sq8)

CEREAL_REGISTER_TYPE(lorann_sq8sq4)
CEREAL_REGISTER_POLYMORPHIC_RELATION(Lorann::LorannBase, lorann_sq8sq4)

CEREAL_REGISTER_TYPE(lorann_sq8sq8)
CEREAL_REGISTER_POLYMORPHIC_RELATION(Lorann::LorannBase, lorann_sq8sq8)
