#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Dense>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "lorann_base.h"
#include "quant.h"
#include "utils.h"
#include <bitset_matrix.h>
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
  explicit Lorann(float *data, int m, int d, int n_clusters, int global_dim, BitsetMatrix& attributes, std::vector<int>& attribute_idxs, int rank = 32,
                  int train_size = 5, bool euclidean = false, bool balanced = false) //, std::vector<std::string>* attributes, std::vector<std::string>* attribute_idxs
      : LorannBase(data, m, d, n_clusters, global_dim, attributes, attribute_idxs, rank + 1, train_size, euclidean, balanced) {
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
  void search(const float *data, const int k, const int clusters_to_search,
              const int points_to_rerank, int *idx_out, attribute_set& filter_attributes, std::string filter_approach, float *dist_out = nullptr, bool verbose=false) const override {
    auto start_prework = std::chrono::high_resolution_clock::now();
    ColVector scaled_query;
    ColVector transformed_query;
    Eigen::Map<const Eigen::VectorXf> data_vec(data, _dim);
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

    std::vector<int> I(clusters_to_search);
    select_nearest_clusters(quantized_query, quantization_factor, principal_axis,
                            compensation_query, clusters_to_search, I.data());

    const int total_pts = _cluster_sizes(I).sum();
    
    ColVector all_distances(total_pts);
    ColVectorInt all_idxs(total_pts); // all_idxs contains the original indexes of all resultant datapoints from the query
    ColVector tmp(_max_rank);

    int current_cumulative_size = 0;
    // int current_cumulative_cluster_size = 0;
    bool use_attr_indexing = (filter_approach == "indexing" || filter_approach == "prefilter");
    int total_smallest_idx_sizes = 0; // temporary, remove
    bool matching_results_found = false;
    double found_ratio_avg;
    int cumulative_cluster_size = 0;
    int cumulative_found_points = 0;
    auto stop_prework = std::chrono::high_resolution_clock::now();
    auto start_clusters = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds total_filter_duration = (std::chrono::nanoseconds) 0;
    std::chrono::microseconds total_filter_preloop_duration = (std::chrono::microseconds) 0;
    std::chrono::microseconds total_prefilter_duration = (std::chrono::microseconds) 0;
    std::chrono::microseconds total_duration_matvec = (std::chrono::microseconds) 0;
    for (int i = 0; i < clusters_to_search; ++i) {
      // std::cout << "searching cluster " << i << "/" << clusters_to_search << std::endl;
      const int cluster = I[i];
      const int sz = _cluster_sizes[cluster];
      if (sz == 0) continue;
      cumulative_cluster_size += sz;
      // std::cout << "sz: " << sz << std::endl;
      std::vector<int> attribute_data_idxs;
      std::vector<int> cluster_attribute_data_idxs;
      if (filter_approach == "indexing") {
        auto start_preloop = std::chrono::high_resolution_clock::now();
        attribute_data_map& this_cluster_attribute_data_map = _cluster_attribute_data_maps[cluster];
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
        attribute_data_idxs.reserve(attribute_idx.size());
        cluster_attribute_data_idxs.reserve(attribute_idx.size());
        total_smallest_idx_sizes += attribute_idx.size();
        auto stop_preloop = std::chrono::high_resolution_clock::now();
        auto start_indexing = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < attribute_idx.size(); ++i) { // for each data point in the smallest index which the datapoints belong to, check if the data point has the other filter attributes as well, if yes then add to filtered list.
          bool filters_match = _attributes.matches(attribute_idx[i], filter_attributes);
          if (filters_match) {
            attribute_data_idxs.push_back(attribute_idx[i]);
            cluster_attribute_data_idxs.push_back(i);
            matching_results_found = true;
          }
        }
        auto stop_indexing = std::chrono::high_resolution_clock::now();
        auto duration_indexing = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_indexing - start_indexing);
        total_filter_duration += duration_indexing;
        
        auto duration_preloop = std::chrono::duration_cast<std::chrono::microseconds>(stop_preloop - start_preloop);
        total_filter_preloop_duration += duration_preloop;
        
      } else if (filter_approach == "prefilter") {
        auto start_prefilter = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < sz; i++) {
          bool filters_match = _attributes.matches(_cluster_map[cluster][i], filter_attributes);
          if (filters_match) {
            attribute_data_idxs.push_back(_cluster_map[cluster][i]);
            matching_results_found = true;
          }
        }
        auto stop_prefilter = std::chrono::high_resolution_clock::now();
        auto duration_prefilter = std::chrono::duration_cast<std::chrono::microseconds>(stop_prefilter - start_prefilter);
        total_prefilter_duration += duration_prefilter;
      }
      int n_filtered_cluster_datapoints = attribute_data_idxs.size();
      cumulative_found_points += n_filtered_cluster_datapoints;
      if (( filter_approach != "postfilter" && n_filtered_cluster_datapoints == 0)) continue;
      const ColMatrixUInt8 &A = _A[cluster];
      const ColMatrixUInt8 &B = _B[cluster];
      const Vector &A_correction = _A_corrections[cluster];
      const Vector &B_correction = _B_corrections[cluster];
      /* compute s = q^T A */
      quant_data.quantized_matvec_product_A(A, quantized_query, A_correction, quantization_factor,
                                            principal_axis, compensation_data, tmp.data());
      
      const float principal_axis_tmp = tmp[0];

      const float tmpfact = quant_query.quantize_vector(tmp.data() + 1, _max_rank - 1,
                                                        quantized_query_doubled.data());
      const float compensation_tmp =
          quantized_query_doubled.cast<float>().sum() * quant_data.compensation_factor;
      /* compute r = s^T B */
      auto start_matvec = std::chrono::high_resolution_clock::now();
      if (filter_approach == "prefilter" || filter_approach == "indexing") {
        quant_data.quantized_matvec_product_B_filter(B, quantized_query_doubled, &cluster_attribute_data_idxs, B_correction, tmpfact,
                                                    principal_axis_tmp, compensation_tmp,
                                                    &all_distances[current_cumulative_size], verbose && (i == 0));
      } else {
        quant_data.quantized_matvec_product_B(B, quantized_query_doubled, B_correction, tmpfact,
                                                    principal_axis_tmp, compensation_tmp,
                                                    &all_distances[current_cumulative_size]);
      }
      auto stop_matvec = std::chrono::high_resolution_clock::now();
      auto duration_matvec = std::chrono::duration_cast<std::chrono::microseconds>(stop_matvec - start_matvec);
      total_duration_matvec += duration_matvec;
      if (_euclidean)
        add_inplace(_cluster_norms[cluster].data(), &all_distances[current_cumulative_size],
                    _cluster_norms[cluster].size());
      if (use_attr_indexing) { // when we use indexing, we process fewer results than the full size of the cluster due to filtering them beforehand.
        std::memcpy(&all_idxs[current_cumulative_size], attribute_data_idxs.data(), attribute_data_idxs.size() * sizeof(int));
        current_cumulative_size += n_filtered_cluster_datapoints;
      } else {
        std::memcpy(&all_idxs[current_cumulative_size], _cluster_map[cluster].data(), sz * sizeof(int));
        current_cumulative_size += sz;
      }
    }
    auto stop_clusters = std::chrono::high_resolution_clock::now();
    auto duration_clusters = std::chrono::duration_cast<std::chrono::microseconds>(stop_clusters - start_clusters);
    auto duration_prework = std::chrono::duration_cast<std::chrono::microseconds>(stop_prework - start_prework);
    if (filter_approach != "postfilter" && !matching_results_found) {
      throw std::runtime_error("No matches found for filter attributes!");
    }
    if (verbose) {
      if (filter_approach == "indexing") std::cout << "total_filter_duration: " << ((double) total_filter_duration.count()) / 1000 << " microseconds" << std::endl;
      if (filter_approach == "indexing") std::cout << "total_filter_preloop_duration: " << total_filter_preloop_duration.count() << " microseconds" << std::endl;
      if (filter_approach == "indexing") std::cout << "duration_matvec: " << total_duration_matvec.count() << " microseconds" << std::endl;
      if (filter_approach == "prefilter") std::cout << "total_prefilter_duration: " << total_prefilter_duration.count() << " microseconds" << std::endl;
      std::cout << "!! Average ratio of satisfactory points to cluster size: " << ((double) cumulative_found_points) / cumulative_cluster_size << std::endl;
      std::cout << "cumulative_found_points: " << cumulative_found_points << std::endl;
      std::cout << "duration_clusters: " << duration_clusters.count() << " microseconds" << std::endl;
      std::cout << "duration_prework: " << duration_prework.count() << " microseconds" << std::endl;
      std::cout << "total_smallest_idx_sizes: " << total_smallest_idx_sizes << std::endl;
    }
    auto start_postwork = std::chrono::high_resolution_clock::now();
    ColVector filtered_distances(current_cumulative_size);
    for (int i = 0; i < current_cumulative_size; i++) { // this is needed because all_distances when using indexing will have more reserved memory than there are filtered datapoints so we need to filter it to include only the number of datapoints we want.
      filtered_distances[i] = all_distances[i];
    }
    Eigen::VectorXi shuffled_out(k); // why is this needed?
    // std::cout << "k: " << k << std::endl;
    // std::cout << "current_cumulative_size 1: " << current_cumulative_size << std::endl;
    // std::cout << "points_to_rerank: " << points_to_rerank << std::endl;
    select_final(_euclidean ? data : scaled_query.data(), k, points_to_rerank, current_cumulative_size,
                 all_idxs.data(), filtered_distances.data(), shuffled_out.data(), dist_out);
    if (filter_approach == "postfilter") {
      std::vector<int>* matched_idxs = new std::vector<int>();
      // std::cout << "_attributes.size(): " << _attributes.size() << std::endl;
      for (int i = 0; i < k; i++) {
        // std::cout << "shuffled_out[" << i << "]: " << shuffled_out[i] << std::endl;
      }
      for (int i = 0; i < k; i++) {
        // std::cout << "loop " << i << " ";
        bool filters_match = _attributes.matches(shuffled_out[i], filter_attributes);
        // bool filters_match = (_attributes[shuffled_out[i]] & filter_attributes) == filter_attributes;
        // for (const auto& attr: filter_attributes) {
        //   if (!_attributes[shuffled_out[i]].count(attr)) {
        //     filters_match = false;
        //     break;
        //   }
        // }
        if (filters_match) {
          matched_idxs->push_back(shuffled_out[i]);
        }
      }
      int matched_k = matched_idxs->size();
      // std::cout << "matched_k: " << matched_k << std::endl;
      // std::cout << "k: " << k << std::endl;
      int new_k = k;
      // if (matched_k < k) {
      //   std::cout << "true" << std::endl;
      // }
      std::vector<std::vector<int>*> ptr_vec;
      while (matched_k < k) { // if not enough datapoints are found in k results, double it and search again
        new_k = new_k * 2;
        if (new_k > _n_samples) new_k = _n_samples;
        // std::cout << "rerunning select_k with k=" << new_k << std::endl;
        Eigen::VectorXi new_out(new_k);
        select_final(_euclidean ? data : scaled_query.data(), new_k, points_to_rerank, current_cumulative_size,
                 all_idxs.data(), filtered_distances.data(), new_out.data(), dist_out);
        // select_k(new_k, new_out.data(), n_datapoints, NULL, dist.data(), dist_out, true);
        std::vector<int>* new_matched_idxs = new std::vector<int>();
        ptr_vec.push_back(new_matched_idxs);
        for (int i = 0; i < new_k; i++) {
          bool filters_match = _attributes.matches(new_out[i], filter_attributes);
          // bool filters_match = (_attributes[new_out[i]] & filter_attributes) == filter_attributes;
          // for (const auto& attr: filter_attributes) {
          //   if (!_attributes[new_out[i]].count(attr)) {
          //     filters_match = false;
          //     break;
          //   }
          // }
          if (filters_match) {
            new_matched_idxs->push_back(new_out[i]);
          }
        }
        matched_k = new_matched_idxs->size();
        matched_idxs = new_matched_idxs;
        // std::cout << "new_matched_idxs, first element " << (*new_matched_idxs)[0] << std::endl;
        // std::cout << "matched_idxs, first element " << (*matched_idxs)[0] << std::endl;
        // std::cout << "matched_idxs addr: " << matched_idxs << std::endl;
        if (new_k == _n_samples) {
          std::cout << "could not find enough samples (found " << matched_k << ")" << std::endl;
          break;
        } 
      }
      // std::cout << "matched_k 2: " << matched_k << std::endl;
      // std::cout << "matched_idxs addr 2: " << matched_idxs << std::endl;
      if (matched_k >= k) {
        for (int i = 0; i < k; i++) {
          // std::cout << "og idxs: " << (*matched_idxs)[i] << " - ";
          idx_out[i] = (*matched_idxs)[i];
          // std::cout << "idx_out[" << i << "]:" << idx_out[i] << " ";
        }
      }
      for (int i = 0; i < ptr_vec.size(); i++) { // deallocate memory
        delete ptr_vec[i];
      }
    } else {
      for (int i = 0; i < k; i++) {
        idx_out[i] = shuffled_out[i];
      }
    }
    auto stop_postwork = std::chrono::high_resolution_clock::now();
    auto duration_postwork = std::chrono::duration_cast<std::chrono::microseconds>(stop_postwork - start_postwork);
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
      // for (const auto& subvec : attr_subvecs) {
      //   std::cout << subvec.size() << " ";
      // }
      std::cout << std::endl;
      for (const auto& attr_subvec : attr_subvecs) {
        BitsetMatrix attribute_subvec_bitset;
        attribute_subvec_bitset.init(1, _n_attributes);
        for (int attribute_idx: attr_subvec) {
          attribute_subvec_bitset.set(0, attribute_idx);
        }
        // attribute_set attribute_subvec_bitset(attr_subvec.begin(), attr_subvec.end()); // turn the attribute partition vector into a set to eliminate duplicates and enable using it as a key for map
        attribute_partition_sets.push_back(attribute_subvec_bitset);
        std::vector<int> attribute_data_idx_vec; // vector of indexes of datapoints that have at least one of the attributes in attribute_subvec_bitset
        for (int i = 0; i<_n_samples; i++) { // for each datapoint
          if (_attributes.any_match(i, attribute_subvec_bitset)) attribute_data_idx_vec.push_back(i);
          // attribute_set attribute_match = (_attributes[i] & attribute_subvec_bitset);
          // if (attribute_match.count() > 0) {
          //   attribute_data_idx_vec.push_back(i);
          // }
          // for (const auto& attr: attribute_subvec_bitset) { // for each attribute in the attribute set
          //   if (_attributes[i].count(attr)) { // check if that datapoint has any of the current filter attributes
          //     attribute_data_idx_vec.push_back(i); // if yes, add it to the map for the corresponding attribute set
          //     break; // break so we do not add the same datapoint multiple times for different attributes
          //   }
          // }
        }
        for (int i = 0; i < _n_attributes; ++i) {
          if (attribute_subvec_bitset.is_set(0, i)) _attribute_index_map.insert({i, attribute_subvec_bitset});
        }
        // for (std::size_t i = attribute_subvec_bitset.find_first(); i != attribute_set::npos; i = attribute_subvec_bitset.find_next(i)) {
        //   _attribute_index_map.insert({i, attribute_subvec_bitset});
        // }
        // for (int i = 0; i < attribute_subvec_bitset.size(); ++i) {
        //   if (attribute_subvec_bitset[i]) _attribute_index_map.insert({i, attribute_subvec_bitset});
        // }
        // for (const auto& attr: attribute_subvec_bitset) {
        //   _attribute_index_map.insert({attr, attribute_subvec_bitset});
        // }
        _attribute_data_map.insert({attribute_subvec_bitset.key(0), attribute_data_idx_vec});
      }
    }

    /* Some printouts to make sure data indexes were stored correctly */
  //   for (int i = 0; i < attribute_partition_sets.size(); i++) {
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
  //   for(int i=0; i < 10; i++){
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
