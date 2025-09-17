#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <random>
#include "lorann.h"
#include <vector>
#include <chrono>
#include <bitset_matrix.h>
#include <H5Cpp.h>
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrix;
BitsetMatrix attribute_bitmatrix;
int _n_attributes;// = attribute_strings.size(); // 30
std::vector<int> attribute_idxs;
std::random_device rd; // obtain a random number from hardware
std::mt19937 gen(42); // seed the generator



std::vector<int> findUnion(Eigen::VectorXi& a, Eigen::VectorXi& b) {
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    std::vector<int> result;
    std::set_intersection(a.begin(), a.end(),
                          b.begin(), b.end(),
                          std::back_inserter(result));
    return result;
}


RowMatrix* load_vectors(
    int n_input_vecs,
    int n_attributes_per_datapoint,
    bool use_hdf5,
    const std::string& file_path)
{
    std::uniform_int_distribution<> attribute_selector_distr(0, _n_attributes - 1);
    std::uniform_int_distribution<> attribute_count_distr(1, n_attributes_per_datapoint);

    attribute_bitmatrix.init(n_input_vecs, _n_attributes);
    std::cout << "Using " << n_input_vecs << " input vectors." << std::endl;

    RowMatrix* ret_ptr = nullptr;

    if (use_hdf5) {
        // ==============================
        // HDF5 READER
        // ==============================
        try {
            H5::H5File file(file_path, H5F_ACC_RDONLY);
            H5::DataSet dataset = file.openDataSet("train");
            H5::DataSpace dataspace = dataset.getSpace();

            hsize_t dims[2];
            dataspace.getSimpleExtentDims(dims, nullptr);
            int n = dims[0];
            int d = dims[1];

            std::cout << "hdf5 n: " << n << " hdf5 d: " << d << std::endl;

            if (n_input_vecs > n) {
                throw std::runtime_error("Requested more vectors than available in HDF5 dataset");
            }

            std::vector<float> buffer(n_input_vecs * d);
            dataset.read(buffer.data(), H5::PredType::NATIVE_FLOAT);

            ret_ptr = new RowMatrix(n_input_vecs, d);

            for (int i = 0; i < n_input_vecs; ++i) {
                int _n_attributes_for_point = attribute_count_distr(gen);
                for (int k = 0; k < _n_attributes_for_point; ++k) {
                    int selected_attr_idx = attribute_selector_distr(gen);
                    attribute_bitmatrix.set(i, selected_attr_idx);
                }
                for (int j = 0; j < d; ++j) {
                    (*ret_ptr)(i, j) = buffer[i * d + j];
                }
            }

            std::cout << "Loaded vectors from HDF5 file." << std::endl;

        } catch (H5::FileIException& e) {
            throw std::runtime_error("Failed to open HDF5 file: " + file_path);
        } catch (H5::DataSetIException& e) {
            throw std::runtime_error("Failed to read dataset from HDF5 file: " + file_path);
        }

    } else {
        // ==============================
        // .VEC TEXT READER
        // ==============================
        std::ios::sync_with_stdio(false);
        std::ifstream fin(file_path);
        if (!fin.is_open()) {
            throw std::runtime_error("Could not open vec file: " + file_path);
        }

        std::string line;
        std::getline(fin, line); // first line is header
        std::istringstream header(line);
        int n, d;
        header >> n >> d;

        ret_ptr = new RowMatrix(n_input_vecs, d);

        int i = 0;
        while (std::getline(fin, line) && i < n_input_vecs) {
            std::istringstream iss(line);
            std::string token;
            iss >> token; // discard word/token

            float value;
            int j = 0;

            int _n_attributes_for_point = attribute_count_distr(gen);
            for (int k = 0; k < _n_attributes_for_point; ++k) {
                int selected_attr_idx = attribute_selector_distr(gen);
                attribute_bitmatrix.set(i, selected_attr_idx);
            }

            while (iss >> value) {
                (*ret_ptr)(i, j) = value;
                ++j;
            }
            ++i;
        }

        std::cout << "Loaded vectors from .vec text file." << std::endl;
    }

    return ret_ptr;
}


// RowMatrix* load_vectors(int n_input_vecs=999994, int n_attributes_per_datapoint=5) {
//   std::uniform_int_distribution<> attribute_selector_distr(0, _n_attributes-1); // define the range
//   std::uniform_int_distribution<> attribute_count_distr(1, n_attributes_per_datapoint); // define the range
//   attribute_bitmatrix.init(n_input_vecs, _n_attributes);
//   std::cout << "Using " << n_input_vecs << " input vectors." << std::endl;
//   std::ios::sync_with_stdio(false);
//   std::ifstream fin("wiki-news-300d-1M.vec");
//   if (!fin.is_open()) {
//     throw std::runtime_error(
//         "Could not open wiki-news-300d-1M.vec. Run `make prepare-data` first.");
//   }
//   std::string line;
//   std::getline(fin, line);
//   std::istringstream header(line);
//   int n, d;
//   header >> n >> d;

//   RowMatrix* ret_ptr = new RowMatrix(n_input_vecs, 300);

//   int i = 0;
//   // int n_this_attribute_points = 0;
//   while (std::getline(fin, line) && i < n_input_vecs) {
//     std::istringstream iss(line);
//     std::string token;
//     iss >> token;
//     int j = 0;
//     float value;
//     int _n_attributes_for_point = attribute_count_distr(gen);
//     for (int k = 0; k < _n_attributes_for_point; ++k) {
//       int selected_attr_idx = attribute_selector_distr(gen);
//       attribute_bitmatrix.set(i, selected_attr_idx);
//     }
//     while (iss >> value) {
//       (*ret_ptr)(i, j) = value;
//       ++j;
//     }
//     ++i;
//   }
//   std::cout << "Loading data complete." << std::endl;
//   return ret_ptr;
// }

Lorann::Lorann<Lorann::SQ4Quantizer>* index_ptr = nullptr;
RowMatrix* Q_ptr;

extern "C" {
  bool build_index(int* filter_attribute_list, int n_attributes, int n_attributes_per_datapoint, int n_attr_idx_partitions, int n_input_vecs, int n_clusters, int global_dim, int rank, int train_size, bool euclidean, bool use_hdf5, char* dataset_file_path) {
    std::cout << "Loading data..." << std::endl;
    std::cout << "use_hdf5: " << use_hdf5 << std::endl;
    std::cout << "dataset_file_path: " << dataset_file_path << std::endl;
    _n_attributes = n_attributes;
    RowMatrix* X = load_vectors(n_input_vecs, n_attributes_per_datapoint, use_hdf5, dataset_file_path);
    Q_ptr = X;
    // RowMatrix Q = X.topRows(1000);
    // Q_ptr =  new RowMatrix(X->topRows(100000));
    for (int i = 0; i < _n_attributes; ++i) {
      attribute_idxs.push_back(i);
    }
    std::cout << "Building the index..." << std::endl;
    index_ptr = new Lorann::Lorann<Lorann::SQ4Quantizer>(X->data(), X->rows(), X->cols(), n_clusters, global_dim, attribute_bitmatrix, attribute_idxs,
                                              rank, train_size, euclidean, false);
    index_ptr->build(true, -1, n_attr_idx_partitions);
    // std::cout << "index_ptr: " << index_ptr << std::endl;
    return true;
  }
}

// extern "C" {
//   /**
//    * filter_approach one of "prefilter", "postfilter", "indexing"
//    */
// float filter(int q_idx, bool exact_search, int k,  int clusters_to_search, int points_to_rerank, const char* filter_attribute, const char* filter_approach) {
//   if (index_ptr == nullptr) {
//     std::cout << "Index has not been initialised. Aborting..." << std::endl;
//     return 0;
//   }
//   Lorann::Lorann<Lorann::SQ4Quantizer> index = *index_ptr;
//   RowMatrix Q = (*Q_ptr).topRows(1000);
//   auto it = std::find(attribute_strings.begin(), attribute_strings.end(), filter_attribute);
//   int filter_idx = it - attribute_strings.begin();
//   BitsetMatrix filter_attributes;
//   filter_attributes.init(1, n_attributes);
//   filter_attributes.set(0, filter_idx);
//   Eigen::VectorXi exact_indices(k);
//   Eigen::VectorXi approx_indices(k);
//   index.exact_search((*Q_ptr).row(q_idx).data(), k, exact_indices.data(), filter_attributes, filter_approach);
//   bool verbose = false;
//   index.search((*Q_ptr).row(q_idx).data(), k, clusters_to_search, points_to_rerank, approx_indices.data(), filter_attributes, filter_approach, nullptr, verbose);
//   std::vector<int> res_union = findUnion(exact_indices, approx_indices);
//   float recall = res_union.size()/float(k);
//   return recall;
// }
// }

// extern "C" {
//   float filter_wrapper(int* idxs, int n_idxs, bool exact_search, int k,  int clusters_to_search, int points_to_rerank, const char* filter_attribute, const char* filter_approach) {

//     std::vector<float> results(n_idxs);
//     std::chrono::microseconds total_duration = (std::chrono::microseconds)0;
//     for ( int i = 0; i < n_idxs; i++) {
//       auto start = std::chrono::high_resolution_clock::now();
//       results[i] = filter(idxs[i], exact_search, k, clusters_to_search, points_to_rerank, filter_attribute, filter_approach);
//       auto stop = std::chrono::high_resolution_clock::now();
//       auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
//       total_duration = total_duration + duration;
//       // std::cout << "loop duration: " << duration.count() << " ms"<< std::endl;
//     }
//     std::chrono::microseconds avg_duration = total_duration / n_idxs;
//     std::cout << "Average query duration: " << avg_duration.count() << " microseconds" << std::endl;
//     float sum = 0;
//     for (size_t i = 0; i < n_idxs; ++i) {
//       sum += results[i];
//     }
//     float avg_recall = sum / n_idxs;
//     std::cout << "average recall: " << avg_recall << " microseconds" << std::endl;
//     return avg_recall;
//   }
// }


extern "C" {
  float fast_filter_wrapper_profiled(
    int* idxs,
    int n_idxs,
    int k,
    int clusters_to_search,
    int points_to_rerank,
    int* int_filter_attributes,
    int n_filter_attributes,
    const char* filter_approach,
    const char* exact_search_approach,
    float* recall,
    int* approx_latency,
    int* exact_latency,
    bool verbose) {
    Lorann::Lorann<Lorann::SQ4Quantizer> index = *index_ptr;
    BitsetMatrix filter_attributes;
    filter_attributes.init(1, _n_attributes);
    for (int i = 0; i<n_filter_attributes; ++i) {
      filter_attributes.set(0, int_filter_attributes[i]);
    }
    // for (int i = 0; i < _n_attributes; ++i) {
    //   std::cout << filter_attributes.is_set(0,i);
    // }
    std::vector<float> recall_vec(n_idxs);
    std::chrono::microseconds total_exact_duration = (std::chrono::microseconds) 0;
    std::chrono::microseconds total_approx_duration = (std::chrono::microseconds) 0;
    std::cout << "Beginning querying..." << std::endl;
    std::vector<Eigen::VectorXi> all_exact_indices(n_idxs);
    std::vector<Eigen::VectorXi> all_approx_indices(n_idxs);
    std::vector<int> bad_Recall_idxs;
    for ( int i = 0; i < n_idxs; i++) {
      Eigen::VectorXi exact_indices(k);
      auto start_exact = std::chrono::high_resolution_clock::now();
      try {
        index.exact_search((*Q_ptr).row(idxs[i]).data(), k, exact_indices.data(), filter_attributes, exact_search_approach);
      } catch (const std::runtime_error &e) {
        std::cout << e.what() << std::endl;
        break;
      }
      auto stop_exact = std::chrono::high_resolution_clock::now();
      auto duration_exact = std::chrono::duration_cast<std::chrono::microseconds>(stop_exact - start_exact);
      total_exact_duration = total_exact_duration + duration_exact;
      all_exact_indices.push_back(exact_indices);
      Eigen::VectorXi approx_indices(k);
      auto start_approx = std::chrono::high_resolution_clock::now();
      try {
        index.search((*Q_ptr).row(idxs[i]).data(), k, clusters_to_search, points_to_rerank, approx_indices.data(), filter_attributes, filter_approach, nullptr, verbose);
      } catch (const std::runtime_error &e) {
        std::cout << e.what() << std::endl;
        break;
      }
      auto stop_approx = std::chrono::high_resolution_clock::now();
      auto duration_approx = std::chrono::duration_cast<std::chrono::microseconds>(stop_approx - start_approx);
      total_approx_duration = total_approx_duration + duration_approx;
      all_approx_indices.push_back(approx_indices);
      std::vector<int> res_union = findUnion(exact_indices, approx_indices);
      float recall = res_union.size()/float(k);
      if (recall < 0.1) bad_Recall_idxs.push_back(idxs[i]); //std::cout << "ALERT Recall: " << recall << " for query index " << idxs[i] << std::endl;
      recall_vec[i] = recall;
      // std::cout << "idx: " << idxs[i] << std::endl;
      // std::cout << "exact indices:" << std::endl;
      // std::cout << exact_indices.transpose() << std::endl;
      // std::cout << "approx indices:" << std::endl;
      // std::cout << approx_indices.transpose() << std::endl;
    }
    int exact_indices_true_matches = 0;
    for (const auto& exact_indices: all_exact_indices) {
      for (const auto& idx: exact_indices) {
        bool matches = attribute_bitmatrix.matches(idx, filter_attributes);
        if (matches) exact_indices_true_matches++;
      }
    }
    double exact_indices_match_rate = ((double) exact_indices_true_matches) / (n_idxs*k);
    if (exact_indices_match_rate < 1) std::cout << "Exact indices match rate: " << exact_indices_match_rate<< std::endl;
    int approx_indices_true_matches = 0;
    for (const auto& approx_indices: all_approx_indices) {
      for (const auto& idx: approx_indices) {
        bool matches = attribute_bitmatrix.matches(idx, filter_attributes);
        if (matches) approx_indices_true_matches++;
      }
    }
    double approx_indices_match_rate = ((double) approx_indices_true_matches) / (n_idxs*k);
    if (approx_indices_match_rate < 1) std::cout << "Approximate indices match rate: " << ((double) approx_indices_true_matches) / (n_idxs*k) << std::endl;
    std::chrono::microseconds avg_exact_duration = total_exact_duration / n_idxs;
    std::chrono::microseconds avg_approx_duration = total_approx_duration / n_idxs;
    std::cout << "Average exact query duration: " << avg_exact_duration.count() << " microseconds" << std::endl;
    std::cout << "Average approx query duration: " << avg_approx_duration.count() << " microseconds" << std::endl;
    float sum = 0;
    for (size_t i = 0; i < n_idxs; ++i) {
      // std::cout << "recall: " << recall_vec[i] << std::endl;
      sum += recall_vec[i];
    }
    float avg_recall = sum / n_idxs;
    // std::cout << "average recall: " << avg_recall << std::endl;
    // std::cout << "[";
    // for (const auto& idx : bad_Recall_idxs) {
    //   std::cout << idx << ", ";
    // }
    // std::cout << "]";
    *recall = avg_recall;
    *approx_latency = avg_approx_duration.count();
    *exact_latency = avg_exact_duration.count();
    return avg_recall;
  }
}

// extern "C" {
//   float fast_filter_wrapper_profiled(
//     int* idxs,
//     int n_idxs,
//     int k,
//     int clusters_to_search,
//     int points_to_rerank,
//     const char** string_filter_attributes,
//     int n_filter_attributes,
//     const char* filter_approach,
//     const char* exact_search_approach,
//     float* recall,
//     int* approx_latency,
//     int* exact_latency) {
//     Lorann::Lorann<Lorann::SQ4Quantizer> index = *index_ptr;
//     BitsetMatrix filter_attributes;
//     filter_attributes.init(1, n_attributes);
//     // std::cout << "n_filter_attributes: " << n_filter_attributes << std::endl;
//     for (int i = 0; i<n_filter_attributes; ++i) {
//       auto it = std::find(attribute_strings.begin(), attribute_strings.end(), string_filter_attributes[i]);
//       int filter_idx = it - attribute_strings.begin();
//       filter_attributes.set(0, filter_idx);
//     }
//     // for (int i = 0; i < n_attributes; ++i) {
//     //   std::cout << filter_attributes.is_set(0,i);
//     // }
//     std::cout << std::endl;
//     std::vector<float> recall_vec(n_idxs);
//     std::chrono::microseconds total_exact_duration = (std::chrono::microseconds) 0;
//     std::chrono::microseconds total_approx_duration = (std::chrono::microseconds) 0;
//     std::cout << "Beginning querying..." << std::endl;
//     std::vector<Eigen::VectorXi> all_exact_indices(n_idxs);
//     std::vector<Eigen::VectorXi> all_approx_indices(n_idxs);
//     std::vector<int> bad_Recall_idxs;
//     for ( int i = 0; i < n_idxs; i++) {
//       Eigen::VectorXi exact_indices(k);
//       auto start_exact = std::chrono::high_resolution_clock::now();
//       try {
//         index.exact_search((*Q_ptr).row(idxs[i]).data(), k, exact_indices.data(), filter_attributes, exact_search_approach);
//       } catch (const std::runtime_error &e) {
//         std::cout << e.what() << std::endl;
//         break;
//       }
//       auto stop_exact = std::chrono::high_resolution_clock::now();
//       auto duration_exact = std::chrono::duration_cast<std::chrono::microseconds>(stop_exact - start_exact);
//       total_exact_duration = total_exact_duration + duration_exact;
//       all_exact_indices.push_back(exact_indices);
//       Eigen::VectorXi approx_indices(k);
//       auto start_approx = std::chrono::high_resolution_clock::now();
//       try {
//         index.search((*Q_ptr).row(idxs[i]).data(), k, clusters_to_search, points_to_rerank, approx_indices.data(), filter_attributes, filter_approach, nullptr, true);
//       } catch (const std::runtime_error &e) {
//         std::cout << e.what() << std::endl;
//         break;
//       }
//       auto stop_approx = std::chrono::high_resolution_clock::now();
//       auto duration_approx = std::chrono::duration_cast<std::chrono::microseconds>(stop_approx - start_approx);
//       total_approx_duration = total_approx_duration + duration_approx;
//       all_approx_indices.push_back(approx_indices);
//       std::vector<int> res_union = findUnion(exact_indices, approx_indices);
//       float recall = res_union.size()/float(k);
//       if (recall < 0.1) bad_Recall_idxs.push_back(idxs[i]); //std::cout << "ALERT Recall: " << recall << " for query index " << idxs[i] << std::endl;
//       recall_vec[i] = recall;
//       // std::cout << "idx: " << idxs[i] << std::endl;
//       // std::cout << "exact indices:" << std::endl;
//       // std::cout << exact_indices.transpose() << std::endl;
//       // std::cout << "approx indices:" << std::endl;
//       // std::cout << approx_indices.transpose() << std::endl;
//     }
//     int exact_indices_true_matches = 0;
//     for (const auto& exact_indices: all_exact_indices) {
//       for (const auto& idx: exact_indices) {
//         bool matches = attribute_bitmatrix.matches(idx, filter_attributes);
//         if (matches) exact_indices_true_matches++;
//       }
//     }
//     double exact_indices_match_rate = ((double) exact_indices_true_matches) / (n_idxs*k);
//     if (exact_indices_match_rate < 1) std::cout << "Exact indices match rate: " << exact_indices_match_rate<< std::endl;
//     int approx_indices_true_matches = 0;
//     for (const auto& approx_indices: all_approx_indices) {
//       for (const auto& idx: approx_indices) {
//         bool matches = attribute_bitmatrix.matches(idx, filter_attributes);
//         if (matches) approx_indices_true_matches++;
//       }
//     }
//     double approx_indices_match_rate = ((double) approx_indices_true_matches) / (n_idxs*k);
//     if (approx_indices_match_rate < 1) std::cout << "Approximate indices match rate: " << ((double) approx_indices_true_matches) / (n_idxs*k) << std::endl;
//     std::chrono::microseconds avg_exact_duration = total_exact_duration / n_idxs;
//     std::chrono::microseconds avg_approx_duration = total_approx_duration / n_idxs;
//     std::cout << "Average exact query duration: " << avg_exact_duration.count() << " microseconds" << std::endl;
//     std::cout << "Average approx query duration: " << avg_approx_duration.count() << " microseconds" << std::endl;
//     float sum = 0;
//     for (size_t i = 0; i < n_idxs; ++i) {
//       // std::cout << "recall: " << recall_vec[i] << std::endl;
//       sum += recall_vec[i];
//     }
//     float avg_recall = sum / n_idxs;
//     // std::cout << "average recall: " << avg_recall << std::endl;
//     // std::cout << "[";
//     // for (const auto& idx : bad_Recall_idxs) {
//     //   std::cout << idx << ", ";
//     // }
//     // std::cout << "]";
//     *recall = avg_recall;
//     std::cout << "avg_recall c++: " << avg_recall;
//     *approx_latency = avg_approx_duration.count();
//     *exact_latency = avg_exact_duration.count();
//     return avg_recall;
//   }
// }

// extern "C" {
//   int filter_proc(int n_attr_idx_partitions, int n_input_vecs, int n_clusters, int global_dim, int rank, int train_size, bool euclidean, int* idxs, int n_idxs, bool exact_search, int k,  int clusters_to_search, int points_to_rerank, const char* filter_attribute, const char* filter_approach) {
//     bool idx_built = build_index(n_attr_idx_partitions, n_input_vecs, n_clusters, global_dim, rank, train_size, euclidean);
//     std::vector<float> results(n_idxs);
//     for ( int i = 0; i < n_idxs; ++i) {
//       results[i] = filter(idxs[i], exact_search, k, clusters_to_search, points_to_rerank, filter_attribute, filter_approach);
//     }
//     float sum = 0;
//     for (size_t i = 0; i < n_idxs; ++i) {
//       sum += results[i];
//     }
//     float avg_recall = sum / n_idxs;
//     std::cout << "average recall: " << avg_recall << std::endl;
//     return 0;
//   }
// }

// int main() {
//   bool idx = build_index(10, 100000, 1024, 256, 32, 5, true);
//   for (int i = 4070; i < 4075; i++) {
//     filter(i, true, 10, 64, 2000, "brown", "indexing");
//   }
//   std::cout << "finished." << std::endl;
//   return 0;
// }