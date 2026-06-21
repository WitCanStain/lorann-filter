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
#include <cstdint>
#include <memory>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrix;
BitsetMatrix attribute_bitmatrix;
std::vector<std::uint32_t> attribute_ints;
std::vector<int> attribute_idxs;
int _n_attributes;// = attribute_strings.size(); // 30
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
    float selectivity,
    bool use_hdf5,
    const std::string& file_path)
{
    std::uniform_int_distribution<> attribute_selector_distr(1, _n_attributes - 1);
    std::uniform_int_distribution<> attribute_count_distr(1, n_attributes_per_datapoint);
    std::uniform_real_distribution<> selectivity_distr(0.0, 1.0);
    attribute_ints.reserve(n_input_vecs);
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

            std::vector<float> buffer(n * d);
            dataset.read(buffer.data(), H5::PredType::NATIVE_FLOAT);
            ret_ptr = new RowMatrix(n_input_vecs, d);

            for (int i = 0; i < n_input_vecs; ++i) {
              int this_attribute_int = 0;
              int _n_attributes_for_point = n_attributes_per_datapoint; //attribute_count_distr(gen);
              bool selectivity_criterion_fulfilled = selectivity_distr(gen) < selectivity;
              if (selectivity_criterion_fulfilled) {
                attribute_bitmatrix.set(i, 0);
                this_attribute_int |= (1u << 0);
              }
              for (int k = 1; k < _n_attributes_for_point; ++k) {
                  int selected_attr_idx = attribute_selector_distr(gen);
                  attribute_bitmatrix.set(i, selected_attr_idx);
                  this_attribute_int |= (1u << selected_attr_idx);
              }
              
              for (int j = 0; j < d; ++j) {
                  (*ret_ptr)(i, j) = buffer[i * d + j];
              }
              attribute_ints.push_back(this_attribute_int);
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
            int this_attribute_int = 0;
            int _n_attributes_for_point = n_attributes_per_datapoint; //attribute_count_distr(gen);
            bool selectivity_criterion_fulfilled = selectivity_distr(gen) < selectivity;
            if (selectivity_criterion_fulfilled) {
              attribute_bitmatrix.set(i, 0);
              this_attribute_int |= (1u << 0);
            }
            for (int k = 1; k < _n_attributes_for_point; ++k) {
                int selected_attr_idx = attribute_selector_distr(gen);
                attribute_bitmatrix.set(i, selected_attr_idx);
                this_attribute_int |= (1u << selected_attr_idx);
            }
            attribute_ints.push_back(this_attribute_int);
            while (iss >> value) {
                (*ret_ptr)(i, j) = value;
                ++j;
            }
            ++i;
        }

        std::cout << "Loaded vectors from .vec text file." << std::endl;
    }

    for (size_t i = 0; i < n_input_vecs; ++i) {
        if (attribute_ints[i] != attribute_bitmatrix.get_attribute_int(i)) {
            std::cout << "Mismatch at index " << i << ": attribute_ints = "
                      << std::bitset<32>(attribute_ints[i])
                      << ", bitmatrix = "
                      << std::bitset<32>(attribute_bitmatrix.get_attribute_int(i))
                      << std::endl;
        }
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

std::unique_ptr<Lorann::Lorann<Lorann::SQ4Quantizer>> index_ptr;
std::unique_ptr<RowMatrix> Q_ptr;

extern "C" {
  bool build_index(int* filter_attribute_list, int n_attributes, int n_attributes_per_datapoint, int n_attr_idx_partitions, float selectivity, int n_input_vecs, int n_clusters, int global_dim, int rank, int train_size, bool euclidean, bool use_hdf5, char* dataset_file_path) {
    std::cout << "Loading data..." << std::endl;
    std::cout << "use_hdf5: " << use_hdf5 << std::endl;
    std::cout << "dataset_file_path: " << dataset_file_path << std::endl;
    index_ptr.reset();
    Q_ptr.reset();
    attribute_ints.clear();
    attribute_idxs.clear();
    _n_attributes = n_attributes;
    RowMatrix* X = load_vectors(n_input_vecs, n_attributes_per_datapoint, selectivity, use_hdf5, dataset_file_path);
    Q_ptr.reset(X); // take ownership of the returned raw pointer
    // RowMatrix Q = X.topRows(1000);
    // Q_ptr =  new RowMatrix(X->topRows(100000));
    for (int i = 0; i < _n_attributes; ++i) {
      attribute_idxs.push_back(i);
    }
    std::cout << "Building the index..." << std::endl;
    index_ptr = std::make_unique<Lorann::Lorann<Lorann::SQ4Quantizer>>(X->data(), X->rows(), X->cols(), n_clusters, global_dim, attribute_bitmatrix, attribute_idxs, attribute_ints,
                                              rank, train_size, euclidean, false);
    index_ptr->build(true, -1, n_attr_idx_partitions);
    // std::cout << "index_ptr: " << index_ptr << std::endl;
    return true;
  }
}

extern "C" {
  float fast_filter_wrapper_profiled(
    int* idxs,
    int n_idxs,
    int k,
    int M,
    int clusters_to_search,
    int points_to_rerank,
    int* int_filter_attributes,
    int n_filter_attributes,
    const char* filter_approach,
    const char* exact_search_approach,
    float* recall,
    int* approx_latency,
    int* exact_latency,
    int* exact_filter_time,
    int* part_time,
    bool verbose) {
    Lorann::Lorann<Lorann::SQ4Quantizer> index = *index_ptr;
    uint32_t filter_attributes = 0;
    uint32_t filter_attributes_int = 0;
    for (int i = 0; i < n_filter_attributes; ++i) {
      filter_attributes |= (1u << int_filter_attributes[i]);
      filter_attributes_int |= (1u << int_filter_attributes[i]);
    }

    std::vector<float> recall_vec(n_idxs);
    std::chrono::microseconds total_exact_duration = std::chrono::microseconds(0);
    std::chrono::microseconds total_exact_filter_duration = std::chrono::microseconds(0);
    std::chrono::microseconds total_approx_duration = std::chrono::microseconds(0);
    std::chrono::microseconds duration_clusters = std::chrono::microseconds(0);
    std::vector<Eigen::VectorXi> all_exact_indices(n_idxs);
    std::vector<Eigen::VectorXi> all_approx_indices(n_idxs);

    std::cout << "Beginning querying..." << std::endl;
    for (int i = 0; i < n_idxs; ++i) {
      Eigen::VectorXi exact_indices(k);
      std::chrono::microseconds exact_filter_duration = std::chrono::microseconds(0);

      auto start_exact = std::chrono::high_resolution_clock::now();
      try {
        index.exact_search(
            (*Q_ptr).row(idxs[i]).data(),
            k,
            exact_indices.data(),
            filter_attributes,
            filter_attributes_int,
            exact_search_approach,
            nullptr,
            &exact_filter_duration,
            verbose);
      } catch (const std::runtime_error &e) {
        std::cout << e.what() << std::endl;
        break;
      }
      auto stop_exact = std::chrono::high_resolution_clock::now();
      auto duration_exact = std::chrono::duration_cast<std::chrono::microseconds>(stop_exact - start_exact);
      total_exact_duration += duration_exact;
      total_exact_filter_duration += exact_filter_duration;
      all_exact_indices[i] = exact_indices;

      Eigen::VectorXi approx_indices(k);
      std::chrono::microseconds duration_cluster = std::chrono::microseconds(0);
      auto start_approx = std::chrono::high_resolution_clock::now();
      try {
        index.search(
            (*Q_ptr).row(idxs[i]).data(),
            k,
            M,
            clusters_to_search,
            points_to_rerank,
            approx_indices.data(),
            filter_attributes,
            filter_attributes_int,
            filter_approach,
            &duration_cluster,
            nullptr,
            verbose);
      } catch (const std::runtime_error &e) {
        std::cout << e.what() << std::endl;
        break;
      }
      auto stop_approx = std::chrono::high_resolution_clock::now();
      duration_clusters += duration_cluster;
      auto duration_approx = std::chrono::duration_cast<std::chrono::microseconds>(stop_approx - start_approx);
      total_approx_duration += duration_approx;
      all_approx_indices[i] = approx_indices;

      std::vector<int> res_union = findUnion(exact_indices, approx_indices);
      recall_vec[i] = res_union.size() / float(k);
    }

    int exact_indices_true_matches = 0;
    for (const auto& exact_indices : all_exact_indices) {
      for (const auto& idx : exact_indices) {
        if (attribute_bitmatrix.matches_int(idx, filter_attributes)) {
          exact_indices_true_matches++;
        }
      }
    }

    int approx_indices_true_matches = 0;
    for (const auto& approx_indices : all_approx_indices) {
      for (const auto& idx : approx_indices) {
        if (attribute_bitmatrix.matches_int(idx, filter_attributes)) {
          approx_indices_true_matches++;
        }
      }
    }

    std::chrono::microseconds avg_exact_duration = total_exact_duration / n_idxs;
    std::chrono::microseconds avg_exact_filter_duration = total_exact_filter_duration / n_idxs;
    std::chrono::microseconds avg_approx_duration = total_approx_duration / n_idxs;
    std::chrono::microseconds avg_cluster_duration = duration_clusters / n_idxs;

    float sum = 0;
    for (float value : recall_vec) {
      sum += value;
    }
    float avg_recall = sum / n_idxs;

    *recall = avg_recall;
    *approx_latency = avg_approx_duration.count();
    *exact_latency = avg_exact_duration.count();
    *exact_filter_time = avg_exact_filter_duration.count();
    *part_time = avg_cluster_duration.count();
    return avg_recall;
  }
}



// int main() {
//   bool idx = build_index(10, 100000, 1024, 256, 32, 5, true);
//   for (int i = 4070; i < 4075; i++) {
//     filter(i, true, 10, 64, 2000, "brown", "indexing");
//   }
//   std::cout << "finished." << std::endl;
//   return 0;
// }