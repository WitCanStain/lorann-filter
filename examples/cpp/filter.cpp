#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <random>
#include "lorann.h"
#include <vector>
#include <chrono>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrix;
std::vector<std::string> attributes;
const int n_attributes = 10;
// const int n_input_vecs = 50000; //999994
std::vector<std::string> attribute_strings = {"red", "green", "blue", "yellow", "orange", "purple", "black", "white", "pink", "brown"};
std::random_device rd; // obtain a random number from hardware
std::mt19937 gen(42); // seed the generator
std::uniform_int_distribution<> distr(0, n_attributes-1); // define the range

std::vector<int> findUnion(Eigen::VectorXi& a, Eigen::VectorXi& b) {
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
  	
    std::vector<int> result;
    
    std::set_intersection(a.begin(), a.end(),
                          b.begin(), b.end(),
                          std::back_inserter(result));
  	
    return result;
}

RowMatrix* load_vectors(int n_input_vecs=999994) {
  std::cout << "Using " << n_input_vecs << " input vectors." << std::endl;
  std::ios::sync_with_stdio(false);
  attributes.reserve(n_input_vecs); //999994
  std::ifstream fin("wiki-news-300d-1M.vec");
  if (!fin.is_open()) {
    throw std::runtime_error(
        "Could not open wiki-news-300d-1M.vec. Run `make prepare-data` first.");
  }
  std::string line;
  std::getline(fin, line);
  std::istringstream header(line);
  int n, d;
  header >> n >> d;

  RowMatrix* ret_ptr = new RowMatrix(n_input_vecs, 300);

  int i = 0;
  while (std::getline(fin, line) && i < n_input_vecs) {
    std::istringstream iss(line);
    std::string token;
    iss >> token;
    int j = 0;
    float value;
    int rand_int = distr(gen);
    std::string random_attribute = attribute_strings[rand_int];
    // std::cout << "random attribute: " << random_attribute << std::endl;
    attributes.push_back(random_attribute);
    while (iss >> value) {
      (*ret_ptr)(i, j) = value;
      ++j;
    }
    ++i;
  }

  return ret_ptr;//().topRows(100000);
}

Lorann::Lorann<Lorann::SQ4Quantizer>* index_ptr = nullptr;
RowMatrix* Q_ptr;

extern "C" {
  bool build_index(int n_attr_partitions, int n_input_vecs, int n_clusters, int global_dim, int rank, int train_size, bool euclidean) {
    std::cout << "Loading data..." << std::endl;
    RowMatrix* X = load_vectors(n_input_vecs);
    Q_ptr = X;
    // RowMatrix Q = X.topRows(1000);
    // Q_ptr =  new RowMatrix(X->topRows(100000));

    std::cout << "Building the index..." << std::endl;
    std::vector<std::string> sliced_attributes(attributes.begin(), attributes.begin()+X->rows());
    index_ptr = new Lorann::Lorann<Lorann::SQ4Quantizer>(X->data(), X->rows(), X->cols(), n_clusters, global_dim, sliced_attributes, attribute_strings,
                                              rank, train_size, euclidean, false);
    index_ptr->build(true, -1, n_attr_partitions);
    std::cout << "index_ptr: " << index_ptr << std::endl;
    return true;
  }
}

extern "C" {
  /**
   * filter_approach one of "prefilter", "postfilter", "indexing"
   */
float filter(int q_idx, bool exact_search, int k,  int clusters_to_search, int points_to_rerank, const char* filter_attribute, const char* filter_approach) {
  if (index_ptr == nullptr) {
    std::cout << "Index has not been initialised. Aborting..." << std::endl;
    return 0;
  }
  Lorann::Lorann<Lorann::SQ4Quantizer> index = *index_ptr;
  RowMatrix Q = (*Q_ptr).topRows(1000);
  // std::cout << "mark filter_attribute: " << filter_attribute << std::endl;
  std::set<std::string> filter_attributes = {filter_attribute};
  // std::cout << "q_idx: " << q_idx << std::endl;
  Eigen::VectorXi exact_indices(k);
  Eigen::VectorXi approx_indices(k);
  // std::cout << "Q.row(q_idx).data() " << Q.row(q_idx).data()[0] << std::endl;

  // std::cout << "mark 3" << std::endl;
  // std::cout << "----" << std::endl;
  // std::cout << "Q rows: " << Q.rows() << std::endl;
  // std::cout << "Q cols: " << Q.cols() << std::endl;

  // std::cout << "Querying the index using exact search..." << std::endl;
  // std::cout << "running exact search with " << filter_approach << " strategy and k: " << k << std::endl;
  index.exact_search((*Q_ptr).row(q_idx).data(), k, exact_indices.data(), filter_attributes, filter_approach);
  
  for (const auto& idx : exact_indices) {
    // std::cout << attributes[idx] << " ";
  }
  // std::cout << std::endl << "Querying the index using approximate search..." << std::endl;
    // std::cout << "running approximate search with " << filter_approach << " strategy and k: " << k << std::endl;
  bool verbose = false;
  index.search((*Q_ptr).row(q_idx).data(), k, clusters_to_search, points_to_rerank, approx_indices.data(), filter_attributes, filter_approach, nullptr, verbose);
  // std::cout << exact_indices.transpose() << std::endl;
  // // std::cout << "approx search results:" << std::endl;
  // std::cout << approx_indices.transpose() << std::endl;
  // for (const auto& idx : approx_indices) {
  //   std::cout << attributes[idx] << " ";
  // }
  // for (const auto& idx : exact_indices) {
  //   std::cout << attributes[idx] << " ";
  // }
  std::vector<int> res_union = findUnion(exact_indices, approx_indices);
  // for (int i = 0; i < res_union.size(); i++) {
  //   std::cout << res_union[i] << " ";
  // }
  float recall = res_union.size()/float(k);
  std::cout << "Recall: " << recall << std::endl;
  
  // if (exact_search) {
    
  // } else {
    
  // }
  // std::cout << std::endl << "Saving the index to disk..." << std::endl;
  std::ofstream output_file("index.bin", std::ios::binary);
  cereal::BinaryOutputArchive output_archive(output_file);
  output_archive(index);
  return recall;
}
}

extern "C" {
  float filter_wrapper(int* idxs, int n_idxs, bool exact_search, int k,  int clusters_to_search, int points_to_rerank, const char* filter_attribute, const char* filter_approach) {
    std::cout << "Entered filter_wrapper" << std::endl;

    std::vector<float> results(n_idxs);
    std::chrono::milliseconds total_duration = (std::chrono::milliseconds)0;
    for ( int i = 0; i < n_idxs; i++) {
      auto start = std::chrono::high_resolution_clock::now();
      results[i] = filter(idxs[i], exact_search, k, clusters_to_search, points_to_rerank, filter_attribute, filter_approach);
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
      total_duration = total_duration + duration;
      std::cout << "loop duration: " << duration.count() << " ms"<< std::endl;
    }
    std::chrono::milliseconds avg_duration = total_duration / n_idxs;
    std::cout << "Average query duration: " << avg_duration.count() << " ms" << std::endl;
    float sum = 0;
    for (size_t i = 0; i < n_idxs; ++i) {
      sum += results[i];
    }
    float avg_recall = sum / n_idxs;
    std::cout << "average recall: " << avg_recall << std::endl;
    return avg_recall;
  }
}

extern "C" {
  int filter_proc(int n_attr_partitions, int n_input_vecs, int n_clusters, int global_dim, int rank, int train_size, bool euclidean, int* idxs, int n_idxs, bool exact_search, int k,  int clusters_to_search, int points_to_rerank, const char* filter_attribute, const char* filter_approach) {
    bool idx_built = build_index(n_attr_partitions, n_input_vecs, n_clusters, global_dim, rank, train_size, euclidean);
    std::vector<float> results(n_idxs);
    for ( int i = 0; i < n_idxs; ++i) {
      results[i] = filter(idxs[i], exact_search, k, clusters_to_search, points_to_rerank, filter_attribute, filter_approach);
    }
    float sum = 0;
    for (size_t i = 0; i < n_idxs; ++i) {
      sum += results[i];
    }
    float avg_recall = sum / n_idxs;
    std::cout << "average recall: " << avg_recall << std::endl;
    return 0;
  }
}

int main() {
  bool idx = build_index(10, 100000, 1024, 256, 32, 5, true);
  for (int i = 4070; i < 4075; i++) {
    filter(i, true, 10, 64, 2000, "brown", "indexing");
  }
  std::cout << "finished." << std::endl;
  return 0;
}