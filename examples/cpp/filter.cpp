#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <random>
#include "lorann.h"
#include <vector>


typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrix;
std::vector<std::string> attributes;
const int n_attributes = 10;
std::vector<std::string> attribute_strings = {"red", "green", "blue", "yellow", "orange", "purple", "black", "white", "pink", "brown"};
std::random_device rd; // obtain a random number from hardware
std::mt19937 gen(rd()); // seed the generator
std::uniform_int_distribution<> distr(0, n_attributes-1); // define the range

RowMatrix* load_vectors() {
  std::ios::sync_with_stdio(false);
  attributes.reserve(999994);
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

  RowMatrix* ret_ptr = new RowMatrix(999994, 300);

  int i = 0;
  while (std::getline(fin, line)) {
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
  std::cout << "test 3 len: " << i << std::endl;

  return ret_ptr;//().topRows(100000);
}

Lorann::Lorann<Lorann::SQ4Quantizer>* index_ptr = nullptr;
RowMatrix* Q_ptr;

extern "C" {
  bool build_index(int n_attr_partitions, int n_clusters, int global_dim, int rank, int train_size, bool euclidean) {
    std::cout << "Loading data..." << std::endl;
    RowMatrix* X = load_vectors();
    Q_ptr = X;
    // RowMatrix Q = X.topRows(1000);
    // Q_ptr =  &Q;//new RowMatrix(X.topRows(1000));

    std::cout << "Building the index..." << std::endl;
    std::vector<std::string> sliced_attributes(attributes.begin(), attributes.begin()+X->rows());
    index_ptr = new Lorann::Lorann<Lorann::SQ4Quantizer>(X->data(), X->rows(), X->cols(), n_clusters, global_dim, sliced_attributes, attribute_strings,
                                              rank, train_size, euclidean, false);
    std::cout << "index pointer: " << index_ptr << std::endl;
    index_ptr->build(true, -1, n_attr_partitions);
    return true;
  }
}

extern "C" {
int* filter(int q_idx, bool exact_search, int k,  int clusters_to_search, int points_to_rerank, const char* filter_attribute, const char* filter_approach) {
  std::cout << "index_ptr 1: " << index_ptr << std::endl;
  if (index_ptr == nullptr) {
    std::cout << "Index has not been initialised. Aborting..." << std::endl;
    return 0;
  }
  std::cout << "filter test 1: " << index_ptr << std::endl;
  Lorann::Lorann<Lorann::SQ4Quantizer> index = *index_ptr;
  std::cout << "filter test 2" << std::endl;
  RowMatrix Q = (*Q_ptr).topRows(1000);
  std::cout << "mark filter_attribute: " << filter_attribute << std::endl;
  std::set<std::string> filter_attributes = {filter_attribute};
  std::cout << "mark k: " << k << std::endl;

  Eigen::VectorXi indices(k), indices_exact(k);
  std::cout << "mark 3" << std::endl;
  std::cout << "----" << std::endl;
  std::cout << Q.row(q_idx).data() << std::endl;
  if (exact_search) {
    std::cout << "Querying the index using exact search..." << std::endl;
    index.exact_search(Q.row(q_idx).data(), k, indices_exact.data(), filter_attributes, filter_approach);
    std::cout << indices_exact.transpose() << std::endl;
    for (const auto& idx : indices_exact) {
      std::cout << attributes[idx] << " ";
    }
  } else {
    std::cout << std::endl << "Querying the index using approximate search..." << std::endl;
    index.search(Q.row(q_idx).data(), k, clusters_to_search, points_to_rerank, indices.data(), filter_attributes, filter_approach);
    std::cout << "approximate search finished." << std::endl;
    std::cout << indices.transpose() << std::endl;
    for (const auto& idx : indices) {
      std::cout << "approx index: " << idx << "| ";
      std::cout << "attr: " << attributes[idx] << std::endl;
    }
  }
  std::cout << std::endl << "Saving the index to disk..." << std::endl;
  std::ofstream output_file("index.bin", std::ios::binary);
  cereal::BinaryOutputArchive output_archive(output_file);
  output_archive(index);
  return indices_exact.data();
}
}
