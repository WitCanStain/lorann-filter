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
std::uniform_int_distribution<> attribute_selector_distr(0, n_attributes-1); // define the range

RowMatrix load_vectors() {
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

  RowMatrix ret(999994, 300);

  int i = 0;
  while (std::getline(fin, line)) {
    std::istringstream iss(line);
    std::string token;
    iss >> token;
    int j = 0;
    float value;
    int rand_int = attribute_selector_distr(gen);
    std::string random_attribute = attribute_strings[rand_int];
    // std::cout << "random attribute: " << random_attribute << std::endl;
    attributes.push_back(random_attribute);
    while (iss >> value) {
      ret(i, j) = value;
      ++j;
    }
    ++i;
  }
  std::cout << "test 3 len: " << i << std::endl;

  return ret.topRows(100000);
}

int main() {
  std::cout << "Loading data..." << std::endl;
  RowMatrix X = load_vectors();
  RowMatrix Q = X.topRows(1000);
  // std::cout <<  "X cols: " << X.cols() << std::endl;
  // std::cout <<  "Q cols: " << Q.cols() << std::endl;

  const int k = 10;
  const int n_attr_partitions = 10;
  const int n_clusters = 1024;
  const int global_dim = 256;
  const int rank = 32;
  const int train_size = 5;
  const bool euclidean = true;

  const int clusters_to_search = 64;
  const int points_to_rerank = 2000;

  std::cout << "Building the index..." << std::endl;
  std::vector<std::string> sliced_attributes(attributes.begin(), attributes.begin()+X.rows());
  Lorann::Lorann<Lorann::SQ4Quantizer> index(X.data(), X.rows(), X.cols(), n_clusters, global_dim, sliced_attributes, attribute_strings,
                                             rank, train_size, euclidean, false);
  index.build(true, -1, n_attr_partitions);
  
  std::set<std::string> filter_attributes = {"brown"};

  Eigen::VectorXi indices(k), indices_exact(k);
  std::cout << "----" << std::endl;
  std::cout << Q.row(0).data() << std::endl;
  std::cout << "Querying the index using exact search..." << std::endl;
  index.exact_search(Q.row(0).data(), k, indices_exact.data(), filter_attributes, "postfilter");
  std::cout << indices_exact.transpose() << std::endl;
  for (const auto& idx : indices_exact) {
    std::cout << sliced_attributes[idx] << " ";
  }
  std::cout << std::endl << "Querying the index using approximate search..." << std::endl;
  index.search(Q.row(0).data(), k, clusters_to_search, points_to_rerank, indices.data(), filter_attributes, "indexing");
  std::cout << "approximate search finished." << std::endl;
  std::cout << indices.transpose() << std::endl;
  for (const auto& idx : indices) {
    std::cout << "approx index: " << idx << "| ";
    std::cout << "attr: " << sliced_attributes[idx] << std::endl;
  }
  std::cout << std::endl << "Saving the index to disk..." << std::endl;
  std::ofstream output_file("index.bin", std::ios::binary);
  cereal::BinaryOutputArchive output_archive(output_file);
  output_archive(index);

  return 0;
}
