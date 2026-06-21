# Makefile

PKG_CONFIG ?= pkg-config

CXXFLAGS_COMMON = -std=c++17 -O3 -march=native -fopenmp -DNDEBUG -I../../lorann -flax-vector-conversions -fPIC
LDFLAGS_COMMON  = -lgomp -lhdf5_cpp -lhdf5

ABSL_CFLAGS := $(shell $(PKG_CONFIG) --cflags absl_hash)
# ABSL_LIBS   := $(shell $(PKG_CONFIG) --libs absl_hash)
ABSL_LIBS = -Wl,--no-as-needed \
  -labsl_raw_hash_set -labsl_hashtablez_sampler  \
  -labsl_hash -labsl_city \
  -labsl_strings -labsl_strings_internal \
  -labsl_base -labsl_spinlock_wait -labsl_throw_delegate \
  -labsl_raw_logging_internal -labsl_log_severity -labsl_int128 \
  -Wl,--as-needed

# Default target
all: example 

prepare-data:
	@echo "Preparing data..."
# 	wget "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip" -q --show-progress
# 	unzip wiki-news-300d-1M.vec.zip
# 	rm wiki-news-300d-1M.vec.zip
# 	wget "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5" -q --show-progress
# 	wget "http://ann-benchmarks.com/deep-image-96-angular.hdf5" -q --show-progress
	wget "http://ann-benchmarks.com/nytimes-256-angular.hdf5" -q --show-progress
# 	wget "http://ann-benchmarks.com/gist-960-euclidean.hdf5" -q --show-progress

example:
	$(CXX) -std=c++17 -O3 -march=native -fopenmp -DNDEBUG -I../../lorann -flax-vector-conversions example.cpp -o example -lgomp

filterdb:
	$(CXX) -std=c++17 -march=native -fopenmp -DNDEBUG -I../../lorann -flax-vector-conversions filter.cpp -o filterdb -lgomp -g
filter:
	$(CXX) $(CXXFLAGS_COMMON) $(ABSL_CFLAGS) filter.cpp -shared -o libfilter.so $(LDFLAGS_COMMON) $(ABSL_LIBS)
	
filter.s:
	$(CXX) -std=c++17 -O3 -march=native -fverbose-asm -fopenmp -DNDEBUG \
	  -I../../lorann -flax-vector-conversions \
	  $(ABSL_CFLAGS) \
	  -S filter.cpp -o filter.s

clean:
	rm example wiki-news-300d-1M.vec index.bin

hold:
	$(CXX) -shared -o libfilter.so filter.o

.PHONY: all prepare-data example clean
