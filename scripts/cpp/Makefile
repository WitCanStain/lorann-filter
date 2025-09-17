# Makefile

# Default target
all: example 

prepare-data:
	@echo "Preparing data..."
	wget "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip" -q --show-progress
	unzip wiki-news-300d-1M.vec.zip
	rm wiki-news-300d-1M.vec.zip
	wget "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5" -q --show-progress
	wget "http://ann-benchmarks.com/deep-image-96-angular.hdf5" -q --show-progress

example:
	$(CXX) -std=c++17 -O3 -march=native -fopenmp -DNDEBUG -I../../lorann -flax-vector-conversions example.cpp -o example -lgomp

filterdb:
	$(CXX) -std=c++17 -march=native -fopenmp -DNDEBUG -I../../lorann -flax-vector-conversions filter.cpp -o filterdb -lgomp -g
filter:
	$(CXX) -std=c++17 -O3 -march=native -fopenmp -DNDEBUG -I../../lorann -flax-vector-conversions filter.cpp -o libfilter.so -lgomp -fPIC -shared -lhdf5_cpp -lhdf5
	

clean:
	rm example wiki-news-300d-1M.vec index.bin

hold:
	$(CXX) -shared -o libfilter.so filter.o

.PHONY: all prepare-data example clean
