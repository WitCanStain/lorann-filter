# Makefile

# Default target
all: example 

prepare-data:
	@echo "Preparing data..."
	wget "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip" -q --show-progress
	unzip wiki-news-300d-1M.vec.zip
	rm wiki-news-300d-1M.vec.zip

example:
	$(CXX) -std=c++17 -O3 -march=native -fopenmp -DNDEBUG -I../../lorann -flax-vector-conversions example.cpp -o example -lgomp

filter:
	$(CXX) -std=c++17 -O3 -march=native -fopenmp -DNDEBUG -I../../lorann -flax-vector-conversions filter.cpp -o filter -lgomp

clean:
	rm example wiki-news-300d-1M.vec index.bin

.PHONY: all prepare-data example clean
