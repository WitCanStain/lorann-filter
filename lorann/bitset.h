#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <functional>
#include <immintrin.h>
class Bitset {
public:
    using Block = uint64_t;
    static constexpr size_t BITS_PER_BLOCK = sizeof(Block) * 8;

private:
    size_t n_bits;
    size_t n_blocks;
    std::vector<Block> data;

public:
    Bitset() = default;

    Bitset(size_t n_bits_, bool initialize_ones = false) : n_bits(n_bits_) {
        n_blocks = (n_bits + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
        data.resize(n_blocks, initialize_ones ? 1 : 0);
    }

    size_t size() const { return n_bits; }

    void set(size_t idx) {
        if (idx >= n_bits) throw std::out_of_range("Bit index out of range");
        data[idx / BITS_PER_BLOCK] |= (Block(1) << (idx % BITS_PER_BLOCK));
    }

    void clear(size_t idx) {
        if (idx >= n_bits) throw std::out_of_range("Bit index out of range");
        data[idx / BITS_PER_BLOCK] &= ~(Block(1) << (idx % BITS_PER_BLOCK));
    }

    bool is_set(size_t idx) const {
        if (idx >= n_bits) throw std::out_of_range("Bit index out of range");
        return (data[idx / BITS_PER_BLOCK] >> (idx % BITS_PER_BLOCK)) & 1;
    }

    // Bitwise AND with another Bitset
    Bitset operator&(const Bitset& other) const {
        Bitset result(n_bits);
        for (size_t i = 0; i < n_blocks; ++i)
            result.data[i] = data[i] & other.data[i];
        return result;
    }

    void inline bitwise_and(const Bitset& other, Bitset& result) const {
        if (n_bits != other.n_bits) throw std::invalid_argument("Size mismatch");
        if (result.n_bits != n_bits) result = Bitset(n_bits);

        size_t blocks = n_blocks;
        size_t i = 0;

        // Process 4 blocks at a time using AVX2
        size_t simd_blocks = blocks / 4 * 4;
        // #pragma omp parallel for
        for (i = 0; i < simd_blocks; i += 4) {
            __m256i a = _mm256_loadu_si256((__m256i*)&data[i]);
            __m256i b = _mm256_loadu_si256((__m256i*)&other.data[i]);
            __m256i c = _mm256_and_si256(a, b);
            _mm256_storeu_si256((__m256i*)&result.data[i], c);
        }

        // Process remaining blocks
        for (i = simd_blocks; i < blocks; ++i) {
            result.data[i] = data[i] & other.data[i];
        }
    }

    std::vector<size_t> get_set_bit_positions() const {
        std::vector<size_t> positions;
        positions.reserve(n_bits / 2); // rough guess to avoid reallocs

        for (size_t block_idx = 0; block_idx < n_blocks; ++block_idx) {
            uint64_t block = data[block_idx];
            while (block) {
                uint64_t t = block & -block;              // isolate lowest set bit
                size_t bit = __builtin_ctzll(block);      // count trailing zeros
                positions.push_back(block_idx * 64 + bit);
                block &= block - 1;                        // clear the lowest set bit
            }
        }
        return positions;
    }

    // Clear all bits
    void clear_all() {
        std::fill(data.begin(), data.end(), 0);
    }

    
};
