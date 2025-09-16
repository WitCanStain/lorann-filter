#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <functional>
#include <immintrin.h>
#include <format>
#include <iostream>
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

    explicit Bitset(size_t n_bits, bool init_ones = false)
        : n_bits(n_bits), n_blocks((n_bits + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK), data((n_bits + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK) 
    {
        if (init_ones) {
            // fill with all ones
            std::fill(data.begin(), data.end(), ~Block(0));

            // mask off extra bits in the last block
            size_t extra_bits = n_bits % 64;
            if (extra_bits != 0) {
                data.back() &= ((Block(1) << extra_bits) - 1);
            }
        }
    }

    size_t size() const { return n_bits; }

    inline void set(size_t idx) {
        if (idx >= n_bits) throw std::out_of_range("set Bit index " + std::to_string(idx) + " out of range");
        data[idx / BITS_PER_BLOCK] |= (Block(1) << (idx % BITS_PER_BLOCK));
    }

    inline void clear(size_t idx) {
        if (idx >= n_bits) throw std::out_of_range("Bit index out of range");
        data[idx / BITS_PER_BLOCK] &= ~(Block(1) << (idx % BITS_PER_BLOCK));
    }

    bool is_set(size_t idx) const {
        if (idx >= n_bits) throw std::out_of_range("is_set Bit index " + std::to_string(idx) + " out of range");
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
        if (n_bits != other.n_bits) throw std::invalid_argument("Size mismatch: " + std::to_string(n_bits) + " and " + std::to_string(other.n_bits));
        if (result.n_bits != n_bits) std::cout << "problem"; //result = Bitset(n_bits);

        // Process 4 blocks at a time using AVX2
        size_t simd_blocks = n_blocks / 4 * 4;
        // #pragma omp parallel for
        for (size_t i = 0; i < simd_blocks; i += 4) {
            __m256i a = _mm256_loadu_si256((__m256i*)&data[i]);
            __m256i b = _mm256_loadu_si256((__m256i*)&other.data[i]);
            __m256i c = _mm256_and_si256(a, b);
            _mm256_storeu_si256((__m256i*)&result.data[i], c);
        }

        // Process remaining blocks
        for (size_t i = simd_blocks; i < n_blocks; ++i) {
            result.data[i] = data[i] & other.data[i];
        }
    }

    inline void get_set_bit_positions(std::vector<size_t> &positions) const {
        positions.reserve(n_bits / 2); // rough guess to avoid reallocs
        for (size_t block_idx = 0; block_idx < n_blocks; ++block_idx) {
            uint64_t block = data[block_idx];
            while (block) {
                size_t bit = __builtin_ctzll(block);      // count trailing zeros
                positions.push_back(block_idx * BITS_PER_BLOCK + bit);
                block &= (block - 1);                        // clear the lowest set bit
            }
        }
    }

    inline void get_set_bit_positions_simd(std::vector<size_t> &positions) const {
        positions.reserve(n_bits / 2); // rough guess
        size_t base = 0;

        // Process 4 blocks (256 bits) at a time
        size_t simd_blocks = (n_blocks / 4) * 4;
        for (size_t i = 0; i < simd_blocks; i += 4) {
            __m256i v = _mm256_loadu_si256((__m256i*)&data[i]);
            if (_mm256_testz_si256(v, v)) {
                base += 256;
                continue; // all zero, skip entire 256-bit chunk
            }

            // Fallback to scalar for each of the 4 blocks
            for (int j = 0; j < 4; ++j) {
                uint64_t block = data[i + j];
                while (block) {
                    int pos = __builtin_ctzll(block);
                    positions.push_back(base + pos);
                    block &= (block - 1);
                }
                base += BITS_PER_BLOCK;
            }
        }

        // Handle remaining blocks (tail)
        for (size_t i = simd_blocks; i < n_blocks; ++i) {
            uint64_t block = data[i];
            while (block) {
                int pos = __builtin_ctzll(block);
                positions.push_back(i * BITS_PER_BLOCK + pos);
                block &= (block - 1);
            }
        }
    }

    // Clear all bits
    void clear_all() {
        std::fill(data.begin(), data.end(), 0);
    }

    void print_bits(bool msb_first = true) const {
        if (msb_first) {
            for (size_t i = 0; i < n_bits; ++i) {
                std::cout << is_set(n_bits - 1 - i);
            }
        } else {
            for (size_t i = 0; i < n_bits; ++i) {
                std::cout << is_set(i);
            }
        }
        std::cout << "\n";
    }

    
};
