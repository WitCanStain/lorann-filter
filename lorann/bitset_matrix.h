#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <iostream>

class BitsetMatrix {
public:
    using Block = uint32_t;
    static constexpr size_t BITS_PER_BLOCK = sizeof(Block) * 8;
    std::vector<Block> data;
    BitsetMatrix() = default;  // allow empty construction

    void init(size_t n_points_, size_t n_attributes_) {
        n_points = n_points_;
        n_attributes = n_attributes_;
        blocks_per_bitset = (n_attributes + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
        data.assign(n_points * blocks_per_bitset, 0);
    }

    inline void set(size_t dp_idx, size_t attr) {
        size_t block = attr / BITS_PER_BLOCK;
        size_t offset = attr % BITS_PER_BLOCK;
        data[dp_idx * blocks_per_bitset + block] |= (Block(1) << offset);
    }

    inline bool is_set(size_t dp_idx, size_t attr) const {
        size_t block = attr / BITS_PER_BLOCK;
        size_t offset = attr % BITS_PER_BLOCK;
        return data[dp_idx * blocks_per_bitset + block] & (Block(1) << offset);
    }

    // inline bool matches(size_t dp_idx, const std::vector<Block>& filter) const {
    //     size_t start = dp_idx * blocks_per_bitset;
    //     for (size_t i = 0; i < blocks_per_bitset; ++i) {
    //         if ((data[start + i] & filter[i]) != filter[i]) {
    //             return false;
    //         }
    //     }
    //     return true;
    // }

    inline bool matches(size_t point, const BitsetMatrix& filter_matrix) const {
        size_t start = point * blocks_per_bitset;
        for (size_t i = 0; i < blocks_per_bitset; ++i) {
            if ((data[start + i] & filter_matrix.data[i]) != filter_matrix.data[i]) {
                return false; // data point does not have all filter attributes
            }
        }
        return true; // all filter attributes matched
    }

    bool matches_view(size_t point, const BitsetMatrix& filter_matrix, size_t filter_row = 0) const {
        // Get the row for the data point
        BitsetView row = view(point);
        // Get the filter row
        BitsetView filter_row_view = filter_matrix.view(filter_row);

        // Compare blocks
        for (size_t i = 0; i < blocks_per_bitset; ++i) {
            if ((row.blocks[i] & filter_row_view.blocks[i]) != filter_row_view.blocks[i]) {
                return false; // data point does not have all filter attributes
            }
        }
        return true; // all filter attributes matched
    }


    

    // inline bool any_match(size_t point, const std::vector<Block>& test) const {
    //     size_t base = point * blocks_per_bitset;
    //     for (size_t i = 0; i < blocks_per_bitset; ++i) {
    //         if (data[base + i] & test[i]) {
    //             return true; // at least one bit overlaps
    //         }
    //     }
    //     return false; // no overlapping bits
    // }


    inline void clear(size_t point, size_t attr) {
        size_t block = attr / BITS_PER_BLOCK;
        size_t offset = attr % BITS_PER_BLOCK;
        data[point * blocks_per_bitset + block] &= ~(Block(1) << offset);
    }

    // Access raw blocks for a given data point
    inline std::vector<Block> get_bitset(size_t point) const {
        return std::vector<Block>(data.begin() + point * blocks_per_bitset,
                                  data.begin() + (point + 1) * blocks_per_bitset);
    }


    // Iterate over set bits of a specific point
    void for_each_set_bit(size_t point, const std::function<void(size_t)>& func) const {
        size_t base = point * blocks_per_bitset;
        for (size_t block_idx = 0; block_idx < blocks_per_bitset; ++block_idx) {
            Block b = data[base + block_idx];
            size_t bit_offset = 0;
            while (b != 0) {
                // Find the least significant set bit
                size_t tz = __builtin_ctzll(b); // count trailing zeros (GCC/Clang)
                func(block_idx * BITS_PER_BLOCK + tz);
                b &= b - 1; // clear the least significant set bit
            }
        }
    }

    inline size_t size() {
        return n_points;
    }

    void to_string() {
        for (int i = 0; i < n_points; ++i) {
            for (int j = 0; j < n_attributes; ++j) {
                std::cout << is_set(i, j);
            }
            std::cout << std::endl;
        }
    }

    struct BitsetView {
        BitsetView(const Block* blocks_, size_t blocks_per_bitset_)
                : blocks(blocks_), blocks_per_bitset(blocks_per_bitset_) {}
        bool operator==(const BitsetView& other) const {
            for (size_t i = 0; i < blocks_per_bitset; ++i) {
                if (blocks[i] != other.blocks[i]) return false;
            }
            return true;
        }
        bool operator<(const BitsetView& other) const {
            for (size_t i = 0; i < blocks_per_bitset; ++i) {
                if (blocks[i] < other.blocks[i]) return true;
                if (blocks[i] > other.blocks[i]) return false;
            }
            return false;
        }
        const Block* blocks;
        size_t blocks_per_bitset;
    };

    inline BitsetView view(size_t point) const {
        return BitsetView(&data[point * blocks_per_bitset], blocks_per_bitset);
    }

    inline bool matches(size_t dp_idx, const BitsetView& filter) const {
        BitsetView row = view(dp_idx);
        for (size_t i = 0; i < blocks_per_bitset; ++i) {
            if ((row.blocks[i] & filter.blocks[i]) != filter.blocks[i])
                return false;
        }
        return true;
    }

    inline bool any_match(size_t point, const BitsetMatrix& filter_matrix) const {
        size_t start_point = point * blocks_per_bitset;
        for (size_t i = 0; i < blocks_per_bitset; ++i) {
            if ((data[start_point + i] & filter_matrix.data[i]) != 0) {
                return true;  // found at least one matching bit
            }
        }
        return false; // no bits matched
    }

    struct BitsetKey {
        std::vector<Block> blocks;

        bool operator==(const BitsetKey& other) const {
            return blocks == other.blocks;
        }
    };

    struct BitsetKeyHash {
        size_t operator()(const BitsetKey& k) const {
            size_t hash = 0;
            for (auto b : k.blocks) {
                // boost::hash_combine style
                hash ^= std::hash<Block>{}(b) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };

    // --- Extract a BitsetKey from a given data point ---
    BitsetKey key(size_t point) const {
        size_t start = point * blocks_per_bitset;
        return BitsetKey{
            std::vector<Block>(data.begin() + start,
                               data.begin() + start + blocks_per_bitset)
        };
    }

private:
    size_t n_points = 0;
    size_t n_attributes = 0;
    size_t blocks_per_bitset = 0;
};


struct BitsetViewHash {
    size_t operator()(const BitsetMatrix::BitsetView& view) const {
        size_t hash = 0;
        for (size_t i = 0; i < view.blocks_per_bitset; ++i) {
            hash ^= std::hash<BitsetMatrix::Block>{}(view.blocks[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

struct BitsetViewEqual {
    bool operator()(const BitsetMatrix::BitsetView& a,
                    const BitsetMatrix::BitsetView& b) const {
        if (a.blocks_per_bitset != b.blocks_per_bitset) return false;
        for (size_t i = 0; i < a.blocks_per_bitset; ++i)
            if (a.blocks[i] != b.blocks[i]) return false;
        return true;
    }
};