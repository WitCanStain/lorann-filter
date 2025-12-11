// gcc/clang:  -O3 -mavx512f -mbmi -mpopcnt
#pragma once
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <vector>

static inline unsigned ctz32(unsigned x) { return (unsigned)__builtin_ctz(x); }
static inline int popcnt32(unsigned x) { return __builtin_popcount(x); }

static inline __mmask16 tail_mask_16(size_t rem) {
  return rem >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << rem) - 1u);
}

static inline __mmask16 subset_mask_epi32(__m512i v, __m512i vx) {
  return _mm512_cmpeq_epi32_mask(_mm512_and_epi32(v, vx), vx);
}

size_t build_subset_masks_avx512(const uint32_t *a, size_t n, uint32_t q,
                                 uint16_t *masks_out) {
  const __m512i vq = _mm512_set1_epi32((int)q);
  size_t num_blocks = (n + 15) >> 4;

  for (size_t b = 0, base = 0; b < num_blocks; ++b, base += 16) {
    __mmask16 tail = tail_mask_16(n - base);
    __m512i v = _mm512_maskz_loadu_epi32(tail, a + base);
    __mmask16 m = subset_mask_epi32(v, vq) & tail;
    masks_out[b] = (uint16_t)m;
  }

  return num_blocks;
}

inline void iterate_hits_from_masks(const uint16_t *masks, size_t num_blocks,
                             std::vector<int> &idx_out) {
  for (size_t b = 0; b < num_blocks; ++b) {
    unsigned mm = masks[b];
    int base = b << 4;
    while (mm) {
      unsigned i = ctz32(mm);
      mm &= mm - 1;
      idx_out.push_back(base + i);
    }
  }
}

// ---------------- example ----------------
#include <stdio.h>
static void print_idx(size_t idx) { printf("%zu\n", idx); }

// int main(void) {
//   enum { N = 1 << 20 };
//   static uint32_t data[N];
//   for (size_t i = 0; i < N; i++)
//     if (i % 12345 == 0)
//       data[i] = 0xF; // demo init

//   uint32_t q = 0x5;

//   // make mask vector
//   size_t num_blocks = (N + 15) >> 4;
//   static uint16_t masks[((1 << 20) + 15) >> 4];
//   build_subset_masks_avx512(data, N, q, masks);

//   // iterate positions
//   // iterate_hits_from_masks(masks, num_blocks, print_idx);

//   return 0;
// }
