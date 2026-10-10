/**
 *  @brief Haswell (AVX2) backend for substring & byte-set search.
 *  @file include/stringzilla/find/haswell.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_HASWELL_H_
#define STRINGZILLA_FIND_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/find/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  AVX2 implementation of the string search algorithms for Haswell processors and newer.
 *  Very minimalistic (compared to AVX-512), but still faster than the serial implementation.
 */
#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,lzcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "bmi", "bmi2", "lzcnt")
#endif

/**
 *  @brief  Searches a haystack shorter than 32 bytes without a byte-by-byte loop: two overlapping 16-byte loads
 *          for 16-31 bytes, two overlapping 8-byte words for 8-15, and two 4-byte words packed into one 64-bit
 *          word for 4-7. Every load stays inside the haystack. Lane @b i of the returned mask stands for byte
 *          @b i for the first word, and for byte `length - width + i` for the second, so the lowest set lane is
 *          the first match and the highest set lane is the last one.
 */
SZ_HELPER_INLINE sz_u32_t sz_find_byte_short_mask_haswell_(sz_u8_t const *text, sz_size_t length, sz_u8_t needle) {
    sz_assert_(length >= 16 && length < 32);
    __m128i const needle_u8x16 = _mm_set1_epi8((char)needle);
    sz_u32_t const head_mask = (sz_u32_t)_mm_movemask_epi8(
        _mm_cmpeq_epi8(_mm_loadu_si128((__m128i const *)text), needle_u8x16));
    sz_u32_t const tail_mask = (sz_u32_t)_mm_movemask_epi8(
        _mm_cmpeq_epi8(_mm_loadu_si128((__m128i const *)(text + length - 16)), needle_u8x16));
    return head_mask | (tail_mask << (length - 16));
}

SZ_API_COMPTIME sz_cptr_t sz_find_byte_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    sz_u8_t const *const text = (sz_u8_t const *)haystack;
    sz_u8_t const needle_byte = *(sz_u8_t const *)needle;

    // Inputs under 32 bytes: overlapping loads that stay inside the haystack, instead of a byte-by-byte loop.
    if (haystack_length < 32) {
        if (haystack_length >= 16) {
            sz_u32_t const matches = sz_find_byte_short_mask_haswell_(text, haystack_length, needle_byte);
            return matches ? haystack + sz_u32_ctz(matches) : SZ_NULL_CHAR;
        }
        sz_u64_vec_t needle_vec;
        needle_vec.u64 = (sz_u64_t)needle_byte * 0x0101010101010101ull;
        if (haystack_length >= 8) {
            sz_u64_vec_t const head = sz_u64_each_byte_equal_(sz_u64_load(haystack), needle_vec);
            if (head.u64) return haystack + sz_u64_ctz(head.u64) / 8;
            sz_u64_vec_t const tail = sz_u64_each_byte_equal_(sz_u64_load(haystack + haystack_length - 8), needle_vec);
            return tail.u64 ? haystack + haystack_length - 8 + sz_u64_ctz(tail.u64) / 8 : SZ_NULL_CHAR;
        }
        if (haystack_length >= 4) {
            // Bytes [0, 4) in the low half, bytes [length - 4, length) in the high half.
            sz_u64_vec_t both;
            both.u64 = (sz_u64_t)sz_u32_load(haystack).u32 |
                       ((sz_u64_t)sz_u32_load(haystack + haystack_length - 4).u32 << 32);
            sz_u64_vec_t const matches = sz_u64_each_byte_equal_(both, needle_vec);
            if (!matches.u64) return SZ_NULL_CHAR;
            sz_size_t const lane = (sz_size_t)sz_u64_ctz(matches.u64) / 8;
            return haystack + (lane < 4 ? lane : haystack_length - 8 + lane);
        }
        return sz_find_byte_serial(haystack, haystack_length, needle);
    }

    __m256i const needle_u8x32 = _mm256_set1_epi8((char)needle_byte);
    sz_u8_t const *const end = text + haystack_length;
    sz_u32_t matches = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i const *)text), needle_u8x32));
    if (matches) return haystack + sz_u32_ctz(matches);
    if (haystack_length <= 64) {
        // One more block, overlapping the first one, finishes the input.
        matches = (sz_u32_t)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i const *)(end - 32)), needle_u8x32));
        return matches ? (sz_cptr_t)(end - 32 + sz_u32_ctz(matches)) : SZ_NULL_CHAR;
    }
    // The second block, before any loop: a match there costs no more than it did with one block per iteration.
    matches = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i const *)(text + 32)), needle_u8x32));
    if (matches) return haystack + 32 + sz_u32_ctz(matches);

    // Inputs of 256 bytes or more continue from the next 32-byte boundary, which the first two blocks have already
    // reached, so that no load straddles two cache lines: worth 1.2-1.6x from 1 KB up, and nothing below 256 bytes.
    // Then test four blocks per iteration and branch once on their union.
    sz_u8_t const *cursor = haystack_length >= 256 ? (sz_u8_t const *)(((sz_size_t)text + 64) & ~(sz_size_t)31)
                                                   : text + 64;
    for (; end - cursor >= 128; cursor += 128) {
        __m256i const first_u8x32 = _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i const *)cursor), needle_u8x32);
        __m256i const second_u8x32 = _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i const *)(cursor + 32)),
                                                       needle_u8x32);
        __m256i const third_u8x32 = _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i const *)(cursor + 64)), needle_u8x32);
        __m256i const fourth_u8x32 = _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i const *)(cursor + 96)),
                                                       needle_u8x32);
        __m256i const any_u8x32 = _mm256_or_si256(_mm256_or_si256(first_u8x32, second_u8x32),
                                                  _mm256_or_si256(third_u8x32, fourth_u8x32));
        if (!_mm256_movemask_epi8(any_u8x32)) continue;
        // Two 64-bit masks, so that locating the match takes at most two tests.
        sz_u64_t const low_matches = (sz_u64_t)(sz_u32_t)_mm256_movemask_epi8(first_u8x32) |
                                     ((sz_u64_t)(sz_u32_t)_mm256_movemask_epi8(second_u8x32) << 32);
        if (low_matches) return (sz_cptr_t)(cursor + sz_u64_ctz(low_matches));
        sz_u64_t const high_matches = (sz_u64_t)(sz_u32_t)_mm256_movemask_epi8(third_u8x32) |
                                      ((sz_u64_t)(sz_u32_t)_mm256_movemask_epi8(fourth_u8x32) << 32);
        return (sz_cptr_t)(cursor + 64 + sz_u64_ctz(high_matches));
    }
    for (; end - cursor >= 32; cursor += 32) {
        matches = (sz_u32_t)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i const *)cursor), needle_u8x32));
        if (matches) return (sz_cptr_t)(cursor + sz_u32_ctz(matches));
    }
    if (cursor == end) return SZ_NULL_CHAR;

    // The last 32 bytes overlap blocks already tested: drop their lanes.
    matches = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i const *)(end - 32)), needle_u8x32));
    matches >>= 32 - (end - cursor);
    return matches ? (sz_cptr_t)(cursor + sz_u32_ctz(matches)) : SZ_NULL_CHAR;
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    sz_u8_t const *const text = (sz_u8_t const *)haystack;
    sz_u8_t const needle_byte = *(sz_u8_t const *)needle;

    // Inputs under 32 bytes: the mirror image of `sz_find_byte_haswell`, taking the highest match.
    if (haystack_length < 32) {
        if (haystack_length >= 16) {
            sz_u32_t const matches = sz_find_byte_short_mask_haswell_(text, haystack_length, needle_byte);
            return matches ? haystack + 31 - sz_u32_clz(matches) : SZ_NULL_CHAR;
        }
        sz_u64_vec_t needle_vec;
        needle_vec.u64 = (sz_u64_t)needle_byte * 0x0101010101010101ull;
        if (haystack_length >= 8) {
            sz_u64_vec_t const tail = sz_u64_each_byte_equal_(sz_u64_load(haystack + haystack_length - 8), needle_vec);
            if (tail.u64) return haystack + haystack_length - 8 + (63 - sz_u64_clz(tail.u64)) / 8;
            sz_u64_vec_t const head = sz_u64_each_byte_equal_(sz_u64_load(haystack), needle_vec);
            return head.u64 ? haystack + (63 - sz_u64_clz(head.u64)) / 8 : SZ_NULL_CHAR;
        }
        if (haystack_length >= 4) {
            sz_u64_vec_t both;
            both.u64 = (sz_u64_t)sz_u32_load(haystack).u32 |
                       ((sz_u64_t)sz_u32_load(haystack + haystack_length - 4).u32 << 32);
            sz_u64_vec_t const matches = sz_u64_each_byte_equal_(both, needle_vec);
            if (!matches.u64) return SZ_NULL_CHAR;
            sz_size_t const lane = (sz_size_t)(63 - sz_u64_clz(matches.u64)) / 8;
            return haystack + (lane < 4 ? lane : haystack_length - 8 + lane);
        }
        return sz_rfind_byte_serial(haystack, haystack_length, needle);
    }

    __m256i const needle_u8x32 = _mm256_set1_epi8((char)needle_byte);
    sz_u8_t const *const end = text + haystack_length;
    sz_u32_t matches = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i const *)(end - 32)), needle_u8x32));
    if (matches) return (sz_cptr_t)(end - 1 - sz_u32_clz(matches));
    if (haystack_length <= 64) {
        matches = (sz_u32_t)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i const *)text), needle_u8x32));
        return matches ? haystack + 31 - sz_u32_clz(matches) : SZ_NULL_CHAR;
    }
    // The second-to-last block, before any loop.
    matches = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i const *)(end - 64)), needle_u8x32));
    if (matches) return (sz_cptr_t)(end - 33 - sz_u32_clz(matches));

    // Inputs of 256 bytes or more continue down from the 32-byte boundary at or above `end - 64`, which the last two
    // blocks have already reached, as in `sz_find_byte_haswell`; shorter ones continue from `end - 64` itself.
    sz_u8_t const *cursor = haystack_length >= 256 ? (sz_u8_t const *)(((sz_size_t)(end - 64) + 31) & ~(sz_size_t)31)
                                                   : end - 64;
    for (; cursor - text >= 128; cursor -= 128) {
        __m256i const first_u8x32 = _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i const *)(cursor - 128)),
                                                      needle_u8x32);
        __m256i const second_u8x32 = _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i const *)(cursor - 96)),
                                                       needle_u8x32);
        __m256i const third_u8x32 = _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i const *)(cursor - 64)), needle_u8x32);
        __m256i const fourth_u8x32 = _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i const *)(cursor - 32)),
                                                       needle_u8x32);
        __m256i const any_u8x32 = _mm256_or_si256(_mm256_or_si256(first_u8x32, second_u8x32),
                                                  _mm256_or_si256(third_u8x32, fourth_u8x32));
        if (!_mm256_movemask_epi8(any_u8x32)) continue;
        // Two 64-bit masks, so that locating the match takes at most two tests.
        sz_u64_t const high_matches = (sz_u64_t)(sz_u32_t)_mm256_movemask_epi8(third_u8x32) |
                                      ((sz_u64_t)(sz_u32_t)_mm256_movemask_epi8(fourth_u8x32) << 32);
        if (high_matches) return (sz_cptr_t)(cursor - 1 - sz_u64_clz(high_matches));
        sz_u64_t const low_matches = (sz_u64_t)(sz_u32_t)_mm256_movemask_epi8(first_u8x32) |
                                     ((sz_u64_t)(sz_u32_t)_mm256_movemask_epi8(second_u8x32) << 32);
        return (sz_cptr_t)(cursor - 65 - sz_u64_clz(low_matches));
    }
    for (; cursor - text >= 32; cursor -= 32) {
        matches = (sz_u32_t)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i const *)(cursor - 32)), needle_u8x32));
        if (matches) return (sz_cptr_t)(cursor - 1 - sz_u32_clz(matches));
    }
    if (cursor == text) return SZ_NULL_CHAR;

    // The first 32 bytes overlap blocks already tested: keep only the lanes below `cursor`.
    matches = (sz_u32_t)_mm256_movemask_epi8(
        _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i const *)text), needle_u8x32));
    matches &= (sz_u32_t)((1ull << (cursor - text)) - 1);
    return matches ? haystack + 31 - sz_u32_clz(matches) : SZ_NULL_CHAR;
}

SZ_API_COMPTIME sz_cptr_t sz_find_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                          sz_size_t needle_length) {

    // Empty needle matches at the start, like `strstr`.
    if (!needle_length) return haystack;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_find_byte_haswell(haystack, haystack_length, needle);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into YMM registers.
    sz_u32_vec_t matches_vec;
    sz_u256_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.ymm = _mm256_set1_epi8(needle[offset_first]);
    n_mid_vec.ymm = _mm256_set1_epi8(needle[offset_mid]);
    n_last_vec.ymm = _mm256_set1_epi8(needle[offset_last]);

    // Scan through the string.
    for (; haystack_length >= needle_length + 32; haystack += 32, haystack_length -= 32) {
        h_first_vec.ymm = _mm256_lddqu_si256((__m256i const *)(haystack + offset_first));
        h_mid_vec.ymm = _mm256_lddqu_si256((__m256i const *)(haystack + offset_mid));
        h_last_vec.ymm = _mm256_lddqu_si256((__m256i const *)(haystack + offset_last));
        matches_vec.i32 = //
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_first_vec.ymm, n_first_vec.ymm)) &
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_mid_vec.ymm, n_mid_vec.ymm)) &
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_last_vec.ymm, n_last_vec.ymm));
        while (matches_vec.u32) {
            int potential_offset = (int)_tzcnt_u32(matches_vec.u32);
            if (sz_equal_haswell(haystack + potential_offset, needle, needle_length))
                return haystack + potential_offset;
            matches_vec.u32 &= matches_vec.u32 - 1;
        }
    }

    return sz_find_serial(haystack, haystack_length, needle, needle_length);
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                           sz_size_t needle_length) {

    // Empty needle matches at the end.
    if (!needle_length) return haystack + haystack_length;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_rfind_byte_haswell(haystack, haystack_length, needle);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into YMM registers.
    sz_u32_vec_t matches_vec;
    sz_u256_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.ymm = _mm256_set1_epi8(needle[offset_first]);
    n_mid_vec.ymm = _mm256_set1_epi8(needle[offset_mid]);
    n_last_vec.ymm = _mm256_set1_epi8(needle[offset_last]);

    // Scan through the string.
    sz_cptr_t haystack_cursor;
    for (; haystack_length >= needle_length + 32; haystack_length -= 32) {
        haystack_cursor = haystack + haystack_length - needle_length - 32 + 1;
        h_first_vec.ymm = _mm256_lddqu_si256((__m256i const *)(haystack_cursor + offset_first));
        h_mid_vec.ymm = _mm256_lddqu_si256((__m256i const *)(haystack_cursor + offset_mid));
        h_last_vec.ymm = _mm256_lddqu_si256((__m256i const *)(haystack_cursor + offset_last));
        matches_vec.i32 = //
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_first_vec.ymm, n_first_vec.ymm)) &
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_mid_vec.ymm, n_mid_vec.ymm)) &
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_last_vec.ymm, n_last_vec.ymm));
        while (matches_vec.u32) {
            int potential_offset = (int)_lzcnt_u32(matches_vec.u32);
            if (sz_equal_haswell(haystack + haystack_length - needle_length - potential_offset, needle, needle_length))
                return haystack + haystack_length - needle_length - potential_offset;
            matches_vec.u32 &= ~(1u << (31 - potential_offset));
        }
    }

    return sz_rfind_serial(haystack, haystack_length, needle, needle_length);
}

SZ_API_COMPTIME sz_cptr_t sz_find_byteset_haswell(sz_cptr_t text, sz_size_t length, sz_byteset_t const *filter) {

    // Let's unzip even and odd elements and replicate them into both lanes of the YMM register.
    // That way when we invoke `_mm256_shuffle_epi8` we can use the same mask for both lanes.
    // Load the 32-byte filter as two 16-byte halves, separate even/odd bytes, pack, and broadcast to YMM.
    sz_u128_vec_t byte_mask_vec;
    sz_u128_vec_t filter_low_vec, filter_high_vec;
    sz_u128_vec_t low_evens_vec, high_evens_vec;
    sz_u128_vec_t low_odds_vec, high_odds_vec;
    sz_u128_vec_t evens_xmm_vec, odds_xmm_vec;
    sz_u256_vec_t filter_even_vec, filter_odd_vec;

    sz_u256_vec_t text_vec;
    sz_u256_vec_t matches_vec;
    sz_u256_vec_t lower_nibbles_vec, higher_nibbles_vec;
    sz_u256_vec_t bitset_even_vec, bitset_odd_vec;
    sz_u256_vec_t bitmask_vec, bitmask_lookup_vec;

    byte_mask_vec.xmm = _mm_set1_epi16(0x00ff);

    filter_low_vec.xmm = _mm_lddqu_si128((__m128i const *)(filter));
    filter_high_vec.xmm = _mm_lddqu_si128((__m128i const *)(filter) + 1);
    low_evens_vec.xmm = _mm_and_si128(filter_low_vec.xmm, byte_mask_vec.xmm);
    high_evens_vec.xmm = _mm_and_si128(filter_high_vec.xmm, byte_mask_vec.xmm);
    low_odds_vec.xmm = _mm_srli_epi16(filter_low_vec.xmm, 8);
    high_odds_vec.xmm = _mm_srli_epi16(filter_high_vec.xmm, 8);

    evens_xmm_vec.xmm = _mm_packus_epi16(low_evens_vec.xmm, high_evens_vec.xmm);
    odds_xmm_vec.xmm = _mm_packus_epi16(low_odds_vec.xmm, high_odds_vec.xmm);
    filter_even_vec.ymm = _mm256_set_m128i(evens_xmm_vec.xmm, evens_xmm_vec.xmm);
    filter_odd_vec.ymm = _mm256_set_m128i(odds_xmm_vec.xmm, odds_xmm_vec.xmm);

    bitmask_lookup_vec.ymm = _mm256_set_epi8(                       //
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, //
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1);

    while (length >= 32) {
        // The following algorithm is a transposed equivalent of the "SIMD-ized check which bytes are in a set"
        // solutions by Wojciech Muła. We populate the bitmask differently and target newer CPUs, so
        // StrinZilla uses a somewhat different approach.
        // http://0x80.pl/articles/simd-byte-lookup.html#alternative-implementation-new
        //
        //      sz_u8_t input = *(sz_u8_t const *)text;
        //      sz_u8_t lo_nibble = input & 0x0f;
        //      sz_u8_t hi_nibble = input >> 4;
        //      sz_u8_t bitset_even = filter_even_vec.u8s[hi_nibble];
        //      sz_u8_t bitset_odd = filter_odd_vec.u8s[hi_nibble];
        //      sz_u8_t bitmask = (1 << (lo_nibble & 0x7));
        //      sz_u8_t bitset = lo_nibble < 8 ? bitset_even : bitset_odd;
        //      if ((bitset & bitmask) != 0) return text;
        //      else { length--, text++; }
        //
        // The nice part about this, loading the strided data is vey easy with Arm NEON,
        // while with x86 CPUs after AVX, shuffles within 256 bits shouldn't be an issue either.
        text_vec.ymm = _mm256_lddqu_si256((__m256i const *)text);
        lower_nibbles_vec.ymm = _mm256_and_si256(text_vec.ymm, _mm256_set1_epi8(0x0f));
        bitmask_vec.ymm = _mm256_shuffle_epi8(bitmask_lookup_vec.ymm, lower_nibbles_vec.ymm);
        //
        // At this point we can validate the `bitmask_vec` contents like this:
        //
        //      for (sz_size_t i = 0; i != 32; ++i) {
        //          sz_u8_t input = *(sz_u8_t const *)(text + i);
        //          sz_u8_t lo_nibble = input & 0x0f;
        //          sz_u8_t bitmask = (1 << (lo_nibble & 0x7));
        //          sz_assert_(bitmask_vec.u8s[i] == bitmask);
        //      }
        //
        // Shift right every byte by 4 bits.
        // There is no `_mm256_srli_epi8` intrinsic, so we have to use `_mm256_srli_epi16`
        // and combine it with a mask to clear the higher bits.
        higher_nibbles_vec.ymm = _mm256_and_si256(_mm256_srli_epi16(text_vec.ymm, 4), _mm256_set1_epi8(0x0f));
        bitset_even_vec.ymm = _mm256_shuffle_epi8(filter_even_vec.ymm, higher_nibbles_vec.ymm);
        bitset_odd_vec.ymm = _mm256_shuffle_epi8(filter_odd_vec.ymm, higher_nibbles_vec.ymm);
        //
        // At this point we can validate the `bitset_even_vec` and `bitset_odd_vec` contents like this:
        //
        //      for (sz_size_t i = 0; i != 32; ++i) {
        //          sz_u8_t input = *(sz_u8_t const *)(text + i);
        //          sz_u8_t const *bitset_ptr = &filter->_u8s[0];
        //          sz_u8_t hi_nibble = input >> 4;
        //          sz_u8_t bitset_even = bitset_ptr[hi_nibble * 2];
        //          sz_u8_t bitset_odd = bitset_ptr[hi_nibble * 2 + 1];
        //          sz_assert_(bitset_even_vec.u8s[i] == bitset_even);
        //          sz_assert_(bitset_odd_vec.u8s[i] == bitset_odd);
        //      }
        //
        __m256i take_first_u8x32 = _mm256_cmpgt_epi8(_mm256_set1_epi8(8), lower_nibbles_vec.ymm);
        bitset_even_vec.ymm = _mm256_blendv_epi8(bitset_odd_vec.ymm, bitset_even_vec.ymm, take_first_u8x32);

        // It would have been great to have an instruction that tests the bits and then broadcasts
        // the matching bit into all bits in that byte. But we don't have that, so we have to
        // `and`, `cmpeq`, `movemask`, and then invert at the end...
        matches_vec.ymm = _mm256_and_si256(bitset_even_vec.ymm, bitmask_vec.ymm);
        matches_vec.ymm = _mm256_cmpeq_epi8(matches_vec.ymm, _mm256_setzero_si256());
        int matches_mask = ~_mm256_movemask_epi8(matches_vec.ymm);
        if (matches_mask) {
            int offset = (int)_tzcnt_u32(matches_mask);
            return text + offset;
        }
        else { text += 32, length -= 32; }
    }

    return sz_find_byteset_serial(text, length, filter);
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_haswell(sz_cptr_t text, sz_size_t length, sz_byteset_t const *filter) {

    // Mirror of `sz_find_byteset_haswell` scanning from the end: the same transposed Muła nibble-bitset
    // classifier, but each iteration tests the trailing 32-byte window and resolves the highest matching
    // lane (the last occurrence) with `~movemask` + `clz`.
    sz_u128_vec_t byte_mask_vec;
    sz_u128_vec_t filter_low_vec, filter_high_vec;
    sz_u128_vec_t low_evens_vec, high_evens_vec;
    sz_u128_vec_t low_odds_vec, high_odds_vec;
    sz_u128_vec_t evens_xmm_vec, odds_xmm_vec;
    sz_u256_vec_t filter_even_vec, filter_odd_vec;

    sz_u256_vec_t text_vec;
    sz_u256_vec_t matches_vec;
    sz_u256_vec_t lower_nibbles_vec, higher_nibbles_vec;
    sz_u256_vec_t bitset_even_vec, bitset_odd_vec;
    sz_u256_vec_t bitmask_vec, bitmask_lookup_vec;

    byte_mask_vec.xmm = _mm_set1_epi16(0x00ff);

    filter_low_vec.xmm = _mm_lddqu_si128((__m128i const *)(filter));
    filter_high_vec.xmm = _mm_lddqu_si128((__m128i const *)(filter) + 1);
    low_evens_vec.xmm = _mm_and_si128(filter_low_vec.xmm, byte_mask_vec.xmm);
    high_evens_vec.xmm = _mm_and_si128(filter_high_vec.xmm, byte_mask_vec.xmm);
    low_odds_vec.xmm = _mm_srli_epi16(filter_low_vec.xmm, 8);
    high_odds_vec.xmm = _mm_srli_epi16(filter_high_vec.xmm, 8);

    evens_xmm_vec.xmm = _mm_packus_epi16(low_evens_vec.xmm, high_evens_vec.xmm);
    odds_xmm_vec.xmm = _mm_packus_epi16(low_odds_vec.xmm, high_odds_vec.xmm);
    filter_even_vec.ymm = _mm256_set_m128i(evens_xmm_vec.xmm, evens_xmm_vec.xmm);
    filter_odd_vec.ymm = _mm256_set_m128i(odds_xmm_vec.xmm, odds_xmm_vec.xmm);

    bitmask_lookup_vec.ymm = _mm256_set_epi8(                       //
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, //
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1);

    while (length >= 32) {
        sz_cptr_t const window = text + length - 32;
        text_vec.ymm = _mm256_lddqu_si256((__m256i const *)window);
        lower_nibbles_vec.ymm = _mm256_and_si256(text_vec.ymm, _mm256_set1_epi8(0x0f));
        bitmask_vec.ymm = _mm256_shuffle_epi8(bitmask_lookup_vec.ymm, lower_nibbles_vec.ymm);
        higher_nibbles_vec.ymm = _mm256_and_si256(_mm256_srli_epi16(text_vec.ymm, 4), _mm256_set1_epi8(0x0f));
        bitset_even_vec.ymm = _mm256_shuffle_epi8(filter_even_vec.ymm, higher_nibbles_vec.ymm);
        bitset_odd_vec.ymm = _mm256_shuffle_epi8(filter_odd_vec.ymm, higher_nibbles_vec.ymm);
        __m256i take_first_u8x32 = _mm256_cmpgt_epi8(_mm256_set1_epi8(8), lower_nibbles_vec.ymm);
        bitset_even_vec.ymm = _mm256_blendv_epi8(bitset_odd_vec.ymm, bitset_even_vec.ymm, take_first_u8x32);
        matches_vec.ymm = _mm256_and_si256(bitset_even_vec.ymm, bitmask_vec.ymm);
        matches_vec.ymm = _mm256_cmpeq_epi8(matches_vec.ymm, _mm256_setzero_si256());
        int matches_mask = ~_mm256_movemask_epi8(matches_vec.ymm);
        if (matches_mask) {
            // Highest set bit = last matching byte in the window.
            int offset = 31 - (int)_lzcnt_u32((sz_u32_t)matches_mask);
            return window + offset;
        }
        else { length -= 32; }
    }

    return sz_rfind_byteset_serial(text, length, filter);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_HASWELL_H_
