#pragma once

// Upstream moved the DFlash rotary table into its own header; this fork still keeps it beside the
// text and vision tables in ops/kernel/rope.cuh. Including both would define the table twice, so
// this shim forwards to the fork's copy. The table is byte-identical to upstream's and the body of
// fixed_sincos<DflashText1D> is upstream's dflash_rope_sincos word for word, so the result is the
// same value, not an approximation of it.

#include "ops/kernel/rope.cuh"

#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ void dflash_rope_sincos(const std::int32_t* positions, int token,
                                                   int pair, float* sine, float* cosine) {
    fixed_sincos<RopeKernelMode::DflashText1D>(positions, 0, token, pair, sine, cosine);
}

} // namespace ninfer::ops
