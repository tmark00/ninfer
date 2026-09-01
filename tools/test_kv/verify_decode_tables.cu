// Exhaustive check that the generated decode tables in e8_root_codec.cuh still agree with the
// arithmetic definition they were generated from.
//
// e8_root_decode_8d_lut reads round(lane * scale) out of c_e8_rad_even_i8 / c_e8_rad_odd_i8
// instead of computing it, so those tables and c_e8_stage1_nib are derived data: editing
// c_e8_radius_scale or c_e8_stage1_i8x8 without regenerating them would silently change what the
// KV cache decodes to. The decode input space is only 256 root codes x 256 radius/axis codes, so
// this walks all of it rather than sampling.
//
// Build (standalone):
//   nvcc -arch=sm_120a -O3 -std=c++17 -I ../../src -I ../../include \
//        verify_decode_tables.cu -o verify_decode_tables

#include "ops/kernel/e8_root_codec.cuh"

#include <cstdint>
#include <cstdio>

namespace {

__global__ void compare_all_code_pairs(int* mismatches, int* first_bad_code) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 65536) { return; }

    const std::uint8_t root_code     = static_cast<std::uint8_t>(idx >> 8);
    const std::uint8_t rad_axis_code = static_cast<std::uint8_t>(idx & 0xFF);

    std::int8_t reference[8];
    std::int8_t shipped[8];
    ninfer::ops::e8_root_decode_8d_reference(root_code, rad_axis_code, reference);
    ninfer::ops::e8_root_decode_8d_lut(root_code, rad_axis_code, shipped);

    if (*reinterpret_cast<std::uint64_t*>(reference) != *reinterpret_cast<std::uint64_t*>(shipped)) {
        if (atomicAdd(mismatches, 1) == 0) { *first_bad_code = idx; }
    }
}

bool cuda_ok(cudaError_t status, const char* what) {
    if (status == cudaSuccess) { return true; }
    std::printf("[e8-decode-tables] %s: %s\n", what, cudaGetErrorString(status));
    return false;
}

} // namespace

int main() {
    int* device_counters = nullptr;
    if (!cuda_ok(cudaMalloc(&device_counters, 2 * sizeof(int)), "cudaMalloc")) { return 2; }
    if (!cuda_ok(cudaMemset(device_counters, 0, 2 * sizeof(int)), "cudaMemset")) { return 2; }

    compare_all_code_pairs<<<256, 256>>>(device_counters, device_counters + 1);
    if (!cuda_ok(cudaDeviceSynchronize(), "kernel")) { return 2; }

    int counters[2] = {0, 0};
    if (!cuda_ok(cudaMemcpy(counters, device_counters, sizeof(counters), cudaMemcpyDeviceToHost),
                 "cudaMemcpy")) {
        return 2;
    }
    cudaFree(device_counters);

    if (counters[0] != 0) {
        std::printf("[e8-decode-tables] FAIL: %d of 65536 code pairs decode differently; "
                    "first at root_code=%d rad_axis_code=%d\n",
                    counters[0], counters[1] >> 8, counters[1] & 0xFF);
        std::printf("[e8-decode-tables] regenerate c_e8_stage1_nib, c_e8_rad_even_i8 and "
                    "c_e8_rad_odd_i8 from c_e8_stage1_i8x8 and c_e8_radius_scale\n");
        return 1;
    }

    std::printf("[e8-decode-tables] OK: all 65536 code pairs decode bit-identically\n");
    return 0;
}
