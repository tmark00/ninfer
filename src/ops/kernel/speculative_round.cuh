#pragma once

// Implements: include/ninfer/ops/speculative_round.h
// Match: contiguous request-major state and BF16 verification logits.
// Algorithm assumptions: small vocabularies use one cooperative block; the
// registered full-vocabulary stochastic route uses the sampling partial/group
// pipeline and caller-owned workspace, while greedy commit remains one thread.

#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

__global__ void speculative_prepare_verify_inputs_kernel(const std::int32_t* anchors,
                                                         const std::int32_t* drafts,
                                                         const std::int32_t* base_positions,
                                                         const std::int32_t* current_extents,
                                                         std::int32_t* verify_ids,
                                                         std::int32_t* positions, std::int32_t k) {
    const int row = static_cast<int>(blockIdx.y);
    const int T   = k + 1;
    int extent    = current_extents[row];
    extent        = extent < 0 ? 0 : (extent > k ? k : extent);
    for (int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x; j < T;
         j += blockDim.x * gridDim.x) {
        const int off = row * T + j;
        verify_ids[off] =
            j == 0 ? anchors[row] : (j <= extent ? drafts[row * k + j - 1] : anchors[row]);
        if (positions != nullptr) {
            positions[off] = base_positions[row] + (j <= extent ? j : extent);
        }
    }
}

template <typename T>
__device__ inline T* speculative_workspace_offset(T* ptr, std::size_t byte_offset) {
    return ptr == nullptr
               ? nullptr
               : reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(ptr) + byte_offset);
}

__device__ inline SamplingWorkspace
speculative_workspace_row(SamplingWorkspace workspace, std::size_t row_stride, std::int32_t row) {
    const std::size_t offset = static_cast<std::size_t>(row) * row_stride;
    workspace.partial_keys   = speculative_workspace_offset(workspace.partial_keys, offset);
    workspace.dist_idx       = speculative_workspace_offset(workspace.dist_idx, offset);
    workspace.dist_prob      = speculative_workspace_offset(workspace.dist_prob, offset);
    workspace.dist_support   = speculative_workspace_offset(workspace.dist_support, offset);
    workspace.group_done     = speculative_workspace_offset(workspace.group_done, offset);
    workspace.speculative_finalize_count =
        speculative_workspace_offset(workspace.speculative_finalize_count, offset);
    return workspace;
}

// Commits the round's accepted tokens plus one correction/bonus token, then
// advances the target length. The greedy branch
// (config temperature <= 0) is bit-identical to the original argmax accept: keep
// the longest draft prefix whose target argmax matches, then take the target
// argmax at the divergence column. The sampling branch (temperature > 0) runs
// distribution-correct speculative rejection sampling over the verify logits with
// a one-hot (greedy) draft: accept drafts[i] with probability p_i(drafts[i]) under
// the truncated target distribution, resample from the masked residual on the
// first rejection, and draw a bonus from the last column when every draft accepts.
// The draft-proposal path stays greedy, so q is one-hot and the accept test
// collapses to `u < p_i(drafts[i])`. Launch with a single block of kSamplerBlock
// threads; only thread 0 performs the sequential accept/commit while the whole
// block cooperates on the per-column truncated-distribution build.
__launch_bounds__(kSamplerBlock) __global__ void speculative_accept_greedy_drafts_kernel(
    const std::int32_t* target_tokens, const __nv_bfloat16* logits, const std::int32_t* drafts,
    const std::int32_t* current_extents, std::int32_t* lengths, std::int32_t* anchors,
    std::int32_t* licensed_tokens, std::int32_t* licensed_counts, std::int32_t* accepted,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t physical_rows,
    std::int32_t k) {
    const int tid                   = threadIdx.x;
    const int row                   = static_cast<int>(blockIdx.x);
    const int cols                  = k + 1;
    int extent                      = current_extents[row];
    extent                          = extent < 0 ? 0 : (extent > k ? k : extent);
    const SamplingConfig cfg        = configs[row];
    const std::int32_t* row_targets = target_tokens + row * cols;
    const std::int32_t* row_drafts  = drafts + row * k;
    std::int32_t* row_tokens        = licensed_tokens + row * cols;
    const __nv_bfloat16* row_logits =
        logits + static_cast<std::int64_t>(row) * cols * physical_rows;

    if (!(cfg.temperature > 0.0f)) {
        if (tid == 0) {
            int a = 0;
            while (a < extent && row_targets[a] == row_drafts[a]) { ++a; }
            const int t_star = row_targets[a];

            for (int i = 0; i <= k; ++i) { row_tokens[i] = 0; }
            for (int i = 0; i < a; ++i) { row_tokens[i] = row_drafts[i]; }
            row_tokens[a] = t_star;

            const int produced   = a + 1;
            licensed_counts[row] = produced;
            accepted[row]        = a;
            anchors[row]         = t_star;
            lengths[row] += produced;
        }
        return;
    }

    __shared__ float red_val[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ float merge_val[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int merge_idx[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int n_support;
    __shared__ int a_sh;
    __shared__ int done_sh;
    __shared__ int tstar_sh;
    __shared__ int L_sh;

    const int partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const int group_count    = sampler_group_count(partial_blocks);
    // No-op when the scratch/group path owns this shape.
    if (sampler_multiblock_ok(token_domain, cols, partial_blocks, group_count)) { return; }

    if (tid == 0) {
        a_sh     = 0;
        done_sh  = 0;
        tstar_sh = 0;
        L_sh     = lengths[row];
    }
    __syncthreads();

    for (int i = 0; i <= extent; ++i) {
        // Column i is only reached when drafts[0..i-1] were all accepted, so the
        // round-local penalty overlay for this column is exactly those i drafts.
        const std::int64_t base = static_cast<std::int64_t>(i) * physical_rows;
        if (token_domain <= kSamplerTileItems) {
            sampling_build_truncated_small(row_logits, base, token_domain, cfg, red_val, red_idx,
                                           cand_val, cand_idx, prob, &n_support, row_drafts, i);
        } else {
            sampling_build_truncated_block_fast(row_logits, base, token_domain, cfg, merge_val,
                                                merge_idx, cand_val, cand_idx, prob, &n_support,
                                                row_drafts, i);
        }
        if (tid == 0 && done_sh == 0) {
            const int L = L_sh;
            if (i < extent) {
                const int d = row_drafts[i];
                float pd    = 0.0f;
                for (int j = 0; j < n_support; ++j) {
                    if (cand_idx[j] == d) {
                        pd = prob[j];
                        break;
                    }
                }
                const float u =
                    sampling_uniform(cfg.seed, L + i + 1, kSamplePurposeSpeculativeAccept, 0u);
                if (u < pd) {
                    a_sh = i + 1; // accept drafts[i], keep verifying
                } else {
                    const float ur = sampling_uniform(cfg.seed, L + i + 1,
                                                      kSamplePurposeSpeculativeCorrection, 0u);
                    tstar_sh       = sampling_pick_from_support(cand_idx, prob, n_support, d, ur);
                    done_sh        = 1;
                }
            } else {
                // Every draft accepted: bonus token from the last verify column.
                const float u =
                    sampling_uniform(cfg.seed, L + extent + 1, kSamplePurposeSpeculativeBonus, 0u);
                tstar_sh = sampling_pick_from_support(cand_idx, prob, n_support, -1, u);
                done_sh  = 1;
            }
        }
        __syncthreads();
        if (done_sh) { break; }
    }

    if (tid == 0) {
        const int a     = a_sh;
        const int tstar = tstar_sh;
        const int L     = L_sh;

        for (int i = 0; i <= k; ++i) { row_tokens[i] = 0; }
        for (int i = 0; i < a; ++i) { row_tokens[i] = row_drafts[i]; }
        row_tokens[a] = tstar;

        const int produced   = a + 1;
        licensed_counts[row] = produced;
        accepted[row]        = a;
        anchors[row]         = tstar;
        lengths[row]         = L + produced;
        if (cfg.token_counts != nullptr) {
            for (int i = 0; i < produced; ++i) { atomicAdd(&cfg.token_counts[row_tokens[i]], 1); }
        }
    }
}

__launch_bounds__(kSamplerBlock) __global__ void speculative_sampling_partial_topk_kernel(
    const __nv_bfloat16* logits, const std::int32_t* drafts, const std::int32_t* current_extents,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t physical_rows,
    std::int32_t cols, std::int32_t k, SamplingWorkspace workspace,
    std::size_t workspace_row_stride) {
    const int row     = static_cast<int>(blockIdx.z);
    const int col     = static_cast<int>(blockIdx.y);
    const int partial = static_cast<int>(blockIdx.x);
    int extent        = current_extents[row];
    extent            = extent < 0 ? 0 : (extent > k ? k : extent);
    if (col > extent) { return; }
    const SamplingConfig cfg = configs[row];
    if (!(cfg.temperature > 0.0f) || token_domain <= kSamplerTileItems) { return; }
    workspace = speculative_workspace_row(workspace, workspace_row_stride, row);
    if (partial == 0 && threadIdx.x == 0) {
        workspace.group_done[col] = 0;
        if (col == 0) { *workspace.speculative_finalize_count = 0; }
    }

    __shared__ typename SamplingPartialSort::TempStorage sort_storage;
    unsigned long long keys[kSamplerItemsPerThread];

    const int cap                  = sampling_candidate_cap(cfg, token_domain);
    const std::int64_t base        = (static_cast<std::int64_t>(row) * cols + col) * physical_rows;
    const std::int32_t* row_drafts = drafts + row * k;
    const int tile_start           = partial * kSamplerPartialTileItems;
    // Column col's penalty overlay is the first `col` drafts (see accept loop);
    // applying it before top-k selection lets it change the candidate set, not
    // just the post-truncation probabilities.
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        if (v < token_domain) {
            const float x = sampling_adjusted_logit(__bfloat162float(logits[base + v]), v, cfg,
                                                    row_drafts, col);
            keys[item]    = sampling_sort_key(x, v);
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingPartialSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int rank = threadIdx.x * kSamplerItemsPerThread + item;
        if (rank < cap) {
            const int off               = sampling_partial_offset(workspace, col, partial, rank);
            workspace.partial_keys[off] = keys[item];
        }
    }
}

__launch_bounds__(kSamplerGroupBlock) __global__ void speculative_sampling_group_finalize_kernel(
    const std::int32_t* target_tokens, const std::int32_t* drafts,
    const std::int32_t* current_extents, std::int32_t* lengths, std::int32_t* anchors,
    std::int32_t* licensed_tokens, std::int32_t* licensed_counts, std::int32_t* accepted,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t cols,
    std::int32_t partial_blocks, std::int32_t group_count, SamplingWorkspace workspace,
    std::size_t workspace_row_stride) {
    const int row   = static_cast<int>(blockIdx.z);
    const int group = static_cast<int>(blockIdx.x);
    const int col   = static_cast<int>(blockIdx.y);
    const int tid   = threadIdx.x;
    const int k     = cols - 1;
    int extent      = current_extents[row];
    extent          = extent < 0 ? 0 : (extent > k ? k : extent);
    if (col > extent) { return; }
    const SamplingConfig cfg        = configs[row];
    const std::int32_t* row_targets = target_tokens + row * cols;
    const std::int32_t* row_drafts  = drafts + row * k;
    std::int32_t* row_tokens        = licensed_tokens + row * cols;
    if (token_domain <= kSamplerTileItems) { return; }

    if (!(cfg.temperature > 0.0f)) {
        if (tid == 0 && col == 0 && group == 0) {
            int a = 0;
            while (a < extent && row_targets[a] == row_drafts[a]) { ++a; }
            const int t_star = row_targets[a];
            for (int i = 0; i <= k; ++i) { row_tokens[i] = 0; }
            for (int i = 0; i < a; ++i) { row_tokens[i] = row_drafts[i]; }
            row_tokens[a]        = t_star;
            const int produced   = a + 1;
            licensed_counts[row] = produced;
            accepted[row]        = a;
            anchors[row]         = t_star;
            lengths[row] += produced;
        }
        return;
    }

    workspace = speculative_workspace_row(workspace, workspace_row_stride, row);

    __shared__ typename SamplingGroupSort::TempStorage sort_storage;
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last_group;
    unsigned long long keys[kSamplerGroupItemsPerThread];

    const int cap = sampling_candidate_cap(cfg, token_domain);
    // The preceding partial launch initializes all caller-owned counters. CUDA
    // stream ordering makes those writes visible before this launch begins.

    const int group_begin = group * kSamplerPartialsPerGroup;
    int group_partials    = partial_blocks - group_begin;
    if (group_partials < 0) { group_partials = 0; }
    if (group_partials > kSamplerPartialsPerGroup) { group_partials = kSamplerPartialsPerGroup; }
    const int group_n = group_partials * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < group_n) {
            const int partial = group_begin + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingGroupSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            const int out_off =
                sampling_partial_offset(workspace, col, partial_blocks + group, rank);
            workspace.partial_keys[out_off] = keys[item];
        }
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence();
        const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
        is_last_group  = (done == group_count) ? 1 : 0;
    }
    __syncthreads();
    if (!is_last_group) { return; }

    const int final_n = group_count * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < final_n) {
            const int partial = partial_blocks + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingGroupSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            cand_val[rank] = sampling_key_float(keys[item]);
            cand_idx[rank] = sampling_key_index(keys[item]);
        }
    }
    __syncthreads();

    sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);

    if (tid == 0) {
        workspace.dist_support[col] = n_support;
        for (int j = 0; j < n_support; ++j) {
            const int off            = sampling_dist_offset(col, j);
            workspace.dist_idx[off]  = cand_idx[j];
            workspace.dist_prob[off] = prob[j];
        }
        workspace.group_done[col] = 0;
        __threadfence();
        const int done_cols = atomicAdd(workspace.speculative_finalize_count, 1) + 1;
        if (done_cols == extent + 1) {
            const int L = lengths[row];
            int a       = 0;
            int tstar   = 0;
            for (int i = 0; i <= extent; ++i) {
                const int n            = workspace.dist_support[i];
                const int* dist_idx    = workspace.dist_idx + sampling_dist_offset(i, 0);
                const float* dist_prob = workspace.dist_prob + sampling_dist_offset(i, 0);
                if (i < extent) {
                    const int d = row_drafts[i];
                    float pd    = 0.0f;
                    for (int j = 0; j < n; ++j) {
                        if (dist_idx[j] == d) {
                            pd = dist_prob[j];
                            break;
                        }
                    }
                    const float u =
                        sampling_uniform(cfg.seed, L + i + 1, kSamplePurposeSpeculativeAccept, 0u);
                    if (u < pd) {
                        a = i + 1;
                        continue;
                    }
                    const float ur = sampling_uniform(cfg.seed, L + i + 1,
                                                      kSamplePurposeSpeculativeCorrection, 0u);
                    tstar          = sampling_pick_from_support(dist_idx, dist_prob, n, d, ur);
                    break;
                }
                const float u =
                    sampling_uniform(cfg.seed, L + extent + 1, kSamplePurposeSpeculativeBonus, 0u);
                tstar = sampling_pick_from_support(dist_idx, dist_prob, n, -1, u);
            }
            for (int i = 0; i <= k; ++i) { row_tokens[i] = 0; }
            for (int i = 0; i < a; ++i) { row_tokens[i] = row_drafts[i]; }
            row_tokens[a]        = tstar;
            const int produced   = a + 1;
            licensed_counts[row] = produced;
            accepted[row]        = a;
            anchors[row]         = tstar;
            lengths[row]         = L + produced;
            if (cfg.token_counts != nullptr) {
                for (int i = 0; i < produced; ++i) {
                    atomicAdd(&cfg.token_counts[row_tokens[i]], 1);
                }
            }
            *workspace.speculative_finalize_count = 0;
        }
    }
}

__global__ void speculative_select_accepted_hidden_kernel(const __nv_bfloat16* hidden,
                                                          const std::int32_t* selectors,
                                                          __nv_bfloat16* out, std::int32_t rows,
                                                          std::int32_t cols) {
    const int batch = static_cast<int>(blockIdx.y);
    const int row   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) { return; }
    const int col = selectors[batch];
    if (col < 0 || col >= cols) { return; }
    out[static_cast<std::int64_t>(batch) * rows + row] =
        hidden[(static_cast<std::int64_t>(batch) * cols + col) * rows + row];
}

__global__ void proposal_remap_token_ids_kernel(std::int32_t* proposal_tokens,
                                                std::int32_t proposal_count,
                                                const std::int32_t* id_map, std::int32_t n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= proposal_count) { return; }
    const int idx = proposal_tokens[i];
    if (idx >= 0 && idx < n) { proposal_tokens[i] = id_map[idx]; }
}

// Context-extended proposal selection. A column's best context candidate travels as one 64-bit
// word: the score as an order-preserving unsigned key in the high half and the complemented row
// index in the low half, so a single atomicMax keeps the highest score and, on a tie, the smallest
// row. Zero is below every key the score encoding produces, so it marks "no candidate".
__device__ __forceinline__ unsigned int proposal_context_score_key(float value) {
    const unsigned int bits = __float_as_uint(value);
    return (bits & 0x80000000u) != 0 ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ float proposal_context_score_value(unsigned int key) {
    return __uint_as_float((key & 0x80000000u) != 0 ? (key & 0x7FFFFFFFu) : ~key);
}

__global__ void proposal_context_reset_kernel(unsigned long long* best, std::int32_t columns) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < columns) { best[t] = 0ULL; }
}

template <int kWarps>
__global__ void __launch_bounds__(kWarps * 32)
    proposal_context_score_kernel(const __nv_bfloat16* __restrict__ hidden,
                                  const __nv_bfloat16* __restrict__ rows,
                                  const std::int32_t* __restrict__ count_ptr, std::int32_t dim,
                                  unsigned long long* __restrict__ best) {
    extern __shared__ __align__(16) unsigned char proposal_context_smem[];
    auto* column = reinterpret_cast<__nv_bfloat16*>(proposal_context_smem);
    const int count = *count_ptr;
    const int first = static_cast<int>(blockIdx.x) * kWarps;
    if (first >= count) { return; }
    const int t = static_cast<int>(blockIdx.y);

    const auto* source = reinterpret_cast<const uint4*>(hidden + static_cast<std::int64_t>(t) * dim);
    auto* staged       = reinterpret_cast<uint4*>(column);
    const int vectors  = dim / 8;
    for (int v = static_cast<int>(threadIdx.x); v < vectors; v += kWarps * 32) { staged[v] = source[v]; }
    __syncthreads();

    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int row  = first + warp;
    if (row >= count) { return; }
    const auto* weights =
        reinterpret_cast<const uint4*>(rows + static_cast<std::int64_t>(row) * dim);
    float sum = 0.0f;
    for (int v = lane; v < vectors; v += 32) {
        const uint4 w = weights[v];
        const uint4 h = staged[v];
        const auto* wp = reinterpret_cast<const __nv_bfloat162*>(&w);
        const auto* hp = reinterpret_cast<const __nv_bfloat162*>(&h);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 wf = __bfloat1622float2(wp[k]);
            const float2 hf = __bfloat1622float2(hp[k]);
            sum = fmaf(wf.x, hf.x, sum);
            sum = fmaf(wf.y, hf.y, sum);
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, offset);
    }
    if (lane == 0) {
        const unsigned long long word =
            (static_cast<unsigned long long>(proposal_context_score_key(sum)) << 32) |
            static_cast<unsigned long long>(0xFFFFFFFFu - static_cast<unsigned int>(row));
        atomicMax(best + t, word);
    }
}

__global__ void proposal_context_finalize_kernel(std::int32_t* proposal_tokens, std::int32_t columns,
                                                 const __nv_bfloat16* __restrict__ shortlist_logits,
                                                 const std::int32_t* __restrict__ id_map,
                                                 std::int32_t n,
                                                 const std::int32_t* __restrict__ context_ids,
                                                 const unsigned long long* __restrict__ best) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= columns) { return; }
    const int idx = proposal_tokens[t];
    if (idx < 0 || idx >= n) { return; }
    std::int32_t token             = id_map[idx];
    const unsigned long long word  = best[t];
    if (word != 0ULL) {
        const float context = proposal_context_score_value(static_cast<unsigned int>(word >> 32));
        const float shortlist =
            __bfloat162float(shortlist_logits[static_cast<std::int64_t>(t) * n + idx]);
        if (context > shortlist) {
            const auto row = static_cast<int>(0xFFFFFFFFu - static_cast<unsigned int>(word));
            token          = context_ids[row];
        }
    }
    proposal_tokens[t] = token;
}

} // namespace ninfer::ops
