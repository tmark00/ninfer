#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Caller-owned transient capacity for every draft-count and batch-size pair in the inclusive
// domains. token_domain is the fixed sampling profile; invalid domains throw.
[[nodiscard]] std::size_t speculative_accept_greedy_drafts_workspace_capacity_bytes(
    std::int32_t token_domain, std::int32_t min_drafts, std::int32_t max_drafts,
    std::int32_t min_batch, std::int32_t max_batch);

/**
 * Op: speculative_prepare_verify_inputs
 *
 * Math / indexing:
 *   For row b and 0<=j<=K:
 *     verify_ids[j,b] = anchors[b]                         when j=0
 *                       drafts[j-1,b]                     when 0<j<=Pcur[b]
 *                       anchors[b]                        otherwise;
 *     positions[j,b]  = base_positions[b] + min(j,Pcur[b]).
 *
 * Logical shapes:
 *   All tensors are contiguous I32. anchors/base_positions/current_extents are [B], drafts is
 *   [K,B] with K>=1 and B>=1, and verify_ids/positions are [K+1,B]. Each current extent is in
 *   [0,K]. Inputs and outputs do not overlap.
 *
 * Effects:
 *   Writes every physical output element, including safe invalid-tail values. Inputs remain
 *   unchanged.
 *
 * Workspace:
 *   None.
 */
void speculative_prepare_verify_inputs(const Tensor& anchors, const Tensor& drafts,
                                       const Tensor& base_positions, const Tensor& current_extents,
                                       Tensor& verify_ids, Tensor& positions, cudaStream_t stream);

/**
 * Prepare only the target verification ids when the caller already owns the matching position
 * matrix. Shapes and id semantics are identical to speculative_prepare_verify_inputs; the
 * existing positions remain untouched.
 */
void speculative_prepare_verify_ids(const Tensor& anchors, const Tensor& drafts,
                                    const Tensor& current_extents, Tensor& verify_ids,
                                    cudaStream_t stream);

/**
 * Op: speculative_accept_greedy_drafts
 *
 * Algorithm:
 *   Independently for each row b, greedy mode accepts the longest available draft prefix matching
 *   target_tokens and commits the target token at the first mismatch (or the bonus column).
 *   Sampling mode applies configs[b] to each valid verification column, accepts draft i with
 *   target probability p_i(draft_i), samples from the residual distribution on first rejection,
 *   and samples a bonus from column Pcur[b] when every available draft is accepted. The draft
 *   proposal distribution is one-hot at each greedy draft token.
 *   RNG domains are the speculative accept/correction/bonus SamplePurpose values and logical
 *   positions derived from the old length.
 *
 * Logical shapes:
 *   All Tensor storage is contiguous. target_tokens/licensed_tokens are I32 [K+1,B], drafts is
 *   I32 [K,B], logits is BF16 [physical_rows,K+1,B], and current_extents/lengths/anchors/
 *   licensed_counts/accepted are I32 [B]. token_domain is in [1,physical_rows], K>=1, B>=1, and
 *   configs points to a device-resident SamplingConfig[B]. Tensor arguments, configs, and
 *   configs[b].token_counts do not overlap except for the explicitly mutated objects.
 *
 * Numeric:
 *   Sampling filtering, penalties, normalization, and RNG semantics are those of sampling.h.
 *
 * Effects:
 *   For each row, let A be the accepted draft count and L=A+1. licensed_tokens[0:A,b] receives
 *   accepted drafts, licensed_tokens[A,b] receives the correction/bonus token, and the remaining
 *   physical slots are zero. licensed_counts[b]=L; accepted[b]=A; anchors[b] becomes the
 *   correction/bonus token; lengths[b]+=L. In sampling mode, each produced token increments
 *   configs[b].token_counts when that pointer is non-null. Greedy mode does not update
 *   token_counts. current_extents and all other inputs remain unchanged. Request statistics are
 *   deliberately outside this Op.
 *
 * Workspace:
 *   Caller-owned transient storage reported by
 *   speculative_accept_greedy_drafts_workspace_capacity_bytes().
 */
void speculative_accept_greedy_drafts(const Tensor& target_tokens, const Tensor& logits,
                                      const Tensor& drafts, const Tensor& current_extents,
                                      Tensor& lengths, Tensor& anchors, Tensor& licensed_tokens,
                                      Tensor& licensed_counts, Tensor& accepted,
                                      std::int32_t token_domain, const SamplingConfig* configs,
                                      WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Op: speculative_select_accepted_hidden
 *
 * Math / indexing:
 *   out[:,b] = hidden[:,selectors[b],b].
 *
 * Shape / numeric / effects:
 *   hidden is contiguous BF16 [D,T,B], selectors is contiguous I32 [B] with every value in [0,T),
 *   and out is distinct contiguous BF16 [D,B]. The Op exactly copies BF16 bits, writes all of out,
 *   and uses no workspace or other state.
 */
void speculative_select_accepted_hidden(const Tensor& hidden, const Tensor& selectors, Tensor& out,
                                        cudaStream_t stream);

/**
 * Op: proposal_remap_token_ids
 *
 * Math / indexing:
 *   proposal_tokens[i]' = id_map[proposal_tokens[i]] for every proposal token.
 *
 * Effects:
 *   Updates the contiguous non-empty I32 proposal_tokens vector in place; every input id is in
 *   [0,count), and id_map is a distinct device I32 array [count]. There is no workspace or other
 *   state side effect.
 */
void proposal_remap_token_ids(Tensor& proposal_tokens, const std::int32_t* id_map,
                              std::int32_t count, cudaStream_t stream);

/**
 * Op: proposal_select_with_context
 *
 * Math / indexing, for every proposal column t with i_t = proposal_tokens[t] and k = *context_count:
 *   s_t   = float(shortlist_logits[i_t,t])
 *   c_j,t = sum_d float(context_rows[d,j]) * float(hidden[d,t])      for j in [0,k)
 *   j*_t  = the smallest j attaining max_j c_j,t
 *   proposal_tokens[t]' = context_ids[j*_t]   if k > 0 and c_{j*_t},t > s_t
 *                       = id_map[i_t]          otherwise.
 *
 * It replaces proposal_remap_token_ids for a shortlist proposal head extended at run time by
 * context rows: the shortlist argmax index is remapped to a token id unless a context row scores
 * strictly higher. The proposal only feeds speculative verification, so the comparison is an
 * implementation profile, not a semantic boundary: c is accumulated in FP32 and compared with the
 * BF16 shortlist logit as stored.
 *
 * Shape / numeric / effects:
 *   hidden is contiguous BF16 [D,T] with D a multiple of 8; shortlist_logits is contiguous BF16
 *   [count,T]; proposal_tokens is contiguous I32 [T] holding shortlist indices in [0,count) and is
 *   updated in place; id_map is a distinct device I32 array [count]. context_rows is contiguous
 *   BF16 [D,C] (column j holds context candidate j), context_ids is contiguous I32 [C], and
 *   context_count is a device I32 scalar in [0,C], read on the device so a captured CUDA Graph
 *   follows later updates. workspace holds at least
 *   proposal_select_with_context_workspace_bytes(T) bytes, 8-byte aligned; its contents are
 *   overwritten. No other state is written.
 */
[[nodiscard]] std::size_t proposal_select_with_context_workspace_bytes(std::int32_t columns);
void proposal_select_with_context(const Tensor& hidden, const Tensor& shortlist_logits,
                                  Tensor& proposal_tokens, const std::int32_t* id_map,
                                  std::int32_t count, const Tensor& context_rows,
                                  const Tensor& context_ids, const Tensor& context_count,
                                  Tensor& workspace, cudaStream_t stream);

} // namespace ninfer::ops
