#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

/**
 * Qwen3.6's target-local mapping from a stable request lane to its state roles: one current
 * state and kRewriteCheckpoints rewrite-checkpoint snapshots per lane.
 */
struct LinearStateSlots {
    // Two snapshots let a request that rewrites an older turn fall back to the boundary before
    // the newest one instead of replaying the whole prefix.
    static constexpr std::uint32_t kRewriteCheckpoints = 2;

    [[nodiscard]] static std::int32_t state_slot_count(std::uint32_t max_concurrency) {
        if (max_concurrency == 0 ||
            max_concurrency > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max() /
                                                         (1 + kRewriteCheckpoints))) {
            throw std::invalid_argument("Qwen3.6 Linear Attention concurrency is invalid");
        }
        return static_cast<std::int32_t>((1U + kRewriteCheckpoints) * max_concurrency);
    }

    [[nodiscard]] static std::int32_t current_state_slot(std::uint32_t lane,
                                                         std::uint32_t max_concurrency) {
        if (lane >= max_concurrency) {
            throw std::out_of_range("Qwen3.6 Linear Attention lane is out of range");
        }
        return static_cast<std::int32_t>(lane);
    }

    [[nodiscard]] static std::int32_t rewrite_checkpoint_state_slot(std::uint32_t lane,
                                                                    std::uint32_t max_concurrency,
                                                                    std::uint32_t checkpoint) {
        if (checkpoint >= kRewriteCheckpoints) {
            throw std::out_of_range("Qwen3.6 rewrite checkpoint index is out of range");
        }
        return static_cast<std::int32_t>((1U + checkpoint) * max_concurrency) +
               current_state_slot(lane, max_concurrency);
    }
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
