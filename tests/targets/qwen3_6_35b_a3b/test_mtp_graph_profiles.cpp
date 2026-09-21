// Contract of Variant::mtp_graph_profiles: one cudaGraphExec_t is instantiated per topology class
// and every other profile of that class is installed through cudaGraphExecUpdate, which cannot
// cross a change of node count. So no profile may span an attention route flip inside its own
// frontier range, and no topology class may carry two routes.

#include "ops/launcher/gqa_attention.h"
#include "targets/qwen3_6_35b_a3b/impl/variant.h"

#include <cstdint>
#include <iostream>
#include <map>
#include <vector>

namespace {

using ninfer::targets::qwen3_6_35b_a3b::detail::TextConfig;
using ninfer::targets::qwen3_6_35b_a3b::detail::Variant;
using Route = ninfer::ops::detail::GqaAttentionRoute;

// The masked batch-one verify of one draft window, as the launcher resolves it.
Route verify_route(std::uint32_t capacity, std::uint32_t draft_window, std::uint32_t frontier,
                   ninfer::DType cache_dtype) {
    const std::uint64_t visible =
        std::min<std::uint64_t>(capacity, static_cast<std::uint64_t>(frontier) + draft_window + 1U);
    const ninfer::ops::GqaExecutionEnvelope envelope{
        .min_visible_keys = 1U, .max_visible_keys = static_cast<std::uint32_t>(visible)};
    return ninfer::ops::detail::gqa_attention_resolve_route(
        TextConfig::query_heads, static_cast<std::int32_t>(draft_window + 1U), 1,
        cache_dtype, true, envelope);
}

int check(std::uint32_t capacity, std::uint32_t draft_window, ninfer::DType cache_dtype) {
    const std::vector<ninfer::targets::qwen3_6::GraphExecutionProfile> profiles =
        Variant::mtp_graph_profiles(capacity, draft_window, 1);
    if (profiles.empty()) {
        std::cerr << "no MTP profiles for capacity " << capacity << " k=" << draft_window << '\n';
        return 1;
    }
    std::map<std::uint32_t, Route> route_of_class;
    for (const auto& profile : profiles) {
        const Route low  = verify_route(capacity, draft_window, profile.min, cache_dtype);
        const Route high = verify_route(capacity, draft_window, profile.max, cache_dtype);
        if (low != high) {
            std::cerr << "profile [" << profile.min << ',' << profile.max << "] spans a route flip"
                      << " (capacity " << capacity << " k=" << draft_window << "): "
                      << ninfer::ops::detail::gqa_attention_route_name(low) << " -> "
                      << ninfer::ops::detail::gqa_attention_route_name(high) << '\n';
            return 1;
        }
        const auto [entry, inserted] = route_of_class.emplace(profile.topology_class, low);
        if (!inserted && entry->second != low) {
            std::cerr << "topology class " << profile.topology_class << " carries two routes"
                      << " (capacity " << capacity << " k=" << draft_window << "): "
                      << ninfer::ops::detail::gqa_attention_route_name(entry->second) << " and "
                      << ninfer::ops::detail::gqa_attention_route_name(low) << '\n';
            return 1;
        }
    }
    return 0;
}

} // namespace

int main() {
    for (const std::uint32_t capacity : {2048U, 16384U, 65536U, 262144U}) {
        for (std::uint32_t draft_window = 1; draft_window <= Variant::maximum_mtp_draft_tokens;
             ++draft_window) {
            for (const ninfer::DType cache_dtype : {ninfer::DType::BF16, ninfer::DType::I8}) {
                if (const int result = check(capacity, draft_window, cache_dtype); result != 0) {
                    return result;
                }
            }
        }
    }
    return 0;
}
