#pragma once

// MSVC's cl rejects a by-value 128-byte-aligned TMA descriptor parameter in the nvcc-generated
// kernel host stub (C2719); Clang and GCC accept it. Every TMA kernel here therefore takes a
// device pointer, and this cache owns what that pointer needs: one persistent host copy, which is
// a stable source for the HtoD memcpy node captured into the decode CUDA graphs, and one device
// copy per distinct descriptor. A descriptor is a pure function of the buffers, the token count
// and the schedule, so once written the device copy stays valid across graph replays.
//
// The key is the descriptor's own bytes rather than the operand pointers. Several routes and
// schedules share one descriptor type, and two schedules over the same buffers tile them
// differently, so keying on the operands alone would hand one schedule the other's descriptor.
//
// One cache per descriptor type, because the static lives in the instantiated template.

#include "core/device.h"

#include <cstring>
#include <vector>

namespace ninfer::ops::detail {

template <class Descriptors>
const Descriptors* tma_descriptors_device(const Descriptors& host_descriptors,
                                         cudaStream_t stream) {
    struct Slot {
        Descriptors* host   = nullptr;
        Descriptors* device = nullptr;
    };
    static std::vector<Slot> slots;

    for (const Slot& slot : slots) {
        if (std::memcmp(slot.host, &host_descriptors, sizeof(Descriptors)) == 0) {
            return slot.device;
        }
    }

    Slot slot;
    slot.host = new Descriptors(host_descriptors);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&slot.device), sizeof(Descriptors)));
    CUDA_CHECK(cudaMemcpyAsync(slot.device, slot.host, sizeof(Descriptors), cudaMemcpyHostToDevice,
                               stream));
    slots.push_back(slot);
    return slot.device;
}

} // namespace ninfer::ops::detail
