#pragma once

// Name of the CUDA device synchronization schedule in force: "spin", "blocking", "yield" or
// "auto". Selected by NINFER_CUDA_SYNC when the first DeviceContext binds its device; declared
// apart from device.h so callers that only report the mode need no CUDA headers.

namespace ninfer {

[[nodiscard]] const char* cuda_sync_schedule_name();

} // namespace ninfer
