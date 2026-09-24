#include "core/device.h"

#include "core/cuda_sync.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer {
namespace {

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

struct SyncSchedule {
    std::string_view name;
    unsigned int flags;
};

// Spin first: it is the default, and it is the one measured fastest on this machine.
constexpr SyncSchedule kSyncSchedules[] = {
    {"spin", cudaDeviceScheduleSpin},
    {"blocking", cudaDeviceScheduleBlockingSync},
    {"yield", cudaDeviceScheduleYield},
    {"auto", cudaDeviceScheduleAuto},
};

unsigned int requested_sync_schedule() {
    const char* value = std::getenv("NINFER_CUDA_SYNC");
    if (value == nullptr || *value == '\0') { return cudaDeviceScheduleSpin; }
    for (const SyncSchedule& schedule : kSyncSchedules) {
        if (schedule.name == value) { return schedule.flags; }
    }
    throw std::invalid_argument("NINFER_CUDA_SYNC must be spin, blocking, yield, or auto");
}

unsigned int active_sync_schedule() {
    unsigned int flags    = 0;
    const cudaError_t err = cudaGetDeviceFlags(&flags);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceFlags failed", err));
    }
    return flags & cudaDeviceScheduleMask;
}

// Without this the device runs cudaDeviceScheduleAuto, which spin-waits whenever the host has
// more cores than contexts: one core sits at 100% for as long as the GPU is busy. Spin is still
// the default because blocking measured 0.37% slower here, but a quiet or battery-powered run
// wants the core back and can say so.
void apply_sync_schedule() {
    const unsigned int requested = requested_sync_schedule();
    const cudaError_t err        = cudaSetDeviceFlags(requested);
    if (err == cudaSuccess) { return; }
    (void)cudaGetLastError(); // don't leak this into later launch checks
    // A later context in the same process cannot set the schedule again. That only matters when
    // the schedule it asks for differs from the one already in force.
    if (err == cudaErrorSetOnActiveProcess && active_sync_schedule() == requested) { return; }
    throw std::runtime_error(cuda_error_message("cudaSetDeviceFlags failed", err));
}

void destroy_stream(cudaStream_t& stream) noexcept {
    if (stream != nullptr) {
        log_cuda_error("cudaStreamDestroy", cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        log_cuda_error("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

} // namespace

const char* cuda_sync_schedule_name() {
    const unsigned int flags = active_sync_schedule();
    for (const SyncSchedule& schedule : kSyncSchedules) {
        if (flags == schedule.flags) { return schedule.name.data(); }
    }
    return "unknown";
}

void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
    std::fprintf(stderr, "%s:%d: CUDA_CHECK(%s) failed: %s: %s\n", file, line, expr,
                 cudaGetErrorName(err), cudaGetErrorString(err));
    std::abort();
}

DeviceContext::DeviceContext(int device_id) : device(device_id) {
    int count       = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceCount failed", err));
    }
    if (count <= 0) { throw std::runtime_error("no CUDA devices available"); }
    if (device_id < 0 || device_id >= count) { throw std::runtime_error("invalid CUDA device id"); }

    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
    }

    apply_sync_schedule();

    err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceProperties failed", err));
    }

    cudaStream_t compute = nullptr;
    cudaStream_t load    = nullptr;
    err                  = cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(stream) failed", err));
    }

    err = cudaStreamCreateWithFlags(&load, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        destroy_stream(compute);
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(load_stream) failed", err));
    }

    stream      = compute;
    load_stream = load;
}

DeviceContext::~DeviceContext() {
    if (stream != nullptr || load_stream != nullptr) {
        log_cuda_error("cudaSetDevice", cudaSetDevice(device));
    }
    destroy_stream(load_stream);
    destroy_stream(stream);
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : device(other.device), stream(other.stream), load_stream(other.load_stream),
      props(other.props) {
    other.stream      = nullptr;
    other.load_stream = nullptr;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this == &other) { return *this; }

    if (stream != nullptr || load_stream != nullptr) {
        log_cuda_error("cudaSetDevice", cudaSetDevice(device));
    }
    destroy_stream(load_stream);
    destroy_stream(stream);

    device      = other.device;
    props       = other.props;
    stream      = other.stream;
    load_stream = other.load_stream;

    other.stream      = nullptr;
    other.load_stream = nullptr;
    return *this;
}

int DeviceContext::sm() const noexcept { return props.major * 10 + props.minor; }

std::size_t DeviceContext::total_vram() const noexcept { return props.totalGlobalMem; }

void DeviceContext::synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream)); }

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx) : stream_(ctx.stream) {
    cudaError_t err = cudaSetDevice(ctx.device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice(timer) failed", err));
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    err               = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreate(start) failed", err));
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        destroy_event(start);
        throw std::runtime_error(cuda_error_message("cudaEventCreate(stop) failed", err));
    }

    start_ = start;
    stop_  = stop;
}

CudaEventTimer::~CudaEventTimer() {
    destroy_event(stop_);
    destroy_event(start_);
}

CudaEventTimer::CudaEventTimer(CudaEventTimer&& other) noexcept
    : stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
}

CudaEventTimer& CudaEventTimer::operator=(CudaEventTimer&& other) noexcept {
    if (this == &other) { return *this; }

    destroy_event(stop_);
    destroy_event(start_);

    stream_ = other.stream_;
    start_  = other.start_;
    stop_   = other.stop_;

    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
    return *this;
}

void CudaEventTimer::start() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }

void CudaEventTimer::record_stop() { CUDA_CHECK(cudaEventRecord(stop_, stream_)); }

float CudaEventTimer::elapsed_ms() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
}

float CudaEventTimer::stop_ms() {
    record_stop();
    CUDA_CHECK(cudaEventSynchronize(stop_));
    return elapsed_ms();
}

} // namespace ninfer
