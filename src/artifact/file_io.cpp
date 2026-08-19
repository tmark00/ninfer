#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <system_error>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {

#ifdef _WIN32
namespace {

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    const auto error = ::GetLastError();
    throw ArtifactError(path.string() + ": " + operation + ": " +
                        std::system_category().message(static_cast<int>(error)));
}

HANDLE open_handle(const std::filesystem::path& path, DWORD flags) {
    return ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                         flags, nullptr);
}

// ReadFile with an explicit offset is the positional read; the handle's file pointer is not
// shared state because every call names its own offset. DWORD caps a single call at 4 GiB.
std::size_t read_at(HANDLE handle, std::uint64_t offset, std::byte* data, std::size_t count,
                    const std::filesystem::path& path, const char* operation) {
    OVERLAPPED overlapped{};
    overlapped.Offset     = static_cast<DWORD>(offset & 0xffffffffULL);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
    DWORD read            = 0;
    if (!::ReadFile(handle, data, static_cast<DWORD>(count), &read, &overlapped)) {
        if (::GetLastError() == ERROR_HANDLE_EOF) { return 0; }
        fail(path, operation);
    }
    return read;
}

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    const HANDLE handle = open_handle(path_, FILE_ATTRIBUTE_NORMAL);
    if (handle == INVALID_HANDLE_VALUE) { fail(path_, "open"); }
    handle_ = handle;

    FILE_STANDARD_INFO info{};
    if (!::GetFileInformationByHandleEx(handle, FileStandardInfo, &info, sizeof(info))) {
        fail(path_, "stat");
    }
    if (info.Directory || info.EndOfFile.QuadPart < 0) {
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(info.EndOfFile.QuadPart);
}

InputFile::~InputFile() {
    if (direct_handle_ != nullptr) { ::CloseHandle(static_cast<HANDLE>(direct_handle_)); }
    if (handle_ != nullptr) { ::CloseHandle(static_cast<HANDLE>(handle_)); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto read  = read_at(static_cast<HANDLE>(handle_), offset, destination.data(), count,
                                   path_, "read");
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += read;
        destination = destination.subspan(read);
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > std::numeric_limits<DWORD>::max() / kPayloadAlignment *
                                 kPayloadAlignment) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_handle_ == nullptr) {
        // FILE_FLAG_NO_BUFFERING is the O_DIRECT counterpart: sector-aligned offsets, sizes and
        // buffers, which the 4096-byte checks above already guarantee. A short final block at
        // EOF is returned as a short count, as with pread.
        const HANDLE handle = open_handle(path_, FILE_FLAG_NO_BUFFERING);
        if (handle == INVALID_HANDLE_VALUE) { fail(path_, "open direct"); }
        direct_handle_ = handle;
    }
    return read_at(static_cast<HANDLE>(direct_handle_), offset, destination.data(),
                   destination.size(), path_, "direct read");
}

#else
namespace {

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { fail(path_, "open"); }

    struct stat status {};

    if (::fstat(fd_, &status) != 0) {
        const auto error = errno;
        ::close(fd_);
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        ::close(fd_);
        fd_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
}

InputFile::~InputFile() {
    if (direct_fd_ >= 0) { ::close(direct_fd_); }
    if (fd_ >= 0) { ::close(fd_); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto read  = ::pread(fd_, destination.data(), count, file_offset(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_fd_ < 0) {
        direct_fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (direct_fd_ < 0) { fail(path_, "open direct"); }
    }
    ssize_t read;
    do {
        read = ::pread(direct_fd_, destination.data(), destination.size(), file_offset(offset));
    } while (read < 0 && errno == EINTR);
    if (read < 0) { fail(path_, "direct pread"); }
    return static_cast<std::size_t>(read);
}
#endif

} // namespace ninfer::artifact
