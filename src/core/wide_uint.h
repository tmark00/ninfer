#pragma once

// A 128-bit unsigned integer for the saturating cost arithmetic in the runtime contract.
//
// GCC and Clang provide 'unsigned __int128' natively and keep using it here. MSVC has no
// 128-bit integer type at all, so it gets a minimal stand-in supporting exactly the
// operations those call sites use: construction from an unsigned 64-bit value, truncating
// multiplication, addition, subtraction, shifts, complement, ordering, and an explicit
// narrowing back to 64 bits.

#include <cstdint>

namespace ninfer {

#if defined(_MSC_VER) && !defined(__clang__)

class UInt128 {
public:
    constexpr UInt128() noexcept = default;
    // Deliberately implicit: the call sites mix 64-bit operands into 128-bit expressions.
    constexpr UInt128(std::uint64_t low) noexcept : low_(low) {}
    constexpr UInt128(std::uint64_t high, std::uint64_t low) noexcept : high_(high), low_(low) {}

    [[nodiscard]] constexpr std::uint64_t high() const noexcept { return high_; }
    [[nodiscard]] constexpr std::uint64_t low() const noexcept { return low_; }

    explicit constexpr operator std::uint64_t() const noexcept { return low_; }
    explicit constexpr operator bool() const noexcept { return (high_ | low_) != 0U; }

    friend constexpr UInt128 operator+(UInt128 left, UInt128 right) noexcept {
        const std::uint64_t low = left.low_ + right.low_;
        return {left.high_ + right.high_ + (low < left.low_ ? 1U : 0U), low};
    }

    friend constexpr UInt128 operator-(UInt128 left, UInt128 right) noexcept {
        const std::uint64_t low = left.low_ - right.low_;
        return {left.high_ - right.high_ - (left.low_ < right.low_ ? 1U : 0U), low};
    }

    // Schoolbook 128x128 -> low 128 bits, matching the wrapping semantics of the native type.
    friend constexpr UInt128 operator*(UInt128 left, UInt128 right) noexcept {
        constexpr std::uint64_t mask = 0xFFFFFFFFULL;
        const std::uint64_t a0       = left.low_ & mask;
        const std::uint64_t a1       = left.low_ >> 32U;
        const std::uint64_t b0       = right.low_ & mask;
        const std::uint64_t b1       = right.low_ >> 32U;
        const std::uint64_t p00      = a0 * b0;
        const std::uint64_t p01      = a0 * b1;
        const std::uint64_t p10      = a1 * b0;
        const std::uint64_t p11      = a1 * b1;
        const std::uint64_t middle   = p10 + (p00 >> 32U) + (p01 & mask);
        const std::uint64_t low      = (middle << 32U) | (p00 & mask);
        const std::uint64_t high     = p11 + (middle >> 32U) + (p01 >> 32U) +
                                   left.low_ * right.high_ + left.high_ * right.low_;
        return {high, low};
    }

    friend constexpr UInt128 operator>>(UInt128 value, unsigned shift) noexcept {
        if (shift == 0U) { return value; }
        if (shift >= 128U) { return {}; }
        if (shift >= 64U) { return {0U, value.high_ >> (shift - 64U)}; }
        return {value.high_ >> shift, (value.low_ >> shift) | (value.high_ << (64U - shift))};
    }

    friend constexpr UInt128 operator<<(UInt128 value, unsigned shift) noexcept {
        if (shift == 0U) { return value; }
        if (shift >= 128U) { return {}; }
        if (shift >= 64U) { return {value.low_ << (shift - 64U), 0U}; }
        return {(value.high_ << shift) | (value.low_ >> (64U - shift)), value.low_ << shift};
    }

    friend constexpr UInt128 operator~(UInt128 value) noexcept {
        return {~value.high_, ~value.low_};
    }

    friend constexpr bool operator==(UInt128 left, UInt128 right) noexcept {
        return left.high_ == right.high_ && left.low_ == right.low_;
    }
    friend constexpr bool operator!=(UInt128 left, UInt128 right) noexcept {
        return !(left == right);
    }
    friend constexpr bool operator<(UInt128 left, UInt128 right) noexcept {
        return left.high_ != right.high_ ? left.high_ < right.high_ : left.low_ < right.low_;
    }
    friend constexpr bool operator>(UInt128 left, UInt128 right) noexcept { return right < left; }
    friend constexpr bool operator<=(UInt128 left, UInt128 right) noexcept {
        return !(right < left);
    }
    friend constexpr bool operator>=(UInt128 left, UInt128 right) noexcept {
        return !(left < right);
    }

private:
    std::uint64_t high_ = 0;
    std::uint64_t low_  = 0;
};

using WideUInt = UInt128;

#else

using WideUInt = unsigned __int128;

#endif

} // namespace ninfer
