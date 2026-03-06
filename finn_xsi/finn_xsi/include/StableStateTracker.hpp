#ifndef STABLESTATETRACKER
#define STABLESTATETRACKER

#include <cstdint>
#include <concepts>

/**
 * Implements an Exponential Moving Average (EMA) tracker with stability detection.
 * The tracker updates its EMA with new unsigned integral values and checks for stability
 * based on relative changes over consecutive updates.
 */
template<double Alpha = 0.3,              // alpha * 100 (20 = 0.20)
         double StabilityThreshold = 0.1,  // threshold * 100 (5 = 0.05)
         uint8_t RequiredStableCount = 3>
    requires (Alpha > 0 && Alpha <= 1) &&
             (StabilityThreshold > 0 && StabilityThreshold < 1) &&
             (RequiredStableCount > 0)
class StableStateTracker {
private:
    static constexpr double InvAlpha = 1.0 - Alpha;
    static constexpr double SquaredStabilityThreshold = StabilityThreshold * StabilityThreshold;

    double ema;
    uint8_t stableCount;

public:
    constexpr StableStateTracker() noexcept
        : ema{0.0}
        , stableCount{0}
    {
    }

    /**
     * Update with new interval value
     * Concepts ensure only unsigned integral types are accepted
     */
    inline void update(std::unsigned_integral auto value) noexcept {
        // First update initializes directly
        if (ema == 0.0) [[unlikely]] {
            ema = static_cast<double>(value);
            stableCount = 0;
            return;
        }

        const double oldEma = ema;
        const double valDouble = static_cast<double>(value);

        // EMA calculation: ema = value + (1-alpha) * (oldEma - value)
        ema = valDouble + InvAlpha * (oldEma - valDouble);

        // Stability check: |change|² / oldEma² < threshold²
        // Avoids sqrt and abs operations
        const double diff = ema - oldEma;
        const double squaredRelativeChange = (diff * diff) / (oldEma * oldEma);

        // Branchless increment/reset using arithmetic
        const bool is_change_small = squaredRelativeChange < SquaredStabilityThreshold;
        stableCount = is_change_small * (stableCount + (stableCount < RequiredStableCount));
    }

    [[nodiscard]] constexpr double get_ema() const noexcept {
        return ema;
    }

    [[nodiscard]] constexpr bool is_stable() const noexcept {
        return stableCount >= RequiredStableCount;
    }

    [[nodiscard]] constexpr uint8_t get_stable_count() const noexcept {
        return stableCount;
    }

    constexpr void reset() noexcept {
        ema = 0.0;
        stableCount = 0;
    }

    // Get compile-time parameters
    static consteval double get_alpha() { return Alpha; }
    static consteval double get_stability_threshold() { return StabilityThreshold; }
    static consteval uint8_t get_required_stable_count() { return RequiredStableCount; }
};

#endif /* STABLESTATETRACKER */
