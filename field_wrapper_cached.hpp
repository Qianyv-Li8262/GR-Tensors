#pragma once

#include "contraction_and_arithmetic.hpp"
#include "num_deriv.hpp"
#include "tensor_class.hpp"

#include <any>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <optional>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

// Cached alternative to field_wrapper.hpp.
//
// Include either field_wrapper.hpp or this file in one translation unit, not
// both.  The public Field/make_field API is intentionally kept compatible so
// the two implementations can be benchmarked with the same calling code.

namespace field_cache_detail {

inline std::size_t next_node_id() {
    static std::atomic<std::size_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

inline std::uint64_t double_bits(double value) noexcept {
    static_assert(sizeof(double) == sizeof(std::uint64_t), "unsupported double representation");
    std::uint64_t bits{};
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

struct CacheKey {
    std::size_t node_id;
    const void* call_type;
    std::array<std::uint64_t, 4> coordinate;

    bool operator==(const CacheKey& other) const noexcept {
        return node_id == other.node_id && call_type == other.call_type && coordinate == other.coordinate;
    }
};

inline std::size_t mix_hash(std::size_t seed, std::size_t value) noexcept {
    // A small hash combiner is sufficient here: keys live only for one root
    // evaluation and the coordinate has just four components.
    return seed ^ (value + static_cast<std::size_t>(0x9e3779b9U) + (seed << 6U) + (seed >> 2U));
}

struct CacheKeyHash {
    std::size_t operator()(const CacheKey& key) const noexcept {
        std::size_t hash = key.node_id;
        hash = mix_hash(hash, reinterpret_cast<std::size_t>(key.call_type));
        for (std::uint64_t component : key.coordinate) {
            hash = mix_hash(hash, static_cast<std::size_t>(component));
            if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
                hash = mix_hash(hash, static_cast<std::size_t>(component >> 32U));
            }
        }
        return hash;
    }
};

class EvaluationContext {
public:
    const std::any* find(const CacheKey& key) const {
        const auto found = values_.find(key);
        return found == values_.end() ? nullptr : &found->second;
    }

    template <typename T>
    void save(const CacheKey& key, T&& value) {
        values_.emplace(key, std::forward<T>(value));
    }

private:
    std::unordered_map<CacheKey, std::any, CacheKeyHash> values_;
};

inline thread_local EvaluationContext* active_context = nullptr;

// The outermost Field call owns the context.  Nested Field calls see the same
// pointer and leave its lifetime alone.  Destruction is exception-safe and
// clears the complete cache in one operation.
class EvaluationScope {
public:
    EvaluationScope() : previous_(active_context) {
        if (previous_ == nullptr) {
            owned_.emplace();
            active_context = &*owned_;
        }
    }

    EvaluationScope(const EvaluationScope&) = delete;
    EvaluationScope& operator=(const EvaluationScope&) = delete;

    ~EvaluationScope() {
        if (owned_) {
            active_context = previous_;
        }
    }

    EvaluationContext& context() const {
        return *active_context;
    }

    bool is_root() const noexcept {
        return owned_.has_value();
    }

private:
    EvaluationContext* previous_;
    std::optional<EvaluationContext> owned_;
};

template <typename T>
struct is_coordinate : std::false_type {};

template <typename Letter>
struct is_coordinate<Tensor<double, up<Letter>>> : std::true_type {};

template <typename T>
struct is_tensor : std::false_type {};

template <typename datatype, typename... Indices>
struct is_tensor<Tensor<datatype, Indices...>> : std::true_type {};

template <typename F, typename Result, typename... Args>
struct cache_call_traits {
    using coordinate_type = void;
    static constexpr bool value = false;
};

template <typename F, typename Result, typename Arg>
struct cache_call_traits<F, Result, Arg> {
    using coordinate_type = std::decay_t<Arg>;
    static constexpr bool value =
        is_coordinate<coordinate_type>::value && is_tensor<Result>::value &&
        std::is_invocable_r_v<Result, const F&, const coordinate_type&> &&
        std::is_copy_constructible_v<Result>;
};

template <typename Coordinate, typename Result>
inline const void* call_type_id() noexcept {
    // A node may be called with different generic-lambda instantiations.  This
    // token keeps differently typed calls out of the same std::any entry.
    static const int token = 0;
    return &token;
}

template <typename Coordinate, typename Result>
inline CacheKey make_key(std::size_t node_id, const Coordinate& x) {
    CacheKey key{node_id, call_type_id<std::decay_t<Coordinate>, Result>(), {}};
    for (std::size_t i = 0; i < key.coordinate.size(); ++i) {
        key.coordinate[i] = double_bits(x.data[i]);
    }
    return key;
}

} // namespace field_cache_detail

// The template shape stays Field<F>, matching field_wrapper.hpp.  Whether a
// node stores its own result is runtime metadata so uncached expression nodes
// do not introduce a second public Field type.
template <typename F>
struct Field {
    F func;

    explicit Field(F f) : Field(std::move(f), true) {
    }

    Field(F f, bool cache_enabled)
        : func(std::move(f)), node_id_(field_cache_detail::next_node_id()), cache_enabled_(cache_enabled) {
    }

    template <typename... Args>
    auto operator()(Args&&... args) const {
        using result_type = std::decay_t<decltype(func(std::forward<Args>(args)...))>;
        using cache_traits = field_cache_detail::cache_call_traits<F, result_type, Args...>;

        field_cache_detail::EvaluationScope scope;

        if constexpr (cache_traits::value) {
            using coordinate_type = typename cache_traits::coordinate_type;
            const auto& x = std::get<0>(std::forward_as_tuple(args...));
            const auto& const_x = static_cast<const coordinate_type&>(x);

            // The root result cannot be reused during the evaluation that is
            // currently creating it.  Skipping that insertion also keeps a
            // standalone field call at essentially the baseline behavior.
            if (cache_enabled_ && !scope.is_root()) {
                const auto key = field_cache_detail::make_key<coordinate_type, result_type>(node_id_, const_x);

                if (const std::any* cached = scope.context().find(key)) {
                    return std::any_cast<const result_type&>(*cached);
                }

                result_type result = func(const_x);
                scope.context().save(key, result);
                return result;
            }

            // A cacheable field is always invoked through a const coordinate.
            // This prevents a callable from changing the object used as a key.
            return result_type(func(const_x));
        } else {
            return func(std::forward<Args>(args)...);
        }
    }

private:
    // Default copy/move operations deliberately preserve both values.  Copies
    // of one Field are the same cache node; separately constructed Fields get
    // different node IDs.
    std::size_t node_id_;
    bool cache_enabled_;
};

template <typename F>
Field(F) -> Field<F>;

// User-created fields are cached automatically for the duration of one root
// evaluation.
template <typename F>
auto make_field(F&& f) {
    return Field<std::decay_t<F>>{std::forward<F>(f), true};
}

// Internal expression nodes are deliberately not cached: looking up a simple
// addition or rename normally costs more than evaluating it again.  Their
// expensive child fields remain cached.
template <typename F>
auto make_expression_field(F&& f) {
    return Field<std::decay_t<F>>{std::forward<F>(f), false};
}

// Optional escape hatch for cheap or stateful user callables.
template <typename F>
auto make_uncached_field(F&& f) {
    return Field<std::decay_t<F>>{std::forward<F>(f), false};
}

template <typename F1, typename F2>
auto operator+(const Field<F1>& lhs, const Field<F2>& rhs) {
    return make_expression_field([lhs, rhs](const auto& x) {
        return lhs(x) + rhs(x);
    });
}

template <typename F1, typename F2>
auto operator-(const Field<F1>& lhs, const Field<F2>& rhs) {
    return make_expression_field([lhs, rhs](const auto& x) {
        return lhs(x) - rhs(x);
    });
}

template <typename F1, typename F2>
auto operator*(const Field<F1>& lhs, const Field<F2>& rhs) {
    // Tensor contraction may be expensive, so a reused product is a cached
    // node of its own.
    return make_field([lhs, rhs](const auto& x) {
        return lhs(x) * rhs(x);
    });
}

template <typename F>
auto operator*(double scalar, const Field<F>& field) {
    return make_expression_field([scalar, field](const auto& x) {
        return scalar * field(x);
    });
}

template <typename F>
auto operator*(const Field<F>& field, double scalar) {
    return scalar * field;
}

template <typename... NewIndices, typename F>
auto rename_field(const Field<F>& field) {
    return make_expression_field([field](const auto& x) {
        return rename<NewIndices...>(field(x));
    });
}

template <typename DerivLetter, typename F>
auto partial_field(const Field<F>& field, double h = 1e-2) {
    // A partial derivative is expensive and can itself be a repeated common
    // subexpression, so its result is cached as a real field node.
    return make_field([field, h](const auto& x) {
        return partial<DerivLetter>(field, x, h);
    });
}

template <typename LetterA, typename LetterB, typename F>
auto trace_field(const Field<F>& field) {
    return make_expression_field([field](const auto& x) {
        return trace<LetterA, LetterB>(field(x));
    });
}

#define DEFINE_TENSOR_FIELD2(field_name, tensor_function)                                                      \
    template <typename FirstIndex, typename SecondIndex>                                                       \
    inline auto field_name = make_field([](const auto& x) {                                                    \
        return tensor_function<FirstIndex, SecondIndex>(x);                                                    \
    })
