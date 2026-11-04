#pragma once

#include "tensor_expression.hpp"
#include "expression_graph_operations.hpp"
#include <cstdint>
#include <optional>

namespace dag_detail {

inline constexpr std::size_t not_found = static_cast<std::size_t>(-1);

template <typename T>
constexpr std::string_view type_name() {
#if defined(_MSC_VER) && !defined(__clang__)
    return __FUNCSIG__;
#else
    return __PRETTY_FUNCTION__;
#endif
}

template <typename L, typename R>
constexpr bool type_less() {
    return type_name<L>() < type_name<R>();
}

// The signature supplies a compiler-local hash, never an identity or persistent ID.
// Hash collisions are resolved by comparing the complete key types in a bucket.
template <typename T>
constexpr std::uint64_t type_hash_value() {
    std::uint64_t hash = 14695981039346656037ull;
    for (char c : type_name<T>()) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 1099511628211ull;
    }
    return hash;
}

template <typename T>
struct TypeHash : std::integral_constant<std::uint64_t, type_hash_value<T>()> {};

struct MissingEntry : std::integral_constant<std::size_t, not_found> {};

template <typename Key, typename Value>
struct HashEntry {
    using key = Key;
    using mapped_type = Value;
};

template <typename Value, typename = void>
struct HashLookup {
    using type = Value;
};

template <typename Value>
struct HashLookup<Value, std::void_t<decltype(Value::value)>> {
    using type = Value;
    static constexpr auto value = Value::value;
};

template <typename Key, typename Bucket>
struct bucket_find;

template <typename Key>
struct bucket_find<Key, type_list<>> : HashLookup<MissingEntry> {};

template <typename Key, typename Head, typename... Tail>
struct bucket_find<Key, type_list<Head, Tail...>>
    : std::conditional_t<std::is_same_v<Key, typename Head::key>,
                         HashLookup<typename Head::mapped_type>,
                         bucket_find<Key, type_list<Tail...>>> {};

template <std::size_t>
using empty_bucket = type_list<>;

template <typename Sequence>
struct empty_buckets;

template <std::size_t... I>
struct empty_buckets<std::index_sequence<I...>> {
    using type = type_list<empty_bucket<I>...>;
};

template <std::size_t I, typename Buckets>
struct list_at;

template <std::size_t I, typename... Buckets>
struct list_at<I, type_list<Buckets...>> {
#if defined(__clang__)
    using type = __type_pack_element<I, Buckets...>;
#else
    using type = typename nth_type<I, type_list<Buckets...>>::result;
#endif
};

// Only instantiate append_back for the selected bucket. An eager
// conditional_t would generate discarded appended types for every bucket.
template <bool Selected, typename Bucket, typename Entry>
struct insert_bucket { using type = Bucket; };

template <typename Bucket, typename Entry>
struct insert_bucket<true, Bucket, Entry> {
    using type = typename append_back<Bucket, Entry>::result;
};

} // namespace dag_detail

// Immutable, compile-time hash table. All buckets, entries and values are types.
template <std::size_t BucketCount = 64,
          template <typename> class Hasher = dag_detail::TypeHash,
          typename Buckets = typename dag_detail::empty_buckets<std::make_index_sequence<BucketCount>>::type>
struct TypeHashTable {
    static_assert(BucketCount > 0, "DAG hash table requires at least one bucket");
    static constexpr std::size_t bucket_count = BucketCount;

    template <typename Key>
    static constexpr std::size_t bucket_index = Hasher<Key>::value % BucketCount;

    template <typename Key>
    using find = dag_detail::bucket_find<Key, typename dag_detail::list_at<bucket_index<Key>, Buckets>::type>;

private:
    template <typename Key, typename Value, std::size_t... I>
    static auto insert_impl(std::index_sequence<I...>) -> TypeHashTable<BucketCount, Hasher,
        type_list<typename dag_detail::insert_bucket<I == bucket_index<Key>,
            typename dag_detail::list_at<I, Buckets>::type, dag_detail::HashEntry<Key, Value>>::type...>>;

public:
    template <typename Key, typename Value>
    struct bind {
        static constexpr bool missing = std::is_same_v<typename find<Key>::type, dag_detail::MissingEntry>;
        static_assert(missing || std::is_same_v<typename find<Key>::type, Value>,
                      "cannot assign a second value to the same DAG key");
        using type = std::conditional_t<missing,
            decltype(insert_impl<Key, Value>(std::make_index_sequence<BucketCount>{})), TypeHashTable>;
    };

    template <typename Key, std::size_t Id>
    using insert = bind<Key, std::integral_constant<std::size_t, Id>>;
};

namespace dag_detail {

template <std::size_t... I>
auto identity_layout(std::index_sequence<I...>) -> index<I...>;

template <std::size_t Rank>
using identity = decltype(identity_layout(std::make_index_sequence<Rank>{}));

// A reference is a node ID plus a logical-output-axis -> stored-axis mapping.
// The mapping is metadata: it never materializes a permuted intermediate array.
template <std::size_t Id, typename Layout>
struct NodeRef {
    static constexpr std::size_t id = Id;
    using layout = Layout;
};

template <auto Function, typename Offset>
struct FieldOp {
    using field_type = FieldExpr<Function, Offset>;
    using scalar_type = typename field_type::scalar_type;
    static constexpr std::size_t rank = field_type::rank;
    using result_type = std::array<scalar_type, pow4(rank)>;
    using point_offset = Offset;

    template <std::size_t... Inputs, typename Context>
    static auto eval(Context& context) {
        static_assert(sizeof...(Inputs) == 0, "field DAG nodes have no inputs");
        return field_type{}.eval_components(context.x, context.h);
    }
};

template <typename Plan, typename Offset, typename Scalar>
struct ContractOp {
    static_assert(std::is_arithmetic_v<Scalar>, "DAG contraction normalization requires arithmetic scalars");
    using plan = Plan;
    using point_offset = Offset;
    using scalar_type = Scalar;
    static constexpr std::size_t rank = Plan::result_rank;
    using result_type = std::array<Scalar, pow4(rank)>;

    template <std::size_t Left, std::size_t Right, typename Context>
    static auto eval(Context& context) {
        // decltype(auto) owns single-use temporaries and borrows shared results.
        decltype(auto) left = context.template get<Left>();
        decltype(auto) right = context.template get<Right>();
        return contract_components<Plan>(left, right);
    }
};

template <typename Operation, std::size_t... Inputs>
struct NodeKey {};

} // namespace dag_detail

template <std::size_t Id, typename Operation, std::size_t... Inputs>
struct DAGNode {
    static constexpr std::size_t id = Id;
    static_assert(((Inputs < Id) && ...), "DAG inputs must precede their consumer");
    using operation = Operation;
    using inputs = std::index_sequence<Inputs...>;
    using result_type = typename Operation::result_type;
    using scalar_type = typename Operation::scalar_type;
    static constexpr std::size_t rank = Operation::rank;

    template <std::size_t Input>
    static constexpr std::size_t input_uses = (std::size_t{0} + ... + (Inputs == Input ? 1 : 0));

    template <typename Context>
    static auto execute(Context& context) {
        return Operation::template eval<Inputs...>(context);
    }
};

namespace dag_detail {

template <typename Plan>
struct transpose_plan;

template <std::size_t L, std::size_t R, typename... Pairs>
struct transpose_plan<contraction_plan<L, R, Pairs...>> {
    using type = contraction_plan<R, L, contractpair<Pairs::right, Pairs::left>...>;
};

template <typename Plan>
struct pair_axes;

template <std::size_t L, std::size_t R, typename... Pairs>
struct pair_axes<contraction_plan<L, R, Pairs...>> {
    inline static constexpr std::array<std::size_t, sizeof...(Pairs)> left{Pairs::left...};
    inline static constexpr std::array<std::size_t, sizeof...(Pairs)> right{Pairs::right...};
};

template <typename A, typename B>
constexpr bool plan_less() {
    for (std::size_t i = 0; i < A::contract_rank; ++i) {
        if (pair_axes<A>::left[i] != pair_axes<B>::left[i])
            return pair_axes<A>::left[i] < pair_axes<B>::left[i];
        if (pair_axes<A>::right[i] != pair_axes<B>::right[i])
            return pair_axes<A>::right[i] < pair_axes<B>::right[i];
    }
    return false;
}

// Axis geometry shared by source-signature ordering and ID-based interning.
// Pair order itself is preserved, including floating-point sum order.
template <typename Plan, typename LeftLayout, typename RightLayout, bool Swap>
struct contraction_layout;

template <std::size_t L, std::size_t R, typename... Pairs,
          typename LeftLayout, typename RightLayout, bool Swap>
struct contraction_layout<contraction_plan<L, R, Pairs...>, LeftLayout, RightLayout, Swap> {
    using original_plan = contraction_plan<L, R, Pairs...>;
    using left_layout = LeftLayout;
    using right_layout = RightLayout;
    using mapped_plan = contraction_plan<L, R,
        contractpair<left_layout::axes[Pairs::left], right_layout::axes[Pairs::right]>...>;
    using transposed_plan = typename transpose_plan<mapped_plan>::type;
    static constexpr bool swapped = Swap;
    using plan = std::conditional_t<swapped, transposed_plan, mapped_plan>;

private:
    static constexpr std::size_t free_position(std::size_t axis, std::size_t side) {
        std::size_t position = side == 0 ? 0 : plan::left_rank - plan::contract_rank;
        for (std::size_t i = 0; i < axis; ++i) {
            if (!plan::is_contracted(i, side)) ++position;
        }
        return position;
    }
    static constexpr auto output_axes() {
        std::array<std::size_t, plan::result_rank> result{};
        std::size_t output = 0;
        for (std::size_t axis = 0; axis < L; ++axis) {
            if (!original_plan::is_contracted(axis, 0))
                result[output++] = free_position(left_layout::axes[axis], swapped ? 1 : 0);
        }
        for (std::size_t axis = 0; axis < R; ++axis) {
            if (!original_plan::is_contracted(axis, 1))
                result[output++] = free_position(right_layout::axes[axis], swapped ? 0 : 1);
        }
        return result;
    }
    inline static constexpr auto axes = output_axes();
    template <std::size_t... I>
    static auto layout_type(std::index_sequence<I...>) -> index<axes[I]...>;

public:
    using layout = decltype(layout_type(std::make_index_sequence<plan::result_rank>{}));
};

// Bottom-up interning orders operands by their already known IDs. This catches
// equivalents that local source-signature ordering did not identify early.
template <typename Plan, typename Left, typename Right>
struct normalize_contract {
    using mapped = contraction_layout<Plan, typename Left::layout, typename Right::layout, false>;
    static constexpr bool swapped = Left::id > Right::id || (Left::id == Right::id &&
        plan_less<typename mapped::transposed_plan, typename mapped::mapped_plan>());
    using normalized = contraction_layout<Plan, typename Left::layout, typename Right::layout, swapped>;
    using plan = typename normalized::plan;
    using layout = typename normalized::layout;
    static constexpr std::size_t left_id = swapped ? Right::id : Left::id;
    static constexpr std::size_t right_id = swapped ? Left::id : Right::id;
};

template <typename Stored, typename Logical>
struct compose_layout;

template <typename Stored, std::size_t... Axes>
struct compose_layout<Stored, index<Axes...>> {
    using type = index<Stored::axes[Axes]...>;
};

template <typename Layout>
struct inverse_layout {
private:
    inline static constexpr auto axes = [] {
        std::array<std::size_t, Layout::rank> result{};
        for (std::size_t i = 0; i < Layout::rank; ++i) result[Layout::axes[i]] = i;
        return result;
    }();
    template <std::size_t... I>
    static auto make(std::index_sequence<I...>) -> index<axes[I]...>;

public:
    using type = decltype(make(std::make_index_sequence<Layout::rank>{}));
};

// Source keys contain expression types, not child IDs. They can be queried
// before descent. NodeKey remains the ID-based index for bottom-up interning;
// both key spaces share one compile-time table and compare full key types.
template <typename Node>
struct SourceKey {};

template <typename Node>
struct normalize_source {
    using node = Node;
    using key = SourceKey<node>;
    using layout = identity<Node::rank>;
};

template <typename L, typename R, typename Plan, typename Offset>
struct normalize_source<ContractExpr<L, R, Plan, Offset>> {
    static constexpr bool swapped = type_less<R, L>() ||
        (std::is_same_v<L, R> && plan_less<typename transpose_plan<Plan>::type, Plan>());
    // No child IDs, child compilation or recursive source normalization here.
    using normalized = contraction_layout<Plan, identity<L::rank>, identity<R::rank>, swapped>;
    using node = ContractExpr<std::conditional_t<swapped, R, L>,
                              std::conditional_t<swapped, L, R>,
                              typename normalized::plan, Offset>;
    using key = SourceKey<node>;
    using layout = typename normalized::layout;
};

template <typename Nodes, typename Table>
struct BuildState {
    using nodes = Nodes;
    using table = Table;
};

template <typename Operation, typename State, typename Inputs, bool Found>
struct intern_impl;

template <typename Operation, typename State, std::size_t... Inputs>
struct intern_impl<Operation, State, std::index_sequence<Inputs...>, true> {
    using state = State;
    using table = typename State::table;
    static constexpr std::size_t id = table::template find<NodeKey<Operation, Inputs...>>::value;
};

template <typename Operation, typename State, std::size_t... Inputs>
struct intern_impl<Operation, State, std::index_sequence<Inputs...>, false> {
    static constexpr std::size_t id = list_size<typename State::nodes>::value;
    using node = DAGNode<id, Operation, Inputs...>;
    using old_table = typename State::table;
    using table = typename old_table::template insert<NodeKey<Operation, Inputs...>, id>::type;
    using state = BuildState<typename append_back<typename State::nodes, node>::result, table>;
};

template <typename Operation, typename State, std::size_t... Inputs>
using intern = intern_impl<Operation, State, std::index_sequence<Inputs...>,
    (State::table::template find<NodeKey<Operation, Inputs...>>::value != not_found)>;

template <typename>
inline constexpr bool unsupported_node = false;

template <typename Node, typename State>
struct build_node {
    static_assert(unsupported_node<Node>, "unsupported expression node in DAG compile");
};

template <typename Node, typename State, typename Source, bool Found>
struct compile_dispatch;

// This specialization deliberately does not mention build_node or compile_node
// for the children. A source hit skips instantiating their compilation paths.
template <typename Node, typename State, typename Source>
struct compile_dispatch<Node, State, Source, true> {
    static constexpr bool source_hit = true;
    using state = State;
    using table = typename State::table;
    using saved = typename table::template find<typename Source::key>::type;
    using ref = NodeRef<saved::id,
        typename compose_layout<typename saved::layout, typename Source::layout>::type>;
};

template <typename Node, typename State, typename Source>
struct compile_dispatch<Node, State, Source, false> {
    static constexpr bool source_hit = false;
    using built = build_node<Node, State>;
    using ref = typename built::ref;
    using old_state = typename built::state;
    using old_table = typename old_state::table;
    // Store canonical-source axes -> graph-storage axes. On a hit the caller's
    // own source layout is composed back in, including for unequal free blocks.
    using saved = NodeRef<ref::id, typename compose_layout<typename ref::layout,
        typename inverse_layout<typename Source::layout>::type>::type>;
    using table = typename old_table::template bind<typename Source::key, saved>::type;
    using state = BuildState<typename old_state::nodes, table>;
};

template <typename Node, typename State>
struct compile_node : compile_dispatch<Node, State, normalize_source<Node>,
    !std::is_same_v<typename State::table::template find<typename normalize_source<Node>::key>::type,
                    MissingEntry>> {};

template <auto Function, typename Offset, typename State>
struct build_node<FieldExpr<Function, Offset>, State> {
    using operation = FieldOp<Function, Offset>;
    using built = intern<operation, State>;
    using state = typename built::state;
    using ref = NodeRef<built::id, identity<operation::rank>>;
};

template <typename L, typename R, typename Plan, typename Offset, typename State>
struct build_node<ContractExpr<L, R, Plan, Offset>, State> {
    using left = compile_node<L, State>;
    using right = compile_node<R, typename left::state>;
    using normalized = normalize_contract<Plan, typename left::ref, typename right::ref>;
    using operation = ContractOp<typename normalized::plan, Offset, typename L::scalar_type>;
    // Only the normalized operation and canonical child IDs enter the hash table.
    using built = intern<operation, typename right::state, normalized::left_id, normalized::right_id>;
    using state = typename built::state;
    using ref = NodeRef<built::id, typename normalized::layout>;
};

template <typename Node, typename Ratio, typename Offset, typename State>
struct build_node<ScaleExpr<Node, Ratio, Offset>, State> {
    using child = compile_node<Node, State>;
    using operation = ScaleOp<typename Ratio::type, typename Node::scalar_type, Node::rank, Offset>;
    using built = intern<operation, typename child::state, child::ref::id>;
    using state = typename built::state;
    using ref = NodeRef<built::id, typename child::ref::layout>;
};

template <typename Left, typename Right, typename Scalar, typename Offset, typename State>
struct add_references {
    using operation = AddOp<typename Left::layout, typename Right::layout, Scalar, Offset>;
    using built = intern<operation, State, Left::id, Right::id>;
    using state = typename built::state;
    using ref = NodeRef<built::id, identity<operation::rank>>;
};

template <typename L, typename R, int... Positions, typename Offset, typename State>
struct build_node<AddExpr<L, R, add_plan<Positions...>, Offset>, State> {
    using left = compile_node<L, State>;
    using right = compile_node<R, typename left::state>;
    // add_plan names the output coordinate read by each right input axis.
    using alignment = typename inverse_layout<index<static_cast<std::size_t>(Positions)...>>::type;
    using right_view = NodeRef<right::ref::id,
        typename compose_layout<typename right::ref::layout, alignment>::type>;
    using added = add_references<typename left::ref, right_view, typename L::scalar_type,
                                  Offset, typename right::state>;
    using state = typename added::state;
    using ref = typename added::ref;
};

template <typename Node, typename Plan, typename Offset, typename State>
struct build_node<PermuteExpr<Node, Plan, Offset>, State> {
    using child = compile_node<Node, State>;
    using state = typename child::state;
    using ref = NodeRef<child::ref::id, typename compose_layout<typename child::ref::layout, Plan>::type>;
};

template <typename Plan, typename Layout>
struct trace_layout {
    using plan = trace_plan<Plan::input_rank, Layout::axes[Plan::axis_a], Layout::axes[Plan::axis_b]>;
private:
    inline static constexpr auto axes = [] {
        std::array<std::size_t, Plan::result_rank> result{};
        std::size_t output = 0;
        for (std::size_t axis = 0; axis < Plan::input_rank; ++axis) {
            if (Plan::is_contracted(axis)) continue;
            std::size_t position = 0;
            for (std::size_t i = 0; i < Layout::axes[axis]; ++i) {
                if (!plan::is_contracted(i)) ++position;
            }
            result[output++] = position;
        }
        return result;
    }();
    template <std::size_t... I>
    static auto make(std::index_sequence<I...>) -> index<axes[I]...>;
public:
    using layout = decltype(make(std::make_index_sequence<Plan::result_rank>{}));
};

template <typename Node, typename Plan, typename Offset, typename State>
struct build_node<TraceExpr<Node, Plan, Offset>, State> {
    using child = compile_node<Node, State>;
    using normalized = trace_layout<Plan, typename child::ref::layout>;
    using operation = TraceOp<typename normalized::plan, typename Node::scalar_type, Offset>;
    using built = intern<operation, typename child::state, child::ref::id>;
    using state = typename built::state;
    using ref = NodeRef<built::id, typename normalized::layout>;
};

template <typename Plan, typename Layout>
struct map_symmetry;

template <std::size_t Rank, std::size_t... Axes, typename Layout>
struct map_symmetry<sym_plan<Rank, Axes...>, Layout> {
    using type = sym_plan<Rank, Layout::axes[Axes]...>;
};

template <std::size_t Rank, std::size_t... Axes, typename Layout>
struct map_symmetry<asym_plan<Rank, Axes...>, Layout> {
    using type = asym_plan<Rank, Layout::axes[Axes]...>;
};

template <typename Node, typename Plan, typename Offset, typename State>
struct build_projection {
    using child = compile_node<Node, State>;
    using mapped_plan = typename map_symmetry<Plan, typename child::ref::layout>::type;
    using operation = SymmetryOp<mapped_plan, typename Node::scalar_type, Offset>;
    using built = intern<operation, typename child::state, child::ref::id>;
    using state = typename built::state;
    using ref = NodeRef<built::id, typename child::ref::layout>;
};

template <typename Node, typename Plan, typename Offset, typename State>
struct build_node<SymExpr<Node, Plan, Offset>, State> : build_projection<Node, Plan, Offset, State> {};

template <typename Node, typename Plan, typename Offset, typename State>
struct build_node<AsymExpr<Node, Plan, Offset>, State> : build_projection<Node, Plan, Offset, State> {};

template <typename Node, typename Offset, typename State>
struct build_node<InverseExpr<Node, Offset>, State> {
    using child = compile_node<Node, State>;
    using operation = InverseOp<typename Node::scalar_type, Offset>;
    using built = intern<operation, typename child::state, child::ref::id>;
    using state = typename built::state;
    // A rank-two view is either identity or transpose, and inverse commutes
    // with transpose. The kernel can therefore operate on stored components.
    using ref = NodeRef<built::id, typename child::ref::layout>;
};

template <typename Accumulator, typename Sample, typename Scalar, typename Offset, typename State>
struct accumulate_sample : add_references<Accumulator, Sample, Scalar, Offset, State> {};

template <typename Sample, typename Scalar, typename Offset, typename State>
struct accumulate_sample<void, Sample, Scalar, Offset, State> {
    using state = State;
    using ref = Sample;
};

template <DiffScheme Scheme, typename Offset, typename Scalar, typename State,
          std::size_t Position, typename Accumulator, typename... Samples>
struct stencil_fold;

template <DiffScheme Scheme, typename Offset, typename Scalar, typename State,
          std::size_t Position, typename Accumulator>
struct stencil_fold<Scheme, Offset, Scalar, State, Position, Accumulator> {
    using operation = DivideStepOp<Scalar, Accumulator::layout::rank, Offset>;
    using built = intern<operation, State, Accumulator::id>;
    using state = typename built::state;
    using ref = NodeRef<built::id, typename Accumulator::layout>;
};

template <DiffScheme Scheme, typename Offset, typename Scalar, typename State,
          std::size_t Position, typename Accumulator, typename Head, typename... Tail>
struct stencil_fold<Scheme, Offset, Scalar, State, Position, Accumulator, Head, Tail...> {
    using sample = compile_node<Head, State>;
    using weight = typename std::ratio<derivative_stencil<Scheme>::weights[Position],
                                      derivative_stencil<Scheme>::denominator>::type;
    using operation = ScaleOp<weight, Scalar, Head::rank, typename Head::point_offset>;
    using scaled = intern<operation, typename sample::state, sample::ref::id>;
    using scaled_ref = NodeRef<scaled::id, typename sample::ref::layout>;
    using added = accumulate_sample<Accumulator, scaled_ref, Scalar, Offset, typename scaled::state>;
    using next = stencil_fold<Scheme, Offset, Scalar, typename added::state,
                              Position + 1, typename added::ref, Tail...>;
    using state = typename next::state;
    using ref = typename next::ref;
};

template <DiffScheme Scheme, typename Offset, typename... Samples, typename State>
struct build_node<StencilSumExpr<Scheme, Offset, Samples...>, State>
    : stencil_fold<Scheme, Offset, typename StencilSumExpr<Scheme, Offset, Samples...>::scalar_type,
                   State, 0, void, Samples...> {};

template <typename D0, typename D1, typename D2, typename D3, typename Offset, typename State>
struct build_node<SampledPartialExpr<D0, D1, D2, D3, Offset>, State> {
    using a = compile_node<D0, State>;
    using b = compile_node<D1, typename a::state>;
    using c = compile_node<D2, typename b::state>;
    using d = compile_node<D3, typename c::state>;
    using operation = PackDirectionsOp<typename D0::scalar_type, Offset,
        typename a::ref::layout, typename b::ref::layout, typename c::ref::layout, typename d::ref::layout>;
    using built = intern<operation, typename d::state, a::ref::id, b::ref::id, c::ref::id, d::ref::id>;
    using state = typename built::state;
    using ref = NodeRef<built::id, identity<operation::rank>>;
};

template <typename Node, DiffScheme Scheme, typename Offset, typename State>
struct build_node<PartialExpr<Node, Scheme, Offset>, State> {
    // Only TYPES are expanded. No sampled expression objects or evaluations
    // are created here. Nested expand accumulates absolute integer offsets.
    using sampled = typename expression_detail::partial_samples<Node, Scheme>::result;
    using lowered = compile_node<sampled, State>;
    using state = typename lowered::state;
    using ref = typename lowered::ref;
};

template <typename Result, typename Ref>
struct GraphOutput {
    using result_type = Result;
    using ref = Ref;
};

template <typename Graph, typename Coordinate>
struct Evaluation {
    const Coordinate& x;
    double h;
    typename Graph::cache_type cache{};

    template <std::size_t Id>
    decltype(auto) get() {
        using node = typename Graph::template node<Id>;
        if constexpr (Graph::template use_count<Id> > 1) {
            auto& slot = std::get<Graph::template cache_index<Id>>(cache);
            if (!slot) slot.emplace(node::execute(*this));
            return static_cast<const typename node::result_type&>(*slot);
        } else {
            return node::execute(*this);
        }
    }
};

template <std::size_t Size, std::size_t... Inputs>
constexpr void count_inputs(std::array<std::size_t, Size>& counts, std::index_sequence<Inputs...>) {
    (++counts[Inputs], ...);
}

template <std::size_t Root, typename... Nodes>
constexpr auto count_uses() {
    std::array<std::size_t, sizeof...(Nodes)> result{};
    (count_inputs(result, typename Nodes::inputs{}), ...);
    ++result[Root];
    return result;
}

// Join type lists without asking std::tuple_cat to instantiate runtime tuple
// machinery for every empty/singleton candidate. Eight-way batches bound depth.
template <typename... Lists>
struct cache_tuple;

template <typename... T>
struct cache_tuple<type_list<T...>> { using type = std::tuple<T...>; };

template <typename... A, typename... B, typename... Rest>
struct cache_tuple<type_list<A...>, type_list<B...>, Rest...>
    : cache_tuple<type_list<A..., B...>, Rest...> {};

template <typename... A, typename... B, typename... C, typename... D,
          typename... E, typename... F, typename... G, typename... H, typename... Rest>
struct cache_tuple<type_list<A...>, type_list<B...>, type_list<C...>, type_list<D...>,
                   type_list<E...>, type_list<F...>, type_list<G...>, type_list<H...>, Rest...>
    : cache_tuple<type_list<A..., B..., C..., D..., E..., F..., G..., H...>, Rest...> {};

} // namespace dag_detail

// Nodes are a topologically ordered type pack, not runtime node objects.
// Execution follows dependencies recursively and memoizes only shared nodes.
template <typename Output, typename... Nodes>
struct ExprGraph {
    using output = Output;
    using result_type = typename Output::result_type;
    using root_ref = typename Output::ref;
    using root_layout = typename root_ref::layout;
    using nodes = type_list<Nodes...>;
    static constexpr std::size_t node_count = sizeof...(Nodes);
    static constexpr std::size_t root_id = root_ref::id;
    static constexpr bool requires_step = (false || ... ||
        dag_detail::operation_uses_step<typename Nodes::operation>::value);

    template <std::size_t Id>
    using node = typename dag_detail::list_at<Id, nodes>::type;

    // Count edges, including repeated inputs to one parent. Returning the root
    // contributes one use. Counts apply to the deduplicated DAG, not the tree.
    inline static constexpr auto uses = dag_detail::count_uses<root_id, Nodes...>();

    template <std::size_t Id>
    static constexpr std::size_t use_count = uses[Id];

    inline static constexpr auto cache_indices = [] {
        std::array<std::size_t, node_count> result{};
        std::size_t slot = 0;
        for (std::size_t i = 0; i < node_count; ++i) {
            result[i] = slot;
            if (uses[i] > 1) ++slot;
        }
        return result;
    }();

    template <std::size_t Id>
    static constexpr std::size_t cache_index = cache_indices[Id];

    using cache_type = typename dag_detail::cache_tuple<type_list<>, std::conditional_t<
        (use_count<Nodes::id> > 1), type_list<std::optional<typename Nodes::result_type>>, type_list<>>...>::type;
    static constexpr std::size_t cache_count = std::tuple_size<cache_type>::value;

    template <typename Coordinate>
    result_type eval(const Coordinate& x, double h = 1e-2) const {
        if constexpr (requires_step) derivative_detail::check_step(x, h);
        dag_detail::Evaluation<ExprGraph, Coordinate> context{x, h, {}};
        result_type result;
        if constexpr (root_layout::is_identity) {
            result.data = context.template get<root_id>();
        } else {
            // Intermediate layouts were absorbed into consumer plans. Only the
            // public result, if necessary, is copied into its original axis order.
            result.data = permute_components<root_layout>(context.template get<root_id>());
        }
        return result;
    }
};

namespace dag_detail {

template <typename Output, typename Nodes>
struct make_graph;

template <typename Output, typename... Nodes>
struct make_graph<Output, type_list<Nodes...>> {
    using type = ExprGraph<Output, Nodes...>;
};

} // namespace dag_detail

// Current supported nodes encode all their data in types. No expression object
// or pointer into the source tree is retained; graphs may outlive that tree.
template <std::size_t BucketCount = 64, typename Node, typename... Indices>
constexpr auto compile(const IndexedExpr<Node, Indices...>&) {
    using initial = dag_detail::BuildState<type_list<>, TypeHashTable<BucketCount>>;
    using built = dag_detail::compile_node<Node, initial>;
    using output = dag_detail::GraphOutput<Tensor<typename Node::scalar_type, Indices...>, typename built::ref>;
    using graph = typename dag_detail::make_graph<output, typename built::state::nodes>::type;
    return graph{};
}
