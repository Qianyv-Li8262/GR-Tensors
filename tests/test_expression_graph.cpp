#include "expression_graph.hpp"
#include <cmath>
#include <iostream>

namespace {

int failures = 0;
int f_calls = 0;
int g_calls = 0;
int scalar_calls = 0;

void check(bool condition, const char* message) {
    if (!condition) { ++failures; std::cerr << message << '\n'; }
}

auto f(const coord<mu>& x) {
    ++f_calls;
    Tensor<double, up<a>, dn<b>> value;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            value(i, j) = 0.3 + 2 * i - 0.7 * j + (i + 1) * x(0) - j * x(2);
    return value;
}

auto g(const coord<mu>& x) noexcept {
    ++g_calls;
    Tensor<double, up<a>, dn<b>> value;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            value(i, j) = 1.2 - 0.4 * i + j * j + j * x(1) + (i + 1) * x(3);
    return value;
}

auto scalar(const coord<mu>& x) {
    ++scalar_calls;
    return Tensor<double>(2 + x(0));
}

auto other_scalar(const coord<mu>& x) { return Tensor<double>(3 - x(1)); }

auto raising_polynomial(const coord<mu>& x) {
    return Tensor<double>(x(0) * x(0) + 3 * x(2) * x(3));
}

auto raising_inverse(const coord<mu>& x) {
    Tensor<double, up<a>, up<b>> value;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            value(i, j) = (i == j ? 2 + x(1) : 0) + 0.25 * (i + 1) * (j + 1);
    return value;
}

auto rank_three(const coord<mu>& x) {
    Tensor<double, up<a>, dn<b>, up<c>> value;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                value(i, j, k) = 1 + 0.2 * i - 0.3 * j + k * k + (j + 1) * x(2);
    return value;
}

template <typename A, typename B>
void compare(const A& actual, const B& expected) {
    static_assert(std::is_same_v<A, B>, "graph and direct engines must return the same tensor type");
    for (std::size_t i = 0; i < actual.data.size(); ++i) {
        check(std::isfinite(actual.data[i]) &&
              std::abs(actual.data[i] - expected.data[i]) < 1e-10 * std::max(1.0, std::abs(expected.data[i])),
              "graph differs from direct evaluation");
    }
}

// Positional test helper: output letters are deliberately unrelated to leaves.
template <typename Node, std::size_t... I>
auto wrap(Node value, std::index_sequence<I...>) {
    return IndexedExpr<Node, up<std::integral_constant<std::size_t, I>>...>{value};
}

template <typename Node>
auto wrap(Node value) { return wrap(value, std::make_index_sequence<Node::rank>{}); }

template <typename Plan, typename L, typename R>
auto contract(L left, R right) {
    return ContractExpr<L, R, Plan, typename L::point_offset>{left, right};
}

template <typename Key>
struct ValueHasher : std::integral_constant<std::size_t, Key::value> {};

void check_hash_table() {
    // One bucket forces every distinct key to collide.
    using Empty = TypeHashTable<1>;
    using First = typename Empty::insert<int, 7>::type;
    using Second = typename First::insert<double, 19>::type;
    using Same = typename Second::insert<int, 7>::type;
    static_assert(Empty::find<int>::value == dag_detail::not_found);
    static_assert(First::find<double>::value == dag_detail::not_found);
    static_assert(Second::find<int>::value == 7 && Second::find<double>::value == 19);
    static_assert(Second::find<float>::value == dag_detail::not_found);
    static_assert(std::is_same_v<Same, Second>);

    using Ref = dag_detail::NodeRef<3, index<1, 0>>;
    using Typed = typename Second::bind<float, Ref>::type;
    static_assert(std::is_same_v<typename Typed::find<float>::type, Ref>);
    static_assert(Typed::find<int>::value == 7 && Typed::find<double>::value == 19);

    // Exercise both directory branches, an uneven bucket count, collisions and
    // persistence. One-bucket tests alone cannot catch a broken directory path.
    using K0 = std::integral_constant<std::size_t, 0>;
    using K1 = std::integral_constant<std::size_t, 1>;
    using K2 = std::integral_constant<std::size_t, 2>;
    using K3 = std::integral_constant<std::size_t, 3>;
    using Three = TypeHashTable<3, ValueHasher>;
    using T0 = typename Three::insert<K0, 10>::type;
    using T1 = typename T0::insert<K1, 11>::type;
    using T2 = typename T1::insert<K2, 12>::type;
    using T3 = typename T2::insert<K3, 13>::type;
    static_assert(T3::find<K0>::value == 10 && T3::find<K1>::value == 11 &&
                  T3::find<K2>::value == 12 && T3::find<K3>::value == 13);
    static_assert(T0::find<K1>::value == dag_detail::not_found &&
                  T2::find<K3>::value == dag_detail::not_found);
    static_assert(std::is_same_v<typename T3::insert<K2, 12>::type, T3>);
}

template <std::size_t Count>
struct filled_node_store {
    using type = typename dag_detail::append_node<typename filled_node_store<Count - 1>::type,
        std::integral_constant<std::size_t, Count - 1>>::type;
};

template <>
struct filled_node_store<0> { using type = dag_detail::NodeStore<>; };

template <std::size_t... I>
auto expected_node_store(std::index_sequence<I...>) -> type_list<std::integral_constant<std::size_t, I>...>;

template <typename Chunks, typename Tail, std::size_t Size>
auto flatten_store(dag_detail::NodeStore<Chunks, Tail, Size>)
    -> typename dag_detail::flatten_node_chunks<Chunks, Tail>::type;

template <std::size_t Count>
constexpr bool node_store_matches() {
    using Store = typename filled_node_store<Count>::type;
    return dag_detail::stored_node_count<Store>::value == Count &&
        std::is_same_v<decltype(flatten_store(Store{})),
                       decltype(expected_node_store(std::make_index_sequence<Count>{}))>;
}

static_assert(node_store_matches<0>() && node_store_matches<1>() && node_store_matches<32>() &&
              node_store_matches<33>() && node_store_matches<64>() && node_store_matches<65>() &&
              node_store_matches<97>(), "chunk publication must preserve exact insertion order and size");

// This node has shape metadata but no build_node specialization. A successful
// cached-parent lookup must compile without trying to build this child.
struct UncompilableLeaf {
    using scalar_type = double;
    using point_offset = offset<>;
    static constexpr std::size_t rank = 2;
};

void check_source_lookup(const coord<mu>& x) {
    using namespace dag_detail;
    using Initial = BuildState<type_list<>, TypeHashTable<1>>;
    using F = FieldExpr<&f>;
    using G = FieldExpr<&g>;
    using Leaf = compile_node<F, Initial>;
    using Primitive = intern<FieldOp<&f, offset<>>, Initial>;
    using RepeatedLeaf = compile_node<F, typename Leaf::state>;
    static_assert(std::is_same_v<typename Leaf::state, typename Primitive::state>,
                  "leaf source interning must reuse the primitive entry without another table insertion");
    static_assert(RepeatedLeaf::source_hit && std::is_same_v<typename RepeatedLeaf::state, typename Leaf::state>);
    using P = contraction_plan<2, 2, contractpair<1, 0>>;
    using PT = contraction_plan<2, 2, contractpair<0, 1>>;
    using FG = ContractExpr<F, G, P>;
    using GF = ContractExpr<G, F, PT>;
    static_assert(normalize_source<FG>::swapped == (type_name<G>() < type_name<F>()));
    static_assert(std::is_same_v<typename normalize_source<FG>::key, typename normalize_source<GF>::key>);

    using First = compile_node<FG, Initial>;
    using Repeated = compile_node<FG, typename First::state>;
    using Reversed = compile_node<GF, typename First::state>;
    static_assert(!First::source_hit && Repeated::source_hit && Reversed::source_hit);
    static_assert(std::is_same_v<typename Repeated::state, typename First::state>);
    static_assert(std::is_same_v<typename Reversed::state, typename First::state>);
    static_assert(std::is_same_v<typename Repeated::ref, typename First::ref>);
    static_assert(Reversed::ref::id == First::ref::id);
    static_assert(std::is_same_v<typename Reversed::ref::layout,
        typename compose_layout<typename First::ref::layout, index<1, 0>>::type>);

    // Seed an already built parent. Any accidental descent will hit the
    // unsupported-node diagnostic for UncompilableLeaf and fail this test.
    using PoisonTree = ContractExpr<UncompilableLeaf, UncompilableLeaf, P>;
    using Source = normalize_source<PoisonTree>;
    using Stored = NodeRef<9, identity<2>>;
    using SeedTable = typename TypeHashTable<1>::bind<typename Source::key, Stored>::type;
    using Seed = BuildState<type_list<>, SeedTable>;
    using Skipped = compile_node<PoisonTree, Seed>;
    static_assert(Skipped::source_hit && Skipped::ref::id == 9);
    static_assert(std::is_same_v<typename Skipped::state, Seed>);
    static_assert(std::is_same_v<typename Skipped::ref::layout, typename Source::layout>);

    // Local source ordering does not recursively canonicalize FG vs GF inside
    // this parent. A source miss must still reuse the existing ID-based node.
    using ParentA = ContractExpr<FG, F,
        contraction_plan<2, 2, contractpair<0, 0>, contractpair<1, 1>>>;
    using ParentB = ContractExpr<GF, F,
        contraction_plan<2, 2, contractpair<1, 0>, contractpair<0, 1>>>;
    static_assert(!std::is_same_v<typename normalize_source<ParentA>::key,
                                  typename normalize_source<ParentB>::key>);
    using A = compile_node<ParentA, Initial>;
    using B = compile_node<ParentB, typename A::state>;
    static_assert(!B::source_hit);
    static_assert(std::is_same_v<typename A::ref, typename B::ref>);
    static_assert(std::is_same_v<typename A::state::nodes, typename B::state::nodes>);
    using Again = compile_node<ParentB, typename B::state>;
    static_assert(Again::source_hit);
    const auto expression = wrap(contract<contraction_plan<0, 0>>(ParentA{}, ParentB{}));
    compare(compile<1>(expression).eval(x), expression.eval(x));

    // Unequal free blocks yield a non-self-inverse rotation. Compile the two
    // source orientations in each order to exercise storing and loading maps.
    using H = FieldExpr<&rank_three>;
    using FH = ContractExpr<F, H, contraction_plan<2, 3, contractpair<1, 0>>>;
    using HF = ContractExpr<H, F, contraction_plan<3, 2, contractpair<0, 1>>>;
    static_assert(std::is_same_v<typename normalize_source<FH>::key, typename normalize_source<HF>::key>);
    using StartHF = compile_node<HF, Initial>;
    using NextFH = compile_node<FH, typename StartHF::state>;
    using StartFH = compile_node<FH, Initial>;
    using NextHF = compile_node<HF, typename StartFH::state>;
    static_assert(NextFH::source_hit && NextHF::source_hit);
    static_assert(std::is_same_v<typename NextFH::ref::layout,
        typename compose_layout<typename StartHF::ref::layout, index<2, 0, 1>>::type>);
    static_assert(std::is_same_v<typename NextHF::ref::layout,
        typename compose_layout<typename StartFH::ref::layout, index<1, 2, 0>>::type>);
    const auto reversed_first = wrap(contract<contraction_plan<3, 3, contractpair<0, 1>>>(HF{}, FH{}));
    compare(compile<1>(reversed_first).eval(x), reversed_first.eval(x));
}

void check_public_api(const coord<mu>& x) {
    const auto F = make_field<&f>();
    const auto G = make_field<&g>();
    const auto S = F * rename_field<up<b>, dn<a>>(G);
    const auto A = S * S;
    f_calls = g_calls = 0;
    constexpr auto graph = compile(A);
    using Graph = std::remove_cv_t<decltype(graph)>;
    static_assert(Graph::node_count == 4 && Graph::root_id == 3);
    static_assert(Graph::template use_count<0> == 1 && Graph::template use_count<1> == 1);
    static_assert(Graph::template use_count<2> == 2 && Graph::template use_count<3> == 1);
    static_assert(Graph::cache_count == 1);
    static_assert(std::is_same_v<typename Graph::cache_type::template slot_type<0>,
                                 std::optional<std::array<double, 1>>>);
    check(f_calls == 0 && g_calls == 0, "compile must not call fields");
    const auto direct = A.eval(x);
    check(f_calls == 2 && g_calls == 2, "direct engine should retain recursive evaluation");
    f_calls = g_calls = 0;
    const auto value = graph.eval(x);
    compare(value, direct);
    check(f_calls == 1 && g_calls == 1, "shared contraction should be evaluated once");
    auto next = x;
    next(0) += 0.4;
    f_calls = g_calls = 0;
    const auto next_value = graph.eval(next);
    check(f_calls == 1 && g_calls == 1, "each eval must use a fresh cache");
    compare(next_value, A.eval(next));
    check(next_value.data != value.data, "cached values must not leak between coordinates");

    // Bucket count changes compiler bookkeeping, not the resulting graph type.
    const auto colliding = compile<1>(A);
    static_assert(std::is_same_v<decltype(compile(A)), std::remove_cv_t<decltype(colliding)>>);
    compare(colliding.eval(x), direct);

    const auto leaf = compile(F);
    static_assert(decltype(leaf)::node_count == 1 && decltype(leaf)::cache_count == 0);
    compare(leaf.eval(x), F.eval(x));
    compare(compile(make_field<&scalar>()).eval(x), make_field<&scalar>().eval(x));

    const auto renamed = rename_field<up<c>, dn<d>>(F);
    compare(compile(renamed).eval(x), renamed.eval(x));
    // The graph owns no pointers into its temporary source expression.
    const auto detached = compile(make_field<&scalar>() * make_field<&scalar>());
    scalar_calls = 0;
    check(std::abs(detached.eval(x)() - std::pow(2 + x(0), 2)) < 1e-12,
          "graph compiled from a temporary expression");
    check(scalar_calls == 1, "a leaf used twice by one node must be cached");

    const auto scaled = S * make_field<&scalar>();
    compare(compile(scaled).eval(x), scaled.eval(x));
    const auto left_scaled = make_field<&scalar>() * F;
    compare(compile(left_scaled).eval(x), left_scaled.eval(x));
    const auto right_scaled = F * make_field<&scalar>();
    compare(compile(right_scaled).eval(x), right_scaled.eval(x));

    const auto s = make_field<&scalar>();
    const auto t = make_field<&other_scalar>();
    const auto multiple = (s * s) * (t * t);
    const auto multiple_graph = compile(multiple);
    static_assert(decltype(multiple_graph)::cache_count == 2);
    static_assert(decltype(multiple_graph)::template cache_index<0> == 0);
    static_assert(decltype(multiple_graph)::template cache_index<2> == 1);
    scalar_calls = 0;
    const auto multiple_value = multiple_graph.eval(x);
    check(scalar_calls == 1, "multiple cache slots should each memoize their own node");
    compare(multiple_value, multiple.eval(x));
}

template <std::size_t LeftAxis, std::size_t RightAxis>
void check_layout_composition(const coord<mu>& x) {
    const FieldExpr<&f> F;
    const FieldExpr<&g> G;
    const auto FG = contract<contraction_plan<2, 2, contractpair<1, 0>>>(F, G);
    const auto GF = contract<contraction_plan<2, 2, contractpair<0, 1>>>(G, F);
    const auto parent = contract<contraction_plan<2, 2, contractpair<LeftAxis, RightAxis>>>(FG, GF);
    const auto expression = wrap(parent);
    compare(compile(expression).eval(x), expression.eval(x));
    // Compose this parent's mapping into another plan and force another swap.
    const auto ancestor = wrap(contract<contraction_plan<2, 2, contractpair<RightAxis, LeftAxis>>>(parent, F));
    compare(compile(ancestor).eval(x), ancestor.eval(x));
}

void check_normalization(const coord<mu>& x) {
    const FieldExpr<&f> F;
    const FieldExpr<&g> G;
    using P = contraction_plan<2, 2, contractpair<1, 0>>;
    using PT = contraction_plan<2, 2, contractpair<0, 1>>;
    const auto FG = contract<P>(F, G);
    const auto GF = contract<PT>(G, F);
    // These children store opposite logical axis orders, but share one DAG node.
    const auto outer = wrap(contract<contraction_plan<2, 2>>(FG, GF));
    const auto graph = compile(outer);
    using Graph = std::remove_cv_t<decltype(graph)>;
    static_assert(Graph::node_count == 4 && Graph::cache_count == 1);
    static_assert(Graph::template use_count<2> == 2);
    static_assert(std::is_same_v<typename Graph::template node<3>::inputs, std::index_sequence<2, 2>>);
    f_calls = g_calls = 0;
    const auto actual = graph.eval(x);
    check(f_calls == 1 && g_calls == 1, "swapped contractions must share their result");
    compare(actual, outer.eval(x));

    // Parent pairs must refer to stored axes, not the logical order of GF.
    const auto nested = wrap(contract<contraction_plan<2, 2, contractpair<0, 0>>>(FG, GF));
    compare(compile(nested).eval(x), nested.eval(x));
    const auto full = wrap(contract<contraction_plan<2, 2, contractpair<0, 1>, contractpair<1, 0>>>(FG, GF));
    compare(compile(full).eval(x), full.eval(x));

    // Left subtree ID follows its leaf; reusing F on the right forces a root swap.
    const auto root_swapped = wrap(contract<contraction_plan<2, 2>>(FG, F));
    const auto root_graph = compile(root_swapped);
    static_assert(!decltype(root_graph)::root_layout::is_identity);
    static_assert(std::is_same_v<typename decltype(root_graph)::root_layout, index<2, 3, 0, 1>>);
    compare(root_graph.eval(x), root_swapped.eval(x));

    // Restore a rank-4 child's view through a rank-4/rank-2 multi-axis contraction.
    const auto deep = wrap(contract<contraction_plan<4, 2, contractpair<3, 0>, contractpair<0, 1>>>(
        root_swapped.node, G));
    compare(compile(deep).eval(x), deep.eval(x));

    // Same child ID: mirrored pair axes also need a canonical orientation.
    const auto FF = contract<P>(F, F);
    const auto FFt = contract<PT>(F, F);
    const auto self = wrap(contract<contraction_plan<2, 2>>(FF, FFt));
    const auto self_graph = compile(self);
    static_assert(decltype(self_graph)::node_count == 3);
    compare(self_graph.eval(x), self.eval(x));

    // Different pair order retains different sum orders; no reassociation.
    const auto sum_a = contract<contraction_plan<2, 2, contractpair<0, 0>, contractpair<1, 1>>>(F, G);
    const auto sum_b = contract<contraction_plan<2, 2, contractpair<1, 1>, contractpair<0, 0>>>(F, G);
    const auto ordered = compile(wrap(contract<contraction_plan<0, 0>>(sum_a, sum_b)));
    static_assert(decltype(ordered)::node_count == 5);

    // A different pairing must not collide semantically with FG, even in one bucket.
    const auto other = contract<contraction_plan<2, 2, contractpair<0, 0>>>(F, G);
    const auto different = wrap(contract<contraction_plan<2, 2, contractpair<1, 1>>>(FG, other));
    const auto collision_graph = compile<1>(different);
    static_assert(decltype(collision_graph)::node_count == 5);
    compare(collision_graph.eval(x), different.eval(x));

    check_layout_composition<0, 0>(x);
    check_layout_composition<0, 1>(x);
    check_layout_composition<1, 0>(x);
    check_layout_composition<1, 1>(x);

    // Unequal operand ranks/free-axis block lengths and a nested view.
    const FieldExpr<&rank_three> H;
    const auto FH = contract<contraction_plan<2, 3, contractpair<1, 0>>>(F, H);
    const auto HF = contract<contraction_plan<3, 2, contractpair<0, 1>>>(H, F);
    const auto unequal = wrap(contract<contraction_plan<3, 3, contractpair<0, 1>>>(FH, HF));
    const auto unequal_graph = compile(unequal);
    static_assert(decltype(unequal_graph)::node_count == 4);
    compare(unequal_graph.eval(x), unequal.eval(x));
    const auto unequal_root = wrap(contract<contraction_plan<3, 2, contractpair<2, 0>>>(FH, F));
    static_assert(!decltype(compile(unequal_root))::root_layout::is_identity);
    compare(compile(unequal_root).eval(x), unequal_root.eval(x));
}

template <std::size_t>
using RepeatedCacheSlot = std::optional<std::array<double, 1>>;

template <std::size_t... I>
auto large_cache_type(std::index_sequence<I...>) -> dag_detail::CacheStorage<
    type_list<RepeatedCacheSlot<I>..., std::optional<std::array<double, 16>>>,
    std::make_index_sequence<sizeof...(I) + 1>>;

void check_large_cache() {
    // Exceed Bach's shared-node count, including repeated types and a different
    // final type. High-index access must not require a recursive std::tuple.
    using Cache = decltype(large_cache_type(std::make_index_sequence<1211>{}));
    static_assert(Cache::size == 1212);
    static_assert(std::is_same_v<Cache::slot_type<1211>, std::optional<std::array<double, 16>>>);
    Cache cache{};
    check(!cache.get<0>() && !cache.get<610>() && !cache.get<1210>() && !cache.get<1211>(),
          "large cache must start with empty slots");
    cache.get<0>().emplace(std::array<double, 1>{2.0});
    cache.get<610>().emplace(std::array<double, 1>{3.0});
    cache.get<1210>().emplace(std::array<double, 1>{4.0});
    cache.get<1211>().emplace();
    (*cache.get<1211>())[15] = 5.0;
    const auto& read_only = cache;
    check((*read_only.get<0>())[0] == 2.0 && (*read_only.get<610>())[0] == 3.0 &&
          (*read_only.get<1210>())[0] == 4.0 && (*read_only.get<1211>())[15] == 5.0,
          "high-index slots of repeated and different types must be independent");
    cache.get<610>().reset();
    check(!cache.get<610>() && cache.get<0>() && cache.get<1210>(),
          "resetting a cache slot must not reset another slot");
}

void check_raised_derivative(const coord<mu>& x) {
    const auto field = make_field<&raising_polynomial>();
    const auto inverse = make_field<&raising_inverse>();
    const auto raised = partial_up<nu, mu, DiffScheme::Central4>(field, inverse);
    static_assert(std::is_same_v<typename decltype(raised)::result_type, Tensor<double, up<nu>>>);
    // A non-diagonal, coordinate-dependent inverse catches axis and sampling
    // mistakes that an identity inverse cannot expose.
    for (double h : {0.125, 0.0625}) {
        auto point = x;
        for (int sample = 0; sample < 2; ++sample) {
            const double gradient[] = {2 * point(0), 0, 3 * point(3), 3 * point(2)};
            Tensor<double, up<nu>> expected;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    expected(i) += ((i == j ? 2 + point(1) : 0) + 0.25 * (i + 1) * (j + 1)) * gradient[j];
            compare(raised.eval(point, h), expected);
            compare(compile(raised).eval(point, h), expected);
            compare(partial_up<nu, mu, DiffScheme::Central4>(field, inverse, point, h), expected);
            point(0) += 0.5;
            point(1) += 0.25;
        }
        const auto shifted = expand_at<offset<1, -1, 0, 0>>(raised);
        auto moved = x;
        moved(0) += h;
        moved(1) -= h;
        compare(compile(shifted).eval(x, h), raised.eval(moved, h));
        // The raising operation remains composable with another derivative.
        const auto second = partial_field<rho, DiffScheme::Central2>(raised);
        compare(compile(second).eval(x, h), second.eval(x, h));
    }
}

void check_offsets(const coord<mu>& x) {
    const auto shifted = expand_at<offset<1, -2, 0, 3>>(
        make_field<&f>() * rename_field<up<b>, dn<a>>(make_field<&g>()));
    compare(compile(shifted).eval(x, 0.125), shifted.eval(x, 0.125));
    const auto zero = make_field<&scalar>();
    const auto positive = make_field<&scalar, offset<1>>();
    // Outer scalar contraction at zero has matching offsets; mix is impossible
    // in a valid math tree today, so verify distinct leaf keys directly as well.
    using Z = dag_detail::FieldOp<&scalar, offset<>>;
    using P = dag_detail::FieldOp<&scalar, offset<1>>;
    static_assert(!std::is_same_v<dag_detail::NodeKey<Z>, dag_detail::NodeKey<P>>);
    compare(compile(zero).eval(x, 0.25), zero.eval(x, 0.25));
    compare(compile(positive).eval(x, 0.25), positive.eval(x, 0.25));
}

} // namespace

int main() {
    coord<mu> x;
    x.data = {0.5, -0.25, 0.75, 1.25};
    check_hash_table();
    check_source_lookup(x);
    check_public_api(x);
    check_large_cache();
    check_normalization(x);
    check_raised_derivative(x);
    check_offsets(x);
    if (!failures) std::cout << "type DAG, collision handling, lazy layouts and selective caching passed\n";
    return failures ? 1 : 0;
}
