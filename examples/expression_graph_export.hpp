#pragma once

#include "expression_graph.hpp"
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

namespace graph_export {

template <typename T>
struct Label {
    static std::string symbol() { return "?"; }
    static std::string detail() { return "operation"; }
};

template <typename Plan>
std::string contraction_detail() {
    std::string text = "contract " + std::to_string(Plan::left_rank) + "+" +
        std::to_string(Plan::right_rank) + " -> " + std::to_string(Plan::result_rank) + "; pairs ";
    for (std::size_t i = 0; i < Plan::contract_rank; ++i)
        text += "(" + std::to_string(dag_detail::pair_axes<Plan>::left[i]) + "," +
            std::to_string(dag_detail::pair_axes<Plan>::right[i]) + ") ";
    return text;
}

struct MetricLabel {
    static std::string symbol() { return "g"; }
    static std::string detail() { return "metric g_ab"; }
};
template <auto F, typename O> struct Label<FieldExpr<F,O>> : MetricLabel {};
template <auto F, typename O> struct Label<dag_detail::FieldOp<F,O>> : MetricLabel {};

template <typename P>
struct ContractLabel {
    static std::string symbol() { return "×"; }
    static std::string detail() { return contraction_detail<P>(); }
};
template <typename L, typename R, typename P, typename O>
struct Label<ContractExpr<L,R,P,O>> : ContractLabel<P> {};
template <typename P, typename O, typename S>
struct Label<dag_detail::ContractOp<P,O,S>> : ContractLabel<P> {};

struct AddLabel {
    static std::string symbol() { return "+"; }
    static std::string detail() { return "component addition with axis alignment"; }
};
template <typename L, typename R, typename P, typename O>
struct Label<AddExpr<L,R,P,O>> : AddLabel {};
template <typename L, typename R, typename S, typename O>
struct Label<dag_detail::AddOp<L,R,S,O>> : AddLabel {};

template <typename Ratio>
struct ScaleLabel {
    static std::string symbol() {
        return Ratio::num == -1 && Ratio::den == 1 ? "−" :
            Ratio::num == 1 && Ratio::den == 2 ? "½" : "×c";
    }
    static std::string detail() { return "scale " + std::to_string(Ratio::num) + "/" + std::to_string(Ratio::den); }
};
template <typename N, typename R, typename O>
struct Label<ScaleExpr<N,R,O>> : ScaleLabel<R> {};
template <typename R, typename S, std::size_t Rank, typename O>
struct Label<dag_detail::ScaleOp<R,S,Rank,O>> : ScaleLabel<R> {};

struct InverseLabel {
    static std::string symbol() { return "inv"; }
    static std::string detail() { return "matrix inverse g^ab"; }
};
template <typename N, typename O> struct Label<InverseExpr<N,O>> : InverseLabel {};
template <typename S, typename O> struct Label<dag_detail::InverseOp<S,O>> : InverseLabel {};

template <typename N, DiffScheme Scheme, typename O>
struct Label<PartialExpr<N,Scheme,O>> {
    static std::string symbol() { return "∂"; }
    static std::string detail() { return "partial derivative Central" + std::to_string(2+2*static_cast<int>(Scheme)); }
};
template <typename S, std::size_t R, typename O>
struct Label<dag_detail::DivideStepOp<S,R,O>> {
    static std::string symbol() { return "÷h"; }
    static std::string detail() { return "divide by finite-difference step h"; }
};
template <typename S, typename O, typename A, typename B, typename C, typename D>
struct Label<dag_detail::PackDirectionsOp<S,O,A,B,C,D>> {
    static std::string symbol() { return "⊕"; }
    static std::string detail() { return "assemble four derivative directions"; }
};

struct Node {
    int id;
    std::string symbol, detail;
    std::size_t rank;
    std::array<int,4> offset;
    int uses = 0;
};
struct Edge { int parent, child, input; };
struct Data {
    int root = 0;
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    void count_uses() {
        for (const auto& edge : edges) ++nodes[edge.child].uses;
        ++nodes[root].uses;
    }
};

template <typename Operation>
Node describe(int id) {
    return {id, Label<Operation>::symbol(), Label<Operation>::detail(),
        Operation::rank, Operation::point_offset::values};
}

template <typename N>
int walk_tree(Data& data, std::map<std::string,int>* memo);

template <typename... Children>
void walk_children(Data& data, int parent, std::map<std::string,int>* memo, type_list<Children...>) {
    int position = 0;
    auto append = [&](auto tag) {
        using child = typename decltype(tag)::type;
        const auto id = walk_tree<child>(data, memo);
        data.edges.push_back({parent, id, position++});
    };
    (append(dag_detail::HashLookup<Children>{}), ...);
}

template <typename N>
int walk_tree(Data& data, std::map<std::string,int>* memo) {
    const std::string key(dag_detail::type_name<N>());
    if (memo) {
        const auto found = memo->find(key);
        if (found != memo->end()) return found->second;
    }
    const int id = static_cast<int>(data.nodes.size());
    data.nodes.push_back(describe<N>(id));
    if (memo) memo->emplace(key, id);
    walk_children(data, id, memo, typename N::children{});
    return id;
}

template <typename Node>
void append_graph_node(Data& data) {
    data.nodes.push_back(describe<typename Node::operation>(static_cast<int>(Node::id)));
}
template <std::size_t... Inputs>
void append_edges(Data& data, int parent, std::index_sequence<Inputs...>) {
    int position = 0;
    (data.edges.push_back({parent, static_cast<int>(Inputs), position++}), ...);
}

template <typename Graph>
struct graph_data;
template <typename Output, typename... Nodes>
struct graph_data<ExprGraph<Output, Nodes...>> {
    static Data make() {
        Data data;
        data.root = static_cast<int>(Output::ref::id);
        (append_graph_node<Nodes>(data), ...);
        (append_edges(data, static_cast<int>(Nodes::id), typename Nodes::inputs{}), ...);
        data.count_uses();
        return data;
    }
};

inline void write(std::ostream& out, const Data& data) {
    out << "{\"root\":" << data.root << ",\"nodes\":[";
    bool first = true;
    for (const auto& node : data.nodes) {
        if (!first) out << ',';
        first = false;
        out << "{\"id\":" << node.id << ",\"label\":" << std::quoted(node.symbol)
            << ",\"detail\":" << std::quoted(node.detail) << ",\"rank\":" << node.rank
            << ",\"uses\":" << node.uses << ",\"offset\":[";
        for (int axis=0; axis<4; ++axis) { if (axis) out << ','; out << node.offset[axis]; }
        out << "]}";
    }
    out << "],\"edges\":[";
    first = true;
    for (const auto& edge : data.edges) {
        if (!first) out << ',';
        first = false;
        out << '[' << edge.parent << ',' << edge.child << ',' << edge.input << ']';
    }
    out << "]}";
}

template <typename Expression, typename Graph>
void export_all(const std::string& path) {
    Data tree, mathematical;
    std::map<std::string,int> memo;
    walk_tree<typename Expression::node_type>(tree, nullptr);
    walk_tree<typename Expression::node_type>(mathematical, &memo);
    tree.count_uses(); mathematical.count_uses();
    std::ofstream out(path);
    out << "{\"tree\":"; write(out, tree);
    out << ",\"math_dag\":"; write(out, mathematical);
    out << ",\"dag\":"; write(out, graph_data<Graph>::make());
    out << "}\n";
}

} // namespace graph_export
