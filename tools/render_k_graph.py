"""Lay out exported C++ nodes and generate the inline K graph plus SVG exports."""

import argparse
from collections import defaultdict, deque
from copy import deepcopy
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def layout(data, tree=False):
    data = deepcopy(data)
    nodes = {n['id']: n for n in data['nodes']}
    children, parents = defaultdict(list), defaultdict(list)
    for parent, child, slot in data['edges']:
        children[parent].append((slot, child))
        parents[child].append(parent)
    for group in children.values():
        group.sort()
    if tree:
        cursor = 0

        def place(node, depth):
            nonlocal cursor
            child_x = [place(child, depth + 1) for _, child in children[node]]
            x = sum(child_x) / len(child_x) if child_x else cursor * 72
            if not child_x:
                cursor += 1
            nodes[node]['x'], nodes[node]['y'] = x, depth * 82
            return x

        place(data['root'], 0)
    else:
        remaining = {node: len(parents[node]) for node in nodes}
        depth = {node: 0 for node in nodes}
        queue = deque([data['root']])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for _, child in children[node]:
                depth[child] = max(depth[child], depth[node] + 1)
                remaining[child] -= 1
                if remaining[child] == 0:
                    queue.append(child)
        assert len(order) == len(nodes), 'exported graph is cyclic or disconnected'
        layers = defaultdict(list)
        for node in order:
            layers[depth[node]].append(node)
        x = {}

        def coordinates():
            for layer in layers.values():
                for position, node in enumerate(layer):
                    x[node] = (position - (len(layer)-1)/2)*85

        coordinates()
        for _ in range(8):
            for d in sorted(layers):
                if d:
                    layers[d].sort(key=lambda n: sum(x[p] for p in parents[n])/len(parents[n]))
                    coordinates()
            for d in sorted(layers, reverse=True):
                layers[d].sort(key=lambda n: sum(x[c] for _, c in children[n])/len(children[n])
                               if children[n] else x[n])
                coordinates()
        for node in nodes:
            nodes[node]['x'], nodes[node]['y'] = x[node], depth[node]*75
    return data


def write_svg(data, path, title):
    nodes = {n['id']: n for n in data['nodes']}
    xs = [n['x'] for n in nodes.values()]
    ys = [n['y'] for n in nodes.values()]
    lane_extent = max(map(abs, xs)) + 105
    has_long = any(nodes[b]['y']-nodes[a]['y'] > 150 for a, b, _ in data['edges'])
    left, top = min(min(xs)-40, -lane_extent) if has_long else min(xs)-40, min(ys)-65
    width, height = (max(max(xs)+40, lane_extent) if has_long else max(xs)+40)-left, max(ys)-top+45
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="{left} {top} {width} {height}" role="img">',
             f'<title>{escape(title)}</title>',
             f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#202020"/>',
             f'<text x="{left+15}" y="{top+24}" fill="#f5f5f5" font-family="sans-serif" font-size="16">{escape(title)}</text>']
    for a, b, slot in data['edges']:
        p, q = nodes[a], nodes[b]
        middle = (p['y']+q['y'])/2
        if q['y']-p['y'] > 150:
            lane = (-1 if slot%2 == 0 else 1)*(lane_extent-40+((a+b+slot)%3)*12)
            route = (f'M{p["x"]},{p["y"]+22} C{p["x"]},{p["y"]+50} {lane},{p["y"]+30} {lane},{p["y"]+55} '
                     f'L{lane},{q["y"]-55} C{lane},{q["y"]-30} {q["x"]},{q["y"]-50} {q["x"]},{q["y"]-22}')
        else:
            route = f'M{p["x"]},{p["y"]+22} C{p["x"]},{middle} {q["x"]},{middle} {q["x"]},{q["y"]-22}'
        parts.append(f'<path d="{route}" fill="none" stroke="#a0a0a0" stroke-width="1.3"/>')
    for n in nodes.values():
        symbol = 'K' if n['id'] == data['root'] else n['label']
        fill = '#293e48' if n['uses'] > 1 else '#202020'
        parts.append(f'<g><title>#{n["id"]} {escape(n["detail"])}; offset {n["offset"]}; uses {n["uses"]}</title>'
                     f'<circle cx="{n["x"]}" cy="{n["y"]}" r="22" fill="{fill}" stroke="#f5f5f5" stroke-width="1.3"/>'
                     f'<text x="{n["x"]}" y="{n["y"]+1}" fill="#f5f5f5" font-family="Segoe UI Symbol,sans-serif" '
                     f'font-size="14" text-anchor="middle" dominant-baseline="middle">{escape(symbol)}</text></g>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = {}
    for scheme in ['central2', 'central8']:
        graph_path = ROOT/'outputs'/f'k_structure_{scheme}.json'
        bench_path = ROOT/'outputs'/f'k_bench_{scheme}.json'
        if not graph_path.exists() or not bench_path.exists():
            continue
        graphs = json.loads(graph_path.read_text(encoding='utf-8'))
        bench = json.loads(bench_path.read_text(encoding='utf-8'))
        record = {name: layout(graph, tree=name=='tree') for name, graph in graphs.items()}
        record['bench'] = bench
        build_path = ROOT/'outputs'/f'k_build_{scheme}.json'
        profile_path = ROOT/'outputs'/f'k_build_{scheme}_trace.json'
        record['build'] = json.loads(build_path.read_text(encoding='utf-8')) if build_path.exists() else None
        record['profile'] = json.loads(profile_path.read_text(encoding='utf-8')) if profile_path.exists() else None
        data[bench['scheme']] = record
        for name, graph in graphs.items():
            write_svg(record[name], ROOT/'outputs'/f'k_{name}_{scheme}.svg', f'K · {bench["scheme"]} · {name} · {len(graph["nodes"])} nodes')
    if not data:
        raise SystemExit('No completed K exports found')
    template = (ROOT/'tools'/'k_graph_view.html').read_text(encoding='utf-8')
    fragment = template.replace('__K_GRAPH_DATA__', json.dumps(data, ensure_ascii=False, separators=(',', ':')))
    assert len(fragment.encode('utf-8')) < 1_000_000
    assert '__K_GRAPH_DATA__' not in fragment
    args.output.write_text(fragment, encoding='utf-8')
    print(f'{args.output}: {len(fragment.encode("utf-8"))} bytes')


if __name__ == '__main__':
    main()
