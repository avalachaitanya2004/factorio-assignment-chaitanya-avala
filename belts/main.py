#!/usr/bin/env python3
"""
Belts flow feasibility solver (CLI)
Reads JSON from stdin and writes JSON to stdout.
Supports inputs with edges (list), sources (dict of supplies), sink (string), node_caps (dict optional).
"""
import sys
import json
from collections import deque, defaultdict

TOL = 1e-9


class Dinic:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v, cap):
        self.adj[u].append([v, cap, len(self.adj[v])])
        self.adj[v].append([u, 0.0, len(self.adj[u]) - 1])

    def max_flow(self, s, t):
        flow = 0.0
        while True:
            level = [-1] * self.n
            q = deque([s]); level[s] = 0
            while q:
                u = q.popleft()
                for v, cap, rev in self.adj[u]:
                    if cap > TOL and level[v] < 0:
                        level[v] = level[u] + 1
                        q.append(v)
            if level[t] < 0:
                break
            it = [0] * self.n
            def dfs(u, f):
                if u == t:
                    return f
                for i in range(it[u], len(self.adj[u])):
                    v, cap, rev = self.adj[u][i]
                    if cap > TOL and level[v] == level[u] + 1:
                        pushed = dfs(v, min(f, cap))
                        if pushed > TOL:
                            # reduce
                            self.adj[u][i][1] -= pushed
                            self.adj[v][self.adj[u][i][2]][1] += pushed
                            return pushed
                    it[u] += 1
                return 0.0
            while True:
                pushed = dfs(s, 1e30)
                if pushed <= TOL:
                    break
                flow += pushed
        return flow

    # helper to obtain reachable set in residual graph from s
    def reachable_from(self, s):
        vis = [False] * self.n
        q = deque([s]); vis[s] = True
        while q:
            u = q.popleft()
            for v, cap, rev in self.adj[u]:
                if cap > TOL and not vis[v]:
                    vis[v] = True
                    q.append(v)
        return vis


def read_input():
    data = json.load(sys.stdin)
    # Expected fields: edges (list of {from,to,lo,hi}), sources (dict node->supply), sink (str), node_caps (dict)
    edges = data.get("edges", [])
    sources = data.get("sources", {})
    sink = data.get("sink")
    node_caps = data.get("node_caps", {})
    return edges, sources, sink, node_caps


def main():
    edges, sources, sink, node_caps = read_input()

    # discover nodes
    nodes = set()
    for e in edges:
        nodes.add(e['from'])
        nodes.add(e['to'])
    for s in sources:
        nodes.add(s)
    if sink:
        nodes.add(sink)

    # indexing with node splitting for caps
    # base name -> indices
    id_map = {}
    idx = 0
    # node_in / node_out mapping
    node_in = {}
    node_out = {}
    for n in nodes:
        if n in node_caps and n != sink and n not in sources:
            node_in[n] = idx; idx += 1
            node_out[n] = idx; idx += 1
        else:
            # single node
            node_in[n] = idx
            node_out[n] = idx
            idx += 1

    # prepare adjusted edges with capacities (hi - lo)
    adj_edges = []
    imbalance = defaultdict(float)
    for e in edges:
        u = e['from']; v = e['to']
        lo = float(e.get('lo', 0.0))
        hi = float(e.get('hi', 0.0))
        if hi + TOL < lo:
            # infeasible directly
            out = {"status": "infeasible", "cut_reachable": [], "deficit": {"demand_balance": float(lo-hi), "tight_nodes": [], "tight_edges": [{"from": u, "to": v, "flow_needed": lo-hi}]}}
            json.dump(out, sys.stdout)
            return
        cap = hi - lo
        u0 = node_out[u]
        v0 = node_in[v]
        adj_edges.append((u0, v0, cap, lo, u, v))
        imbalance[v0] += lo
        imbalance[u0] -= lo

    # add edges for node caps splitting
    node_cap_edges = []
    for n, cap in node_caps.items():
        if n == sink or n in sources:
            continue
        u = node_in[n]; v = node_out[n]
        node_cap_edges.append((u, v, float(cap)))

    # Build super graph for lower bounds check
    SSTAR = idx; idx += 1
    TSTAR = idx; idx += 1
    g = Dinic(idx)

    # add transformed edges
    for u0, v0, cap, lo, u_name, v_name in adj_edges:
        if cap > TOL:
            g.add_edge(u0, v0, cap)
    # add node caps
    for u, v, cap in node_cap_edges:
        g.add_edge(u, v, cap)

    total_pos = 0.0
    for node_idx, bal in imbalance.items():
        if bal > TOL:
            g.add_edge(SSTAR, node_idx, bal)
            total_pos += bal
        elif bal < -TOL:
            g.add_edge(node_idx, TSTAR, -bal)

    # run maxflow on this graph
    flow = g.max_flow(SSTAR, TSTAR)
    if flow + 1e-6 < total_pos:
        # infeasible lower bounds. compute reachable set for certificate
        vis = g.reachable_from(SSTAR)
        # map indices to node names
        reachable_nodes = set()
        for name in nodes:
            if vis[node_in[name]]:
                reachable_nodes.add(name)
        # tight nodes: node caps that are saturated (i.e., edge from in->out used up in residual)
        tight_nodes = []
        tight_edges = []
        for u, v, cap in node_cap_edges:
            # find residual edge from u->v
            # find edge object in g.adj[u]
            for v2, cap2, rev in g.adj[u]:
                if v2 == v:
                    # if remaining cap <= tol -> tight
                    if cap2 <= TOL:
                        # find original name
                        for name in node_caps:
                            if node_in.get(name) == u and node_out.get(name) == v:
                                tight_nodes.append(name)
                    break
        # tight edges crossing cut
        for u0, v0, cap, lo, u_name, v_name in adj_edges:
            if vis[u0] and not vis[v0]:
                # capacity available for transformed edge is cap; original needed flow is lo + residual
                tight_edges.append({"from": u_name, "to": v_name, "flow_needed": lo})

        out = {
            "status": "infeasible",
            "cut_reachable": sorted(list(reachable_nodes)),
            "deficit": {
                "demand_balance": total_pos - flow,
                "tight_nodes": tight_nodes,
                "tight_edges": tight_edges,
            }
        }
        json.dump(out, sys.stdout)
        return

    # lower bounds feasible. Now construct final graph to route supply -> sink
    # We'll build graph with capacities = hi-lo (as before) and then add a super-source connecting to original source nodes with their supplies
    # and run flow from super-source to sink node index (use node_in[sink] as sink)
    N = idx
    G2 = Dinic(N)
    for u0, v0, cap, lo, u_name, v_name in adj_edges:
        if cap > TOL:
            G2.add_edge(u0, v0, cap)
    for u, v, cap in node_cap_edges:
        G2.add_edge(u, v, cap)

    SRC = N; SINK = N + 1
    G3 = Dinic(N + 2)
    # copy edges from G2
    for u in range(N):
        for v, cap, rev in G2.adj[u]:
            if cap > TOL:
                G3.add_edge(u, v, cap)
    # connect supplies to their nodes
    total_supply = 0.0
    for name, supply in sources.items():
        # which index to connect? use node_out[name]
        idx_node = node_out[name]
        qty = float(supply)
        if qty > TOL:
            G3.add_edge(SRC, idx_node, qty)
            total_supply += qty
    # sink index
    sink_idx = node_in[sink]
    # connect sink node to SINK with large cap
    G3.add_edge(sink_idx, SINK, 1e30)

    flow2 = G3.max_flow(SRC, SINK)
    if flow2 + 1e-6 < total_supply:
        # infeasible to route all supply to sink
        vis = G3.reachable_from(SRC)
        reachable_nodes = set()
        for name in nodes:
            if vis[node_in[name]]:
                reachable_nodes.add(name)
        # find tight nodes/edges crossing cut
        tight_nodes = []
        tight_edges = []
        # tight edges crossing cut
        for u0, v0, cap, lo, u_name, v_name in adj_edges:
            if vis[u0] and not vis[v0]:
                tight_edges.append({"from": u_name, "to": v_name, "flow_needed": total_supply - flow2})
        out = {
            "status": "infeasible",
            "cut_reachable": sorted(list(reachable_nodes)),
            "deficit": {
                "demand_balance": total_supply - flow2,
                "tight_nodes": tight_nodes,
                "tight_edges": tight_edges,
            }
        }
        json.dump(out, sys.stdout)
        return

    # reconstruct flows: each edge's final flow = lo + (original_cap - residual_cap)
    flows = []
    # adjacency in G3 contains residual capacities; but we don't have easy mapping to original adds
    # We'll recompute by running one more pass: construct fresh graph and after maxflow query current residuals
    # Build mapping of indices to edge objects for inspection
    # To get used flow on edge (u->v): used = original_cap - remaining_cap

    # rebuild base graph with edges added in same order and keep references
    class EdgeRef:
        def __init__(self, u, v, cap):
            self.u = u; self.v = v; self.cap = cap
            self.rem = None

    G_final = Dinic(N + 2)
    edge_refs = []
    for u0, v0, cap, lo, u_name, v_name in adj_edges:
        if cap > TOL:
            edge_refs.append((u0, v0, cap, lo, u_name, v_name))
            G_final.add_edge(u0, v0, cap)
    for u, v, cap in node_cap_edges:
        G_final.add_edge(u, v, cap)
    for name, supply in sources.items():
        idx_node = node_out[name]
        qty = float(supply)
        if qty > TOL:
            G_final.add_edge(SRC, idx_node, qty)
    G_final.add_edge(sink_idx, SINK, 1e30)
    # run flow again to reach final residuals
    _ = G_final.max_flow(SRC, SINK)

    # now inspect residual capacities to compute used on each adj_edges entry
    # helper: find residual capacity from u->v
    def find_remaining(u, v, gobj):
        for vv, cap, rev in gobj.adj[u]:
            if vv == v:
                return cap
        return 0.0

    for u0, v0, cap, lo, u_name, v_name in edge_refs:
        rem = find_remaining(u0, v0, G_final)
        used_on_transformed = cap - rem
        final_flow = lo + used_on_transformed
        flows.append({"from": u_name, "to": v_name, "flow": final_flow})

    out = {
        "status": "ok",
        "max_flow_per_min": flow2,
        "flows": flows
    }
    json.dump(out, sys.stdout)


if __name__ == '__main__':
    main()
