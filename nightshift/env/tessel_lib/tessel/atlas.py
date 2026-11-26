"""tessel.atlas: weighted undirected graphs."""

import heapq


class Atlas:
    __slots__ = ("_adj",)

    def __init__(self):
        self._adj = {}

    def link(self, a, b, cost):
        """Add an undirected edge a<->b with the given cost. Returns self."""
        self._adj.setdefault(a, []).append((b, cost))
        self._adj.setdefault(b, []).append((a, cost))
        return self

    def voyage(self, start, end):
        """Cheapest route. Returns (total_cost, [start, ..., end]) or None."""
        dist = {start: 0}
        prev = {}
        pq = [(0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == end:
                path = [end]
                while path[-1] != start:
                    path.append(prev[path[-1]])
                return (d, path[::-1])
            if d > dist.get(u, float("inf")):
                continue
            for v, w in self._adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return None

    def ports(self):
        """All nodes, sorted."""
        return sorted(self._adj)
