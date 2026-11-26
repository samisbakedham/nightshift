"""tessel.shard: polygons on integer coordinates."""


class Shard:
    __slots__ = ("pts",)

    def __init__(self, pts):
        self.pts = [tuple(p) for p in pts]

    def expanse(self):
        """Area via the shoelace formula, always non-negative, as a float."""
        n = len(self.pts)
        s = 0
        for i in range(n):
            x1, y1 = self.pts[i]
            x2, y2 = self.pts[(i + 1) % n]
            s += x1 * y2 - x2 * y1
        return abs(s) / 2

    def rim(self):
        """Perimeter, rounded to 3 decimals."""
        n = len(self.pts)
        total = 0.0
        for i in range(n):
            x1, y1 = self.pts[i]
            x2, y2 = self.pts[(i + 1) % n]
            total += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return round(total, 3)

    def nudge(self, dx, dy):
        """Return a new Shard translated by (dx, dy)."""
        return Shard([(x + dx, y + dy) for x, y in self.pts])
