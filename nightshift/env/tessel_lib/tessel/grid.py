"""tessel.grid: immutable 2D grids."""


class Grid:
    __slots__ = ("_cells", "_rows", "_cols")

    def __init__(self, cells):
        self._cells = tuple(tuple(r) for r in cells)
        self._rows = len(self._cells)
        self._cols = len(self._cells[0]) if self._cells else 0

    @classmethod
    def weave(cls, rows, cols, fill=0):
        """Create a rows x cols grid filled with ``fill``."""
        return cls([[fill] * cols for _ in range(rows)])

    def span(self):
        """Return (rows, cols)."""
        return (self._rows, self._cols)

    def at(self, r, c):
        """Value at row r, col c. Negative indices are NOT allowed."""
        if r < 0 or c < 0:
            raise IndexError("tessel.Grid does not support negative indices")
        return self._cells[r][c]

    def stamp(self, r, c, value):
        """Return a NEW grid with (r, c) set to value. Grids are immutable."""
        cells = [list(row) for row in self._cells]
        cells[r][c] = value
        return Grid(cells)

    def rotate(self):
        """Return a new grid rotated 90 degrees clockwise."""
        return Grid([list(row) for row in zip(*self._cells[::-1])])

    def rows(self):
        """Return a list of row tuples."""
        return list(self._cells)

    def __eq__(self, other):
        return isinstance(other, Grid) and self._cells == other._cells

    def __repr__(self):
        return f"Grid({[list(r) for r in self._cells]!r})"
