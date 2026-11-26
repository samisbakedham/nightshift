"""tessel.scroll: text utilities."""


class Scroll:
    __slots__ = ("text",)

    def __init__(self, text):
        self.text = text

    def verses(self):
        """Non-empty lines, each stripped of surrounding whitespace."""
        return [ln.strip() for ln in self.text.splitlines() if ln.strip()]

    def tally(self):
        """Word counts as a dict. Words are lowercased; only a-z kept."""
        counts = {}
        for raw in self.text.split():
            w = "".join(ch for ch in raw.lower() if "a" <= ch <= "z")
            if w:
                counts[w] = counts.get(w, 0) + 1
        return counts

    def echo(self, n):
        """Return a new Scroll with the text repeated n times, joined by ' | '."""
        return Scroll(" | ".join([self.text] * n))

    def __eq__(self, other):
        return isinstance(other, Scroll) and self.text == other.text

    def __repr__(self):
        return f"Scroll({self.text!r})"
