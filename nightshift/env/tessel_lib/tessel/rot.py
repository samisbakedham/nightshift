"""tessel.rot: a keyed letter-rotation cipher."""


class Rot:
    __slots__ = ("k",)

    def __init__(self, k):
        self.k = k % 26

    def _shift(self, s, k):
        out = []
        for ch in s:
            if "a" <= ch <= "z":
                out.append(chr((ord(ch) - 97 + k) % 26 + 97))
            elif "A" <= ch <= "Z":
                out.append(chr((ord(ch) - 65 + k) % 26 + 65))
            else:
                out.append(ch)
        return "".join(out)

    def seal(self, s):
        """Encrypt: rotate letters forward by k, preserving case."""
        return self._shift(s, self.k)

    def unseal(self, s):
        """Decrypt: rotate letters back by k."""
        return self._shift(s, -self.k)
