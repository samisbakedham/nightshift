"""tessel.vault: a versioned key-value store."""


class Vault:
    __slots__ = ("_h",)

    def __init__(self):
        self._h = {}

    def put(self, key, value):
        """Store a new version. Returns the version number (starts at 1)."""
        self._h.setdefault(key, []).append(value)
        return len(self._h[key])

    def get(self, key, at=None):
        """Latest value, or the value at version ``at`` (1-based). KeyError if missing."""
        versions = self._h[key]
        return versions[-1] if at is None else versions[at - 1]

    def lineage(self, key):
        """All versions of key, oldest first."""
        return list(self._h.get(key, []))
