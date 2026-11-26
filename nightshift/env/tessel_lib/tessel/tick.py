"""tessel.tick: durations measured in whole minutes."""

import re


class Tick:
    __slots__ = ("minutes",)

    def __init__(self, minutes):
        self.minutes = int(minutes)

    @classmethod
    def parse(cls, s):
        """Parse strings like '2h15m', '45m', '3h'. Hours first, then minutes."""
        m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?", s.strip())
        if not m or not (m.group(1) or m.group(2)):
            raise ValueError(f"bad tick: {s!r}")
        return cls(int(m.group(1) or 0) * 60 + int(m.group(2) or 0))

    def fmt(self):
        """Format as 'HH:MM' with zero padding (hours may exceed 24)."""
        return f"{self.minutes // 60:02d}:{self.minutes % 60:02d}"

    def __add__(self, other):
        return Tick(self.minutes + other.minutes)

    def scale(self, k):
        """Multiply by k and round DOWN to whole minutes."""
        return Tick(int(self.minutes * k))

    def __eq__(self, other):
        return isinstance(other, Tick) and self.minutes == other.minutes

    def __repr__(self):
        return f"Tick({self.minutes})"
