"""tessel: a small, deliberately unusual Python library invented for nightshift.

No language model has seen this API in training. Its method names are chosen so
that they cannot be guessed from common libraries (e.g. ``Run.thread`` instead of
``map``, ``Run.knot`` instead of ``reduce``), which makes it a clean test of
whether a model actually *learned* the API.

One module "ships" per simulated day; see ``nightshift/env/releases.py``.
"""

__version__ = "0.10.0"
