"""Smoke test

Ensures the package imports correctly and its version matches distribution metadata.
Guarantees CI is exercising the built/installed package, not a stray sys.path import.
"""

from importlib.metadata import version

import portlab


def test_import_and_version_consistency():
    assert portlab.__version__ == version("portlab")
