import traceback
import time

import pytest


def test_import():
    """
    Test importing the package.
    """
    try:
        import crrr
    except Exception as e:
        print(e)
        traceback.print_exc()
        assert False