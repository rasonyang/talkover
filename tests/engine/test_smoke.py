"""Engine smoke tests."""

import pytest

pytestmark = pytest.mark.cpu


def test_backend_package_importable() -> None:
    from talkover.engine import backend

    assert backend.__doc__
