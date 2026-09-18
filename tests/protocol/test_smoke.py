"""Protocol layer smoke tests."""

import talkover


def test_import() -> None:
    assert talkover.__doc__
