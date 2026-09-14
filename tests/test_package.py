import vindex


def test_version_is_a_string() -> None:
    assert isinstance(vindex.__version__, str)
