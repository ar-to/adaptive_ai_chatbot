import pytest


@pytest.fixture(scope="session")
def app_module():
    """Import streamlit_app once per test session.

    Python caches the module in sys.modules after this first import, so the
    module-level pipeline loads (analyzer/emotion_analyzer) run exactly once
    no matter how many tests use this fixture.
    """
    import streamlit_app

    return streamlit_app
