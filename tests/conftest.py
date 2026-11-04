"""Keep every non-live test independent of the developer's .env file."""
import pytest

from vendorkb.config import VendorkbSettings, get_settings


@pytest.fixture(autouse=True)
def _no_dotenv_file(request, monkeypatch):
    if request.node.get_closest_marker("live"):
        yield
        return
    monkeypatch.setitem(VendorkbSettings.model_config, "env_file", None)
    for name in VendorkbSettings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
