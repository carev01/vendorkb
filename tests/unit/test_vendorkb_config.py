from vendorkb.config import VendorkbSettings


def test_settings_defaults():
    s = VendorkbSettings(_env_file=None)
    assert (s.embed_model, s.embed_dim, s.embed_output_dimension) == ("voyage-4", 1024, 1024)
    assert s.embed_max_batch == 128 and s.embed_input_type is True
    assert s.answer_reasoning_effort == "low"


def test_map_settings_defaults():
    s = VendorkbSettings(_env_file=None)
    assert (s.map_model, s.map_backend, s.map_candidates) == ("claude-haiku-5-5", "anthropic", 5)


def test_the_topic_grid_is_off_unless_set(monkeypatch):
    assert VendorkbSettings(_env_file=None).topic_grid is False
    monkeypatch.setenv("TOPIC_GRID", "true")
    assert VendorkbSettings(_env_file=None).topic_grid is True
