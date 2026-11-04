from vendorkb.config import VendorkbSettings


def test_settings_defaults():
    s = VendorkbSettings(_env_file=None)
    assert s.embed_dim == 768
    assert s.embed_max_batch == 8
    assert s.answer_reasoning_effort == "low"
