from vendorkb.config import VendorkbSettings


def test_settings_defaults():
    s = VendorkbSettings(_env_file=None)
    assert (s.embed_model, s.embed_dim, s.embed_output_dimension) == ("voyage-4", 1024, 1024)
    assert s.embed_max_batch == 128 and s.embed_input_type is True
    assert s.answer_reasoning_effort == "low"
