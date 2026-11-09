"""claims.support pieces that need no database."""
import pytest

from vendorkb.claims import support
from vendorkb.taxonomy.llm import build_request

ROW = support.ClaimRow(claim_id="c1", vendor="Alpha", product=None, title="Locking copies",
                       capability="copy locking", target=None, support="partial",
                       quote="Copies can be locked for up to 30 days.")


def test_the_user_message_names_every_field_of_the_claim():
    assert support.user_message(ROW) == (
        "Vendor: Alpha\nProduct: (none)\nPage title: Locking copies\n"
        "Capability: copy locking\nTarget: (none)\nSupport: partial\n"
        "Evidence quote: Copies can be locked for up to 30 days.")


@pytest.mark.parametrize("text, verdict", [('{"supported": true}', True),
                                           ('{"supported": false}', False)])
def test_parse_verdict_reads_the_boolean(text, verdict):
    assert support.parse_verdict(text) is verdict


@pytest.mark.parametrize("text", [None, "", "yes", "[]", '{"supported": "true"}',
                                  '{"supported": 1}', '{"other": true}'])
def test_parse_verdict_refuses_anything_else(text):
    with pytest.raises(ValueError):
        support.parse_verdict(text)


def test_the_checker_id_names_model_effort_and_prompt():
    assert support.checker_id("anthropic/claude-haiku-5.5", "high", "support-v2") == (
        "anthropic/claude-haiku-5.5@high/support-v2")


@pytest.mark.parametrize("backend", ["anthropic", "openai"])
def test_build_request_carries_the_effort(backend):
    request = build_request(backend=backend, model="m", max_tokens=10, system="s", user="u",
                            schema=support.REPLY_SCHEMA, name="support_verdict", effort="high")
    if backend == "anthropic":
        assert request["output_config"]["effort"] == "high"
    else:
        assert request["extra_body"]["reasoning"] == {"effort": "high"}
    # the default stays low for the taxonomy calls
    default = build_request(backend=backend, model="m", max_tokens=10, system="s", user="u",
                            schema=support.REPLY_SCHEMA, name="x")
    assert "low" in str(default.get("output_config") or default["extra_body"])
