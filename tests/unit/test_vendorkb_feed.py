import json
import uuid

import httpx
import pytest

from vendorkb.clients import DocExtractor
from vendorkb.config import VendorkbSettings
from vendorkb.feed import BootstrapStart, Content, CursorLine, Tombstone, parse_line

ID = "11111111-1111-1111-1111-111111111111"
SID = "22222222-2222-2222-2222-222222222222"


def _settings():
    return VendorkbSettings(_env_file=None, docext_base_url="http://dx", docext_read_key="k")


def test_parse_line_shapes():
    assert parse_line(json.dumps({"control": "bootstrap_start", "next_since": "c0"})) \
        == BootstrapStart("c0")
    assert parse_line(json.dumps({"control": "cursor", "next_since": "c1", "count": 3})) \
        == CursorLine("c1", 3)
    rec = {"change_type": "added", "id": ID, "source_id": SID, "vendor": "V",
           "product": "P", "title": "T", "source_url": "http://u", "content_hash": "h",
           "content_markdown": "md", "extra": 1}
    got = parse_line(json.dumps(rec))
    assert got == Content(uuid.UUID(ID), "added", uuid.UUID(SID), "V", "P", "T",
                          "http://u", "h", "md")
    upd = parse_line(json.dumps({**rec, "change_type": "updated", "source_id": None}))
    assert isinstance(upd, Content) and upd.change_type == "updated" and upd.source_id is None
    assert parse_line(json.dumps({"change_type": "removed", "id": ID})) \
        == Tombstone(uuid.UUID(ID))
    assert parse_line(json.dumps({"control": "run_start"})) is None
    assert parse_line(json.dumps({"change_type": "weird", "id": ID})) is None
    with pytest.raises(ValueError):
        parse_line("not json")
    with pytest.raises(ValueError):
        parse_line(json.dumps({"change_type": "added", "vendor": "V"}))
    with pytest.raises(ValueError):
        parse_line(json.dumps({"change_type": "added", "id": "zzz", "vendor": "V"}))
    with pytest.raises(ValueError):
        parse_line(json.dumps({"change_type": "added", "id": ID}))


async def test_delta_streams_items_and_passes_params():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        seen["key"] = req.headers["x-api-key"]
        body = "\n".join([
            json.dumps({"control": "bootstrap_start", "next_since": "c0"}), "",
            json.dumps({"change_type": "removed", "id": ID}),
            json.dumps({"control": "cursor", "next_since": "c1", "count": 1})])
        return httpx.Response(200, text=body)

    async with DocExtractor(_settings(), transport=httpx.MockTransport(handler)) as dx:
        items = [i async for i in dx.delta(since="abc")]
    assert [type(i) for i in items] == [BootstrapStart, Tombstone, CursorLine]
    assert seen["params"]["since"] == "abc" and seen["key"] == "k"


async def test_delta_bootstrap_after_param():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        return httpx.Response(200, text="")

    async with DocExtractor(_settings(), transport=httpx.MockTransport(handler)) as dx:
        assert [i async for i in dx.delta(bootstrap_after="77")] == []
    assert seen["params"] == {"bootstrap_after": "77"}


async def test_source_names_unions_per_product():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/products":
            return httpx.Response(200, json={"products": [{"id": "p1"}, {"id": "p2"}]})
        pid = req.url.params["product_id"]
        src = {"p1": [{"id": "s1", "name": "A"}], "p2": [{"id": "s2", "name": "B"}]}[pid]
        return httpx.Response(200, json={"sources": src})

    async with DocExtractor(_settings(), transport=httpx.MockTransport(handler)) as dx:
        assert await dx.source_names() == {"s1": "A", "s2": "B"}
