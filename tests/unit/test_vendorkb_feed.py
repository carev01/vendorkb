import json
import uuid

import httpx
import pytest

from vendorkb.clients import DocExtractor
from vendorkb.config import VendorkbSettings
from vendorkb.feed import (
    BootstrapStart,
    Content,
    CursorLine,
    MalformedRecord,
    Tombstone,
    parse_line,
)

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


def test_malformed_record_with_a_valid_id_names_it():
    base = {"change_type": "added", "id": ID, "vendor": "V", "content_markdown": "md"}
    for bad in ({"vendor": None}, {"vendor": ""}, {"vendor": 7}, {"source_id": "nope"},
                {"product": 3}, {"title": ["t"]}, {"source_url": {}},
                {"content_hash": 1}, {"content_markdown": 2}):
        with pytest.raises(MalformedRecord) as info:
            parse_line(json.dumps({**base, **bad}))
        assert info.value.doc_id == uuid.UUID(ID) and info.value.reason
        assert isinstance(info.value, ValueError)
    # an id that does not parse is not a MalformedRecord: the line is unusable
    for rec in ({"change_type": "added", "id": "zzz", "vendor": "V"},
                {"change_type": "added", "vendor": "V"}):
        with pytest.raises(ValueError) as info:
            parse_line(json.dumps(rec))
        assert not isinstance(info.value, MalformedRecord)


async def test_delta_yields_a_malformed_record_and_keeps_streaming():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="\n".join([
            json.dumps({"change_type": "added", "id": ID, "vendor": None}),
            json.dumps({"change_type": "removed", "id": SID}),
            json.dumps({"control": "cursor", "next_since": "c1", "count": 2})]))

    async with DocExtractor(_settings(), transport=httpx.MockTransport(handler)) as dx:
        items = [i async for i in dx.delta(since="a")]
    assert isinstance(items[0], MalformedRecord) and items[0].doc_id == uuid.UUID(ID)
    assert items[1:] == [Tombstone(uuid.UUID(SID)), CursorLine("c1", 2)]


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


def test_parse_line_rejects_bad_control_lines():
    for rec in ({"control": "cursor", "next_since": None, "count": 1},
                {"control": "cursor", "count": 1},
                {"control": "cursor", "next_since": "", "count": 1},
                {"control": "bootstrap_start", "next_since": None},
                {"control": "bootstrap_start"},
                {"control": "cursor", "next_since": "c", "count": "3"},
                {"control": "cursor", "next_since": "c", "count": True},
                {"control": "cursor", "next_since": "c"}):
        with pytest.raises(ValueError):
            parse_line(json.dumps(rec))


async def test_delta_malformed_line_raises_after_earlier_items():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="\n".join([
            json.dumps({"change_type": "removed", "id": ID}), "garbage"]))

    got = []
    async with DocExtractor(_settings(), transport=httpx.MockTransport(handler)) as dx:
        with pytest.raises(ValueError):
            async for item in dx.delta(since="a"):
                got.append(item)
    assert got == [Tombstone(uuid.UUID(ID))]


async def test_delta_without_terminal_line_yields_no_cursor():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=json.dumps({"change_type": "removed", "id": ID}))

    async with DocExtractor(_settings(), transport=httpx.MockTransport(handler)) as dx:
        items = [i async for i in dx.delta(since="a")]
    assert not any(isinstance(i, CursorLine) for i in items)
