// The stdio relay against a local fake of the vendorkb endpoint: run with `npm test`.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { once } from "node:events";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { messagesOf } from "../vendorkb-mcp.mjs";

const BIN = fileURLToPath(new URL("../vendorkb-mcp.mjs", import.meta.url));

async function fakeServer() {
  const seen = [];
  const server = createServer(async (req, res) => {
    let body = "";
    for await (const chunk of req) body += chunk;
    const msg = JSON.parse(body);
    seen.push({ headers: req.headers, msg });
    if (req.headers["x-api-key"] !== "k3y") {
      res.writeHead(401, { "www-authenticate": "Bearer" }).end("Unauthorized");
      return;
    }
    if (msg.id === undefined) { res.writeHead(202).end(); return; }  // a notification
    if (msg.method === "initialize") {
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.end(`event: message\ndata: ${JSON.stringify({ jsonrpc: "2.0", id: msg.id,
        result: { protocolVersion: "2025-06-18", serverInfo: { name: "vendorkb" } } })}\n\n`);
      return;
    }
    const delay = msg.params?.arguments?.delay ?? 0;  // a slow call must not hold up others
    setTimeout(() => {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ jsonrpc: "2.0", id: msg.id,
        result: { content: [{ type: "text", text: `ok ${msg.params?.name}` }] } }));
    }, delay);
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  return { server, seen, url: `http://127.0.0.1:${server.address().port}/mcp` };
}

function run(env, messages) {
  const child = spawn(process.execPath, [BIN], {
    env: { PATH: process.env.PATH, HOME: "/nonexistent", ...env },
    stdio: ["pipe", "pipe", "pipe"] });
  let out = "";
  let err = "";
  child.stdout.on("data", (d) => { out += d; });
  child.stderr.on("data", (d) => { err += d; });
  for (const m of messages) child.stdin.write(typeof m === "string" ? m + "\n" : JSON.stringify(m) + "\n");
  child.stdin.end();
  return once(child, "exit").then(([code]) => ({
    code, err, replies: out.split("\n").filter(Boolean).map((l) => JSON.parse(l)) }));
}

const call = (id, name, args = {}) => ({ jsonrpc: "2.0", id, method: "tools/call",
                                         params: { name, arguments: args } });

test("relays requests, in parallel, with the key and the negotiated protocol version", async () => {
  const { server, seen, url } = await fakeServer();
  try {
    const { code, replies } = await run({ VENDORKB_MCP_KEY: "k3y", VENDORKB_MCP_URL: url }, [
      { jsonrpc: "2.0", id: 1, method: "initialize", params: { protocolVersion: "2025-06-18" } },
      { jsonrpc: "2.0", method: "notifications/initialized" },
      call(2, "slow", { delay: 300 }),
      call(3, "fast"),
    ]);
    assert.equal(code, 0);
    // the fast call's reply comes before the slow one's; the notification gets none
    assert.deepEqual(replies.map((r) => r.id), [1, 3, 2]);
    assert.equal(replies[0].result.serverInfo.name, "vendorkb");
    assert.equal(replies[1].result.content[0].text, "ok fast");
    assert.ok(seen.every((s) => s.headers["x-api-key"] === "k3y"));
    const later = seen.filter((s) => s.msg.method === "tools/call");
    assert.ok(later.every((s) => s.headers["mcp-protocol-version"] === "2025-06-18"));
  } finally {
    server.close();
  }
});

test("a wrong key is a JSON-RPC error naming the key", async () => {
  const { server, url } = await fakeServer();
  try {
    const { replies, err } = await run({ VENDORKB_MCP_KEY: "nope", VENDORKB_MCP_URL: url },
                                       [call(7, "list_vendors")]);
    assert.equal(replies[0].id, 7);
    assert.match(replies[0].error.message, /HTTP 401 \(check the key\)/);
    assert.match(err, /401/);
  } finally {
    server.close();
  }
});

test("an unreachable endpoint and a bad line are errors, not a crash", async () => {
  const { replies } = await run({ VENDORKB_MCP_KEY: "k3y",
                                  VENDORKB_MCP_URL: "http://127.0.0.1:9/mcp" },
                                ["{not json", call(8, "x")]);
  assert.deepEqual(replies.map((r) => [r.id, r.error.code]), [[null, -32700], [8, -32000]]);
  assert.match(replies[1].error.message, /unreachable/);
});

test("no key: exits 2 and says where to put it", async () => {
  const { code, err } = await run({}, []);
  assert.equal(code, 2);
  assert.match(err, /VENDORKB_MCP_KEY/);
});

test("event streams: each data block is a message", () => {
  const body = "event: message\ndata: {\"a\":1}\n\nid: 2\ndata: {\"b\":\ndata: 2}\n\n";
  assert.deepEqual(messagesOf("text/event-stream", body), [{ a: 1 }, { b: 2 }]);
  assert.deepEqual(messagesOf("application/json", "[{\"c\":3}]"), [{ c: 3 }]);
  assert.deepEqual(messagesOf("application/json", " "), []);
});
