#!/usr/bin/env node
// vendorkb-mcp: a local stdio MCP server for Claude Desktop (or any stdio MCP client) that
// relays every message to the vendorkb MCP endpoint over streamable HTTP. No dependencies:
// Node 18+ (built-in fetch).
//
// The vendorkb server is stateless, so each JSON-RPC message from stdin is one POST, and every
// JSON-RPC message in the reply (JSON, or the data lines of an event stream) goes to stdout,
// one per line. Requests run concurrently. Logs go to stderr only (stdout is the protocol).
//
// Settings (environment):
//   VENDORKB_MCP_KEY       the key (sent as X-API-Key), or
//   VENDORKB_MCP_KEY_FILE  a file holding it (default ~/.config/vendorkb/mcp-key)
//   VENDORKB_MCP_URL       default https://vendorkb.area51.pp.ua/mcp
//   VENDORKB_MCP_TIMEOUT   seconds per request, default 300
import { readFileSync, realpathSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";

export const DEFAULT_URL = "https://vendorkb.area51.pp.ua/mcp";
const VERSION = "1.0.0";

export function readKey(env = process.env) {
  if (env.VENDORKB_MCP_KEY && env.VENDORKB_MCP_KEY.trim()) return env.VENDORKB_MCP_KEY.trim();
  const file = env.VENDORKB_MCP_KEY_FILE || join(homedir(), ".config", "vendorkb", "mcp-key");
  try {
    const key = readFileSync(file, "utf8").trim();
    if (key) return key;
  } catch {
    // fall through
  }
  throw new Error(`no key: set VENDORKB_MCP_KEY or put it in ${file}`);
}

// Every JSON-RPC message of a reply body: a JSON object or array, or an event stream's data.
export function messagesOf(contentType, body) {
  if (!body.trim()) return [];
  if ((contentType || "").startsWith("text/event-stream")) {
    const out = [];
    for (const event of body.split(/\r?\n\r?\n/)) {
      const data = event.split(/\r?\n/).filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).replace(/^ /, "")).join("\n");
      if (data.trim()) out.push(JSON.parse(data));
    }
    return out;
  }
  const parsed = JSON.parse(body);
  return Array.isArray(parsed) ? parsed : [parsed];
}

function errorFor(message, code, text) {
  // a reply only to a request (it has an id); a failed notification is just logged
  return message && message.id !== undefined && message.id !== null
    ? [{ jsonrpc: "2.0", id: message.id, error: { code, message: text } }]
    : [];
}

export function makeRelay({ url, key, timeoutMs, fetchImpl = fetch, log = () => {} }) {
  let protocolVersion = null; // from the initialize reply, sent on every later request
  let sessionId = null; // the server is stateless, but honour one if it ever sets it
  let initializing = null; // later messages wait for the initialize reply (its version)
  async function relay(message) {
    if (message && message.method === "initialize") {
      initializing = send(message);
      return initializing;
    }
    if (initializing) await initializing.catch(() => {});
    return send(message);
  }
  return relay;

  async function send(message) {
    const headers = {
      "Content-Type": "application/json",
      Accept: "application/json, text/event-stream",
      "X-API-Key": key,
      "User-Agent": `vendorkb-mcp-node/${VERSION}`,
    };
    if (protocolVersion) headers["MCP-Protocol-Version"] = protocolVersion;
    if (sessionId) headers["Mcp-Session-Id"] = sessionId;
    let res;
    try {
      res = await fetchImpl(url, { method: "POST", headers, body: JSON.stringify(message),
                                   signal: AbortSignal.timeout(timeoutMs) });
    } catch (e) {
      const why = e && e.name === "TimeoutError"
        ? `vendorkb did not answer within ${timeoutMs / 1000} s` : `vendorkb unreachable: ${e}`;
      log(why);
      return errorFor(message, -32000, why);
    }
    const body = await res.text();
    if (res.headers.get("mcp-session-id")) sessionId = res.headers.get("mcp-session-id");
    if (!res.ok) {
      // 401: a wrong key; 421: a host vendorkb does not answer; 429: the rate limit;
      // 524/502: the public route's proxy gave up (it cuts a request at 100 s)
      const why = `vendorkb HTTP ${res.status}${res.status === 401 ? " (check the key)" : ""}`
        + (body.trim() ? `: ${body.trim().slice(0, 200)}` : "");
      log(why);
      // an error reply in JSON-RPC form is still the server's answer: pass it on
      try {
        const msgs = messagesOf(res.headers.get("content-type"), body);
        if (msgs.length && msgs.every((m) => m && m.jsonrpc)) return msgs;
      } catch {
        // not JSON
      }
      return errorFor(message, -32000, why);
    }
    let msgs;
    try {
      msgs = messagesOf(res.headers.get("content-type"), body);
    } catch (e) {
      log(`unreadable reply: ${e}`);
      return errorFor(message, -32700, "vendorkb sent a reply that is not JSON-RPC");
    }
    for (const m of msgs) {
      if (m && m.result && m.result.protocolVersion && message.method === "initialize") {
        protocolVersion = m.result.protocolVersion;
      }
    }
    return msgs;
  }
}

export function main() {
  const log = (text) => process.stderr.write(`vendorkb-mcp: ${text}\n`);
  let key;
  try {
    key = readKey();
  } catch (e) {
    log(e.message);
    process.exit(2);
  }
  const url = process.env.VENDORKB_MCP_URL || DEFAULT_URL;
  const timeoutMs = 1000 * Number(process.env.VENDORKB_MCP_TIMEOUT || 300);
  const relay = makeRelay({ url, key, timeoutMs, log });
  const write = (m) => process.stdout.write(JSON.stringify(m) + "\n");
  log(`relaying to ${url}`);
  let pending = 0;
  let closed = false;
  const finish = () => { if (closed && pending === 0) process.exit(0); };
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
  lines.on("line", async (line) => {
    if (!line.trim()) return;
    let message;
    try {
      message = JSON.parse(line);
    } catch {
      write({ jsonrpc: "2.0", id: null, error: { code: -32700, message: "Parse error" } });
      return;
    }
    pending += 1;
    try {
      for (const m of await relay(message)) write(m);
    } finally {
      pending -= 1;
      finish();
    }
  });
  lines.on("close", () => { closed = true; finish(); });
}

// run as a program (directly, or through npm's bin link), not when imported by the tests
const isMain = (() => {
  try {
    return realpathSync(process.argv[1]) === realpathSync(fileURLToPath(import.meta.url));
  } catch {
    return false;
  }
})();
if (isMain) main();
