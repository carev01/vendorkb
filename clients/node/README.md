# vendorkb-mcp (Node)

A local MCP server for Claude Desktop that relays to the vendorkb MCP endpoint
(`https://vendorkb.area51.pp.ua/mcp`). Use it where custom (remote) connectors are not allowed:
Claude Desktop starts it as a local stdio server, and it forwards every MCP message to vendorkb
with your key. One file, no dependencies, Node 18 or later.

It holds no logic of its own: the tools, their descriptions and their answers are the server's,
so a server update needs no new version of this file.

## Install

Pick one:

- **The file alone.** Copy `vendorkb-mcp.mjs` anywhere, e.g. `~/tools/vendorkb-mcp.mjs`.
- **As a CLI.** `npm install -g ./vendorkb-mcp-1.0.0.tgz` (built with `npm pack` in this
  folder). It installs nothing from the registry, and puts `vendorkb-mcp` on the PATH.

## The key

Which key decides what you see: `MCP_FULL_ACCESS_TOKEN` sees every vendor, `MCP_BEARER_TOKEN`
hides Cohesity (docs/deploy/k3s.md §7c). Put the key in a file that only you can read, so it
stays out of Claude Desktop's config:

- macOS / Linux: `~/.config/vendorkb/mcp-key`, then `chmod 600 ~/.config/vendorkb/mcp-key`
- Windows: `%USERPROFILE%\.config\vendorkb\mcp-key`

Or set `VENDORKB_MCP_KEY` in the config's `env` (simpler, but the key then sits in that file).

## Claude Desktop

Settings → Developer → Edit Config opens `claude_desktop_config.json`. Add the server under
`mcpServers` and restart Claude Desktop:

```json
{
  "mcpServers": {
    "vendorkb": {
      "command": "node",
      "args": ["/Users/you/tools/vendorkb-mcp.mjs"]
    }
  }
}
```

On Windows the path is written with double backslashes:
`"args": ["C:\\Users\\you\\tools\\vendorkb-mcp.mjs"]`. Installed as a CLI, use
`"command": "vendorkb-mcp"` and no `args` (on Windows, `"command": "vendorkb-mcp.cmd"`). If
Claude Desktop cannot find `node`, give its full path (`which node` / `where node`).

The vendorkb tools then appear under the tools menu of a chat. For a first check, ask "Use
list_vendors from vendorkb".

## Settings

| variable | default | |
|---|---|---|
| `VENDORKB_MCP_KEY` | | the key; overrides the key file |
| `VENDORKB_MCP_KEY_FILE` | `~/.config/vendorkb/mcp-key` | |
| `VENDORKB_MCP_URL` | `https://vendorkb.area51.pp.ua/mcp` | the LAN route works too: `http://vendorkb-mcp.k3s.home.lan/mcp` |
| `VENDORKB_MCP_TIMEOUT` | `300` | seconds per call |

## Troubleshooting

The relay logs to stderr, which Claude Desktop writes to `mcp-server-vendorkb.log` (macOS
`~/Library/Logs/Claude/`, Windows `%APPDATA%\Claude\logs\`).

- `HTTP 401 (check the key)`: a wrong or rotated key.
- `HTTP 429`: the public route's rate limit; wait a minute.
- `HTTP 524` or `502`: the public route cut a call at 100 s (a long `answer_question`); ask
  again, or use `gather_evidence`, which returns in seconds.
- `vendorkb unreachable`: no network route to the host (a corporate proxy: Node's fetch does
  not use the system proxy; set `NODE_USE_ENV_PROXY=1` and `HTTPS_PROXY` in `env`, Node 22.21 or 24.5 and later).

Test by hand (each line in is one JSON-RPC message, each line out one reply):

```
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | node vendorkb-mcp.mjs
```

## Development

`npm test` runs the relay against a local fake endpoint (no network, no key).
