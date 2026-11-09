# vendorkb in Copilot Studio (BACKLOG 26)

Status: designed with the user on 2026-11-09, awaiting their review.

## What and for whom

The user will build a Copilot Studio agent in Veeam's Microsoft 365 tenant that answers
questions about competing products from vendorkb.

**Users.**
- First, the Competitive Intelligence team.
- Possibly Product Management later.

**Questions.** Mostly the topics of the comparison tables (does vendor X support Y, how do X and
Z compare on immutability), with occasional narrower ones.

**Rollout.** The user tests it alone first, then shares it with the team and collects feedback.
Improvements follow from that feedback.

## What must not happen

The anti-goal is the one the past days' work was aimed at: wrong answers, hallucinated facts,
facts from bad sources.

Copilot Studio adds a model of its own between vendorkb and the reader. It could:
- paraphrase loosely;
- merge vendors;
- fill gaps from its general knowledge.

The design keeps that model on vendorkb's evidence, and accepts a less conversational agent for
it (the user agreed).

## Decided with the user

- **Access** is controlled in Copilot Studio, by whom the agent is shared with. The server
  checks one shared secret. No per-user sign-in, no OAuth.
- **Address:** `vendorkb.area51.pp.ua`, in the Cloudflare zone that already serves DocExtractor
  (`docs.area51.pp.ua`).
- **Answers:** the agent mostly relays vendorkb's own cited answers (`answer_question`) rather
  than composing its own from raw facts.
- **Full documents:** on request, the user can get the full text of a cited page, including pages
  behind a login wall that DocExtractor fetched.

## What exists today

**The MCP server.** `vendorkb mcp` (`mcp_app.py`, `mcp_server.py`) serves streamable HTTP at
`/mcp`. It runs one replica in k3s, on the LAN host `vendorkb-mcp.k3s.home.lan`.

Every request must carry `Authorization: Bearer <MCP_BEARER_TOKEN>`. `MCP_ALLOWED_HOSTS` guards
the Host header.

Its seven read-only tools:

| tool | what it does |
|---|---|
| `list_vendors` | the vendors |
| `list_dimensions` | the taxonomy's dimensions |
| `list_targets` | the taxonomy's targets |
| `query_capabilities` | capability facts matching filters |
| `search_docs` | passages from the documentation |
| `read_section` | a passage in context, up to its whole section |
| `answer_question` | a cited answer, which refuses when the documentation does not support one |

**DocExtractor's public route** is the pattern to copy:
- a second Ingress (`docextractor-public`) on Traefik's `websecure` entrypoint;
- TLS from the secret `cloudflare-origin-ca`, a Cloudflare Origin CA certificate for
  `*.area51.pp.ua` valid to 2041. It lives in the `docextractor` namespace.

The host is proxied by Cloudflare to the home connection. `vendorkb.area51.pp.ua` has no DNS
record yet.

## What to build

0. **Feasibility first.** The user does not know whether Veeam's tenant allows a Copilot Studio
   agent to connect to an outside MCP server: data-loss-prevention policies or connector
   governance may block it, or need an admin. So the order is:
   1. the public route and the rotated token (items 1–3);
   2. the user connects a test agent in Copilot Studio.

   If the tenant blocks the connection, work pauses there until the user clears it. Items 4 and
   5 are built only once a test agent has listed the tools. The same test checks whether
   Copilot Studio needs an `Mcp-Session-Id` (the server is stateless) and what tool timeout and
   result size it applies.

1. **Public route.**
   - A second, public Ingress for the MCP Service: host `vendorkb.area51.pp.ua`, entrypoint
     `websecure`, TLS from a copy of `cloudflare-origin-ca` in the `vendorkb` namespace.
   - The LAN ingress stays.
   - The host joins `MCP_ALLOWED_HOSTS`.
   - **Only `/mcp` is routed.**
   - Helm values gain a `mcp.publicIngress` block.
2. **Cloudflare, free plan** (the user does these; the doc gives exact steps):
   - the proxied DNS record;
   - the free plan's one rate-limiting rule on the host, because `answer_question` and
     `search_docs` spend money on the answer model and on Voyage reranks. It counts per IP over
     10 seconds. Copilot Studio calls come from a few Microsoft egress addresses, so the limit
     is effectively global: start near 50 requests per 10 seconds, block for 10 seconds;
   - a WAF custom rule (the free plan has five) that blocks the host except `/mcp`;
   - the zone's SSL mode stays Full (strict), as DocExtractor's Origin CA certificate already
     requires. The builder confirms it.
3. **The secret as Copilot Studio sends it.**
   - Copilot Studio's MCP connection offers API-key authentication with a header name and a
     value. The builder confirms whether the value can be `Bearer <token>` in the
     `Authorization` header.
   - If it cannot, the gate also accepts `X-API-Key: <token>`, with the same constant-time
     comparison and still exactly one secret.
   - Rotate `MCP_BEARER_TOKEN` before going public. The current one has only been on the LAN.
4. **`read_document`, a new read-only tool.**
   - It takes a `chunk_id` from any citation and returns the full text of that page.
   - The text comes from vendorkb's stored chunks, in order, under the page's title, URL and
     version, the way `read_section` writes them.
   - It serves vendorkb's copy, which the sync keeps current every 30 minutes, rather than
     fetching live from DocExtractor (my call): one path, no new dependency, and the same text
     the facts were extracted from.
   - Long pages are returned in parts (`part`, `parts`) under a size the builder measures
     against Copilot Studio's tool-result limit.
   - A page whose newer version replaced the chunk says so, as `read_section` does.
5. **Agent instructions** (a text the user pastes into Copilot Studio, kept in
   `docs/deploy/copilot-studio.md` with the setup steps):
   - answer with `answer_question`, and relay its answer and its Sources list;
   - never add facts from general knowledge;
   - say "not found in the documentation" when vendorkb refuses;
   - use `read_section` and `read_document` when the user asks for the source text;
   - keep vendor names exactly as the answer gives them.
6. **Checks before the user tests it:**
   - a public smoke from outside the LAN: a wrong token gets a 401, a right token lists the
     tools, and one `answer_question` call works;
   - the same smoke through Copilot Studio's test pane;
   - `read_document` on a long page and on a login-walled one.

## Settled after the builder check

- **Deploying** is done by Claude under the standing vendorkb approvals. The token lives in
  `vendorkb-secret`. Rotating it breaks only the local evaluation environment, which gets the
  new value.
- **Timeouts.** `answer_question` takes about 13 s; topic answers take 27–80 s (BACKLOG 35). That
  is under Cloudflare's 100-second proxy timeout. Copilot Studio's own tool timeout is measured
  in step 0; if it is shorter, faster topic answers become the next item.
- **Full text.** vendorkb stores each page's complete text, login-walled pages included, so
  `read_document` can return whole pages. The user asked for this. Anyone the agent is shared
  with can read those pages through it.

## Left to the builder

- Whether the public Ingress is a second template or a list in `mcp.ingress`.
- The part size for `read_document`.
- The rate-limit numbers, once a Copilot call pattern is seen.
- Whether Copilot Studio still needs to see tools other than `answer_question`, `read_section`
  and `read_document`. If its orchestrator picks raw tools and composes its own answers, hiding
  `search_docs` and `query_capabilities` behind a setting is an option. Measure before deciding.

## Not in scope

- Per-user sign-in or OAuth.
- Teams or Microsoft 365 Copilot publishing beyond what Copilot Studio does by itself.
- Live fetches from DocExtractor.
- Answer-quality work beyond the instructions above; feedback from the team drives that later.
