# vendorkb in Copilot Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve vendorkb's MCP server at `https://vendorkb.area51.pp.ua/mcp` for a Copilot Studio
agent, behind one shared secret, with a `read_document` tool and the agent's setup instructions.

**Architecture:**
- **Route.** A second, public Traefik Ingress for the existing MCP Service carries only `/mcp`.
  It uses the wildcard Cloudflare Origin CA certificate DocExtractor already uses.
- **Secret.** The bearer gate also accepts the same secret as `X-API-Key`, because Copilot
  Studio's API-key authentication may not send `Authorization: Bearer`.
- **`read_document`.** It returns a cited page's full stored text in parts, using the same
  version rules as `read_section`.
- **Order.** Code tasks 1–2 ship first. Operations O1 then put the route live and the user
  connects a test agent (the feasibility gate). Tasks 3–4 follow only if that works.

**Tech Stack:** Python 3.12, Starlette, the MCP SDK (FastMCP, streamable HTTP), psycopg 3 async,
Helm 4 on k3s with Traefik, Cloudflare (free plan).

**Spec:** `docs/superpowers/specs/2026-11-09-copilot-mcp-design.md`

## Global Constraints

- **Public host:** exactly `vendorkb.area51.pp.ua`. TLS from a secret named
  `cloudflare-origin-ca` in namespace `vendorkb`, copied from namespace `docextractor`. Traefik
  entrypoint `websecure`.
- **Public route:** only `/mcp`. The LAN ingress (`vendorkb-mcp.k3s.home.lan`, path `/`) stays
  as it is.
- **One secret:** `MCP_BEARER_TOKEN`, accepted as `Authorization: Bearer <token>` or
  `X-API-Key: <token>`, compared in constant time. No OAuth, no per-user sign-in.
- **Rotation:** rotate `MCP_BEARER_TOKEN` before the public route goes live. Never print it,
  never commit it.
- **Tools:** all stay read-only (`_RO` annotations). `read_document` serves vendorkb's stored
  chunks, never a live DocExtractor fetch.
- **Helm:** helm-managed fields change only through `helm upgrade --reset-then-reuse-values
  --set ...`, never `kubectl patch`.
- **Gates:** CI must pass (`uv run ruff check src tests`, `uv run mypy src`, `uv run --extra dev
  pytest`, helm lint and template). Run gitleaks before every push:
  `gitleaks detect --source . --log-opts "origin/main..HEAD" --no-banner && git push`.

## Review Focus

1. **A request that carries both headers, one of them wrong, is let through** when either
   matches. A request with neither, or with both wrong, gets a 401. Tested in Task 1.
2. **The public host is not in `MCP_ALLOWED_HOSTS`:** every public call would fail the
   DNS-rebinding guard with a 421. Task 2 renders the allowed host into the values, and O1's
   smoke checks it.
3. **`read_document` on a chunk of an earlier version of its page** must return that version's
   chunks, marked as from an earlier version. It must never mix in chunks of the current page.
   Tested in Task 3.
4. **One chunk longer than a part** must go out alone as its own part, never cut and never
   dropped. Tested in Task 3.
5. **A `part` number outside 1..parts** must raise a `ToolError` that names the valid range,
   never return empty text. Tested in Task 3.

---

### Task 1: The gate accepts the secret as `X-API-Key`

**Files:**
- Modify: `src/vendorkb/mcp_app.py` (`BearerTokenGate`)
- Test: `tests/unit/test_mcp_app.py`

**Interfaces:**
- Produces: `BearerTokenGate(app, token)`, unchanged signature. It passes a request when
  `authorization == b"Bearer " + token` or `x-api-key == token`; each comparison uses
  `hmac.compare_digest`. The 401 response is unchanged.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("headers", [{"x-api-key": "s3cret"},
                                     {"x-api-key": "s3cret", "authorization": "Bearer nope"},
                                     {"authorization": "Bearer s3cret", "x-api-key": "nope"}])
async def test_the_secret_as_an_api_key_reaches_the_server(headers):
    r = await _post(_app(), {**HDRS, **headers})
    assert r.status_code == 200

@pytest.mark.parametrize("headers", [{"x-api-key": "nope"}, {"x-api-key": ""},
                                     {"x-api-key": "Bearer s3cret"},
                                     {"x-api-key": "nope", "authorization": "Bearer nope"}])
async def test_a_wrong_api_key_is_401(headers):
    r = await _post(_app(), {**HDRS, **headers})
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
```

- [ ] **Step 2:** `uv run --extra dev pytest tests/unit/test_mcp_app.py -q`. Expected: the new
  tests FAIL.
- [ ] **Step 3: Implement the check in `BearerTokenGate.__call__`.** Keep a second expected value
  (`token.encode()`) and pass if either header matches. Update the class docstring and the module
  docstring.
- [ ] **Step 4:** run the same command. Expected: all PASS.
- [ ] **Step 5:** commit: `feat(mcp): accept the secret as X-API-Key (Copilot Studio)`.

### Task 2: A public ingress for `/mcp` in the Helm chart

**Files:**
- Modify: `deploy/helm/vendorkb/templates/mcp.yaml` (a second Ingress after the existing one)
- Modify: `deploy/helm/vendorkb/values.yaml` (`mcp.publicIngress`)
- Modify: `docs/deploy/k3s.md` §7b (two lines on the public ingress)

**Interfaces:**
- Produces these values:

```yaml
publicIngress:
  enabled: false
  className: traefik
  host: ""          # vendorkb.area51.pp.ua
  tlsSecret: ""     # cloudflare-origin-ca
  annotations: {traefik.ingress.kubernetes.io/router.entrypoints: websecure}
```

- Produces this Ingress:
  - named `vendorkb-mcp-public`;
  - one rule for `host`: path `/mcp`, `pathType: Exact`, backend `vendorkb-mcp:8080`;
  - `tls` with `tlsSecret`, required when the ingress is enabled (`required` with a message).
- When the ingress is enabled, the chart appends `,<publicIngress.host>` to the
  `MCP_ALLOWED_HOSTS` env value in `templates/mcp.yaml` (line ~33, `{{ $m.allowedHosts | quote }}`).

- [ ] **Step 1:** `helm template v deploy/helm/vendorkb --set image.tag=sha-0000000 --set
  mcp.enabled=true --set mcp.publicIngress.enabled=true --set
  mcp.publicIngress.host=vendorkb.area51.pp.ua --set mcp.publicIngress.tlsSecret=cloudflare-origin-ca`.
  Expected before the change: no `vendorkb-mcp-public` in the output.
- [ ] **Step 2: Add the template and the values.**
- [ ] **Step 3:** rerun Step 1. Expected:
  - an Ingress `vendorkb-mcp-public` with host `vendorkb.area51.pp.ua`, path `/mcp` `Exact`, tls
    secret `cloudflare-origin-ca` and the `websecure` annotation;
  - `MCP_ALLOWED_HOSTS` ends with `,vendorkb.area51.pp.ua`.

  Also run with `publicIngress.enabled=true` and no `tlsSecret`. Expected: helm fails with the
  `required` message.
- [ ] **Step 4:** `helm lint deploy/helm/vendorkb --set image.tag=sha-0000000`. Expected:
  `0 chart(s) failed`.
- [ ] **Step 5:** commit: `feat(helm): public ingress for /mcp`.

### Operations O1: Go live and the feasibility gate (controller and user, after Tasks 1–2 merge)

- [ ] **O1.1.** Open a PR, wait for green CI, review it, merge it, and wait for `image.yml` to
  build `sha-<merge>`.
- [ ] **O1.2. Copy the certificate:**
  `kubectl -n docextractor get secret cloudflare-origin-ca -o json | jq '{apiVersion, kind, type, data, metadata: {name: .metadata.name, namespace: "vendorkb"}}' | kubectl apply -f -`.
  It is not helm-managed.
- [ ] **O1.3. Rotate the token.**
  - Generate it: `python3 -c 'import secrets; print(secrets.token_urlsafe(32))'`.
  - Put it into `vendorkb-secret` key `MCP_BEARER_TOKEN` (`kubectl patch secret`; the secret
    is not helm-managed). Never echo the value.
- [ ] **O1.4. Deploy:** `helm upgrade vendorkb deploy/helm/vendorkb -n vendorkb
  --reset-then-reuse-values --set image.tag=sha-<merge> --set mcp.publicIngress.enabled=true
  --set mcp.publicIngress.host=vendorkb.area51.pp.ua --set
  mcp.publicIngress.tlsSecret=cloudflare-origin-ca`. The new pod reads the rotated token.
- [ ] **O1.5. The user, in Cloudflare:**
  - a proxied DNS record `vendorkb` → the same target as `docs`;
  - one rate-limiting rule on host `vendorkb.area51.pp.ua`: 50 requests per 10 s per IP, block
    for 10 s;
  - one WAF custom rule that blocks the host when the URI path is not `/mcp`;
  - SSL mode Full (strict), unchanged.
- [ ] **O1.6. Public smoke, from outside the LAN** (any off-LAN shell, or Cloudflare's path):
  - no token: 401;
  - the wrong `X-API-Key`: 401;
  - the right token as `Authorization: Bearer` and as `X-API-Key`: `initialize` gives 200,
    then `tools/list` lists 7 tools;
  - `/healthz` and `/`: blocked by the WAF rule (403).
- [ ] **O1.7. The gate.** The user creates a test agent in Copilot Studio with an MCP tool for
  `https://vendorkb.area51.pp.ua/mcp` and API-key authentication. Record:
  - whether the tenant allows it;
  - which header form works;
  - whether the tools list;
  - the time one `answer_question` takes, against Copilot's timeout;
  - the largest tool result Copilot shows uncut.

  **If the tenant blocks it, stop here and report.**

### Task 3: `read_document`

**Files:**
- Modify: `src/vendorkb/search.py`: `read_section`'s `expand` gains `"page"`, meaning every
  chunk of the anchor's version of the page, by ordinal.
- Modify: `src/vendorkb/mcp_backend.py`: the `Expand` literal and the `Backend` protocol, which
  stay `section(chunk_id, expand)`.
- Modify: `src/vendorkb/mcp_server.py`: the new tool and `PART_CHARS`.
- Test: `tests/integration/test_supersede.py` (the page expand), `tests/unit/test_mcp_tools.py`
  (the tool).

**Interfaces:**
- Consumes: `backend.section(chunk_id, "page") -> list[Hit]` (the version rules of
  `read_section`).
- Produces:
  - the constant `PART_CHARS = 20_000`. O1.7's measured limit replaces it when smaller.
  - the tool `read_document(chunk_id: str, part: int = 1) -> CallToolResult`, read-only.
    - Its text: `"{vendor} · {product} — {title}\nsource: {url}{EARLIER_VERSION if superseded}\npart {part} of {parts}\n\n"`,
      then that part's chunk texts joined by blank lines.
    - Its structured content: `{"chunk_ids": [...], "part": n, "parts": m, "url": ...}`.
    - Parts are built by adding whole chunks in order while the part stays at or under
      `PART_CHARS`. A chunk alone over `PART_CHARS` is a part by itself.
    - An unknown chunk raises a `ToolError` with the same wording as `read_section`'s.
- Tool description: `"The full text of the page a passage comes from (any chunk_id from
  search_docs, read_section or answer_question's sources), in parts: start at part 1; the
  reply says how many parts there are. Use it when the user asks for the source document."`

- [ ] **Step 1: Write the failing tests.**
  - In `test_supersede.py`: `test_read_section_page_returns_the_whole_version_of_the_page`.
    Seed a page with 3 current chunks plus 2 chunks of an earlier version, using that file's
    seeding helpers. Assert:
    - `[h.chunk_id for h in await read_section(conn, current_id, "page")]` is the 3 current
      ids in ordinal order;
    - for an earlier-version id, the result is that version's 2 ids, all with
      `superseded=True`.
  - In `test_mcp_tools.py`, extend `FakeBackend.section` so that `expand == "page"` returns,
    for `"k2"`: hits of 9,000, 9,000, 25,000 and 100 chars; for `"old"`: 2 superseded hits.
    Add:
    - `test_read_document_parts_whole_chunks`: part 1 holds k1 and k2 (18,000 chars), part 2
      holds the 25,000-char chunk alone, part 3 holds the last one. `parts == 3` in each, and
      the text says `part 2 of 3`.
    - `test_read_document_out_of_range_part_names_the_range`: `part=4` and `part=0` raise a
      `ToolError` whose message contains `1-3`.
    - `test_read_document_marks_an_earlier_version`: the text of `"old"` contains
      `EARLIER_VERSION`.
    - `test_read_document_unknown_chunk_is_an_error`.
    - Update `test_the_tools_are_listed_read_only` to expect 8 tools, `read_document` among
      them.
- [ ] **Step 2:** `uv run --extra dev pytest tests/unit/test_mcp_tools.py
  tests/integration/test_supersede.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement.** The `"page"` branch is `base` with no extra predicate. Then add the
  tool beside `read_section`, reusing its header code through a helper both share.
- [ ] **Step 4:** rerun. Expected: PASS. Then run the full gates (Global Constraints).
- [ ] **Step 5:** commit: `feat(mcp): read_document, a cited page's full text in parts`.

### Task 4: The Copilot Studio setup and agent instructions

**Files:**
- Create: `docs/deploy/copilot-studio.md`
- Modify:
  - `CLAUDE.md`: the MCP bullet names the public host, `read_document` and the new doc;
  - `BACKLOG.md`: item 26 is marked built (date, PR), and the remaining feedback loop is kept;
  - `docs/deploy/k3s.md` §7b: a link to the new doc.

**Interfaces:** none (docs).

- [ ] **Step 1: Write `docs/deploy/copilot-studio.md`** with these sections:
  1. **What the agent is.** For the competitive intelligence team; access is whom the agent is
     shared with.
  2. **Cloudflare.** The O1.5 rules, with their exact values.
  3. **Connecting the MCP tool.** The URL `https://vendorkb.area51.pp.ua/mcp`, API-key
     authentication, and the header form O1.7 found working. Where the secret comes from
     (`vendorkb-secret` `MCP_BEARER_TOKEN`) and how to rotate it (O1.3 plus a pod restart).
  4. **Agent instructions to paste.** Exactly these rules, in this order:
     - answer every product question with `answer_question` and relay its answer and its
       Sources list unchanged;
     - never add a fact, vendor or version from general knowledge;
     - when vendorkb refuses or finds nothing, say it is not found in the documentation;
     - when the user asks for a source document, call `read_document` with the cited chunk and
       relay every part they ask for; use `read_section` for the passage in context;
     - keep vendor and product names exactly as vendorkb gives them.
  5. **Measured limits.** The O1.7 numbers: timeout, largest uncut result, `PART_CHARS`.
  6. **Feedback.** Where the team's reports go: BACKLOG item 26's follow-ups.
- [ ] **Step 2:** commit: `docs: Copilot Studio setup and agent instructions`.

### Operations O2: Ship and hand over (controller and user)

- [ ] **O2.1.** PR for Tasks 3–4, green CI, review, merge, deploy `sha-<merge>` with
  `--reset-then-reuse-values --set image.tag=...`.
- [ ] **O2.2. Smoke through Copilot Studio's test pane:**
  - one comparison-table question;
  - one "which vendors" question;
  - one `read_document` request on a long page, getting part 2;
  - one on a login-walled page.

  The user runs them; the controller checks the MCP pod logs for the calls.
- [ ] **O2.3.** The user tests alone, then shares the agent with the team. Their feedback
  becomes BACKLOG follow-ups.
