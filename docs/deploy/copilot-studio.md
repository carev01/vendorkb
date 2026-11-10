# vendorkb in Copilot Studio

Spec: `docs/superpowers/specs/2026-11-09-copilot-mcp-design.md`.

## 1. What the agent is

A Copilot Studio agent in Veeam's tenant answers questions about competing products from
vendorkb's facts and documentation. It is for the Competitive Intelligence team, and possibly
Product Management later.

**Access.**
- Who may use the agent is decided in Copilot Studio, by whom it is shared with.
- The server checks one shared secret, which every call from the agent carries.

**Answers.** The agent relays vendorkb's own cited answers rather than composing its own, and
can fetch the full text of any cited page.

## 2. The route

**Path of a request.**
- The URL is `https://vendorkb.area51.pp.ua/mcp`.
- Cloudflare proxies it (zone `area51.pp.ua`, SSL mode Full strict) to Traefik on k3s.
- The Ingress `vendorkb-mcp-public` routes only `/mcp`; any other path gets a 404.

**TLS.** The certificate is `cloudflare-origin-ca` in namespace `vendorkb`: a copy of
DocExtractor's wildcard Cloudflare Origin CA certificate for `*.area51.pp.ua`, valid until 2041.
To copy it again:

```
kubectl -n docextractor get secret cloudflare-origin-ca -o json \
  | jq '{apiVersion, kind, type, data, metadata: {name: .metadata.name, namespace: "vendorkb"}}' \
  | kubectl apply -f -
```

**Helm values** (also in the deploy commands):

```
--set mcp.publicIngress.enabled=true --set mcp.publicIngress.host=vendorkb.area51.pp.ua
--set mcp.publicIngress.tlsSecret=cloudflare-origin-ca
```

**Rate limits** (Traefik `RateLimit` middlewares, `mcp.publicIngress.rateLimit`; Cloudflare's free
plan, as configured for this zone, offers no WAF or rate-limiting rules). An over-limit request
gets a 429.

| bucket | keyed on | limit |
|---|---|---|
| per client | Cloudflare's `CF-Connecting-IP` | 30 requests a minute, bursts of 20 |
| global | the host (one bucket) | 120 requests a minute, bursts of 60 |

The global bucket bounds total spend even if a client reaches the origin directly and forges
the header. The secret still guards everything: a request without it gets a 401 before anything
is spent.

## 3. The secret

**Where it lives.** It is `MCP_BEARER_TOKEN` in `vendorkb-secret`: 32 random bytes, last
rotated 2026-11-09. The server accepts it as `X-API-Key: <token>`, which is what Copilot Studio
sends, or as `Authorization: Bearer <token>`.

**Read it** in your own terminal, never in a shared session:

```
kubectl -n vendorkb get secret vendorkb-secret -o jsonpath='{.data.MCP_BEARER_TOKEN}' | base64 -d
```

**Rotate it.** Patch the secret, restart the pod, then update the Copilot Studio connection:

```
T=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
kubectl -n vendorkb patch secret vendorkb-secret --type merge \
  -p "{\"stringData\":{\"MCP_BEARER_TOKEN\":\"$T\"}}"; unset T
kubectl -n vendorkb delete pod -l app.kubernetes.io/name=vendorkb-mcp
```

## 4. Connecting the agent

1. In Copilot Studio, open the agent: Tools → Add a tool → Model Context Protocol.
2. Server URL: `https://vendorkb.area51.pp.ua/mcp`.
3. Authentication: API key, in a header named `X-API-Key`. Paste the secret when you create the
   connection.
4. The agent should list nine tools:

| tool | |
|---|---|
| `list_vendors` | |
| `list_dimensions` | |
| `list_targets` | |
| `query_capabilities` | |
| `search_docs` | |
| `read_section` | |
| `read_document` | |
| `answer_question` | |
| `gather_evidence` | |

**Agent model.** Use Claude Haiku. The default GPT model refused to relay a full page as
copyrighted content; Haiku relays it (user, 2026-11-09).

## 5. Agent instructions

Paste these into the agent's instructions:

```
You answer questions about backup and data-protection products using only the vendorkb tools.

1. Answer every product or vendor question with answer_question. Pass it the user's question
   word for word: add no instructions, detail, examples or formatting requests. Relay its
   answer and its Sources list unchanged: keep every [N] marker and every source link.
2. Never add a fact, vendor, product, version or number from general knowledge or from earlier
   in the conversation that the tools did not return.
3. When answer_question refuses or finds nothing, say that it is not found in the documentation
   vendorkb covers. Do not guess.
4. When the user asks for a source document, call read_document with the cited chunk_id and
   relay the parts they ask for, starting at part 1. Each reply says how many parts the page
   has. Use read_section for the passage in its context.
5. Keep vendor and product names exactly as vendorkb gives them.
6. When you summarise a fetched document, cite its source link and state nothing the fetched
   text does not say.
```

Rule 1's "word for word" matters: vendorkb retrieves on the text it is sent. An agent that
appended "List all vendors documented, distinguishing supported versus partial support…" to a
"which vendors" question made a vendor fall under the relevance floor (BACKLOG 39). The MCP
server logs each question as it arrived (`answer_question {"question": …}` in the
`vendorkb-mcp` pod's log), so a rewritten question can be seen there.

### Writing from the evidence instead (BACKLOG 42: measured, not recommended)

The 2026-11-10 pilot (`docs/superpowers/bypass-eval-2026-11-10.md`) found this as accurate as
relaying `answer_question`, but with 3–6% of claims left without a marker, about 30% fewer
claims, and no gain in time. Keep the rules above unless you have a reason to try it.

`gather_evidence` returns what `answer_question` writes from, without the written answer: the
numbered passages by vendor or aspect, and what was found nothing for. The agent then writes the
answer itself, so one model stands between the evidence and the reader instead of two. To try
it, replace rules 1 and 3 with:

```
1. Answer every product or vendor question with gather_evidence. Pass it the user's question
   word for word: add no instructions, detail, examples or formatting requests. When the reply
   says "Part 1 of N", call it again with part=2 … N (same question) before you answer.
   Write the answer from the numbered passages only. Put the [N] of the passage after every
   statement it supports, and end with a Sources list: [N] title — link, for each [N] used.
   State for a vendor only what its own passages show.
3. When gather_evidence finds no passage, or none for a vendor or aspect, say that it is not
   found in the documentation vendorkb covers; never say the product lacks it. Do not guess.
```

vendorkb cannot check the agent's text: a statement without a marker, or a marker on the wrong
statement, is no longer caught in code.
Sizes on 2026-11-10: a two-vendor ransomware comparison is 37 passages in 3 parts (56k
characters); a "which vendors" question is about 45 passages in 4 parts (66k).

Every Sources line of an `answer_question` reply ends with `(chunk <id>)`. That id is what rule 4
passes to `read_document`.

## 6. Measured limits

Measured on 2026-11-09 through Cloudflare:

| | value |
|---|---|
| a single-vendor `answer_question` | about 25 s, 2.6k characters |
| "which vendors" topic answers | 27–80 s (BACKLOG 35); Cloudflare's proxy timeout is 100 s |
| `gather_evidence` | 6–9 s single or "which vendors", about 45 s for a topic comparison; parts of 20,000 characters |
| `read_document` part size | 20,000 characters (`mcp_server.PART_CHARS`) |

Lower `PART_CHARS` if Copilot Studio shows a part cut off.

## 7. Feedback

The team's reports go to BACKLOG 26 follow-ups: wrong or unsupported answers with the question
asked, missing vendors, timeouts.
