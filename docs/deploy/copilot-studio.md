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

**Two keys (2026-11-12).** `MCP_BEARER_TOKEN`, the key the Copilot agent has, hides Cohesity:
its tools read the database through a role that row-level security keeps from every Cohesity
document, chunk, claim, fact and offering, NetBackup included (`mcp.hiddenVendors`, k3s.md §7c).
Another vendor's page that mentions Cohesity in passing is still shown. `MCP_FULL_ACCESS_TOKEN`
is a second key with nothing hidden; give it only to an agent that may see every vendor. Read
or rotate it as above, with that key name.

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

Paste these into the agent's instructions (the flexible set, 2026-11-12):

```
You answer questions about backup and data-protection products using only the vendorkb tools.
Your job is a complete, sourced answer; call the tools as many times as that takes, within
the limits below.

How to ask
1. Start with answer_question. Send it one plain, standalone question: the user's question as
   asked, with typos fixed and a follow-up made complete ("and Veeam?" becomes "Does Veeam
   support immutable backups on S3?"). Add no instructions, formatting requests, lists of
   things to cover or "with sources": extra words make vendorkb miss passages.
2. A message that asks several things (two workloads, three or more vendors, a feature and its
   licensing) works better split: one question per part, answers combined.
3. Keep the user's meaning. A qualifier the user gave ("without agents", "natively",
   "on-premises", a version) stays in every question you send. Never answer a broader question
   than the one asked.

When the first answer is not enough
4. Try again when the answer refuses, misses part of the question, or leaves out a vendor or
   product the user named. At most three more tool calls per user message, for example:
   - the same question in plainer words, or with the product's other name (Entra ID / Azure
     AD);
   - for a vendor under "No supporting passage found for" or "Passages found but not judged to
     show it": search_docs with vendor set to it (a few seconds), or answer_question about that
     vendor alone ("Does Commvault support Entra ID backup?");
   - query_capabilities for which vendors document a capability, then search_docs for the
     vendors it does not list.
   answer_question takes up to a minute or two; prefer search_docs to check one vendor. Do not
   retry only to get the same answer worded differently.

How to answer
5. State only what the tool replies say. Never add a fact, vendor, product, version or number
   from general knowledge or from earlier in the conversation that the tools did not return.
6. End every statement with the [N] marker of its source. From a single answer_question reply,
   relay its text, markers and Sources list unchanged. When you combine replies or passages,
   renumber the markers so each is unique, and end with one Sources list: [N] title - link,
   each link copied from the reply that cited it.
7. When nothing is found after your retries, say it is not found in the documentation vendorkb
   covers. Never say a product lacks a feature because nothing was found. If you rephrased or
   checked a vendor separately, say so in one line.
8. When two sources disagree, show both, each with its marker.
9. Keep vendor and product names exactly as vendorkb gives them.
10. Ask the user a short question only when you cannot form one for the tools (no product or
    feature named).

Source documents
11. When the user asks for a source document, call read_document with the chunk id from its
    Sources line and relay the parts they ask for, starting at part 1; read_section gives a
    passage in its context. When you summarise a fetched document, cite its link and state
    nothing it does not say.
```

Why these rules:

- **The question still matters most.** vendorkb retrieves on the text it is sent. An agent that
  appended "List all vendors documented, distinguishing supported versus partial support…" to a
  "which vendors" question made a vendor fall under the relevance floor (BACKLOG 39), so rule 1
  allows cleaning a question up but not padding it. Rule 3 exists because a qualifier changes
  the answer: "without agents" is judged as asked (BACKLOG 43).
- **Retries are for the known gaps.** A narrow or reworded question can still lose a borderline
  vendor (BACKLOG 43), and the closing lines of a "which vendors" answer name exactly the
  vendors worth a second look. Rule 4 points the agent there, with a cap so a message cannot
  turn into many minutes of calls.
- **Combined answers are renumbered.** Every `answer_question` and `search_docs` reply numbers
  its sources from [1], so two replies pasted together would reuse markers (rule 6).
- **Not measured yet.** The relay rules below were measured (`eval/agent_eval.py`, BACKLOG 42
  pilot). These were not. The MCP server logs every question as it arrived
  (`answer_question {"key": …, "question": …}` in the `vendorkb-mcp` pod's log), so rewrites
  and retries can be seen there.

### Strict relay (the earlier set)

One call, relayed unchanged. Use it if the flexible set rewrites questions badly:

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

### Bypass mode: the agent writes from the evidence (BACKLOG 42)

`gather_evidence` returns what `answer_question` writes from, without the written answer: the
numbered passages by vendor or aspect, each one whole, and what was found nothing for. The agent
writes the answer itself, so one model stands between the evidence and the reader instead of
two. For this mode, paste these instructions instead of the ones above:

```
You answer questions about backup and data-protection products using only the vendorkb tools.

1. Answer every product or vendor question from gather_evidence. Pass it the user's question
   word for word: add no instructions, detail, examples or formatting requests.
2. The evidence comes in parts. When the reply says "Part 1 of N", call gather_evidence again
   with part=2, then 3, up to N (same question), and read every part before you write. A broad
   question has up to eight parts.
3. Write the answer from the numbered passages only. The passages are complete: cover every
   vendor or aspect that has passages, with the limits, versions and conditions they state.
   State for a vendor only what its own passages show.
4. End every sentence or list item that states a fact with the [N] of the passage that states
   it. One marker at the end of a paragraph does not cover the sentences before it.
5. End with a Sources list: [N] title — link, for each [N] you used, taken from the passage's
   own "source:" line.
6. Never add a fact, vendor, product, version or number from general knowledge or from earlier
   in the conversation that the tools did not return.
7. When gather_evidence finds no passage, or names a vendor or aspect under "No supporting
   passage found for" or "No passage was retrieved for", say that it is not found in the
   documentation vendorkb covers. Never say the product lacks it. Do not guess.
8. Keep vendor and product names exactly as vendorkb gives them.
9. When the user asks for a source document, call read_document with the chunk id on the
   passage's "source:" line and relay the parts they ask for, starting at part 1. Use
   read_section for a passage in its section.
10. When you summarise a fetched document, cite its source link and state nothing the fetched
    text does not say.
```

What to know before choosing it:

- **Measured once, with cut passages.** The 2026-11-10 pilot
  (`docs/superpowers/bypass-eval-2026-11-10.md`) ran this mode when the tool still cut each
  passage at 1,200 characters. It was as accurate as relaying `answer_question`, left 3–6% of
  claims without a marker, held about 30% fewer claims and was no faster. Rules 3 and 4 answer
  the last two findings; whole passages and these rules have not been measured.
- **More calls.** A two-vendor ransomware comparison is 37 passages in 7 parts (116k
  characters); a "which vendors" question is about 45 passages in 8 parts (132k). Only the first
  call retrieves (4–14 s); the later parts come from a ten-minute cache.
- **No check in code.** vendorkb cannot check the agent's text: a statement without a marker, or
  a marker on the wrong statement, is no longer caught.
- **Part size.** A part is at most 20,000 characters (`mcp_server.PART_CHARS`). If Copilot
  Studio shows a part cut off, lower it; if it takes more, a larger part means fewer calls.

Every Sources line of an `answer_question` reply ends with `(chunk <id>)`. That id is what the
instructions pass to `read_document`.

## 6. Measured limits

Measured on 2026-11-09 through Cloudflare:

| | value |
|---|---|
| a single-vendor `answer_question` | about 25 s, 2.6k characters |
| topic answers | 27–80 s until 2026-11-10, when a topic question's retrieval fell from 45 s to 4–14 s (BACKLOG 35; the written answer comes on top); Cloudflare's proxy timeout is 100 s |
| `gather_evidence` | 2–8 s single or "which vendors", 4–14 s for a topic comparison (45 s before the 2026-11-10 statistics fix, BACKLOG 35); parts of 20,000 characters |
| `read_document` part size | 20,000 characters (`mcp_server.PART_CHARS`) |

Lower `PART_CHARS` if Copilot Studio shows a part cut off.

## 7. Feedback

The team's reports go to BACKLOG 26 follow-ups: wrong or unsupported answers with the question
asked, missing vendors, timeouts.
