# vendorkb in Copilot Studio

Spec: `docs/superpowers/specs/2026-11-09-copilot-mcp-design.md`.

## 1. What the agent is

A Copilot Studio agent in Veeam's tenant answers questions about competing products from
vendorkb's facts and documentation. It is for the Competitive Intelligence team, and possibly
Product Management later.

**Access.**
- Who may use the agent is decided in Copilot Studio, by whom it is shared with.
- The server checks one shared secret, which every call from the agent carries.

**Answers.** The agent writes its own cited answer from the evidence `gather_evidence` returns
(§5; since 2026-11-16, the user's choice), and can fetch the full text of any cited page. It
can instead relay vendorkb's own written answer (§5, the alternative sets).

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

Paste these into the agent's instructions (the evidence set, 2026-11-16). The agent writes the
answer itself from `gather_evidence`, and may rephrase, split and retry when the first evidence
falls short:

```
You answer questions about backup and data-protection products using only the vendorkb tools.
You write the answer yourself from the evidence they return. Your job is a complete, sourced
answer; call the tools as many times as that takes, within the limits below.

Before the first question
1. At the start of a conversation, call list_vendors once to learn which vendors and products
   vendorkb covers. Use it to:
   - tell the user at once when a vendor or product they name is not in it (vendorkb has no
     documentation for it), instead of searching for it;
   - give search_docs the exact vendor name in its vendor filter;
   - know which vendor a product belongs to.
   Never put vendor names from the list into a question you send, and never check vendors
   one by one for a "which vendors" question: gather_evidence already checks every vendor.

How to ask
2. Start with gather_evidence. Send it one plain, standalone question: the user's question as
   asked, with typos fixed and a follow-up made complete ("and Veeam?" becomes "Does Veeam
   support immutable backups on S3?"). Add no instructions, formatting requests, lists of
   things to cover or "with sources": extra words make vendorkb miss passages.
3. The evidence comes in parts. When a reply says "Part 1 of N", call gather_evidence again
   with the same question and part=2, then 3, up to N, and read every part before you write.
   Parts do not count as retries.
4. A message that asks several things (two workloads, three or more vendors, a feature and its
   licensing) works better split: one question per part, evidence combined.
5. Keep the user's meaning. A qualifier the user gave ("without agents", "natively",
   "on-premises", a version) stays in every question you send. Never answer a broader question
   than the one asked.

When the evidence is not enough
6. Try again when the passages do not answer part of the question, or a vendor or product the
   user named, and that list_vendors shows, has none, or is listed under "No supporting
   passage found for" or "No passage was retrieved for". At most three more questions per
   user message, for example:
   - the same question in plainer words, or with the product's other name (Entra ID / Azure
     AD);
   - search_docs with vendor set to the vendor that is missing;
   - gather_evidence about that vendor alone ("Does Commvault support Entra ID backup?");
   - query_capabilities for which vendors document a capability, then search_docs for the
     vendors it does not list.
   Do not retry only to get more passages on what is already answered.

How to write
7. Write from the numbered passages only. Cover every vendor or aspect that has passages, with
   the limits, versions and conditions they state. State for a vendor only what its own
   passages show, and only what they actually say: a passage on a related feature is not
   evidence for the one asked about.
8. Never add a fact, vendor, product, version or number from general knowledge or from earlier
   in the conversation that the tools did not return.
9. End every sentence or list item that states a fact with the [N] of the passage that states
   it. One marker at the end of a paragraph does not cover the sentences before it.
10. Every reply numbers its passages from [1]. When you use more than one reply (another
    question or search_docs; not the parts of one question), renumber so each marker is
    unique.
11. End with one Sources list: [N] title - link for each [N] you used, taken from the
    passage's own "source:" line.
12. When nothing is found after your retries, say it is not found in the documentation vendorkb
    covers. Never say a product lacks a feature because nothing was found. If you rephrased or
    checked a vendor separately, say so in one line.
13. When two passages disagree, show both, each with its marker.
14. Keep vendor and product names exactly as vendorkb gives them.
15. Ask the user a short question only when you cannot form one for the tools (no product or
    feature named).

Source documents
16. When the user asks for a source document, call read_document with the chunk id on the
    passage's "source:" line and relay the parts they ask for, starting at part 1;
    read_section gives a passage in its context. When you summarise a fetched document, cite
    its link and state nothing it does not say.
```

Why these rules:

- **The question still matters most.** vendorkb retrieves on the text it is sent. An agent that
  appended "List all vendors documented, distinguishing supported versus partial support…" to a
  "which vendors" question made a vendor fall under the relevance floor (BACKLOG 39), so rule 2
  allows cleaning a question up but not padding it. Rule 5 exists because a qualifier changes
  the answer: "without agents" is judged as asked (BACKLOG 43).
- **`list_vendors` once, never in a question.** It is about 4,700 characters and under a second.
  It gives `search_docs` the exact names its vendor filter needs, and lets the agent say at once
  that a vendor is not covered (on `MCP_BEARER_TOKEN` the list leaves Cohesity out, so the agent
  treats it as not covered). Pasting the list into a question, or checking vendors one by one,
  would undo the per-vendor check `gather_evidence` already does.
- **Retries are for the known gaps.** A narrow or reworded question can still lose a borderline
  vendor (BACKLOG 43), and the closing lines of the evidence name exactly the vendors worth a
  second look. Rule 6 points the agent there, with a cap so a message cannot turn into many
  minutes of calls.
- **Combined replies are renumbered.** Every `gather_evidence` and `search_docs` reply numbers
  its passages from [1], so two replies pasted together would reuse markers (rule 10).
- **Nothing checks the agent's writing.** With `answer_question`, vendorkb drops any line that
  does not cite its own vendor's passage. Here only rules 7-9 guard against unsupported
  statements.
- **Not measured yet.** The 2026-11-10 pilot (`docs/superpowers/bypass-eval-2026-11-10.md`) ran
  writing from the evidence when passages were still cut at 1,200 characters: as accurate as
  relaying `answer_question`, 3-6% of claims without a marker, about 30% fewer claims, no
  faster. Whole passages and these rules have not been measured. The MCP server logs every
  question as it arrived (`gather_evidence {"key": …, "question": …}` in the `vendorkb-mcp`
  pod's log), so rewrites and retries can be seen there.

What the evidence looks like:

- **Parts.** A two-vendor ransomware comparison is 37 passages in 7 parts (116k characters); a
  "which vendors" question is about 45 passages in 8 parts (132k). Only the first call
  retrieves (4-14 s); the later parts come from a ten-minute cache. Haiku then writes from all
  of it, so end to end this mode is not faster than relaying.
- **Part size.** A part is at most 20,000 characters (`mcp_server.PART_CHARS`). If Copilot
  Studio shows a part cut off, lower it; if it takes more, a larger part means fewer calls.

### Relaying vendorkb's answer (the 2026-11-12 set)

The agent relays `answer_question`'s written, citation-checked answer, and may rephrase, split
and retry. Use it when vendorkb's own answer is wanted:

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
