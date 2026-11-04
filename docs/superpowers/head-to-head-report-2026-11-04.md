# Head-to-head report: vendorkb (chunk retrieval) vs. the Graphiti answer-api

Run 2026-11-04. Spec: `specs/2026-11-03-vendorkb-head-to-head-design.md`; plan:
`plans/2026-11-03-vendorkb-head-to-head.md`. Raw answers, judge tasks and judgements are in
`eval/headtohead/runs/2026-11-04/`. Questions are in `eval/headtohead/questions.yaml`,
frozen at e443136.

## Outcome

**vendorkb wins: 28 wins to 0, two-sided sign test p = 7.5e-9 (threshold 0.05).**

The decision rule was fixed before the run (spec §7). It covers the 100 new questions. A
question is a win for a side whose answer is perfect when the other side's is not.

| | Questions |
|---|---|
| vendorkb perfect, graph not | 28 (24 single-vendor, 3 multi-article, 1 unanswerable) |
| graph perfect, vendorkb not | 0 |
| both perfect | 22 (19 unanswerable, 3 single-vendor) |
| neither perfect | 50 |

Of the graph's 22 perfect answers, 19 are correct refusals on unanswerable questions. On
answerable questions the graph produced 3 perfect answers and vendorkb 30.

## Setup

- **Both sides use the same models.** Answers come from `z-ai/glm-5.3-flash` at reasoning
  effort `low`, and reranking is Voyage `rerank-3`. The graph side was the deployed pod,
  `ghcr.io/carev01/graph-rag:sha-3b30597`. The runner checked that the graph was frozen
  and that the models matched before starting.
- **vendorkb** indexes 20,899 articles, the same articles the graph extracted, as 32,282
  section chunks (`eval/headtohead/stats.json`). It searches by keyword (Postgres
  full-text) and by vector (pgvector HNSW), merges the two lists with reciprocal rank
  fusion (k = 60), and reranks the top 40 with Voyage.
- **Version drift.** 4,246 of the 20,899 articles changed upstream after the graph was
  built (`changed_since_graph`). vendorkb indexes their current versions. No question
  takes a gold fact or gold source from a changed article, as spec §3 requires.
- **Questions.** There are 119 questions:
  - 100 new deciding questions: 50 single-vendor, 15 multi-article, 15 cross-vendor and 20
    unanswerable. Opus subagents wrote them and the user reviewed them.
  - 19 existing questions, reported but not deciding. The spec planned 36. The other 17
    were dropped because their expected articles had changed or were missing (ruling R22).
- **Judging.** Opus subagents judged blind:
  - The two answers came under shuffled labels.
  - Each citation came with its source document text: the chunk for vendorkb, and for the
    graph, the episode text behind each cited fact.
  - Every question was judged in two passes. Six questions got a third pass because the
    first two disagreed.
- **Truncation.** The judge context was capped at 60,000 characters of source text per
  answer. The cap cut the source text for 6 graph answers and 3 vendorkb answers (pass 1).
  No citation resolved to zero evidence text on either side.

## Results, deciding set (100)

| type / side | n | perfect | fact recall | claim precision | citation precision | abstention correct | gold article cited | mean latency (s) |
|---|---|---|---|---|---|---|---|---|
| all / graph | 100 | 22 | 0.419 | 0.872 | 0.529 | 0.981 | 0.800 | 30.9 |
| all / vendorkb | 100 | **50** | **0.719** | **0.971** | **0.932** | 0.956 | 0.775 | **3.9** |
| single-vendor / graph | 50 | 3 | 0.418 | 0.888 | 0.520 | 0.980 | 0.760 | 20.6 |
| single-vendor / vendorkb | 50 | 27 | 0.771 | 0.994 | 0.961 | 0.912 | 0.740 | 3.5 |
| multi-article / graph | 15 | 0 | 0.337 | 0.867 | 0.344 | 1.000 | 1.000 | 18.5 |
| multi-article / vendorkb | 15 | 3 | 0.744 | 0.982 | 0.875 | 1.000 | 0.867 | 4.9 |
| cross-vendor / graph | 15 | 0 | 0.537 | 0.852 | 0.281 | 1.000 | 0.733 | 86.6 |
| cross-vendor / vendorkb | 15 | 0 | 0.520 | 0.926 | 0.812 | 1.000 | 0.800 | 5.5 |
| unanswerable / graph | 20 | 19 | - | 0.500 | 0.900 | 0.950 | - | 24.3 |
| unanswerable / vendorkb | 20 | 20 | - | - | 1.000 | 1.000 | - | 2.7 |

How to read the columns:
- Fact recall and claim precision are pooled over passes.
- Citation precision is the share of answers in which every citation supports what it is
  attached to.
- Abstention correct is the share of answers that abstained exactly when the question was
  unanswerable.

**Latency.** Median latency was 21.0 s for the graph and 3.4 s for vendorkb. At the 90th
percentile it was 60.8 s and 6.2 s.

**Cost.** vendorkb's answer calls came to $0.0014 per question ($0.167 for the 119
questions), measured. The graph side's cost was not measured: its default tier,
gpt-5-mini on Azure, sits outside the OpenRouter counter. Voyage cost was not measured
for either side.

## Results, existing questions (19, not deciding)

| type / side | n | perfect | fact recall | claim precision | citation precision | abstention correct | gold article cited | mean latency (s) |
|---|---|---|---|---|---|---|---|---|
| all / graph | 19 | 0 | 0.248 | 0.800 | 0.158 | 1.000 | 0.842 | 68.5 |
| all / vendorkb | 19 | 1 | 0.432 | 0.964 | 0.895 | 0.842 | 0.526 | 4.8 |

Per type, both sides were weak on these questions. Their gold facts were written from
older expected-article lists, so many of the articles behind them are now missing or
changed. Read this table as a direction only.

## Why each side loses

Counts are over the 160 pass-1 and pass-2 judgements of the 80 answerable deciding
questions per side.

**Graph: the facts are often there, but the cited text does not back them.**
- 19 of 160 graph answers contain every gold fact. 11 of those 19 still fail, on an
  unsupported claim or a bad citation.
- The usual pattern is a correct fact, cited to an episode whose text does not say it. In
  sv-d-01, for example, all four gold facts are present, but the cited chunks are firewall
  rules and a table of contents.
- On top of that, the graph adds unsupported claims at about 1.2 per answer, against 0.17
  for vendorkb. Its answers are longer: median 965 characters and 7 citations, against 701
  and 3.

**vendorkb: it refuses when retrieval misses.**
- 64 of 160 vendorkb answers contain every gold fact, and every one of the 64 is perfect.
- Its main failure is an incomplete but fully supported answer (67 of 160). Next come 8
  refusals (of the 160 judgements) on answerable questions, such as sv-d-01, sv-d-06 and sv-b-12, where the
  graph's answer had more of the gold facts.
- That is why vendorkb's abstention correctness (0.956) is a little below the graph's
  (0.981).

**Cross-vendor.** Neither side produced a perfect answer on any of the 15 cross-vendor
questions.
- vendorkb usually names the vendors it found and stops there. The graph's
  "Checked / Documented for" line names every vendor, but its claims about them are less
  often supported.
- Part of this comes from the scoring rules, not the systems. See the hand-check below.

**Ties.** Among the 53 answerable questions that tied, vendorkb had the higher fact recall
on 25, the graph on 11, and the two were equal on 17. Counting only perfect answers
therefore understates vendorkb's lead rather than overstating it.

## Illustrative cases

The graph has no wins to show. The first five cases are vendorkb wins, drawn at random
(seed 7). The last five are the questions where the graph's fact recall was furthest
ahead of vendorkb's. The full answers are in `answers.jsonl`.

### vendorkb wins

1. **sv-a-02, AWS Backup: on-premises VMware backups fail with "Abort signal detected".**
   - **vendorkb:** all 4 gold facts in both passes: the 100 Mbps minimum, at most 4 jobs
     per gateway, queued jobs failing once the window passes, and the remedies.
   - **Graph:** lists the causes but never states the 100 Mbps minimum or the queuing
     behaviour. In pass 2, its "increase the backup window" remedy was not in the cited
     text, which recommends more bandwidth.
2. **sv-a-10, Azure Blob backup: version retention vs. backup retention.**
   - **vendorkb:** all 4 facts.
   - **Graph:** correct but incomplete. It omits the vaulted-only guidance and the
     "higher of the two" rule.
3. **sv-b-09, limits of Veeam Agent for Windows backups to object storage (6 gold facts).**
   - **vendorkb:** 6 of 6.
   - **Graph:** 2 of 6, plus two unsupported claims. It applied an "immediately" from the
     network-share case, and turned a 1 MB recommendation into an object-storage limit.
4. **sv-b-11, VB365 modern-auth device code: how long it lasts, and what to do when it
   expires.**
   - **Both:** have the facts (15 minutes, then Refresh).
   - **Graph:** cited the Refresh claim to a basic-auth CAS-server step.
5. **sv-d-03, Google Cloud SQL restore to the original location.**
   - **vendorkb:** 4 of 4.
   - **Graph:** 3 of 4. It omits that by default the restore uses the service account that
     protected the source.

### Graph's best cases (all ties)

1. **sv-d-01, Veeam Recon: editions, endpoints and Coveware users.**
   - **Graph:** all 4 gold facts, but the cited chunks are firewall rules and a table of
     contents, so the judges marked every claim unsupported.
   - **vendorkb:** refused.
2. **sv-d-06, Kasten on NFS volumes without CSI snapshots.**
   - **Graph:** 3 of 4 facts, but the GSB claims cite a multi-cluster passage that says
     nothing about GSB.
   - **vendorkb:** refused. It found the Generic Volume Backup Helm options but not the
     NFS guidance.
3. **cv-08, which vendors can protect IBM AIX.**
   - **Graph:** 4 of 4 facts, with "Documented for: Cohesity, Veeam". Passes 1 and 3
     flagged one claim: FortKnox "recovering" AIX, which its own table marks as
     unsupported.
   - **vendorkb:** covered Veeam well but missed Cohesity.
4. **sv-b-12, rights needed to restore with Veeam Explorer for OneDrive.**
   - **Graph:** had the backup-server roles. In passes 1 and 3 it missed the detail of
     the custom-role minimum.
   - **vendorkb:** found only the launch rights, then refused.
5. **cv-06, which vendors back up Microsoft 365.**
   - **Graph:** 4 of 4 facts, but it cited FortKnox's Microsoft 365 support to a page of
     example chatbot questions.
   - **vendorkb:** covered Veeam and Microsoft, but was silent on Cohesity and AWS.

## Hand-check

The user reviewed 10 questions drawn at random (seed 20261104) and **agreed with 7 of
the 10 final verdicts**:
- Agreed: ma-b-04, ma-b-07, sv-a-12, sv-b-04, sv-c-08, un-01 and un-15.
- Disagreed: cv-03, sv-b-06 and sv-d-08.

All three disagreements were with **gold facts that asked for more than the question
did**. In none of them did the user dispute how the judges applied those facts.

- **cv-03 (which vendors back up Hyper-V).**
  - Cohesity's gold fact rests on FortKnox listing Hyper-V as a workload. But FortKnox is
    a vault target; the Hyper-V backup itself is DataProtect.
  - Leaving AWS out of a "which vendors" answer already says AWS is not covered, yet both
    judges scored it as a missing fact.
  - The user calls vendorkb's answer good, and the graph's worse, because it invented AWS
    support.
- **sv-b-06.** The question asks only what happens when "Restore only latest version" is
  left unticked. The gold fact about the ticked case goes beyond that. Without it, the
  graph's answer is perfect.
- **sv-d-08.** The question asks what happens when the token expires and what period to
  set. The gold fact about the 2-day default for the welcome email is not asked for.
  Without it, vendorkb's answer is perfect in both passes.

**The official score stays as judged.** The rules were fixed before the run. Applying the
user's corrections as a sensitivity check:
- vendorkb gains sv-d-08 and probably cv-03, so the score becomes 29 or 30 to 0.
- Perfect answers: the graph goes from 22 to 23 (sv-b-06), and vendorkb from 50 to 51 or 52.
  vendorkb's hedged answer on sv-b-06 is conservatively not counted as perfect.

The same rubric weakness affects the other cross-vendor questions:
- A vendor with "no evidence in the corpus" counts as missed when an answer simply leaves
  it out.
- Statements like "only these vendors offer it" get flagged as uncited.

Re-scoring the five affected questions (cv-03, 04, 06, 08 and 09) with every
"no evidence" fact counted as found leaves the result at 28 to 0. Both weaknesses cut
against vendorkb, which is the side that leaves vendors out.

## Limits of this result

- **Graph cost and Voyage cost are unmeasured.** The latency gap, 8× at the median, is
  measured.
- **About 20% of the articles (4,246) have a newer version in vendorkb than in the
  graph.** Questions were written only from unchanged articles, but vendorkb could still
  retrieve newer text on topics next to them.
- **One run per side.** Answer variance between runs is not measured. The margin, 28 to
  0, is far beyond what judge noise has produced in this project before.
- **Partial-gold leniency varied by judge.** This is what the third passes were for.
- **The hedge rule.** An item the graph flags "needs validation" earns no gold credit.
  That affected one graph answer (sv-a-01, which scored 0 of 5 gold facts in both passes)
  and does not change any tally.

## Recommendation

- **Adopt chunk retrieval (vendorkb) as the answer path.** Keep the graph parked: no
  re-extraction and no rebuild. On the user's bar of perfect answers, vendorkb gives more
  than twice as many perfect answers (50 vs 22, and 30 vs 3 on answerable questions). It
  is more faithful to its sources (0.93 vs 0.53 citation precision) and about 8× faster,
  and it never lost a question.
- **Next for vendorkb: retrieval recall, then the agent step.**
  - Its losses are refusals and incomplete answers where the right article was not
    retrieved or not used, as in sv-d-01, sv-d-06 and sv-b-12.
  - That is what the deferred agent, which calls `search_docs` / `read_section` itself, is
    meant to address. It is the next experiment, run against this same question set.
- **Fix the question set before reusing it.**
  - Trim gold facts that go beyond the question; the three hand-check disagreements show
    the pattern.
  - Change the cross-vendor rule so that leaving out a "no evidence" vendor is not scored
    as a miss.
- **The graph's real failure is attribution, not knowledge.** Its facts are often right,
  but citations through fact → episode → chunk point at text that does not state them.
  This is worth knowing for any future graph or capability-table work (DESIGN M3–M5):
  claims must keep the exact source span they came from.
