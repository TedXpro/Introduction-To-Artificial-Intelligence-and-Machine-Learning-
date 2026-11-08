# Colab edits: brief v2

These edits change only the **text and code of sections 8 to 13**. Sections 1 to 7 stay exactly as they are.

**What the edits do**

1. The search always shows the writer the 3 biggest groups of copied complaints in the chosen week, company or topic.
2. Python writes the Scale line from the real counts. The AI writes only What, Who and When.
3. Template blanks in [square brackets] are removed before the AI reads a complaint.
4. The old brief stays in the notebook as **brief v1**, and both versions get the same tests, so the scorecard shows v1 against v2 (like the v1 and v2 prompts in Assignment 4).

**How to use this guide**

- There are 15 steps. Do them in order, from top to bottom of the notebook.
- Each step tells you what to search for. Press **Ctrl+F** in Colab and type that text to jump to the cell. If that doesn't find it, use **Edit → Find and replace** (Ctrl+H) instead. Every search text appears in exactly one cell.
- In every step except step 11, you replace the **whole** cell: select everything in it and paste the new text.
- Use the copy button on each grey block, so nothing is missed.
- Don't run anything until all 15 steps are done.

---

## Step 1: Section 8 introduction

**Find it:** search for `Explain: why is this problem surging`

It is a **text cell**. Double-click the text to open it, click inside the edit box, press **Ctrl+A**, then **Delete**, paste the text below, and press **Shift+Enter**.

```markdown
## 8. Explain: why is this problem surging? (main method 3)

**What it does.** When a topic alarm fires, a manager wants to know what is behind it before deciding what to do. They click the alarm, or ask in plain English, for example *"What were Capital One customers complaining about in the week of 13 January?"*, and the assistant answers from that week's complaints and cites them. It works like our Assignment 3 RAG pipeline:

1. **Agent** (Qwen3-4B, Session 8): reads the question and returns JSON: whether to search, a short search query, and the filters the question asks for (dates, product, company, topic).
2. **Filtered search** (Session 7): FAISS on the sentence embeddings from section 4, limited with `IDSelectorBatch` to the complaints that match the filters, plus BM25 on the same complaints. A cross-encoder re-ranks the candidates, as in Assignment 3.
3. **"I don't know"**: if no complaint matches the filters, or even the best one is not relevant enough, the assistant says so instead of guessing.
4. **Brief** (Qwen3-4B): what customers report, which companies and when, citing complaints as [1], [2], ... Copies of one template are shown once, with the number of times they were filed. The Scale line (how many complaints match, how many are copies) is counted from the database, not written by the model.

**What we learned from our first version.** Our first RAG assistant wrote a brief for every single complaint, ending with a "Next step" line of advice for the agent. Its RAGAs faithfulness was 0.75, below the 0.80 target, and 0.94 when that line was left out: advice cannot be backed by any complaint. The new brief states only facts from the complaints; deciding what to do stays with the manager. For each single complaint, the agent now gets a case card with real records instead (8.2).

**Why an open model writes and GPT only grades.** In production Qwen3-4B runs on the bank's own GPU, so complaint text does not leave the bank. `gpt-4o-mini` is only used to evaluate, on public CFPB text, and a model should not grade its own answers.

**Evaluation.**

- **Search engine** (8.1): precision@5 for finding past cases with the same issue, compared with keyword search, plain FAISS, a filter on the routed team, hybrid search, and the Word2Vec document vectors from our midterm.
- **Agent** (8.3): a test suite in the style of Assignment 4: 16 questions with known correct filters, a first prompt (v1) against an improved prompt (v2), and the share of checks each passes.
- **Briefs** (8.5): two brief versions (v1 and v2, see 8.4), one brief per alarm, with and without the filters. Does it name the company behind the surge (taken from the complaint records, never given to the model as a label)? Are the three January events named in that week's briefs? Then RAGAs faithfulness and answer relevancy, and an LLM judge spot-checked by a person.
```

## Step 2: 8.4 text

**Find it:** search for `8.4 Filtered search and the brief`

It is a **text cell**. Double-click the text to open it, click inside the edit box, press **Ctrl+A**, then **Delete**, paste the text below, and press **Shift+Enter**.

```markdown
### 8.4 Filtered search and the brief

The search works like Assignment 3's `retrieve()`. The agent's filters select the allowed complaints; FAISS (with `IDSelectorBatch`) and BM25 each propose candidates from those complaints only; copies of the same template are kept once; and the cross-encoder re-ranks the candidates against the question and the agent's keyword query, keeping the better score.

**When the answer is "I don't know based on the complaints".** If no complaint matches the filters (for example a question about 2023), there is nothing to answer from. If the question sets no filters at all, Assignment 3's relevance threshold decides: the best candidate must score at least −3. We checked this threshold on real complaints before using it: the re-ranker scores a question that names the right company or problem above 0 and an unrelated question around −11, but it also scores open questions such as "what were Chime customers complaining about?" around −8 even on the right complaints. So when filters are set, the complaints that match them are the answer, and the threshold is not applied.

**Brief v1 and brief v2.** Our first full run showed three problems with the first brief (v1). The judge and the human check found them:

| Problem in brief v1 | Example from the first run | Change in brief v2 |
|---|---|---|
| The biggest group was missed | The 13 Jan "Late payment fees" alarm was 84% Navy Federal<br>overdraft letters, but the brief named other banks | The 3 largest groups of copies that match the filters<br>(each at least 2% of them) are always shown to the writer |
| The Scale line was wrong | "Scale: 1, 1, 1, 1", or "2 times" when every complaint<br>was filed once | Python counts the complaints and copies in the database<br>and writes the Scale line; the model writes the other lines |
| Template blanks read as facts | A blank such as "[describe the specific deceptive practice,<br>e.g. unauthorized charges]" was reported as a customer's words | Blanks in square brackets are removed before the model<br>reads the complaint |

Both versions use the same filters and the same "I don't know" rule, and both prompts allow only facts from the numbered complaints and no advice. Both are kept and scored with the same tests in 8.5, as with the agent prompts above and the v1 and v2 prompts of Assignment 4. Brief v2 was designed after reading the v1 briefs of these same alarms, so its scores here are development results: a pilot has to confirm them on new weeks.
```

## Step 3: 8.4 code (search and brief)

**Find it:** search for `RELEVANCE_THRESHOLD = -3.0`

It is a **code cell**. Click inside the code, press **Ctrl+A**, then **Delete**, and paste the code below. Do not run it yet.

```python
bm25_all = BM25Okapi(df["tokens"].tolist())
RELEVANCE_THRESHOLD = -3.0
NO_ANSWER = "I don't know based on the complaints."
NO_FILTERS = {"date_from": None, "date_to": None, "product": None, "company": None, "topic": None}
GROUP_OF_ROW = df["group"].to_numpy()
BIG_GROUP_SLOTS = 3
BIG_GROUP_SHARE = 0.02
BIG_GROUP_MIN = max(3, round(10 * SCALE))
PLACEHOLDER = re.compile(r"\[[^\]]*(?:\b(?:describe|insert|enter|specify|explain|your)\b|e\.g\.)[^\]]*\]",
                         re.IGNORECASE)


def allowed_rows(filters):
    mask = np.ones(len(df), dtype=bool)
    if filters["date_from"] is not None:
        mask &= (df["date_received"] >= filters["date_from"]).to_numpy()
    if filters["date_to"] is not None:
        mask &= (df["date_received"] <= filters["date_to"]).to_numpy()
    if filters["product"] is not None:
        mask &= (df["product"] == filters["product"]).to_numpy()
    if filters["company"] is not None:
        mask &= df["company"].isin(resolve_companies(filters["company"])).to_numpy()
    if filters["topic"] is not None:
        mask &= (df["topic"] == filters["topic"]).to_numpy()
    return np.where(mask)[0].astype("int64")


def group_counts(allowed):
    return pd.Series(GROUP_OF_ROW[allowed]).value_counts()


def big_groups(allowed):
    counts = group_counts(allowed)
    counts = counts[counts >= max(BIG_GROUP_MIN, BIG_GROUP_SHARE * len(allowed))]
    return [int(allowed[np.argmax(GROUP_OF_ROW[allowed] == g)]) for g in counts.index[:BIG_GROUP_SLOTS]]


def search_complaints(question, search_query, allowed, top_k=6, n_candidates=30, keep_big_groups=False):
    if len(allowed) == 0:
        return []
    selector = faiss.IDSelectorBatch(len(allowed), faiss.swig_ptr(allowed))
    candidates = []
    for query in dict.fromkeys(q for q in (question, search_query) if q):
        vector = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        _, ids = index_all.search(vector, min(n_candidates, len(allowed)), params=faiss.SearchParameters(sel=selector))
        candidates += [int(i) for i in ids[0] if i != -1]
        scores = bm25_all.get_scores(list(bigram[tokenize(clean_text(query))]))[allowed]
        candidates += [int(allowed[i]) for i in np.argsort(-scores)[:n_candidates] if scores[i] > 0]
    must = big_groups(allowed) if keep_big_groups else []
    unique_hits, seen = [], set()
    for i in dict.fromkeys(must + candidates):
        if GROUP_OF_ROW[i] not in seen:
            seen.add(GROUP_OF_ROW[i])
            unique_hits.append(i)
    if not unique_hits:
        return []
    texts = [" ".join(df.at[i, "text"].split()[:200]) for i in unique_hits]
    relevance = reranker.predict([(question, text) for text in texts])
    if search_query and search_query != question:
        relevance = np.maximum(relevance, reranker.predict([(search_query, text) for text in texts]))
    copies = group_counts(allowed)
    order = list(range(len(must))) + [j for j in np.argsort(-relevance) if j >= len(must)]
    return [{"row": unique_hits[j], "score": float(relevance[j]), "copies": int(copies[GROUP_OF_ROW[unique_hits[j]]]),
             "big group": j < len(must)} for j in order[:top_k]]


BRIEF_PROMPT_V1 = """You help a bank's complaints manager understand a group of customer complaints.
Use ONLY the numbered complaints below. Each one shows its company, its date, its product and how many times
the same text was filed in the period searched.
Write exactly these four lines and nothing else:
What customers report: the main problem in one or two sentences, citing complaints like [1] or [3].
Who: the companies named in the complaints, the most frequent first.
When: the dates of the complaints.
Scale: how many times the main texts were filed, using the numbers shown.
Do not give advice. Do not add any fact that is not in the complaints.
Describe what these complaints say, even if they answer the question only in part.

Complaints:
{context}"""

BRIEF_PROMPT_V2 = """You help a bank's complaints manager understand a group of customer complaints.
Use ONLY the numbered complaints below. Each one shows its company, its date, its product and how many times
the same text was filed in the period searched. A text filed many times is a large group of customers.
Write exactly these three lines and nothing else:
What customers report: the main problems in one or two sentences, starting with the texts filed most often,
citing complaints like [1] or [3].
Who: the companies in the complaints, the one with the most filings first.
When: the dates of the complaints.
Do not give advice. Do not add any fact that is not in the complaints.
Describe what these complaints say, even if they answer the question only in part.

Complaints:
{context}"""

SMALL_TALK = ("You are the assistant of a bank's complaints team. Reply in one or two sentences. Do not state any "
              "facts. Invite the user to ask about the customer complaints of 2025.")


def context_parts(hits, remove_blanks=False):
    parts = []
    for n, h in enumerate(hits, start=1):
        row = h["row"]
        text = PLACEHOLDER.sub(" ", df.at[row, "text"]) if remove_blanks else df.at[row, "text"]
        parts.append(f"[{n}] {df.at[row, 'company']} | {df.at[row, 'date_received']:%d %b %Y} | "
                     f"{SHORT[df.at[row, 'product']]} | filed {h['copies']} time{'s' if h['copies'] > 1 else ''}\n"
                     + " ".join(text.split()[:150]))
    return parts


def scale_line(allowed, filtered):
    if not filtered:
        return f"Scale: no filters were set, so all {len(df):,} complaints were searched."
    counts = group_counts(allowed)
    shared = int(counts[counts > 1].sum())
    line = f"Scale: {len(allowed):,} complaint{'s' if len(allowed) > 1 else ''} match the filters"
    if shared:
        line += (f"; {shared / len(allowed):.0%} of them share their text with another complaint, "
                 f"and the largest group was filed {int(counts.iloc[0]):,} times")
    return line + "."


def has_filters(filters):
    return any(filters[key] is not None for key in NO_FILTERS)


def run_explain(question, use_filters=True, decision=None, check_relevance=True, version=2):
    decision = decision or agent_decide(question)
    result = {"question": question, "decision": decision, "version": version, "hits": [], "parts": [],
              "scale": None, "written": NO_ANSWER, "answer": NO_ANSWER}
    if not decision["retrieve"]:
        result["written"] = result["answer"] = generate(
            [{"role": "system", "content": SMALL_TALK}, {"role": "user", "content": question}], max_new_tokens=60)
        return result
    filters = decision if use_filters else NO_FILTERS
    filtered = has_filters(filters)
    allowed = allowed_rows(filters)
    hits = search_complaints(question, decision["search_query"], allowed, keep_big_groups=(version == 2 and filtered))
    needs_threshold = check_relevance and not filtered
    if hits and (not needs_threshold or max(h["score"] for h in hits) >= RELEVANCE_THRESHOLD):
        result["hits"] = hits
        result["parts"] = context_parts(hits, remove_blanks=(version == 2))
        prompt = BRIEF_PROMPT_V2 if version == 2 else BRIEF_PROMPT_V1
        written = generate([{"role": "system", "content": prompt.format(context="\n\n".join(result["parts"]))},
                            {"role": "user", "content": question}], max_new_tokens=250)
        if version == 2:
            written = "\n".join(line for line in written.splitlines() if not re.match(r"\W*scale\b", line, re.IGNORECASE))
            result["scale"] = scale_line(allowed, filtered)
        result["written"] = written.strip()
        result["answer"] = result["written"] + (f"\n{result['scale']}" if result["scale"] else "")
    return result


def describe_filters(decision):
    parts = []
    if decision["date_from"] is not None or decision["date_to"] is not None:
        start = f"{decision['date_from']:%d %b %Y}" if decision["date_from"] is not None else "any date"
        end = f"{decision['date_to']:%d %b %Y}" if decision["date_to"] is not None else "any date"
        parts.append(f"dates {start} to {end}")
    if decision["product"] is not None:
        parts.append(f"product {SHORT[decision['product']]}")
    if decision["company"] is not None:
        parts.append(f"company {decision['company']}")
    if decision["topic"] is not None:
        parts.append(f"topic {topic_names[decision['topic']]}")
    return ", ".join(parts) or "no filters"


def show_answer(result):
    decision = result["decision"]
    text = (f"**Question:** {result['question']}\n\n"
            f"**Agent:** search = {decision['retrieve']}, query = \"{decision['search_query']}\", "
            f"{describe_filters(decision)}\n\n" + result["answer"].replace("\n", "\n\n"))
    if result["hits"]:
        text += "\n\n**Sources:** " + ", ".join(f"[{n}] complaint {df.at[h['row'], 'complaint_id']}"
                                               for n, h in enumerate(result["hits"], start=1))
    display(Markdown(text.replace("$", "USD ")))
```

## Step 4: 8.5 text

**Find it:** search for `8.5 Does it explain real surges`

It is a **text cell**. Double-click the text to open it, click inside the edit box, press **Ctrl+A**, then **Delete**, paste the text below, and press **Shift+Enter**.

```markdown
### 8.5 Does it explain real surges?

On the dashboard a manager clicks an alarm, so the alarm itself gives the filters: its topic and its week. The question is *"What were customers complaining about under the topic [topic] in the week of [date]?"*, and the topic's top words are the search query. Each alarm is answered three times: brief v2 with the alarm's filters, brief v1 with the same filters (to see what v2 changes), and brief v2 with no filters at all, the whole year (to see what the filters add, as we did with the source and date filters in Assignment 3). The relevance threshold is skipped in all three, because the alarm already says which complaints need explaining. Free-text questions go through the agent, as in the demo above and the event checks below.

Two automatic checks use facts from the complaint records that the model is never given as labels:

- **Right company:** the company with the most complaints in that topic and week must appear in the lines the model wrote.
- **Right week:** the share of the brief's sources that come from the alarm week.
```

## Step 5: 8.5 code (surge briefs)

**Find it:** search for `MAX_ALARM_BRIEFS`

It is a **code cell**. Click inside the code, press **Ctrl+A**, then **Delete**, and paste the code below. Do not run it yet.

```python
MAX_ALARM_BRIEFS = 12 if QUICK_RUN else 30
BRAND_KEYS = {"Block, Inc.": ["cash app", "cashapp", "block"], "Early Warning Services, LLC": ["zelle", "early warning"],
              "JPMORGAN CHASE & CO.": ["chase", "jpmorgan"], "CITIBANK, N.A.": ["citi", "citibank"],
              "U.S. BANCORP": ["u.s. bank", "us bank", "u.s. bancorp"],
              "UNITED SERVICES AUTOMOBILE ASSOCIATION": ["usaa", "united services"]}
BRIEF_RUNS = [("v2 with filters", 2, True), ("v2 no filters", 2, False), ("v1 with filters", 1, True)]


def company_keys(company):
    words = name_words(company)
    return BRAND_KEYS.get(company) or ([words[0]] if words else [company.lower()])


def mentions(text, keys):
    text = text.lower()
    return any(re.search(r"\b" + re.escape(k) + r"\b", text) for k in keys)


surge_rows = []
if RUN_QWEN and len(alarm_review):
    todo = (alarm_review.sort_values("times normal", ascending=False).head(MAX_ALARM_BRIEFS)
            .sort_values(["week", "topic"]))
    start = time.time()
    for _, alarm in todo.iterrows():
        question = (f"What were customers complaining about under the topic '{alarm['topic name']}' "
                    f"in the week of {alarm['week']:%d %B %Y}?")
        decision = {"retrieve": True, "search_query": " ".join(top_words[alarm["topic"]][:8]),
                    "date_from": alarm["week"], "date_to": alarm["week"] + pd.Timedelta(days=6), "product": None,
                    "company": None, "topic": alarm["topic"], "valid_json": True, "reply": ""}
        for mode, version, use_filters in BRIEF_RUNS:
            result = run_explain(question, use_filters, decision, check_relevance=False, version=version)
            hit_rows = [h["row"] for h in result["hits"]]
            surge_rows.append({
                "week": alarm["week"], "topic name": alarm["topic name"], "main company": alarm["main company"],
                "mode": mode, "question": question, "filters": describe_filters(decision),
                "answer": result["answer"], "scale": result["scale"], "parts": result["parts"],
                "names main company": mentions(result["written"], company_keys(alarm["main company"])),
                "sources from alarm week": float(np.mean([df.at[r, "week"] == alarm["week"] for r in hit_rows]))
                                           if hit_rows else 0.0,
                "answered": bool(result["hits"]) and NO_ANSWER.lower() not in result["written"].lower(),
            })
    print(f"{len(todo)} alarms, {len(BRIEF_RUNS)} briefs each, in {time.time() - start:.0f} s")
surge = pd.DataFrame(surge_rows)

COMPANY_HIT_FILTERED = COMPANY_HIT_UNFILTERED = COMPANY_HIT_V1 = WEEK_SHARE_FILTERED = WEEK_SHARE_UNFILTERED = None
if len(surge):
    surge_summary = (surge.groupby("mode")[["names main company", "sources from alarm week", "answered"]].mean()
                     .reindex([mode for mode, _, _ in BRIEF_RUNS]))
    COMPANY_HIT_FILTERED, COMPANY_HIT_UNFILTERED, COMPANY_HIT_V1 = surge_summary["names main company"]
    WEEK_SHARE_FILTERED, WEEK_SHARE_UNFILTERED = surge_summary["sources from alarm week"].iloc[:2]
    display(surge_summary.style.format("{:.0%}"))
    v2_rows = surge[surge["mode"] == "v2 with filters"].reset_index(drop=True)
    v1_rows = surge[surge["mode"] == "v1 with filters"].reset_index(drop=True)
    display(pd.DataFrame({"week": v2_rows["week"].dt.strftime("%d %b %Y"), "topic name": v2_rows["topic name"],
                          "main company": v2_rows["main company"], "named by brief v1": v1_rows["names main company"],
                          "named by brief v2": v2_rows["names main company"]}))

    fig, ax = plt.subplots(figsize=(10, 4.4))
    metrics = ["names main company", "sources from alarm week"]
    y = np.arange(len(metrics))
    bars = [("v2 with filters", SERIES[0], "Brief v2, agent's filters"),
            ("v1 with filters", SERIES[1], "Brief v1, agent's filters"),
            ("v2 no filters", MUTED, "No filters (whole year)")]
    for k, (mode, color, label) in enumerate(bars):
        offset = (k - 1) * 0.27
        values = surge_summary.loc[mode, metrics].to_numpy(dtype=float)
        ax.barh(y + offset, values, height=0.25, color=color, label=label)
        for yy, value in zip(y, values):
            ax.text(value + 0.01, yy + offset, f"{value:.0%}", va="center", color=INK_2, fontsize=9)
    ax.set_yticks(y, ["Brief names the company\nbehind the surge", "Sources come from\nthe alarm week"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.set_title(f"Briefs naming the company behind the surge: v2 {COMPANY_HIT_FILTERED:.0%}, "
                 f"v1 {COMPANY_HIT_V1:.0%}, no filters {COMPANY_HIT_UNFILTERED:.0%}")
    save_fig(fig, "fig13_briefs_with_and_without_filters")
else:
    print("Surge briefs skipped (needs RUN_QWEN = True and at least one alarm)")
```

## Step 6: January events text

**Find it:** search for `The January events.`

It is a **text cell**. Double-click the text to open it, click inside the edit box, press **Ctrl+A**, then **Delete**, paste the text below, and press **Shift+Enter**.

```markdown
**The January events.** For each dated event, a manager asks an open question that names the company and the week but not the cause. The brief passes if the lines written by the model name the cause (keywords checked automatically; the counted Scale line is left out). Both brief versions are checked. The October copy-paste campaign from section 7 is checked the same way, as an extra row.
```

## Step 7: January events code

**Find it:** search for `EVENT_CHECKS = [`

It is a **code cell**. Click inside the code, press **Ctrl+A**, then **Delete**, and paste the code below. Do not run it yet.

```python
EVENT_CHECKS = [
    ("January event", "Outage at FIS delays Capital One deposits (15–19 Jan)",
     "What were Capital One customers complaining about in the week of 13 January 2025?", ["deposit", "deposits"]),
    ("January event", "CFPB sues Capital One over 360 Savings (14 Jan)",
     "What did Capital One savings account customers report in the week of 13 January 2025?",
     ["360", "interest", "rate", "rates"]),
    ("January event", "CFPB orders Block (Cash App) to fix fraud handling (16 Jan)",
     "What were Cash App customers complaining about in the week of 13 January 2025?",
     ["fraud", "unauthorized", "dispute", "disputes", "customer service", "support", "refund"]),
    ("October campaign", "Copy-paste dispute letter against Zelle and Chime",
     "What were Chime customers complaining about in the week of 6 October 2025?", ["dispute", "disputes", "lawsuit"]),
]

EVENTS_FOUND = EVENTS_FOUND_UNFILTERED = EVENTS_FOUND_V1 = None
event_results = []
if RUN_QWEN:
    for kind, event, question, keywords in EVENT_CHECKS:
        decision = agent_decide(question)
        row = {"kind": kind, "event": event, "question": question, "filters": describe_filters(decision)}
        for mode, version, use_filters in BRIEF_RUNS:
            result = run_explain(question, use_filters, decision, version=version)
            row[f"found ({mode})"] = mentions(result["written"], keywords)
            row[f"brief ({mode})"] = result["answer"]
            row[f"parts ({mode})"] = result["parts"]
            row[f"scale ({mode})"] = result["scale"]
        event_results.append(row)
    event_table = pd.DataFrame(event_results)
    january = event_table["kind"] == "January event"
    EVENTS_FOUND, EVENTS_FOUND_UNFILTERED, EVENTS_FOUND_V1 = [int(event_table.loc[january, f"found ({mode})"].sum())
                                                              for mode, _, _ in BRIEF_RUNS]
    display(event_table[["kind", "event", "filters"] + [f"found ({mode})" for mode, _, _ in BRIEF_RUNS]])
    print(f"January events named in the briefs: brief v2 {EVENTS_FOUND} of 3, brief v1 {EVENTS_FOUND_V1} of 3, "
          f"no filters {EVENTS_FOUND_UNFILTERED} of 3")
    for row in event_results:
        display(Markdown(f"**{row['event']}**\n\n*{row['question']}*\n\n"
                         + "**Brief v1:**\n\n" + row["brief (v1 with filters)"].replace("\n", "\n\n").replace("$", "USD ")
                         + "\n\n**Brief v2:**\n\n" + row["brief (v2 with filters)"].replace("\n", "\n\n").replace("$", "USD ")))
else:
    print("Event checks skipped: Qwen3-4B needs a GPU runtime (RUN_QWEN = False)")
```

## Step 8: RAGAs text

**Find it:** search for `RAGAs and the LLM judge`

It is a **text cell**. Double-click the text to open it, click inside the edit box, press **Ctrl+A**, then **Delete**, paste the text below, and press **Shift+Enter**.

```markdown
### RAGAs and the LLM judge

As in Assignment 3, RAGAs scores **faithfulness** (is every statement in the brief backed by the complaints?) and **answer relevancy**, with `gpt-4o-mini` as the judge (Session 11). Every brief written with filters is scored, for both brief versions: one per alarm, plus the event questions. For brief v2, the database count behind the Scale line is given to RAGAs and to the judge as one more context, so that line can be checked too.
```

## Step 9: RAGAs code

**Find it:** search for `graded = []`

It is a **code cell**. Click inside the code, press **Ctrl+A**, then **Delete**, and paste the code below. Do not run it yet.

```python
def with_scale(parts, scale):
    return list(parts) + ([scale] if isinstance(scale, str) else [])


def graded_briefs(mode):
    items = []
    if len(surge):
        for _, r in surge[(surge["mode"] == mode) & surge["answered"]].iterrows():
            items.append({"question": r["question"], "answer": r["answer"], "parts": with_scale(r["parts"], r["scale"])})
    for r in event_results:
        if r[f"brief ({mode})"] != NO_ANSWER:
            items.append({"question": r["question"], "answer": r[f"brief ({mode})"],
                          "parts": with_scale(r[f"parts ({mode})"], r[f"scale ({mode})"])})
    return items


graded = graded_briefs("v2 with filters")
graded_v1 = graded_briefs("v1 with filters")

FAITHFULNESS = RELEVANCY = FAITHFULNESS_V1 = RELEVANCY_V1 = None
ragas_table = ragas_table_v1 = None
if graded and RUN_OPENAI:
    from ragas import evaluate, EvaluationDataset, RunConfig
    from ragas.metrics import faithfulness, answer_relevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_core.embeddings import Embeddings
    from langchain_openai import ChatOpenAI

    judge = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))

    class MpnetEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return embedder.encode(texts, normalize_embeddings=True).tolist()

        def embed_query(self, text):
            return embedder.encode(text, normalize_embeddings=True).tolist()

    def ragas_scores(items):
        dataset = EvaluationDataset.from_list([
            {"user_input": g["question"], "response": g["answer"], "retrieved_contexts": g["parts"]} for g in items])
        scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy], llm=judge,
                          embeddings=LangchainEmbeddingsWrapper(MpnetEmbeddings()),
                          run_config=RunConfig(max_workers=2, timeout=600, max_retries=5, max_wait=60))
        table = scores.to_pandas()
        table.insert(0, "question", [g["question"] for g in items])
        return table

    ragas_table = ragas_scores(graded)
    FAITHFULNESS, RELEVANCY = ragas_table["faithfulness"].mean(), ragas_table["answer_relevancy"].mean()
    if graded_v1:
        ragas_table_v1 = ragas_scores(graded_v1)
        FAITHFULNESS_V1, RELEVANCY_V1 = ragas_table_v1["faithfulness"].mean(), ragas_table_v1["answer_relevancy"].mean()
    ragas_compare = pd.DataFrame({"briefs": [len(graded_v1), len(graded)],
                                  "faithfulness": [FAITHFULNESS_V1, FAITHFULNESS],
                                  "answer relevancy": [RELEVANCY_V1, RELEVANCY]}, index=["brief v1", "brief v2"])
    print(f"RAGAs over {len(graded)} briefs (v2): faithfulness {FAITHFULNESS:.2f}, answer relevancy {RELEVANCY:.2f}")
    display(ragas_compare.style.format({"faithfulness": "{:.2f}", "answer relevancy": "{:.2f}"}, na_rep="–"))
    display(ragas_table[["question", "faithfulness", "answer_relevancy"]]
            .style.format({"faithfulness": "{:.2f}", "answer_relevancy": "{:.2f}"}))
else:
    print("RAGAs skipped (needs briefs and RUN_OPENAI = True)")
```

## Step 10: Judge code

**Find it:** search for `JUDGE_SYSTEM`

It is a **code cell**. Click inside the code, press **Ctrl+A**, then **Delete**, and paste the code below. Do not run it yet.

```python
JUDGE_SYSTEM = """You review short briefs written for a bank's complaints manager about a group of complaints. Be strict.
You get the manager's question, the complaints the brief was written from, then the brief.
When the complaints end with a line that starts with "Scale:", that line is a count taken from the complaint
database: treat it as correct.
First list every statement in the brief that is not supported by the complaints.
Then score usefulness for the manager on this scale:
1 = wrong or misleading, 2 = vague or generic, 3 = correct but the manager still has to read the complaints,
4 = correct and specific: the manager understands what the complaints are about,
5 = correct, specific and complete: what happened, which companies, when and how many.
Return JSON with these keys:
"unsupported_statements": a list of strings (empty if none),
"usefulness": 1 to 5,
"reason": one short sentence."""


def judge_briefs(items):
    verdicts = []
    for n, g in enumerate(items):
        reply, _, _ = ask_openai(JUDGE_SYSTEM, f"QUESTION:\n{g['question']}\n\nCOMPLAINTS:\n" + "\n\n".join(g["parts"])
                                 + "\n\nBRIEF:\n" + g["answer"], max_tokens=400, json_mode=True)
        result = json.loads(reply)
        problems = result.get("unsupported_statements") or []
        verdicts.append({"brief": n, "question": g["question"], "usefulness": int(result.get("usefulness", 0)),
                         "unsupported statements": len(problems), "example": problems[0] if problems else "",
                         "reason": result.get("reason", "")})
    return pd.DataFrame(verdicts).set_index("brief")


JUDGE_MEAN = JUDGE_MEAN_V1 = UNSUPPORTED = UNSUPPORTED_V1 = None
judge_table = judge_table_v1 = None
if graded and RUN_OPENAI:
    judge_table = judge_briefs(graded)
    JUDGE_MEAN = judge_table["usefulness"].mean()
    UNSUPPORTED = int((judge_table["unsupported statements"] > 0).sum())
    if graded_v1:
        judge_table_v1 = judge_briefs(graded_v1)
        JUDGE_MEAN_V1 = judge_table_v1["usefulness"].mean()
        UNSUPPORTED_V1 = int((judge_table_v1["unsupported statements"] > 0).sum())
    judge_compare = pd.DataFrame({
        "briefs": [len(graded_v1), len(graded)],
        "mean usefulness (1 to 5)": [JUDGE_MEAN_V1, JUDGE_MEAN],
        "briefs with an unsupported statement": [UNSUPPORTED_V1, UNSUPPORTED]}, index=["brief v1", "brief v2"])
    display(judge_compare.style.format({"mean usefulness (1 to 5)": "{:.2f}"}, na_rep="–"))
    print(f"Brief v2: usefulness {JUDGE_MEAN:.2f} / 5 (range {judge_table['usefulness'].min()}–"
          f"{judge_table['usefulness'].max()}), briefs with unsupported statements: {UNSUPPORTED} of {len(judge_table)}")
    display(judge_table)
else:
    print("Judge skipped (needs briefs and RUN_OPENAI = True)")
```

## Step 11: Clear your old brief scores

**Find it:** search for `HUMAN_USEFULNESS = {0:`

It is a **code cell**. Change **only the first line**. Your old scores are for the old briefs, so the cell has to start empty again.

Change this line:

```python
HUMAN_USEFULNESS = {0: 3, 1: 1, 2: 1, 3: 4, 4: 5}
```

to this:

```python
HUMAN_USEFULNESS = {}
```

Leave the rest of the cell as it is.

## Step 12: Section 9 Responsible AI

**Find it:** search for `9. Responsible AI`

It is a **text cell**. Double-click the text to open it, click inside the edit box, press **Ctrl+A**, then **Delete**, paste the text below, and press **Shift+Enter**.

```markdown
## 9. Responsible AI

| Risk | What we did |
|---|---|
| Privacy | The CFPB removes personal details before publishing (XXXX).<br>In production the open Qwen3-4B model runs inside the bank, so<br>complaint text is not sent to an outside service. `gpt-4o-mini` is<br>only used here, on public CFPB text, to grade the results. |
| Unfair treatment | The payout model is checked for Older American and<br>Servicemember complaints (section 6) against the 5-point target. |
| Made-up answers | Briefs use only the retrieved complaints and cite them. Counts<br>come from the database, not the model, and template blanks<br>are removed. The assistant says "I don't know" when nothing<br>relevant is found. RAGAs, the judge and a person check for<br>unsupported statements. |
| Automation without oversight | Every output is a suggestion: an agent confirms the route and<br>reviews the risk, and a manager decides what an alarm means<br>and what to do about it. |
| Campaigns | A copy-paste campaign is still real customers using their rights.<br>Copies are grouped for one reviewed answer, never discarded. |
| Label quality | Products and issues are chosen by consumers, so some labels<br>are wrong. Scores are a fair estimate, not an exact one. |
| Regulation | Complaint handling is not a high-risk use under the EU AI Act;<br>it would need human oversight and logging, which this design<br>already has. |
```

## Step 13: Section 10 scorecard code

**Find it:** search for `def verdict(ok):`

It is a **code cell**. Click inside the code, press **Ctrl+A**, then **Delete**, and paste the code below. Do not run it yet.

```python
def verdict(ok):
    if isinstance(ok, str):
        return ok
    return "not measured yet" if ok is None else ("✅ met" if ok else "❌ not met")


def shown(value, spec):
    return "–" if value is None else format(value, spec)


TUNED_NMI = best_setting["agreement with CFPB issues (NMI)"]
hidden_text = "none"
if SHOWCASE_WEEK is not None:
    hidden_text = (f"{len(HIDDEN_FROM_COUNTS)} weeks; e.g. {SHOWCASE_WEEK:%d %b}: all complaints "
                   f"{lens.at[SHOWCASE_WEEK, 'all complaints']:.1f}×, one topic {lens.at[SHOWCASE_WEEK, 'busiest topic']:.0f}×")
scorecard = pd.DataFrame([
    ("Clean", "Copy pairs confirmed by a person", "–", "≥ 18 of 20",
     f"{HUMAN_CONFIRMED} of {len(human_check)}" if HUMAN_CONFIRMED is not None else "–",
     None if HUMAN_CONFIRMED is None else HUMAN_CONFIRMED >= 18),
    ("Route", "Accuracy on held-out complaints", f"Keyword rules {ACC_KW:.0%}", "≥ 80% and above keywords",
     f"{ACC_ROUTER:.0%} (macro-F1 {F1_ROUTER:.2f})", bool(ACC_ROUTER >= 0.8 and ACC_ROUTER > ACC_KW)),
    ("Prioritise", "Payouts caught in the riskiest 20%", "Random 20%", "≥ 40%",
     f"{CAUGHT_20:.0%}", bool(CAUGHT_20 >= 0.4)),
    ("Watch", "Topic coherence (c_v)", f"Untuned {UNTUNED_COHERENCE:.2f}", "≥ 0.40 and above untuned",
     f"{LDA_COHERENCE:.2f}", bool(LDA_COHERENCE >= 0.4 and LDA_COHERENCE > UNTUNED_COHERENCE)),
    ("Watch", "Agreement with CFPB issues (NMI)", f"Untuned {UNTUNED_NMI:.2f}", "no target (diagnostic)",
     f"{TUNED_NMI:.2f}", "ℹ️ diagnostic"),
    ("Watch", "Alarm within 7 days of the January events", "–", "yes", "yes" if ALARM_OK else "no", ALARM_OK),
    ("Watch", "Weeks that only the topic alarm catches", "Counting all complaints or per team",
     "no target (diagnostic)", hidden_text, "ℹ️ diagnostic"),
    ("Watch", "Alarms explained by a dated event or a copy campaign", "–", "no target (diagnostic)",
     f"{ALARMS_EXPLAINED} of {len(alarm_review)}", "ℹ️ diagnostic"),
    ("Explain", "Precision@5 (same issue), best search", f"BM25 {P5_BM25:.2f}", "above BM25",
     f"{P5_OURS:.2f} ({P5_BEST_NAME})", bool(P5_OURS > P5_BM25)),
    ("Explain", "Agent test checks passed", f"Prompt v1 {shown(AGENT_V1, '.0%')}", "≥ 90% and above v1",
     shown(AGENT_V2, ".0%"), None if AGENT_V2 is None else bool(AGENT_V2 >= 0.9 and AGENT_V2 > AGENT_V1)),
    ("Explain", "Surge briefs that name the company behind the surge",
     f"No filters {shown(COMPANY_HIT_UNFILTERED, '.0%')}; brief v1 {shown(COMPANY_HIT_V1, '.0%')}",
     "≥ 80% and above no filters", shown(COMPANY_HIT_FILTERED, ".0%"),
     None if COMPANY_HIT_FILTERED is None else bool(COMPANY_HIT_FILTERED >= 0.8
                                                     and COMPANY_HIT_FILTERED > COMPANY_HIT_UNFILTERED)),
    ("Explain", "January events named in that week's briefs",
     f"No filters {EVENTS_FOUND_UNFILTERED} of 3; brief v1 {EVENTS_FOUND_V1} of 3" if EVENTS_FOUND is not None else "–",
     "3 of 3", f"{EVENTS_FOUND} of 3" if EVENTS_FOUND is not None else "–",
     None if EVENTS_FOUND is None else EVENTS_FOUND == 3),
    ("Explain", "RAGAs faithfulness of the briefs", f"Brief v1 {shown(FAITHFULNESS_V1, '.2f')}", "≥ 0.80",
     shown(FAITHFULNESS, ".2f"), None if FAITHFULNESS is None else bool(FAITHFULNESS >= 0.8)),
    ("Explain", "RAGAs answer relevancy", f"Brief v1 {shown(RELEVANCY_V1, '.2f')}", "no target (diagnostic)",
     shown(RELEVANCY, ".2f"), "ℹ️ diagnostic"),
    ("Explain", "Judge usefulness of the briefs (1 to 5)", f"Brief v1 {shown(JUDGE_MEAN_V1, '.2f')}", "above brief v1",
     shown(JUDGE_MEAN, ".2f"), None if JUDGE_MEAN is None or JUDGE_MEAN_V1 is None else bool(JUDGE_MEAN > JUDGE_MEAN_V1)),
    ("Explain", "Briefs with a statement the judge found unsupported",
     f"Brief v1 {UNSUPPORTED_V1} of {len(judge_table_v1)}" if judge_table_v1 is not None else "–",
     "no target (diagnostic)", f"{UNSUPPORTED} of {len(judge_table)}" if judge_table is not None else "–",
     "ℹ️ diagnostic"),
    ("Explain", "Judge within 1 point of a person (5 briefs)", "–", "no target (diagnostic)",
     shown(JUDGE_AGREEMENT, ".0%"), "ℹ️ diagnostic"),
    ("Fairness", "Shortfall in payouts caught (tagged vs untagged)", "–", "≤ 5 points",
     f"{FAIRNESS_GAP * 100:.1f} points", bool(FAIRNESS_GAP <= 0.05)),
], columns=["stage", "metric", "baseline", "target", "result", "met"])
scorecard["met"] = scorecard["met"].map(verdict)
scorecard.to_csv(os.path.join(FIG_DIR, "scorecard.csv"), index=False)
display(scorecard.set_index("stage"))
```

## Step 14: Section 12 findings code

**Find it:** search for `def pct(value):`

It is a **code cell**. Click inside the code, press **Ctrl+A**, then **Delete**, and paste the code below. Do not run it yet.

```python
def pct(value):
    return "not measured" if value is None else f"{value:.0%}"


def num(value):
    return "not measured" if value is None else f"{value:.2f}"


targets = scorecard[scorecard["met"].isin(["✅ met", "❌ not met", "not measured yet"])]
missed = targets.loc[targets["met"] == "❌ not met", "metric"].tolist()
pending = targets.loc[targets["met"] == "not measured yet", "metric"].tolist()
llm_sample = routing_results.loc["LLM zero-shot (gpt-4o-mini)", f"accuracy ({N_LLM} sample)"] if pred_llm is not None else None
w2v_sample = routing_results.loc["Word2Vec + logistic regression", f"accuracy ({N_LLM} sample)"]
largest = campaigns.iloc[0] if len(campaigns) else None
human_text = (f"A person confirmed {HUMAN_CONFIRMED} of 20 flagged pairs." if HUMAN_CONFIRMED is not None
              else "The 20-pair human check is not entered yet.")
events_text = (f"{EVENTS_FOUND} of the 3 January events were named in that week's briefs (brief v1: {EVENTS_FOUND_V1})"
               if EVENTS_FOUND is not None else "the January event check was not run")
agreement_text = (f" A person's scores were within 1 point of the judge's on {JUDGE_AGREEMENT:.0%} of 5 briefs."
                  if JUDGE_AGREEMENT is not None else "")

findings = [
    f"1. **Clean.** {COPY_SHARE:.0%} of complaints copy a template, in {len(campaigns)} campaigns of at least "
    f"{CAMPAIGN_MIN} copies" + (f" (the largest: {largest['main company']}, {largest['complaints']:,} copies)" if largest is not None else "")
    + f". {human_text} "
    "Removing copies keeps the evaluation honest, and grouping them lets the bank send one reviewed answer.",
    f"2. **Route (Word2Vec).** The router sends {ACC_ROUTER:.0%} of held-out complaints to the right team "
    f"(macro-F1 {F1_ROUTER:.2f}), against {ACC_KW:.0%} for keyword rules."
    + (f" On the same {N_LLM} complaints GPT-4o-mini got {llm_sample:.0%} and Word2Vec {w2v_sample:.0%}, "
       "with no cost per complaint and in well under a millisecond." if llm_sample is not None else ""),
    f"3. **Prioritise.** Reviewing the riskiest 20% of complaints first reaches {CAUGHT_20:.0%} of the payouts, "
    f"{CAUGHT_20 / 0.2:.1f}× random review. Complaints from older Americans and servicemembers are not missed more "
    f"often (largest shortfall {FAIRNESS_GAP * 100:.1f} points, target at most 5).",
    f"4. **Watch (LDA).** Tuned topics are more coherent than untuned ones (c_v {LDA_COHERENCE:.2f} vs "
    f"{UNTUNED_COHERENCE:.2f}) and line up with the CFPB issue labels (NMI {NMI:.2f}, random topics {NMI_RANDOM:.2f}). "
    f"The alarm fired within 7 days of {int(EVENTS['alarm within 7 days'].sum())} of the 3 January events, and "
    f"{ALARMS_EXPLAINED} of {len(alarm_review)} alarms are explained by a dated event or a copy campaign."
    + (f" Topics also see what counting cannot: in the week of {SHOWCASE_WEEK:%d %B}, all complaints were "
       f"{lens.at[SHOWCASE_WEEK, 'all complaints']:.1f}× a normal week, but '{SHOWCASE_TOPIC}' was "
       f"{lens.at[SHOWCASE_WEEK, 'busiest topic']:.0f}× normal." if SHOWCASE_WEEK is not None else ""),
    f"5. **Explain (RAG).** The agent passed {pct(AGENT_V2)} of its test checks (first prompt: {pct(AGENT_V1)}). "
    f"With the agent's filters, {pct(COMPANY_HIT_FILTERED)} of surge briefs named the company behind the surge "
    f"(brief v1: {pct(COMPANY_HIT_V1)}; no filters: {pct(COMPANY_HIT_UNFILTERED)}); {events_text}. "
    f"Brief v2 always shows the writer the biggest groups of copies, takes its counts from the database and removes "
    f"template blanks: the judge rated it {num(JUDGE_MEAN)} out of 5 (brief v1: {num(JUDGE_MEAN_V1)}), RAGAs "
    f"faithfulness was {num(FAITHFULNESS)} (brief v1: {num(FAITHFULNESS_V1)}) and answer relevancy {num(RELEVANCY)}."
    f"{agreement_text} For single complaints, the best search found past cases with the same issue in {P5_OURS:.0%} "
    f"of the top 5 (keyword search {P5_BM25:.0%}).",
    f"6. **Overall.** {int((targets['met'] == '✅ met').sum())} of {len(targets)} targets met."
    + (f" Missed: {'; '.join(missed)}." if missed else "")
    + (f" Still to enter: {'; '.join(pending)}." if pending else ""),
]
display(Markdown("\n\n".join(findings)))
```

## Step 15: Section 13 export code

**Find it:** search for `tables = {"routing_results"`

It is a **code cell**. Click inside the code, press **Ctrl+A**, then **Delete**, and paste the code below. Do not run it yet.

```python
tables = {"routing_results": routing_results, "payout_results": payout_results, "fairness": fairness,
          "lda_tuning": tuning, "alarm_three_ways": comparison, "alarm_review": alarm_review, "retrieval": retrieval}
if agent_results is not None:
    tables["agent_tests"] = agent_results
if len(surge):
    tables["surge_briefs"] = surge.drop(columns="parts")
if event_results:
    event_export = pd.DataFrame(event_results)
    tables["event_checks"] = event_export.drop(columns=[c for c in event_export.columns if c.startswith("parts")])
for name, table in [("ragas", ragas_table), ("ragas_v1", ragas_table_v1), ("judge", judge_table),
                    ("judge_v1", judge_table_v1)]:
    if table is not None:
        tables[name] = table
for name, table in tables.items():
    table.to_csv(os.path.join(FIG_DIR, f"{name}.csv"))
shutil.make_archive("figures", "zip", FIG_DIR)
print("Charts, the scorecard and result tables are in figures.zip:")
print("\n".join(sorted(os.listdir(FIG_DIR))))
```

---

## After the 15 steps: run it

1. **Runtime → Change runtime type → T4 GPU.**
2. Upload `cfpb_bank_complaints_2025.zip` again if the file panel on the left doesn't show it.
3. **Runtime → Run all.** It takes about 1 hour 25 minutes, about 10 minutes longer than last time, because every brief is now written twice (v1 and v2).
4. **Copy check (section 4):** `HUMAN_CONFIRMED = 19` is still in the cell. Look at the 20 pairs above it. If they look the same as last time, leave 19. If they changed, count again and type your new number.
5. **Brief check (section 8):** read briefs 0 to 4 in the cell above `HUMAN_USEFULNESS`. These are now the v2 briefs. Score each from 1 to 5 as before, without looking at the judge's scores, and type them in, for example `HUMAN_USEFULNESS = {0: 4, 1: 3, 2: 5, 3: 4, 4: 2}`. Press **Shift+Enter**.
6. Click the code cell under **10. Scorecard** (it starts with `def verdict(ok):`) and choose **Runtime → Run cell and below**.
7. Download the notebook (**File → Download → Download .ipynb**) and `figures.zip` (right-click it in the file panel on the left).

**What's new in the outputs:** the surge table and chart (fig13) now show three bars: brief v2, brief v1 and no filters. The event check prints brief v1 and brief v2 one after the other. RAGAs and the judge each print a small v1 against v2 table. The scorecard has two new rows: judge usefulness, and briefs with an unsupported statement. `figures.zip` also holds `ragas_v1.csv` and `judge_v1.csv`.
