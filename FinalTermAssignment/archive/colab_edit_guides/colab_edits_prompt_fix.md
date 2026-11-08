# Colab edits: brief prompt fix

**Why.** In the 8 November run, Qwen copied its own instructions into 8 of the 21 briefs and into the demo, for example "What customers report: the main problems in one or two sentences, starting with the texts filed most often...". The search fix worked, so it stays. Two changes:

1. The v2 prompt goes back to v1's wording, plus one sentence.
2. A safety net: if a draft repeats a line of the instructions, the model is asked once to write it again. The notebook counts how often that happens.

**What changes.** Three cells in section 8 and one line in the human check. Nothing else.

Same as last time: press **Ctrl+F** to find each cell, select everything in it with **Ctrl+A**, press **Delete**, and paste with the copy button on the grey block.

---

## Step 1: 8.4 text

**Find it:** search for `8.4 Filtered search and the brief`

It is a **text cell**. Double-click it, click inside the edit box, press **Ctrl+A**, then **Delete**, paste the text below, and press **Shift+Enter**. The only new part is the paragraph that starts with "A prompt lesson".

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

**A prompt lesson.** In our first run of brief v2, its prompt also reworded the three lines (for example "starting with the texts filed most often"). Qwen3-4B then copied those instructions word for word into 8 of the 21 briefs, and the judge scored v2 below v1 (3.05 against 3.71), even though v2 named the company behind the surge more often (94% against 82%). Looking back, v1 had done the same in 5 of its 17 briefs written without filters, always when the complaints were mostly copied template letters, which v2 now puts first. So two changes:

- The v2 prompt keeps v1's wording for its three lines and adds only one sentence: a text filed many times stands for many customers, so describe it first.
- A safety net for both versions: a draft that repeats a line of the instructions is sent back to the model once, with a short correction. Section 8.5 counts how often this happens.

As in Assignment 4, even a small change to a prompt needs the whole test suite again.

Both versions use the same filters and the same "I don't know" rule, and both prompts allow only facts from the numbered complaints and no advice. Both are kept and scored with the same tests in 8.5, as with the agent prompts above and the v1 and v2 prompts of Assignment 4. Brief v2 was designed after reading the v1 briefs of these same alarms, so its scores here are development results: a pilot has to confirm them on new weeks.
```

## Step 2: 8.4 code

**Find it:** search for `RELEVANCE_THRESHOLD = -3.0`

It is a **code cell**. Click inside the code, press **Ctrl+A**, then **Delete**, and paste the code below. Changed: `BRIEF_PROMPT_V2` and the safety net in `run_explain`. Do not run it yet.

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
the same text was filed in the period searched. A text filed many times stands for many customers, so describe it first.
Write exactly these three lines and nothing else:
What customers report: the main problem in one or two sentences, citing complaints like [1] or [3].
Who: the companies named in the complaints, the most frequent first.
When: the dates of the complaints.
Do not give advice. Do not add any fact that is not in the complaints.
Describe what these complaints say, even if they answer the question only in part.

Complaints:
{context}"""

SMALL_TALK = ("You are the assistant of a bank's complaints team. Reply in one or two sentences. Do not state any "
              "facts. Invite the user to ask about the customer complaints of 2025.")
INSTRUCTION_PHRASES = ["in one or two sentences", "citing complaints like", "the most frequent first",
                       "the dates of the complaints", "filed most often", "the most filings first"]
RETRY_NOTE = ("You repeated the instructions instead of answering. Answer again with the same lines, filled with "
              "what the numbered complaints actually say.")


def repeats_prompt(text):
    return any(phrase in text.lower() for phrase in INSTRUCTION_PHRASES)


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
              "scale": None, "written": NO_ANSWER, "answer": NO_ANSWER, "retried": False}
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
        messages = [{"role": "system", "content": prompt.format(context="\n\n".join(result["parts"]))},
                    {"role": "user", "content": question}]
        written = generate(messages, max_new_tokens=250)
        result["retried"] = repeats_prompt(written)
        if result["retried"]:
            written = generate(messages + [{"role": "assistant", "content": written},
                                           {"role": "user", "content": RETRY_NOTE}], max_new_tokens=250)
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

## Step 3: 8.5 code (surge briefs)

**Find it:** search for `MAX_ALARM_BRIEFS`

It is a **code cell**. Click inside the code, press **Ctrl+A**, then **Delete**, and paste the code below. Changed: two new columns that count drafts that repeated the prompt. Do not run it yet.

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
                "first draft repeated the prompt": result["retried"],
                "final brief repeats the prompt": repeats_prompt(result["written"]),
            })
    print(f"{len(todo)} alarms, {len(BRIEF_RUNS)} briefs each, in {time.time() - start:.0f} s")
surge = pd.DataFrame(surge_rows)

COMPANY_HIT_FILTERED = COMPANY_HIT_UNFILTERED = COMPANY_HIT_V1 = WEEK_SHARE_FILTERED = WEEK_SHARE_UNFILTERED = None
if len(surge):
    surge_summary = (surge.groupby("mode")[["names main company", "sources from alarm week", "answered",
                                            "first draft repeated the prompt", "final brief repeats the prompt"]].mean()
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

## Step 4: Clear your brief scores

**Find it:** search for `HUMAN_USEFULNESS = {0:`

Change only the first line, from

```python
HUMAN_USEFULNESS = {0: 5, 1: 2, 2: 3, 3: 1, 4: 1}
```

to

```python
HUMAN_USEFULNESS = {}
```

---

## Then run it

1. **Runtime → Change runtime type → T4 GPU**, and upload `cfpb_bank_complaints_2025.zip` if it isn't in the file panel.
2. **Runtime → Run all** (about 1 hour 25 minutes).
3. **Copy check:** `HUMAN_CONFIRMED = 19` stays, as long as the 20 pairs look the same.
4. **Brief check:** score briefs 0 to 4 into `HUMAN_USEFULNESS` before looking at the judge, then press **Shift+Enter**.
5. Click the code cell under **10. Scorecard** and choose **Runtime → Run cell and below**.
6. Download the .ipynb and `figures.zip`.
