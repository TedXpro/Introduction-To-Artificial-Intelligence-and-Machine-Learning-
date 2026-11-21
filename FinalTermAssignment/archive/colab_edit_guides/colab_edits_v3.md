# Colab edits v3: validation split, campaigns, model comparison

**Why.** These edits answer the professor's feedback from 11 November. Every edit replaces a whole cell, from the top of the notebook to the bottom, so you can work downwards. They were tested locally on a 6,000-complaint run before this guide was written.

| Professor's comment | What changes in the notebook | Steps |
|---|---|---|
| Finding duplicates is a goal, not only removing them | Section 4 becomes "Find copy-paste campaigns", with the payout rate of campaigns | 1, 5, 6 |
| Size of the dataset, random or stratified? | A sample check against all 183,544 complaints; the model splits are stratified | 3, 4, 8 |
| Train and test sizes on one slide; add validation | 70 / 10 / 20 training / validation / test split, with a size table; every setting is chosen on validation | 7 to 22 |
| What are we measuring? A table | "What we measure" table in section 1 | 2 |
| One model comparison near the end | Section 10 starts with a model comparison table with 95% intervals | 24, 25 |
| Architecture slide, routing, RAG too early | Slides only; step 1 adds an end-to-end paragraph | 1 |

---

## Before you start

1. Open the latest notebook in Colab (the one with the run from 8 November).
2. **File → Save a copy in Drive**, so you can always go back.
3. **Runtime → Change runtime type → T4 GPU**.
4. Check that `OPENAI_API_KEY` is still in **Secrets** (the key icon on the left) with notebook access on.

**How to replace a cell**

- **Code cell:** click inside the code, press **Ctrl+A**, then **Delete**, and paste the new code.
- **Text cell:** double-click it, click inside the edit box, press **Ctrl+A**, then **Delete**, paste the new text, and press **Shift+Enter**.
- **Finding a cell:** use **Edit → Find and replace** (Ctrl+H) and type the search text. Each search text matches exactly one cell.

Don't run anything until all 28 steps are done.

---

## Step 1 of 28: Title (text cell)

**Find it:** search for `What we built`

**What changes:** Stage "Clean" renamed to "Find campaigns"; the Prove it row names the training / validation / test split; a new **End to end** paragraph under the table.

```markdown
# Complaint Intelligence for Retail Banks

**MCSBT · Introduction to AI and Machine Learning · Final Term Project**

Group: Yohannes Belay Woldeyes, Vidak Vukadinovic, Jorge Plaza de los Reyes Álvarez

**The problem in plain words.** When a US customer is unhappy with their bank, they can file a written complaint with the government's consumer watchdog, the CFPB, and the bank has 15 days to respond. A large bank receives thousands of these a year, and people read them one by one. Serious complaints wait in the same queue as small ones, and when something breaks for many customers at once, the pattern is noticed late.

**What we built.** A tool that reads every complaint as it arrives, finds copy-paste campaigns, sends each complaint to the right team, flags the ones likely to end in money paid back, warns managers when one kind of problem suddenly grows, and explains each warning from the complaints themselves.

| Stage | Question it answers | Course method |
|---|---|---|
| Find campaigns | Which complaints are copy-paste campaigns? | Deduplication (Assignment 1),<br>sentence embeddings + FAISS (Session 7) |
| Route | Which team should handle this complaint? | **Word2Vec** (Session 5, Assignment 2, Midterm),<br>compared with an LLM (Session 6) |
| Prioritise | Which complaints will end in a payout? | Supervised prediction (Session 3) |
| Watch | Which problems are surging this week? | **LDA topic modeling** (Sessions 3 and 5) |
| Explain | Why is this problem surging? | **RAG** with an agent and filters, Qwen3-4B<br>(Sessions 6–8, Assignment 3), prompt tests (Assignment 4) |
| Prove it | Does each part work? | A training / validation / test split, dated real events,<br>RAGAs, LLM-as-judge with a human spot-check (Session 11) |

**End to end.** Each new complaint is checked for copies of other complaints, routed to a team and given a payout risk, one complaint at a time. Every week, LDA counts the topics of all complaints and raises an alarm when one jumps; the assistant then reads that week's complaints and writes a brief that explains the alarm. A manager sees the routed queue, the risk order, the alarms and the briefs.

The three main AI methods are **Word2Vec**, **LDA topic modeling** and **RAG with an LLM**, and each one feeds the next: Word2Vec handles every single complaint, LDA watches the weekly mix, and RAG explains each alarm LDA raises. Every stage is checked against data we did not create ourselves: the product and issue the consumer chose, how the company actually closed the complaint, and real events with a date.
```

## Step 2 of 28: 1. Business case (text cell)

**Find it:** search for `The AI Canvas`

**What changes:** Everything up to and including the AI Canvas is unchanged. After it come the new **How we split the data** and **What we measure** tables, then the targets table, which now says Find campaigns and test split, and adds the judge row.

```markdown
## 1. Business case

**The problem.** Every complaint a bank receives through the CFPB has to be read, sent to the right product team and answered: the CFPB expects a response within 15 days and a final answer within 60. Today this is done by hand. Problems that hit many customers at once (an outage, a fee error, a scam wave) first show up as a sudden wave of complaints, and a team reading them one by one notices late.

**Background: what the complaint waves of early 2025 were about.** Each of these shows up in our data (sections 4 and 7):

| Date | What happened | Money involved |
|---|---|---|
| Nov 2024 | The CFPB orders Navy Federal Credit Union to refund<br>surprise overdraft fees | More than USD 95 million<br>(the order was ended in July 2025) |
| Dec 2024 | The CFPB sues Zelle's operator and three large banks<br>over how they handled scams | USD 870 million in customer losses claimed<br>(the case was dropped in March 2025) |
| 14 Jan 2025 | The CFPB sues Capital One: its 360 Savings account stayed<br>at 0.30% while a near-identical account paid up to 4.35% | More than USD 2 billion in lost interest claimed;<br>a related class action settled for USD 425 million |
| 15–19 Jan 2025 | A power failure at the vendor FIS delays Capital One<br>direct deposits and payments for several days | Capital One refunded late fees that customers<br>paid elsewhere because of the delay |
| 16 Jan 2025 | The CFPB orders Block (Cash App) to fix its fraud handling<br>and customer service | Up to USD 175 million<br>in refunds and penalties |

Sources: CFPB press releases and enforcement pages; NBC News; American Banker; ConsumerAffairs.

Complaint waves sit next to problems worth hundreds of millions of dollars, and they arrive with the evidence (who, what, since when) in the customers' own words. A bank that reads its complaints as a whole, every week, finds its problems earlier.

**Business model.**

| | |
|---|---|
| Product | Complaint Intelligence: a service that reads every complaint, routes it,<br>scores its payout risk, and sends a weekly surge watch with evidence briefs |
| Customers | US banks and credit unions that receive CFPB complaints, starting with<br>mid-size banks that have no complaint analytics team of their own |
| What they pay for | A yearly subscription priced by complaint volume, plus a fee for adding<br>the bank's own channels (emails, web chats, call-centre notes) |
| Our costs | A small Word2Vec model (runs on a normal CPU), one GPU for the open<br>Qwen3-4B model, and analysts who review new alarms |
| Value for the customer | Hours of manual triage saved, payouts handled first, and problems<br>spotted in days instead of weeks |
| Why us | Trained on public CFPB data, so it works for a new bank on day one; the<br>market-wide view shows a bank when a problem is surging at its peers |

**The AI Canvas** (Session 2, slide 10):

| Box | Our answer |
|---|---|
| Prediction | The right team for each complaint, the chance it ends in a payout,<br>its topic, and an explanation for each topic surge |
| Judgment | Missing a payout case or a surge costs more than reading one<br>extra complaint, so we favour catching cases (recall) |
| Action | Route automatically, review high-risk complaints first, alert managers<br>when a topic surges, and answer their questions from the complaints |
| Outcome | Routing accuracy, payouts caught in the riskiest 20%, surges detected,<br>briefs that are faithful and name the right cause |
| Input | Complaint text, product, issue, company, date |
| Training | 2025 CFPB complaints with their real product, issue and outcome |
| Feedback | Agents correct routes; managers mark alarms as real or not; each<br>month's closed complaints are added to the training data |

**How we split the data.** Copies of a template are grouped first (section 4), and one complaint per group is kept for modeling, so the same text cannot appear in two splits. The unique complaints are then split at random, with the same mix of product and payout in every split (stratified):

| Split | Share | Used for |
|---|---|---|
| Training | 70% | Fitting the models; the knowledge base for search |
| Validation | 10% | Choosing settings: Word2Vec type and window, model strength (C),<br>LDA words and number of topics, the search engine |
| Test | 20% | Measured once at the end; never used to choose anything |

**What we measure.** Every number is checked against an answer key we did not make ourselves.

| Stage | Metric | What it tells us | Answer key | Data |
|---|---|---|---|---|
| Find campaigns | Pairs a person confirms | Are flagged copies really copies? | A person reads them | 20 flagged pairs |
| Route | Accuracy, macro-F1 | Share sent to the right team | Product the consumer chose | Test split |
| Prioritise | Payouts caught in the riskiest 20%,<br>PR-AUC | How many payouts a 20% review finds | How the company<br>closed the complaint | Test split |
| Watch | Coherence (c_v), NMI | Are topics readable, and do they<br>match real problems? | CFPB issue labels | Test split |
| Watch | Alarm within 7 days | Does an alarm follow real events? | Dated news events | All weeks |
| Explain | Precision@5 | Are the 5 past cases about the same issue? | CFPB issue labels | 400 test complaints |
| Explain | Agent checks passed | Does the agent pick the right filters? | Answers we wrote<br>before running it | 16 test and<br>8 held-out questions |
| Explain | Company and events named | Does the brief find the cause? | Complaint records,<br>dated events | Alarm weeks |
| Explain | Faithfulness, judge score | Is the brief backed by the complaints? | GPT-4o-mini judge,<br>checked by a person | All briefs |
| Fairness | Shortfall in payouts caught | Are tagged groups missed more often? | Tags and outcomes | Test split |

Section 10 gives every main result with a 95% interval, so small differences are not read as wins.

**Success targets.** The targets for Find campaigns, Route, Prioritise, Watch and Fairness were fixed before the first run. The Explain stage replaces our first RAG assistant (section 8 says why), and its targets were fixed before it was run.

| Stage | Metric | Compared with | Target |
|---|---|---|---|
| Find campaigns | Flagged copy pairs a person confirms (20-pair check) | – | at least 18 of 20 |
| Route | Accuracy on the test split | Keyword rules | at least 80%,<br>and better than keyword rules |
| Prioritise | Payouts caught when reviewing the riskiest 20% | Random review (20%) | at least 40% (2× random) |
| Watch | Topic coherence (c_v) on the test split | Untuned LDA | at least 0.40,<br>and better than untuned |
| Watch | Surge alarm after the dated January 2025 events | – | an alarm within 7 days |
| Explain | Past cases with the same issue in the top 5 (precision@5),<br>engine chosen on the validation split | Keyword search (BM25) | better than BM25 |
| Explain | Agent test suite: questions turned into the right filters | First prompt (v1) | at least 90% of checks,<br>and better than v1 |
| Explain | Surge briefs that name the company behind the surge | The same briefs<br>without filters | at least 80%,<br>and better than without |
| Explain | The three January events named in that week's briefs | – | 3 of 3 |
| Explain | RAGAs faithfulness of the briefs | – | at least 0.80 |
| Explain | Judge usefulness of the briefs (1 to 5) | Brief v1 | better than brief v1 |
| Fairness | Shortfall in payouts caught for Older American<br>and Servicemember complaints | Complaints without a tag | at most 5 percentage points |
```

## Step 3 of 28: 3. Data (text cell)

**Find it:** search for `Data: source and procurement`

**What changes:** New paragraph: **Is the sample representative?** (random sample, not stratified, and why).

```markdown
## 3. Data: source and procurement

| | |
|---|---|
| Source | CFPB Consumer Complaint Database, official narratives archive<br>(US government open data, listed on Data.gov) |
| Link | [consumerfinance.gov narratives archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/) |
| Files | The six 2025 exports, January–February to November–December:<br>5.4 million complaints, 1.2 million with a written story |
| Privacy | The CFPB removes names, account numbers and dates before<br>publishing; they appear as XXXX |
| Why 2025 | The CFPB's product and issue categories did not change between<br>August 2023 and October 2026, so every label means the same thing |
| Scope | The five product lines a retail bank runs: checking/savings,<br>credit cards, money transfers, mortgages, vehicle loans |
| Size | 183,544 complaints in scope; a simple random sample<br>of 60,000 (seed 42) keeps Colab fast |
| Build script | `prepare_dataset.py` reads the six zip files and writes<br>`cfpb_bank_complaints_2025.csv` |

| Column | Meaning |
|---|---|
| `complaint_id` | CFPB complaint number (can be looked up on consumerfinance.gov) |
| `date_received` | Date the CFPB received the complaint |
| `product`, `sub_product` | Product chosen by the consumer |
| `issue`, `sub_issue` | Problem chosen by the consumer |
| `narrative` | The consumer's own description |
| `company`, `state` | Company the complaint is about, consumer's state |
| `tags` | "Older American", "Servicemember" or empty |
| `company_response` | How the company closed it, for example "Closed with monetary relief" |
| `timely_response` | Whether the company answered on time |

**Is the sample representative?** The sample was drawn at random, not stratified, so we check it against all 183,544 complaints in scope (the table after the overview below): the product shares, the payout rate, the two tags and the share filed in January and February all differ by at most 0.2 percentage points. With 60,000 complaints, chance alone moves a share by about 0.3 points at most (95%), so stratifying the sample would change little. The smaller splits used to train and test the models are stratified (section 5).

Upload `cfpb_bank_complaints_2025.zip` to the Colab file panel before running the next cell.
```

## Step 4 of 28: 3. Data, overview table (code cell)

**Find it:** search for `top_company=`

**What changes:** Adds the sample check: our 60,000 against all 183,544 complaints, with the difference in points.

```python
overview = (df.groupby("product")
              .agg(complaints=("complaint_id", "size"),
                   median_words=("n_words", "median"),
                   payout_rate=("payout", "mean"),
                   on_time=("timely_response", lambda s: (s == "Yes").mean()),
                   top_company=("company", lambda s: s.value_counts().index[0]))
              .reindex(PRODUCTS))
overview.insert(1, "share", overview["complaints"] / len(df))
display(overview.style.format({"complaints": "{:,}", "share": "{:.1%}", "median_words": "{:.0f}",
                               "payout_rate": "{:.1%}", "on_time": "{:.1%}"}))

FULL_2025 = {"complaints": 183_544, "Checking or savings account": 0.2803, "Credit card": 0.2262,
             "Money transfer, virtual currency, or money service": 0.3534, "Mortgage": 0.0764,
             "Vehicle loan or lease": 0.0637, "payout rate": 0.0744, "Older American tag": 0.0621,
             "Servicemember tag": 0.1036, "January and February": 0.4060}
sample_shares = {"complaints": len(df), **{p: (df["product"] == p).mean() for p in PRODUCTS},
                 "payout rate": df["payout"].mean(),
                 "Older American tag": df["tags"].str.contains("Older American").mean(),
                 "Servicemember tag": df["tags"].str.contains("Servicemember").mean(),
                 "January and February": df["date_received"].dt.month.isin([1, 2]).mean()}
check_rows = [("complaints", f"{FULL_2025['complaints']:,}", f"{len(df):,}", "")]
for key in list(FULL_2025)[1:]:
    gap = (sample_shares[key] - FULL_2025[key]) * 100
    check_rows.append((SHORT.get(key, key), f"{FULL_2025[key]:.1%}", f"{sample_shares[key]:.1%}", f"{gap:+.1f}"))
sample_check = pd.DataFrame(check_rows, columns=["measure", "all 2025 complaints in scope", "our random sample",
                                                 "difference (points)"]).set_index("measure")
SAMPLE_MAX_GAP = max(abs(sample_shares[k] - FULL_2025[k]) for k in list(FULL_2025)[1:]) * 100
display(sample_check)
print(f"Largest difference between the sample and all 183,544 complaints: {SAMPLE_MAX_GAP:.1f} percentage points")
```

## Step 5 of 28: 4. Find copy-paste campaigns (text cell)

**Find it:** search for `Many complaints are not written from scratch`

**What changes:** Section renamed from "Clean". Finding campaigns is now a result in itself (the professor's comment), not only cleaning.

```markdown
## 4. Find copy-paste campaigns

Many complaints are not written from scratch. People share templates online and file the same text, and some people file one complaint against several companies. Finding these campaigns is a result in itself, not only a cleaning step:

1. **Business:** a campaign is one problem shared by many customers. The bank can answer it once with a reviewed response, and see which company and week it started in. The campaign table below also shows how often campaign complaints end in a payout, compared with all other complaints.
2. **Early warning:** a sudden campaign is itself a signal that something happened. Sections 7 and 8 use it to explain alarms.
3. **Honest evaluation:** if one copy lands in the training set and another in the test set, the model has seen the answer and its score looks better than it really is. So only one complaint per group goes into the training, validation and test splits. The low-scoring example project had no such check.

Copies are grouped, never deleted: every copy still counts in the weekly volumes and the alarms.

**Design choices.**

- Exact copies are found after cleaning the text, as in Assignment 1. Cleaning lowercases the text, replaces amounts with the word "amount" and removes apostrophes before other punctuation, so "didn't" stays one word ("didnt") instead of breaking into "didn" and "t".
- Near copies (same template, different names or amounts) are found by meaning *and* by wording. Each complaint is turned into a sentence embedding (`all-mpnet-base-v2`, Session 7, Assignment 3), and FAISS finds its 10 closest complaints.
- A pair counts as a copy only if both checks agree: cosine similarity of at least 0.95, *and* at least half of their three-word sequences are shared.
- The same embeddings are reused later for the RAG assistant, so they are computed once.
```

## Step 6 of 28: 4. Campaign table (code cell)

**Find it:** search for `CAMPAIGN_MIN = max(5`

**What changes:** New **paid out** column, and a line comparing the payout rate of campaign complaints with all other complaints.

```python
CAMPAIGN_MIN = max(5, round(20 * SCALE))
big = df[df["group_size"] >= CAMPAIGN_MIN].groupby("group")
campaigns = pd.DataFrame({
    "complaints": big.size(),
    "main company": big["company"].agg(lambda s: s.value_counts().index[0]),
    "product": big["product"].agg(lambda s: SHORT[s.value_counts().index[0]]),
    "first seen": big["date_received"].min().dt.strftime("%d %b %Y"),
    "busiest week": big["week"].agg(lambda s: s.value_counts().index[0]).dt.strftime("%d %b %Y"),
    "paid out": big["payout"].mean(),
}).sort_values("complaints", ascending=False)
campaigns["opening words"] = [" ".join(df.at[g, "text"].split()[:18]) + " ..." for g in campaigns.index]
in_campaign = (df["group_size"] >= CAMPAIGN_MIN).to_numpy()
CAMPAIGN_PAYOUT = df.loc[in_campaign, "payout"].mean()
OTHER_PAYOUT = df.loc[~in_campaign, "payout"].mean()
print(f"{len(campaigns)} campaigns with at least {CAMPAIGN_MIN} copies, {in_campaign.sum():,} complaints in total")
print(f"Paid out: {CAMPAIGN_PAYOUT:.1%} of campaign complaints, against {OTHER_PAYOUT:.1%} of all other complaints")
display(campaigns.head(10).reset_index(drop=True).style.format({"complaints": "{:,}", "paid out": "{:.1%}"}))
```

## Step 7 of 28: 5. Route, text (text cell)

**Find it:** search for `Route: Word2Vec`

**What changes:** New **How we split the data** (70 / 10 / 20, stratified by product and payout); Word2Vec and the router's C are chosen on validation.

```markdown
## 5. Route: Word2Vec (main method 1)

**What it does.** It reads a complaint and sends it to one of the five product teams.

**How it would be used.** CFPB complaints come with a label: the consumer picks the product when filing. A bank's own channels (emails, web chats, call-centre notes) have no such label, and today a person reads each message to decide the team. So the router learns from the labelled CFPB complaints and is then applied to the bank's unlabelled messages.

**How we split the data.** Copies of a template are grouped first (section 4), and one complaint per group is kept for modeling, so the same text cannot appear in two splits. The unique complaints are split 70% training, 10% validation and 20% test, stratified by product *and* payout, so every split has the same mix of teams and the same payout rate (the table below checks it). Sections 5 to 8 all use this split:

- **Training:** fits the models, and is the knowledge base for search.
- **Validation:** chooses every setting: which Word2Vec, how strong a model, which LDA, which search engine.
- **Test:** measured once at the end, for the results in section 10.

**Design choices.**

- **Word2Vec trained on the complaints themselves** (Session 5, Assignment 2, Midterm), so it learns bank language: "zelle" sits next to "scam" and "cash_app", "escrow" sits next to "mortgage". It is trained only on the training split.
- **Word pairs** such as `direct_deposit` and `credit_card` are learned with gensim `Phrases`. This is the "n-gram values" from the brief.
- **Settings chosen on validation:** skip-gram or CBOW, with a context window of 5 or 10 words. All use 100 dimensions and 10 passes, the settings that fixed underfitting in the midterm. Each candidate is scored by how well a simple classifier on its complaint vectors predicts the product of the validation complaints. Word2Vec runs on one thread, so a rerun gives exactly the same vectors.
- A complaint becomes the average of its word vectors.
- The router's strength (C in logistic regression; a lower C means a simpler model) is also chosen on validation.

**What we compare.**

1. **Keyword rules:** a JSON keyword list per team, which is what a simple rules engine does today.
2. **Word2Vec + taxonomy:** the same keyword list, matched by meaning instead of exact words. It uses no labels at all (the brief's taxonomy idea).
3. **Word2Vec + logistic regression:** learns from the labelled training complaints. This is our proposed router.
4. **LLM zero-shot:** `gpt-4o-mini` reads the complaint and picks a team (Session 6). It runs on 300 test complaints because it costs money per call.

The ground truth is the product the consumer chose when filing, which none of the models can see.
```

## Step 8 of 28: 5. The split (code cell)

**Find it:** search for `STOP_WORDS = set(ENGLISH_STOP_WORDS)`

**What changes:** The train/test split becomes a stratified training / validation / test split, with a table showing the size and mix of each split.

```python
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def tokenize(text):
    return [w for w in text.split() if len(w) > 2 and w not in STOP_WORDS]


df["tokens"] = df["clean"].map(tokenize)
unique = df[~df["is_copy"]]
strata = unique["product"] + " | " + unique["payout"].astype(str)
rest_idx, test_idx = train_test_split(unique.index.to_numpy(), test_size=0.2, stratify=strata, random_state=SEED)
train_idx, val_idx = train_test_split(rest_idx, test_size=0.125, stratify=strata.loc[rest_idx], random_state=SEED)
train_idx, val_idx, test_idx = np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)

SPLITS = {"training": train_idx, "validation": val_idx, "test": test_idx}
split_rows = {}
for name, ids in SPLITS.items():
    part = df.loc[ids]
    split_rows[name] = {"complaints": f"{len(ids):,}", "share of unique complaints": f"{len(ids) / len(unique):.0%}",
                        **{SHORT[p]: f"{(part['product'] == p).mean():.1%}" for p in PRODUCTS},
                        "payout rate": f"{part['payout'].mean():.1%}"}
split_summary = pd.DataFrame(split_rows)
display(split_summary)

phrases = Phrases(df.loc[train_idx, "tokens"].tolist(), min_count=max(5, round(20 * SCALE)), threshold=10)
bigram = phrases.freeze()
df["tokens"] = [list(bigram[t]) for t in df["tokens"]]

learned_pairs = Counter(t for doc in df.loc[train_idx, "tokens"] for t in doc if "_" in t)
print(f"Training: {len(train_idx):,} | validation: {len(val_idx):,} | test: {len(test_idx):,} unique complaints")
print("Most frequent word pairs:", ", ".join(w for w, _ in learned_pairs.most_common(20)))
```

## Step 9 of 28: 5. Word2Vec (code cell)

**Find it:** search for `w2v = Word2Vec(`

**What changes:** Four Word2Vec settings (skip-gram or CBOW, window 5 or 10) are compared on validation and the best is kept. One thread, so a rerun gives the same numbers. Takes about 6 to 8 minutes.

```python
W2V_GRID = [("skip-gram", 1, 5), ("skip-gram", 1, 10), ("CBOW", 0, 5), ("CBOW", 0, 10)]
train_sentences = df.loc[train_idx, "tokens"].tolist()


def doc_vectors(model, rows):
    out = np.zeros((len(rows), model.vector_size), dtype="float32")
    for k, tokens in enumerate(df.loc[rows, "tokens"]):
        known = [t for t in tokens if t in model.wv]
        if known:
            out[k] = np.mean([model.wv[t] for t in known], axis=0)
    return out


w2v_models, w2v_rows = [], []
for name, sg, window in W2V_GRID:
    start = time.time()
    model = Word2Vec(sentences=train_sentences, vector_size=100, window=window, min_count=5, sg=sg, epochs=10,
                     workers=1, seed=SEED)
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    probe.fit(doc_vectors(model, train_idx), df.loc[train_idx, "product"])
    accuracy = accuracy_score(df.loc[val_idx, "product"], probe.predict(doc_vectors(model, val_idx)))
    w2v_models.append(model)
    w2v_rows.append({"Word2Vec setting": f"{name}, window {window}", "validation accuracy": accuracy,
                     "seconds": time.time() - start})
    print(f"{name:<9} window {window:>2}: validation accuracy {accuracy:.1%} ({time.time() - start:.0f} s)")
w2v_tuning = pd.DataFrame(w2v_rows)
BEST_W2V = int(w2v_tuning["validation accuracy"].idxmax())
w2v = w2v_models[BEST_W2V]
W2V_SETTING = w2v_tuning.at[BEST_W2V, "Word2Vec setting"]
print(f"Chosen on validation: {W2V_SETTING}; vocabulary {len(w2v.wv):,} words and word pairs")

PROBES = ["zelle", "overdraft", "repossession", "chargeback", "escrow", "cash_app", "direct_deposit", "late_fee"]
probe_table = pd.DataFrame({w: [f"{s} ({score:.2f})" for s, score in w2v.wv.most_similar(w, topn=5)]
                            for w in PROBES if w in w2v.wv}, index=[f"#{i}" for i in range(1, 6)]).T
display(probe_table)
```

## Step 10 of 28: 5. Routers (code cell)

**Find it:** search for `def doc_vector(tokens):`

**What changes:** The router's strength C (0.1, 0.3, 1, 3) is chosen on validation. The rest is the same.

```python
X_w2v = doc_vectors(w2v, df.index)
y_product = df["product"].to_numpy()
y_test = y_product[test_idx]

TAXONOMY = {
    "Checking or savings account": ["checking", "savings", "overdraft", "debit", "deposit", "deposits",
                                    "atm", "branch", "withdrawal", "nsf"],
    "Credit card": ["credit_card", "card", "cards", "statement", "apr", "limit", "annual", "rewards",
                    "purchase", "merchant"],
    "Money transfer, virtual currency, or money service": ["zelle", "cash_app", "cashapp", "paypal", "venmo",
                                                          "wire", "transfer", "crypto", "bitcoin", "coinbase"],
    "Mortgage": ["mortgage", "escrow", "modification", "foreclosure", "refinance", "servicer", "home",
                 "property", "heloc", "appraisal"],
    "Vehicle loan or lease": ["car", "vehicle", "auto", "repossession", "repossessed", "lease", "dealership",
                              "dealer", "title", "repo"],
}
with open("taxonomy.json", "w") as f:
    json.dump(TAXONOMY, f, indent=2)

MOST_COMMON = pd.Series(y_product[train_idx]).mode()[0]


def keyword_route(tokens):
    bag = Counter(tokens + [part for t in tokens if "_" in t for part in t.split("_")])
    counts = {p: sum(bag[k] for k in words) for p, words in TAXONOMY.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else MOST_COMMON


centroids = np.vstack([np.mean([w2v.wv[k] for k in words if k in w2v.wv], axis=0) for words in TAXONOMY.values()])
centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
TEAM_NAMES = np.array(list(TAXONOMY))


def taxonomy_route(X):
    X_norm = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    return TEAM_NAMES[np.argmax(X_norm @ centroids.T, axis=1)]


ROUTER_C = [0.1, 0.3, 1.0, 3.0]
router_rows = []
for C in ROUTER_C:
    candidate = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=C))
    candidate.fit(X_w2v[train_idx], y_product[train_idx])
    router_rows.append({"C (lower = simpler model)": C,
                        "validation accuracy": accuracy_score(y_product[val_idx], candidate.predict(X_w2v[val_idx]))})
router_tuning = pd.DataFrame(router_rows)
ROUTER_BEST_C = float(router_tuning.loc[router_tuning["validation accuracy"].idxmax(), "C (lower = simpler model)"])
ACC_VAL_ROUTER = float(router_tuning["validation accuracy"].max())
display(router_tuning.style.format({"validation accuracy": "{:.1%}"}))

router = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=ROUTER_BEST_C))
router.fit(X_w2v[train_idx], y_product[train_idx])

timings = {}
start = time.time()
pred_kw = np.array([keyword_route(t) for t in df.loc[test_idx, "tokens"]])
timings["Keyword rules (today)"] = (time.time() - start) / len(test_idx) * 1000
start = time.time()
pred_tax = taxonomy_route(X_w2v[test_idx])
timings["Word2Vec + taxonomy (no labels)"] = (time.time() - start) / len(test_idx) * 1000
start = time.time()
pred_lr = router.predict(X_w2v[test_idx])
timings["Word2Vec + logistic regression"] = (time.time() - start) / len(test_idx) * 1000
scored_rows = np.concatenate([val_idx, test_idx])
route_pred = dict(zip(scored_rows, router.predict(X_w2v[scored_rows])))
print(f"Router: C = {ROUTER_BEST_C} chosen on validation ({ACC_VAL_ROUTER:.1%}), "
      f"evaluated on {len(test_idx):,} test complaints")
```

## Step 11 of 28: 6. Prioritise, text (text cell)

**Find it:** search for `which complaints will end in a payout`

**What changes:** One new bullet: the payout model's C is chosen on validation.

```markdown
## 6. Prioritise: which complaints will end in a payout?

**What it does.** It gives every new complaint a payout risk, so the team reviews the riskiest ones first.

**Ground truth.** How the company really closed the complaint: "Closed with monetary relief" or not. Only about 1 complaint in 13 ends in a payout, so accuracy would be misleading: a model that always says "no payout" would be 93% accurate. We use these measures instead:

- **Payouts caught in the riskiest 20%:** if staff review the 20% highest-risk complaints, what share of all payouts do they catch? Random review catches 20%.
- **PR-AUC** (average precision) and **ROC-AUC**.

**Design choices.**

- Logistic regression on the Word2Vec complaint vector plus the product and the length of the story.
- Balanced class weights, because payouts are rare. The low-scoring example dealt with imbalance by throwing data away (down-sampling) and then reported accuracy.
- Compared with a product-only model, to show what the text adds beyond the product.
- The model's strength (C) is chosen on the validation split by PR-AUC; the test split only gives the final numbers.
```

## Step 12 of 28: 6. Payout model (code cell)

**Find it:** search for `product_onehot = pd.get_dummies`

**What changes:** C (0.1, 0.5, 2) is chosen on validation by PR-AUC instead of being fixed at 0.5.

```python
product_onehot = pd.get_dummies(df["product"]).reindex(columns=PRODUCTS, fill_value=0).to_numpy(dtype=float)
X_pay = np.hstack([X_w2v, product_onehot, np.log1p(df["n_words"].to_numpy(dtype=float))[:, None]])
y_pay = df["payout"].to_numpy()
y_pay_test = y_pay[test_idx]


def caught_at(y, scores, share=0.2):
    top = np.argsort(-scores, kind="stable")[:int(round(len(scores) * share))]
    return y[top].sum() / y.sum()


PAYOUT_C = [0.1, 0.5, 2.0]
payout_rows = []
for C in PAYOUT_C:
    candidate = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=C))
    candidate.fit(X_pay[train_idx], y_pay[train_idx])
    val_scores = candidate.predict_proba(X_pay[val_idx])[:, 1]
    payout_rows.append({"C (lower = simpler model)": C,
                        "validation PR-AUC": average_precision_score(y_pay[val_idx], val_scores),
                        "validation payouts caught in riskiest 20%": caught_at(y_pay[val_idx], val_scores)})
payout_tuning = pd.DataFrame(payout_rows)
PAYOUT_BEST_C = float(payout_tuning.loc[payout_tuning["validation PR-AUC"].idxmax(), "C (lower = simpler model)"])
display(payout_tuning.style.format({"validation PR-AUC": "{:.3f}", "validation payouts caught in riskiest 20%": "{:.1%}"}))

product_model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(product_onehot[train_idx], y_pay[train_idx])
payout_model = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=3000, class_weight="balanced", C=PAYOUT_BEST_C))
payout_model.fit(X_pay[train_idx], y_pay[train_idx])

rng = np.random.default_rng(SEED)
jitter = rng.random(len(test_idx)) * 1e-9
payout_scores = {
    "Random review": rng.random(len(test_idx)),
    "Product only": product_model.predict_proba(product_onehot[test_idx])[:, 1] + jitter,
    "Word2Vec text + product": payout_model.predict_proba(X_pay[test_idx])[:, 1],
}

payout_results = pd.DataFrame([{
    "model": name,
    "PR-AUC": average_precision_score(y_pay_test, s),
    "ROC-AUC": roc_auc_score(y_pay_test, s),
    "payouts caught in riskiest 20%": caught_at(y_pay_test, s),
    "payout rate in riskiest 20%": y_pay_test[np.argsort(-s, kind="stable")[:int(round(len(s) * 0.2))]].mean(),
} for name, s in payout_scores.items()]).set_index("model")
print(f"Payout model: C = {PAYOUT_BEST_C} chosen on validation. "
      f"Payout rate in the test set: {y_pay_test.mean():.1%} ({y_pay_test.sum():,} payouts)")
display(payout_results.style.format("{:.3f}", subset=["PR-AUC", "ROC-AUC"])
                      .format("{:.1%}", subset=["payouts caught in riskiest 20%", "payout rate in riskiest 20%"]))
CAUGHT_20 = payout_results.loc["Word2Vec text + product", "payouts caught in riskiest 20%"]
```

## Step 13 of 28: 7. Watch, text (text cell)

**Find it:** search for `main method 2`

**What changes:** New first bullet on how LDA settings are chosen (validation) and scored (test). The number of topics is now chosen on coherence and agreement with CFPB issues together.

```markdown
## 7. Watch: LDA topic modeling (main method 2)

**What it does.** It finds the topics customers complain about and counts them week by week. When a topic jumps to 3× its normal weekly volume, managers get an alarm, and section 8 explains it.

**Design choices (the tuning the brief asks for).**

- **How settings are chosen:** each candidate is fitted on 8,000 training complaints and scored on the validation complaints. The chosen setting is then fitted on all training and validation complaints and scored once on the test complaints, against an untuned model fitted on the same complaints. Coherence is measured on complaints the model did not see; a topic whose top words never appear in them cannot be scored and is left out of the average.
- **N-grams and number of topics:** single words, and single words plus learned word pairs, each tried with 6 to 20 topics. Each combination gets two scores: topic coherence (c_v: the top words of a topic tend to appear together, so people can read it) and agreement with the CFPB issue labels (NMI, see below). Each score ranks the combinations, and the best total rank wins. Coherence alone tends to prefer a few broad topics of common words, too broad to raise a useful alarm; the tuning cell prints what coherence alone would have chosen.
- **Minimum and maximum document frequency:** words in fewer than 10 complaints are dropped, and the share of complaints above which a word is too common to help (0.2, 0.4, 0.7) is tuned for the winning combination in the same way.
- **Untuned baseline:** only standard English stop words removed, no frequency filtering, 10 topics. This is the low-scoring example's setup, whose topics were full of "like", "get" and "im".
- **Words that describe no problem are removed:** contractions ("didnt", "cant"), words about the conversation itself ("told", "saying", "gave"), and the names of banks, apps and credit bureaus (Wells Fargo, Cash App, Zelle, Experian, ...). Names are removed so the topics describe *problems*, not companies; a surge at one bank still shows up, because its complaints share one problem. Our first run still had "didn" and "american express" among the top words, the same kind of mistake as the low-scoring example.
- Trained on unique complaints, so a campaign does not take over a topic. Then applied to *all* complaints, because the volume of copies is part of the alarm signal.

**Evaluation (on the test complaints).**

1. Coherence compared with the untuned baseline. Coherence alone can be fooled, because filler words that appear everywhere also appear together, so we also measure:
2. Agreement with the CFPB's own issue labels (normalized mutual information): do the topics separate real problems? Compared with the untuned model and with random topics.
3. Whether the alarm fires after real, dated events in January 2025.
4. Whether watching topics adds anything to simply counting complaints, which needs no AI at all.
```

## Step 14 of 28: 7. LDA setup (code cell)

**Find it:** search for `LDA_STOP = {`

**What changes:** The word lists are unchanged. The coherence function now skips a topic that cannot be scored, and the last lines fit on training complaints and score on validation complaints.

```python
LDA_STOP = {
    "amount", "told", "said", "say", "says", "called", "call", "calls", "calling", "time", "times", "day", "days",
    "would", "could", "also", "get", "got", "getting", "back", "even", "one", "two", "still", "never", "want",
    "wanted", "made", "make", "know", "like", "since", "asked", "ask", "asking", "received", "receive", "help",
    "helped", "contacted", "contact", "phone", "number", "please", "company", "customer", "service", "just",
    "going", "went", "did", "does", "way", "need", "needed", "dont", "didnt", "able", "use", "used", "new",
    "month", "months", "year", "years", "week", "weeks", "today", "however", "multiple", "without", "another",
    "regarding", "request", "requested", "stated", "state", "let", "take", "took", "tried", "try", "trying",
    "sent", "send", "provided", "provide", "informed", "issue", "issues", "problem", "complaint", "complaints",
    "file", "filed", "filing", "cfpb", "bureau", "consumer", "financial", "protection", "letter", "email",
    "information", "thank", "thanks", "sincerely", "currently", "regards", "understand", "despite", "review",
    "consumers", "documentation", "required", "accordance", "pursuant", "formally", "writing", "concerns",
    "wells", "fargo", "wells_fargo", "bank_america", "america", "chase", "citi", "citibank", "capital",
    "capital_one", "navy", "federal", "navy_federal", "sofi", "synchrony", "discover", "american_express",
    "amex", "barclays", "pnc", "truist", "ally", "santander", "goldman", "sachs", "chime", "usaa", "citizens",
    "huntington", "regions", "comerica", "flagstar", "shellpoint", "rocket", "carvana", "westlake",
}
CONTRACTIONS = {"cant", "wont", "doesnt", "isnt", "wasnt", "werent", "arent", "havent", "hasnt", "hadnt", "couldnt",
                "wouldnt", "shouldnt", "ive", "youre", "theyre", "weve", "theyve", "youve", "ill"}
CONVERSATION = {"saying", "tell", "telling", "talk", "talked", "talking", "spoke", "speak", "gave", "give", "giving",
                "doing", "happened", "right", "explained", "explain"}
NAMES = {"paypal", "venmo", "zelle", "zelles", "cashapp", "cash_app", "coinbase", "robinhood", "experian", "equifax",
         "transunion", "bread", "cooper", "ocwen", "pennymac", "bridgecrest", "bmo", "usbank", "ews", "comenity",
         "bancorp", "affirm", "klarna", "afterpay", "varo", "american", "express", "bofa", "boa", "jpmorgan", "morgan",
         "keybank", "mtb", "suntrust", "lakeview", "nationstar", "newrez", "loancare", "carrington", "selene",
         "midfirst", "exeter", "toyota", "honda", "ford", "nissan", "hyundai", "kia", "tesla"}
LDA_STOP |= CONTRACTIONS | CONVERSATION | NAMES
NAME_PATTERN = re.compile(r"\b(cash ?app|american express|bank of america|wells fargo|capital one|navy federal|"
                          r"goldman sachs|early warning( services)?|credit one|td bank|us bank|fifth third|"
                          r"mr cooper|credit acceptance|apple card|apple pay|google pay)\b")


def lda_tokens(tokens):
    return [t for t in tokens if t not in LDA_STOP and not any(part in LDA_STOP for part in t.split("_"))]


def unigram_tokens(i):
    return [t for t in tokenize(NAME_PATTERN.sub(" ", df.at[i, "clean"])) if t not in LDA_STOP]


def build_corpus(texts, no_below, no_above, keep_n=20000):
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    return dictionary, [dictionary.doc2bow(t) for t in texts]


def fit_lda(dictionary, corpus, k, passes=4):
    return LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=passes, chunksize=2000,
                    alpha="auto", eta="auto", random_state=SEED, eval_every=None)


def coherence(model, texts, dictionary):
    with np.errstate(divide="ignore", invalid="ignore"):
        per_topic = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence="c_v",
                                   processes=1).get_coherence_per_topic()
    return float(np.nanmean(per_topic))


df["lda_tokens"] = df["tokens"].map(lda_tokens)
unique_ids = np.sort(unique.index.to_numpy())
TUNE_N = min(8000, len(train_idx))
tune_ids = np.random.default_rng(SEED).choice(train_idx, TUNE_N, replace=False)
tune_pairs = [df.at[i, "lda_tokens"] for i in tune_ids]
tune_single = [unigram_tokens(i) for i in tune_ids]
tune_raw = [tokenize(df.at[i, "clean"]) for i in tune_ids]
val_pairs = [df.at[i, "lda_tokens"] for i in val_idx]
val_single = [unigram_tokens(i) for i in val_idx]
val_raw = [tokenize(df.at[i, "clean"]) for i in val_idx]
val_issues = df.loc[val_idx, "issue"].to_numpy()
NO_BELOW = 10
print(f"Settings are fitted on {TUNE_N:,} training complaints and scored on {len(val_idx):,} validation complaints")
```

## Step 15 of 28: 7. LDA tuning (code cell)

**Find it:** search for `K_GRID = [6, 8, 10, 12, 15, 20]`

**What changes:** Every setting is scored on the validation complaints on two measures, coherence and agreement with CFPB issues (NMI); the best total rank wins. It also prints what coherence alone would have chosen. The chart (fig07) now has two panels.

```python
K_GRID = [6, 8, 10, 12, 15, 20]
TEXTS = {"single": tune_single, "pairs": tune_pairs}
VAL_TEXTS = {"single": val_single, "pairs": val_pairs}
LABELS = {"single": "Single words", "pairs": "Words + word pairs"}
NMI_COL = "agreement with CFPB issues (NMI)"


def issue_nmi(model, dictionary, texts, issues):
    dominant = [max(model.get_document_topics(dictionary.doc2bow(t), minimum_probability=0.0), key=lambda x: x[1])[0]
                for t in texts]
    return normalized_mutual_info_score(issues, dominant)


def validation_scores(model, dictionary, kind):
    return {"coherence": coherence(model, VAL_TEXTS[kind], dictionary),
            NMI_COL: issue_nmi(model, dictionary, VAL_TEXTS[kind], val_issues)}


def best_rank(table):
    total = table["coherence"].rank(ascending=False) + table[NMI_COL].rank(ascending=False)
    return table.assign(total_rank=total).sort_values(["total_rank", "coherence"], ascending=[True, False]).index[0]


sweep = []
for kind, texts in TEXTS.items():
    d, c = build_corpus(texts, NO_BELOW, 0.4)
    for k in K_GRID:
        start = time.time()
        sweep.append({"texts": kind, "topics": k, **validation_scores(fit_lda(d, c, k), d, kind)})
        print(f"{LABELS[kind]:<18} k={k:>2}: validation c_v {sweep[-1]['coherence']:.3f}, "
              f"NMI {sweep[-1][NMI_COL]:.3f} ({time.time() - start:.0f} s)")
sweep = pd.DataFrame(sweep)
best = sweep.loc[best_rank(sweep)]
BEST_K, best_kind = int(best["topics"]), best["texts"]
coherence_only = sweep.loc[sweep["coherence"].idxmax()]
print(f"Coherence alone would choose: {LABELS[coherence_only['texts']].lower()}, {int(coherence_only['topics'])} topics. "
      f"Both measures together choose: {LABELS[best_kind].lower()}, {BEST_K} topics.")

tuning = []
for no_above in [0.2, 0.4, 0.7]:
    d, c = build_corpus(TEXTS[best_kind], NO_BELOW, no_above)
    tuning.append({"setting": f"{LABELS[best_kind]}, {BEST_K} topics, max share {no_above}", "texts": best_kind,
                   "no_above": no_above, "topics": BEST_K, **validation_scores(fit_lda(d, c, BEST_K), d, best_kind)})
untuned_dictionary, c = build_corpus(tune_raw, 1, 1.0, None)
untuned = fit_lda(untuned_dictionary, c, 10)
tuning.append({"setting": "Untuned: standard stop words only, no filtering, 10 topics", "texts": "raw",
               "no_above": 1.0, "topics": 10, "coherence": coherence(untuned, val_raw, untuned_dictionary),
               NMI_COL: issue_nmi(untuned, untuned_dictionary, val_raw, val_issues)})
tuning = pd.DataFrame(tuning)
print("Scores on the validation complaints:")
display(tuning[["setting", "coherence", NMI_COL]].style.format({"coherence": "{:.3f}", NMI_COL: "{:.3f}"}))

UNTUNED_COHERENCE = tuning.loc[tuning["texts"] == "raw", "coherence"].iloc[0]
UNTUNED_NMI = tuning.loc[tuning["texts"] == "raw", NMI_COL].iloc[0]
best_setting = tuning.loc[best_rank(tuning[tuning["texts"] != "raw"])]
print(f"Chosen on validation: {best_setting['setting']} (c_v {best_setting['coherence']:.3f}, "
      f"NMI {best_setting[NMI_COL]:.3f}; untuned {UNTUNED_COHERENCE:.3f} and {UNTUNED_NMI:.3f})")
print("\nTop words of the untuned topics:")
for t in range(10):
    print(f"  {t}: {', '.join(w for w, _ in untuned.show_topic(t, topn=10))}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, metric, label, untuned_value in [(axes[0], "coherence", "Coherence (c_v)", UNTUNED_COHERENCE),
                                         (axes[1], NMI_COL, "Agreement with CFPB issues (NMI)", UNTUNED_NMI)]:
    for color, kind in zip(SERIES, TEXTS):
        part = sweep[sweep["texts"] == kind]
        ax.plot(part["topics"], part[metric], color=color, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=LABELS[kind])
    ax.axhline(untuned_value, color=MUTED, linestyle="--", linewidth=1.5, label="Untuned baseline (10 topics)")
    ax.scatter([BEST_K], [best[metric]], s=260, facecolor="none", edgecolor=INK, linewidth=1.5, zorder=4)
    ax.set_xticks(K_GRID)
    ax.set_xlabel("Number of topics")
    ax.set_title(f"{label}, validation complaints", fontsize=11)
axes[0].legend(loc="best")
fig.suptitle(f"Chosen on validation: {LABELS[best_kind].lower()}, {BEST_K} topics (circled), the best rank "
             "on both measures", x=0.01, y=1.05, ha="left", fontsize=13, fontweight="bold")
save_fig(fig, "fig07_lda_coherence")
```

## Step 16 of 28: 7. Final LDA (code cell)

**Find it:** search for `use_pairs = best_setting`

**What changes:** The final LDA is fitted on training + validation and scored once on test, against an untuned LDA fitted on the same complaints. Topic naming is unchanged.

```python
use_pairs = best_setting["texts"] == "pairs"
texts_all = df["lda_tokens"].tolist() if use_pairs else [unigram_tokens(i) for i in range(len(df))]
fit_ids = np.sort(np.concatenate([train_idx, val_idx]))
NO_BELOW_FINAL = max(NO_BELOW, round(20 * SCALE))
PASSES = 4 if QUICK_RUN else 8
final_dictionary, final_corpus = build_corpus([texts_all[i] for i in fit_ids], NO_BELOW_FINAL,
                                              best_setting["no_above"])
start = time.time()
lda = fit_lda(final_dictionary, final_corpus, BEST_K, passes=PASSES)
print(f"Final LDA on {len(fit_ids):,} training and validation complaints in {time.time() - start:.0f} s")

untuned_final_dictionary, untuned_final_corpus = build_corpus([tokenize(df.at[i, "clean"]) for i in fit_ids],
                                                              1, 1.0, None)
untuned_final = fit_lda(untuned_final_dictionary, untuned_final_corpus, 10, passes=PASSES)
test_raw = [tokenize(df.at[i, "clean"]) for i in test_idx]
LDA_COHERENCE = coherence(lda, [texts_all[i] for i in test_idx], final_dictionary)
UNTUNED_TEST_COHERENCE = coherence(untuned_final, test_raw, untuned_final_dictionary)
UNTUNED_TEST_NMI = issue_nmi(untuned_final, untuned_final_dictionary, test_raw, df.loc[test_idx, "issue"].to_numpy())
print(f"On {len(test_idx):,} test complaints: c_v {LDA_COHERENCE:.3f}, "
      f"untuned model fitted on the same complaints {UNTUNED_TEST_COHERENCE:.3f}")

top_words = {t: [w for w, _ in lda.show_topic(t, topn=12)] for t in range(BEST_K)}
NAMING_SYSTEM = ("You name topics found in bank customer complaints. Reply with ONE short name of at most 4 words "
                 "that describes the customer's problem, for example: Overdraft fees. No lists, no line breaks.")
topic_names = {}
for t, words in top_words.items():
    if RUN_OPENAI:
        taken = ", ".join(topic_names.values()) or "none"
        name, _, _ = ask_openai(NAMING_SYSTEM, f"Most important words: {', '.join(words)}\nNames already used: {taken}",
                                max_tokens=12)
        name = name.strip().splitlines()[0].strip().strip('"').strip(".")
        topic_names[t] = " ".join(name.split()[:4])
    else:
        topic_names[t] = " / ".join(words[:3])
print(topic_names)
```

## Step 17 of 28: 7. Topics vs CFPB issues (code cell)

**Find it:** search for `u = df.loc[unique_ids]`

**What changes:** Only the first lines change: the NMI is measured on the test complaints, next to the untuned model.

```python
u = df.loc[test_idx]
NMI = normalized_mutual_info_score(u["issue"], u["topic"])
NMI_RANDOM = normalized_mutual_info_score(u["issue"], np.random.default_rng(SEED).permutation(u["topic"].to_numpy()))
print(f"Agreement with the CFPB issue labels on test complaints (NMI): {NMI:.3f}, "
      f"untuned {UNTUNED_TEST_NMI:.3f}, random topics {NMI_RANDOM:.3f}")

top_issues = u["issue"].value_counts().index[:8]
heat = pd.crosstab(u["topic"], u["issue"], normalize="index").reindex(columns=top_issues, fill_value=0)
heat = heat.reindex(topic_table.index, fill_value=0)
fig, ax = plt.subplots(figsize=(11, 0.55 * BEST_K + 2.2))
ax.imshow(heat.to_numpy(), cmap=BLUES, vmin=0, vmax=1, aspect="auto")
ax.grid(False)
ax.set_yticks(range(len(heat)), [topic_names[t] for t in heat.index])
ax.set_xticks(range(len(top_issues)), [i if len(i) <= 32 else i[:30] + "…" for i in top_issues], rotation=30, ha="right")
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        value = heat.iat[i, j]
        if value >= 0.1:
            ax.text(j, i, f"{value:.0%}", ha="center", va="center", color="white" if value > 0.5 else INK, fontsize=9)
ax.set_title(f"Topics line up with the CFPB issue labels (NMI {NMI:.2f} vs {NMI_RANDOM:.2f} for random topics)")
save_fig(fig, "fig08_topics_vs_cfpb_issues")
```

## Step 18 of 28: 8. Explain, text (text cell)

**Find it:** search for `main method 3`

**What changes:** Two bullets updated: the search engine is chosen on validation, and the agent gets 8 held-out questions.

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

- **Search engine** (8.1): precision@5 for finding past cases with the same issue, compared with keyword search, plain FAISS, a filter on the routed team, hybrid search, and the Word2Vec document vectors from our midterm. The engine is chosen on validation complaints and measured on test complaints.
- **Agent** (8.3): a test suite in the style of Assignment 4: 16 questions with known correct filters, a first prompt (v1) against an improved prompt (v2), and the share of checks each passes, plus 8 held-out questions written after prompt v2 was fixed.
- **Briefs** (8.5): two brief versions (v1 and v2, see 8.4), one brief per alarm, with and without the filters. Does it name the company behind the surge (taken from the complaint records, never given to the model as a label)? Are the three January events named in that week's briefs? Then RAGAs faithfulness and answer relevancy, and an LLM judge spot-checked by a person.
```

## Step 19 of 28: 8.1 Choosing the search engine, text (text cell)

**Find it:** search for `Choosing the search engine`

**What changes:** Says that the engine is chosen on validation and measured on test.

```markdown
### 8.1 Choosing the search engine

The assistant first needs good search. We test search methods on a task with a known answer: for a new complaint, find the 5 most similar past complaints (training split) and count how many have the same CFPB issue (precision@5). The engine is chosen on 400 validation complaints and then measured on 400 different test complaints, so the choice cannot flatter the result. The knowledge base is the training split only, so a complaint never finds itself. The newest row is our midterm's method: each complaint as the average of its Word2Vec vectors, compared by cosine similarity.
```

## Step 20 of 28: 8.1 Search evaluation (code cell)

**Find it:** search for `EVAL_N = 100 if QUICK_RUN`

**What changes:** Every engine is scored on 400 validation and 400 test complaints; the best on validation is chosen and reported on test.

```python
EVAL_N = 100 if QUICK_RUN else 400
val_rows = pd.Series(val_idx).sample(EVAL_N, random_state=SEED).to_numpy()
eval_rows = pd.Series(test_idx).sample(EVAL_N, random_state=SEED).to_numpy()
rng = np.random.default_rng(SEED)
search_methods = {
    "Random past cases (routed team)": lambda r: random_search(r, route_pred[r]),
    "Keyword search (BM25)": lambda r: bm25_search(r),
    "FAISS (meaning)": lambda r: dense_search(r),
    "FAISS + routed-team filter": lambda r: dense_search(r, route_pred[r]),
    "Hybrid FAISS + BM25": lambda r: hybrid_search(r),
    "Word2Vec document vectors (midterm)": lambda r: word2vec_search(r),
}


def share_same(row, hits, field):
    return np.mean([df.at[h, field] == df.at[row, field] for h in hits]) if hits else 0.0


retrieval_rows, p5_rows = [], {}
for name, search in search_methods.items():
    start = time.time()
    val_hits = [(r, search(r)) for r in val_rows]
    test_hits = [(r, search(r)) for r in eval_rows]
    p5_rows[name] = np.array([share_same(r, h, "issue") for r, h in test_hits])
    retrieval_rows.append({"method": name,
                           "same issue, validation": np.mean([share_same(r, h, "issue") for r, h in val_hits]),
                           "same issue, test": p5_rows[name].mean(),
                           "same product, test": np.mean([share_same(r, h, "product") for r, h in test_hits]),
                           "ms per query": (time.time() - start) / (2 * EVAL_N) * 1000})
retrieval = pd.DataFrame(retrieval_rows).set_index("method")
print(f"Precision@5 on {EVAL_N} validation and {EVAL_N} test complaints: "
      "the share of the 5 past cases found that have the same issue (or product)")
display(retrieval.style.format({"same issue, validation": "{:.3f}", "same issue, test": "{:.3f}",
                                "same product, test": "{:.3f}", "ms per query": "{:.1f}"}))
P5_BEST_NAME = retrieval.drop(index="Random past cases (routed team)")["same issue, validation"].idxmax()
P5_OURS = retrieval.loc[P5_BEST_NAME, "same issue, test"]
P5_W2V = retrieval.loc["Word2Vec document vectors (midterm)", "same issue, test"]
P5_FAISS = retrieval.loc["FAISS (meaning)", "same issue, test"]
P5_BM25 = retrieval.loc["Keyword search (BM25)", "same issue, test"]
P5_RANDOM = retrieval.loc["Random past cases (routed team)", "same issue, test"]

values = retrieval["same issue, test"].sort_values()
colors = [SERIES[0] if name == P5_BEST_NAME else MUTED for name in values.index]
fig, ax = plt.subplots(figsize=(9, 3.8))
bars = ax.barh(values.index, values.values, color=colors, height=0.6)
ax.bar_label(bars, labels=[f"{v:.2f}" for v in values.values], padding=4, color=INK_2)
ax.set_xlim(0, 1)
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
ax.set_xlabel(f"Share of the top 5 past cases with the same CFPB issue ({EVAL_N} test complaints; "
              "blue = chosen on validation)")
ax.set_title(f"{P5_BEST_NAME}: same issue in {P5_OURS:.0%} of the top 5 cases; keyword search {P5_BM25:.0%}")
save_fig(fig, "fig11_retrieval_precision")

w2v_note = ("averaging word vectors loses detail that sentence embeddings keep" if P5_W2V < P5_FAISS
            else "on par with sentence embeddings")
display(Markdown(f"**Chosen on the validation complaints, for the case card below: {P5_BEST_NAME}** "
                 f"(test precision@5 {P5_OURS:.2f}, BM25 {P5_BM25:.2f}). "
                 f"Our midterm's Word2Vec document vectors reach {P5_W2V:.2f}: {w2v_note}."))
```

## Step 21 of 28: 8.3 Agent test suite, text (text cell)

**Find it:** search for `16 questions with their correct answers`

**What changes:** New paragraph at the end: **Held-out questions**.

```markdown
**Agent test suite.** 16 questions with their correct answers: two small-talk messages, questions with a company, a product, a week, a month, a topic from section 7, and a question about 2023 (outside our data), where the right behaviour is to search and then find nothing. Each answer gets six automatic checks: valid JSON, search or not, product, company, dates (within 3 days) and topic.

**Held-out questions.** Prompt v2 was improved while looking at these 16 questions, so its score on them is a development result. Eight more questions were written after v2 was fixed and were never used to change it: a greeting, a bank by its short name (Chase), a product and a week, a company and a product together, a company with no dates, a month, another topic from section 7, and a year outside the data. They get the same six checks.
```

## Step 22 of 28: 8.3 Agent tests (code cell)

**Find it:** search for `def week_of(day):`

**What changes:** Adds 8 held-out questions, written after prompt v2 was fixed. The 16 test questions and the chart are unchanged.

```python
def week_of(day):
    start = pd.Timestamp(day)
    return start, start + pd.Timedelta(days=6)


def month_of(year, month):
    start = pd.Timestamp(year=year, month=month, day=1)
    return start, start + pd.offsets.MonthEnd(0)


ranked_alarms = alarm_review.sort_values("times normal", ascending=False)
topic_a = int(ranked_alarms.iloc[0]["topic"]) if len(ranked_alarms) else int(df["topic"].mode()[0])
week_a = ranked_alarms.iloc[0]["week"] if len(ranked_alarms) else pd.Timestamp("2025-01-13")
topic_b = next(int(t) for t in df["topic"].value_counts().index if t != topic_a)
topic_c = next(int(t) for t in df["topic"].value_counts().index if t not in (topic_a, topic_b))

AGENT_TESTS = [
    ("Hello! What can you do?", False, None, None, None, None),
    ("Thanks, that was helpful.", False, None, None, None, None),
    ("Why were Capital One customers complaining in the week of 13 January?", True, None,
     "CAPITAL ONE FINANCIAL CORPORATION", week_of("2025-01-13"), None),
    ("What did Cash App users report in January 2025?", True, None, "Block, Inc.", month_of(2025, 1), None),
    ("What problems did mortgage customers have in March?", True, PRODUCTS[3], None, month_of(2025, 3), None),
    ("Show me complaints about Zelle from the week of 6 October.", True, None, "Early Warning Services, LLC",
     week_of("2025-10-06"), None),
    ("What are people saying about overdraft fees at Navy Federal?", True, None, "NAVY FEDERAL CREDIT UNION", None, None),
    ("Which vehicle loan complaints mention repossession in June?", True, PRODUCTS[4], None, month_of(2025, 6), None),
    ("What did credit card customers complain about in the week of 17 November?", True, PRODUCTS[1], None,
     week_of("2025-11-17"), None),
    ("What did customers say about Wells Fargo in 2023?", True, None, "WELLS FARGO & COMPANY",
     (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")), None),
    (f"Why did complaints about {topic_names[topic_a].lower()} jump in the week of {week_a:%d %B}?", True, None, None,
     week_of(week_a), topic_a),
    (f"What are customers saying about {topic_names[topic_b].lower()}?", True, None, None, None, topic_b),
    ("Were there complaints about Chime disputes in early October?", True, None, "Chime Financial Inc",
     (pd.Timestamp("2025-10-01"), pd.Timestamp("2025-10-10")), None),
    ("What happened with money transfer complaints between 13 and 26 January?", True, PRODUCTS[2], None,
     (pd.Timestamp("2025-01-13"), pd.Timestamp("2025-01-26")), None),
    ("How did Bank of America customers describe checking account problems in April?", True, PRODUCTS[0],
     "BANK OF AMERICA, NATIONAL ASSOCIATION", month_of(2025, 4), None),
    ("Give me complaints about PayPal from the week of 3 March.", True, None, "Paypal Holdings, Inc",
     week_of("2025-03-03"), None),
]
AGENT_TESTS_HELDOUT = [
    ("Good morning!", False, None, None, None, None),
    ("What did Chase customers complain about in February?", True, None, "JPMORGAN CHASE & CO.", month_of(2025, 2), None),
    ("Were there mortgage complaints about escrow in the week of 8 September?", True, PRODUCTS[3], None,
     week_of("2025-09-08"), None),
    ("What did Citibank credit card customers report in July?", True, PRODUCTS[1], "CITIBANK, N.A.",
     month_of(2025, 7), None),
    ("What are Synchrony customers unhappy about?", True, None, "SYNCHRONY FINANCIAL", None, None),
    ("Which vehicle loan complaints came in during December?", True, PRODUCTS[4], None, month_of(2025, 12), None),
    (f"What were customers saying about {topic_names[topic_c].lower()} in May?", True, None, None, month_of(2025, 5),
     topic_c),
    ("What did Capital One customers report in 2024?", True, None, "CAPITAL ONE FINANCIAL CORPORATION",
     (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")), None),
]
CHECKS = ["valid JSON", "search or not", "product", "company", "dates", "topic"]


def dates_ok(decision, expected):
    if expected is None:
        return decision["date_from"] is None and decision["date_to"] is None
    if decision["date_from"] is None or decision["date_to"] is None:
        return False
    slack = pd.Timedelta(days=3)
    return abs(decision["date_from"] - expected[0]) <= slack and abs(decision["date_to"] - expected[1]) <= slack


def run_checks(decision, case):
    _, retrieve, product, company, dates, topic = case
    return {
        "valid JSON": decision["valid_json"],
        "search or not": decision["retrieve"] == retrieve,
        "product": decision["product"] == product,
        "company": decision["company"] is None if company is None else company in resolve_companies(decision["company"]),
        "dates": dates_ok(decision, dates),
        "topic": decision["topic"] == topic,
    }


def run_suite(tests):
    rows = []
    for version, prompt in [("v1", AGENT_PROMPT_V1), ("v2", AGENT_PROMPT_V2)]:
        for n, case in enumerate(tests):
            decision = agent_decide(case[0], prompt)
            rows.append({"prompt": version, "question": n, "text": case[0], **run_checks(decision, case),
                         "reply": decision["reply"]})
    results = pd.DataFrame(rows)
    rates = results.groupby("prompt")[CHECKS].mean().T
    rates.loc["all checks"] = rates.mean()
    return results, rates


AGENT_V1 = AGENT_V2 = AGENT_HELDOUT_V1 = AGENT_HELDOUT_V2 = None
agent_results = heldout_results = None
if RUN_QWEN:
    start = time.time()
    agent_results, pass_rates = run_suite(AGENT_TESTS)
    heldout_results, heldout_rates = run_suite(AGENT_TESTS_HELDOUT)
    AGENT_V1, AGENT_V2 = pass_rates.loc["all checks", "v1"], pass_rates.loc["all checks", "v2"]
    AGENT_HELDOUT_V1, AGENT_HELDOUT_V2 = heldout_rates.loc["all checks", "v1"], heldout_rates.loc["all checks", "v2"]
    print(f"{len(agent_results) + len(heldout_results)} agent answers in {time.time() - start:.0f} s")
    display(pd.concat({f"{len(AGENT_TESTS)} test questions": pass_rates,
                       f"{len(AGENT_TESTS_HELDOUT)} held-out questions": heldout_rates}, axis=1).style.format("{:.0%}"))

    v2 = pd.concat([agent_results.assign(questions="test"), heldout_results.assign(questions="held-out")])
    v2 = v2[v2["prompt"] == "v2"].copy()
    v2["failed checks"] = v2[CHECKS].apply(lambda r: ", ".join(c for c in CHECKS if not r[c]), axis=1)
    v2_failed = v2[v2["failed checks"] != ""]
    print(f"Prompt v2 missed at least one check on {len(v2_failed)} of {len(v2)} questions")
    if len(v2_failed):
        display(v2_failed[["questions", "text", "failed checks", "reply"]].assign(
            reply=v2_failed["reply"].str.slice(0, 220)))

    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = np.arange(len(pass_rates))
    ax.barh(y - 0.2, pass_rates["v2"], height=0.38, color=SERIES[0], label="Prompt v2 (rules and examples)")
    ax.barh(y + 0.2, pass_rates["v1"], height=0.38, color=MUTED, label="Prompt v1 (first draft)")
    for yy, value in zip(y, pass_rates["v2"]):
        ax.text(value + 0.01, yy - 0.2, f"{value:.0%}", va="center", color=INK_2, fontsize=9)
    ax.set_yticks(y, pass_rates.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.set_title(f"The agent passes {AGENT_V2:.0%} of its test checks with prompt v2 (v1: {AGENT_V1:.0%})")
    save_fig(fig, "fig12_agent_test_suite")
else:
    print("Agent tests skipped: Qwen3-4B needs a GPU runtime (RUN_QWEN = False)")
```

## Step 23 of 28: 8.5 Human spot-check scores (code cell)

**Find it:** search for `HUMAN_USEFULNESS = {0: 1`

**What changes:** Your old scores are cleared, because the briefs will be new. You fill them in again after the run (see the end of this guide).

```python
HUMAN_USEFULNESS = {}
JUDGE_AGREEMENT = None
if HUMAN_USEFULNESS and judge_table is not None:
    compare = pd.DataFrame({"human": pd.Series(HUMAN_USEFULNESS), "judge": judge_table["usefulness"]}).dropna()
    compare["difference"] = compare["judge"] - compare["human"]
    JUDGE_AGREEMENT = (compare["difference"].abs() <= 1).mean()
    display(compare)
    print(f"Judge within 1 point of the human on {JUDGE_AGREEMENT:.0%} of briefs, "
          f"mean difference {compare['difference'].mean():+.2f}")
else:
    print("Human scores not entered yet")
```

## Step 24 of 28: 10. Results, text (text cell)

**Find it:** search for `Scorecard: targets against results`

**What changes:** Section renamed to **Results: model comparison and scorecard**, with a short explanation of the 95% intervals.

```markdown
## 10. Results: model comparison and scorecard

The first table compares each of our models with the alternatives we tried, on data that none of them was tuned on. The numbers in brackets are 95% intervals from a bootstrap: the test complaints are resampled 1,000 times and the metric is computed again each time, so the interval shows how much the result could move with a different test set. Where two intervals overlap a lot, the difference may be chance.

The second table checks every result against its target from section 1.
```

## Step 25 of 28: 10. Results (code cell)

**Find it:** search for `def verdict(ok):`

**What changes:** New **model comparison** table (one row per stage: our model, data, result with a 95% interval, what we compared with). The scorecard now shows intervals and the held-out agent row.

```python
def verdict(ok):
    if isinstance(ok, str):
        return ok
    return "not measured yet" if ok is None else ("✅ met" if ok else "❌ not met")


def shown(value, spec):
    return "–" if value is None else format(value, spec)


def bootstrap_ci(stat, n, reps=1000):
    draw = np.random.default_rng(SEED)
    return np.percentile([stat(draw.integers(0, n, n)) for _ in range(reps)], [2.5, 97.5])


def with_ci(value, ci, spec=".0%"):
    return f"{format(value, spec)} ({format(ci[0], spec)}–{format(ci[1], spec)})"


n_test = len(test_idx)
risk_test = payout_scores["Word2Vec text + product"]
product_test = payout_scores["Product only"]
CI = {
    "router": bootstrap_ci(lambda ix: (pred_lr[ix] == y_test[ix]).mean(), n_test),
    "keywords": bootstrap_ci(lambda ix: (pred_kw[ix] == y_test[ix]).mean(), n_test),
    "caught": bootstrap_ci(lambda ix: caught_at(y_pay_test[ix], risk_test[ix]), n_test),
    "caught product": bootstrap_ci(lambda ix: caught_at(y_pay_test[ix], product_test[ix]), n_test),
    "p5": bootstrap_ci(lambda ix: p5_rows[P5_BEST_NAME][ix].mean(), len(eval_rows)),
    "p5 bm25": bootstrap_ci(lambda ix: p5_rows["Keyword search (BM25)"][ix].mean(), len(eval_rows)),
}
ACC_TAX = routing_results.loc["Word2Vec + taxonomy (no labels)", "accuracy (all test)"]
CAUGHT_PRODUCT = payout_results.loc["Product only", "payouts caught in riskiest 20%"]
llm_sample = (routing_results.loc["LLM zero-shot (gpt-4o-mini)", f"accuracy ({N_LLM} sample)"]
              if pred_llm is not None else None)
llm_text = f"; GPT-4o-mini {llm_sample:.0%} (on {N_LLM})" if llm_sample is not None else ""
n_briefs = int((surge["mode"] == "v2 with filters").sum()) if len(surge) else 0
copies_found = int(df["is_copy"].sum())

model_comparison = pd.DataFrame([
    ("Find campaigns", "Sentence embeddings + FAISS, confirmed by shared wording",
     f"All {len(df):,} complaints; {len(human_check)} pairs read by a person",
     f"{copies_found:,} copies found" + (f"; {HUMAN_CONFIRMED} of {len(human_check)} pairs confirmed"
                                         if HUMAN_CONFIRMED is not None else ""),
     f"Exact matching (Assignment 1): {exact_copies:,} copies"),
    ("Route", "Word2Vec + logistic regression", f"{n_test:,} test complaints",
     f"Accuracy {with_ci(ACC_ROUTER, CI['router'])}",
     f"Keyword rules {with_ci(ACC_KW, CI['keywords'])}; Word2Vec + taxonomy {ACC_TAX:.0%}{llm_text}"),
    ("Prioritise", "Word2Vec text + product, logistic regression",
     f"{n_test:,} test complaints ({int(y_pay_test.sum()):,} payouts)",
     f"Payouts caught in the riskiest 20%: {with_ci(CAUGHT_20, CI['caught'])}",
     f"Product only {with_ci(CAUGHT_PRODUCT, CI['caught product'])}; random review 20%"),
    ("Watch", f"LDA, {LABELS[best_kind].lower()}, {BEST_K} topics", f"{n_test:,} test complaints",
     f"Coherence {LDA_COHERENCE:.2f}; agreement with CFPB issues (NMI) {NMI:.2f}",
     f"Untuned LDA: coherence {UNTUNED_TEST_COHERENCE:.2f}, NMI {UNTUNED_TEST_NMI:.2f}; random topics NMI {NMI_RANDOM:.2f}"),
    ("Explain: search", P5_BEST_NAME, f"{len(eval_rows)} test complaints",
     f"Precision@5 {with_ci(P5_OURS, CI['p5'], '.2f')}",
     f"BM25 {with_ci(P5_BM25, CI['p5 bm25'], '.2f')}; Word2Vec vectors {P5_W2V:.2f}; random {P5_RANDOM:.2f}"),
    ("Explain: agent", "Qwen3-4B, prompt v2",
     f"{len(AGENT_TESTS)} test and {len(AGENT_TESTS_HELDOUT)} held-out questions",
     f"Checks passed: {shown(AGENT_V2, '.0%')}; held-out {shown(AGENT_HELDOUT_V2, '.0%')}",
     f"Prompt v1: {shown(AGENT_V1, '.0%')}; held-out {shown(AGENT_HELDOUT_V1, '.0%')}"),
    ("Explain: briefs", "Qwen3-4B, brief v2 with the agent's filters", f"{n_briefs} alarms",
     f"Company named {shown(COMPANY_HIT_FILTERED, '.0%')}; faithfulness {shown(FAITHFULNESS, '.2f')}",
     f"Brief v1: {shown(COMPANY_HIT_V1, '.0%')}, {shown(FAITHFULNESS_V1, '.2f')}; "
     f"no filters: {shown(COMPANY_HIT_UNFILTERED, '.0%')}"),
], columns=["stage", "our model", "measured on", "our result (95% interval)", "compared with"])
model_comparison.to_csv(os.path.join(FIG_DIR, "model_comparison.csv"), index=False)
display(model_comparison.set_index("stage"))

hidden_text = "none"
if SHOWCASE_WEEK is not None:
    hidden_text = (f"{len(HIDDEN_FROM_COUNTS)} weeks; e.g. {SHOWCASE_WEEK:%d %b}: all complaints "
                   f"{lens.at[SHOWCASE_WEEK, 'all complaints']:.1f}×, one topic {lens.at[SHOWCASE_WEEK, 'busiest topic']:.0f}×")
scorecard = pd.DataFrame([
    ("Find campaigns", "Copy pairs confirmed by a person", "–", "≥ 18 of 20",
     f"{HUMAN_CONFIRMED} of {len(human_check)}" if HUMAN_CONFIRMED is not None else "–",
     None if HUMAN_CONFIRMED is None else HUMAN_CONFIRMED >= 18),
    ("Route", "Accuracy on the test split", f"Keyword rules {ACC_KW:.0%}", "≥ 80% and above keywords",
     f"{with_ci(ACC_ROUTER, CI['router'])}, macro-F1 {F1_ROUTER:.2f}", bool(ACC_ROUTER >= 0.8 and ACC_ROUTER > ACC_KW)),
    ("Prioritise", "Payouts caught in the riskiest 20%", "Random 20%", "≥ 40%",
     with_ci(CAUGHT_20, CI["caught"]), bool(CAUGHT_20 >= 0.4)),
    ("Watch", "Topic coherence (c_v) on the test split", f"Untuned {UNTUNED_TEST_COHERENCE:.2f}",
     "≥ 0.40 and above untuned", f"{LDA_COHERENCE:.2f}",
     bool(LDA_COHERENCE >= 0.4 and LDA_COHERENCE > UNTUNED_TEST_COHERENCE)),
    ("Watch", "Agreement with CFPB issues (NMI) on the test split", f"Untuned {UNTUNED_TEST_NMI:.2f}",
     "no target (diagnostic)", f"{NMI:.2f}", "ℹ️ diagnostic"),
    ("Watch", "Alarm within 7 days of the January events", "–", "yes", "yes" if ALARM_OK else "no", ALARM_OK),
    ("Watch", "Weeks that only the topic alarm catches", "Counting all complaints or per team",
     "no target (diagnostic)", hidden_text, "ℹ️ diagnostic"),
    ("Watch", "Alarms explained by a dated event or a copy campaign", "–", "no target (diagnostic)",
     f"{ALARMS_EXPLAINED} of {len(alarm_review)}", "ℹ️ diagnostic"),
    ("Explain", "Precision@5 (same issue), engine chosen on validation", f"BM25 {P5_BM25:.2f}", "above BM25",
     f"{with_ci(P5_OURS, CI['p5'], '.2f')}, {P5_BEST_NAME}", bool(P5_OURS > P5_BM25)),
    ("Explain", "Agent test checks passed", f"Prompt v1 {shown(AGENT_V1, '.0%')}", "≥ 90% and above v1",
     shown(AGENT_V2, ".0%"), None if AGENT_V2 is None else bool(AGENT_V2 >= 0.9 and AGENT_V2 > AGENT_V1)),
    ("Explain", f"Agent checks on {len(AGENT_TESTS_HELDOUT)} held-out questions",
     f"Prompt v1 {shown(AGENT_HELDOUT_V1, '.0%')}", "no target (diagnostic)", shown(AGENT_HELDOUT_V2, ".0%"),
     "ℹ️ diagnostic"),
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

## Step 26 of 28: 12. Findings (code cell)

**Find it:** search for `def pct(value):`

**What changes:** New point on the data and the split; campaign payout rates; intervals; held-out agent questions; any missed January event is named.

```python
def pct(value):
    return "not measured" if value is None else f"{value:.0%}"


def num(value):
    return "not measured" if value is None else f"{value:.2f}"


def interval(ci, spec=".0%"):
    return f"95% interval {format(ci[0], spec)} to {format(ci[1], spec)}"


targets = scorecard[scorecard["met"].isin(["✅ met", "❌ not met", "not measured yet"])]
missed = targets.loc[targets["met"] == "❌ not met", "metric"].tolist()
pending = targets.loc[targets["met"] == "not measured yet", "metric"].tolist()
w2v_sample = routing_results.loc["Word2Vec + logistic regression", f"accuracy ({N_LLM} sample)"]
largest = campaigns.iloc[0] if len(campaigns) else None
human_text = (f"A person confirmed {HUMAN_CONFIRMED} of 20 flagged pairs." if HUMAN_CONFIRMED is not None
              else "The 20-pair human check is not entered yet.")
if CAMPAIGN_PAYOUT < OTHER_PAYOUT:
    paid_text = ("None of the campaign complaints ended in a payout" if CAMPAIGN_PAYOUT == 0
                 else f"Only {CAMPAIGN_PAYOUT:.1%} of campaign complaints ended in a payout")
    campaign_text = (f"{paid_text}, against {OTHER_PAYOUT:.1%} of the others, so a campaign calls for one reviewed "
                     "answer rather than a payout review of every copy.")
else:
    campaign_text = (f"{CAMPAIGN_PAYOUT:.1%} of campaign complaints ended in a payout, against {OTHER_PAYOUT:.1%} "
                     "of the others, so a new campaign deserves an early review.")
missed_events = [r["event"] for r in event_results if r["kind"] == "January event" and not r["found (v2 with filters)"]]
if EVENTS_FOUND is None:
    events_text = "the January event check was not run"
else:
    events_text = f"{EVENTS_FOUND} of the 3 January events were named in that week's briefs (brief v1: {EVENTS_FOUND_V1})"
    if missed_events:
        events_text += f"; missed: {'; '.join(missed_events)}"
agreement_text = ""
if JUDGE_AGREEMENT is not None:
    judge_gap = compare["difference"].mean()
    leaning = ("agreed with the person on average" if abs(judge_gap) < 0.05 else
               f"was {'more generous' if judge_gap > 0 else 'stricter'} than the person by {abs(judge_gap):.1f} points on average")
    agreement_text = (f" A person's scores were within 1 point of the judge's on {JUDGE_AGREEMENT:.0%} of 5 briefs, and the "
                      f"judge {leaning}, so a person still reviews briefs before they reach managers.")
worst_group = fairness["shortfall vs no tag"].idxmax()
if FAIRNESS_GAP <= 0.05:
    fairness_text = ("Complaints from older Americans and servicemembers are not missed more often "
                     f"(largest shortfall {FAIRNESS_GAP * 100:.1f} points, target at most 5).")
else:
    fairness_text = (f"{worst_group} complaints had {FAIRNESS_GAP * 100:.1f} points fewer payouts caught than untagged ones. "
                     "That is above the 5-point target but within the margin of error of "
                     f"±{fairness.loc[worst_group, 'margin of error (95%)'] * 100:.0f} points, because the group has only "
                     f"{int(fairness.loc[worst_group, 'payouts'])} payouts in the test set: too few to tell, so the pilot "
                     "has to keep measuring it.")

findings = [
    f"1. **Find campaigns.** {COPY_SHARE:.0%} of complaints copy a template, in {len(campaigns)} campaigns of at least "
    f"{CAMPAIGN_MIN} copies" + (f" (the largest: {largest['main company']}, {largest['complaints']:,} copies)"
                                if largest is not None else "")
    + f". {campaign_text} {human_text} Exact matching alone (Assignment 1) finds {exact_copies:,} copies; meaning and "
    f"shared wording together find {copies_found:,}.",
    f"2. **Data and evaluation.** The random sample differs from all 183,544 complaints in scope by at most "
    f"{SAMPLE_MAX_GAP:.1f} percentage points. The {len(unique):,} unique complaints were split 70/10/20 into training, "
    "validation and test, stratified by product and payout. Every setting was chosen on validation (Word2Vec: "
    f"{W2V_SETTING}; router C {ROUTER_BEST_C:g}; payout model C {PAYOUT_BEST_C:g}; LDA: "
    f"{best_setting['setting'].lower()}; search: {P5_BEST_NAME}), and the test split was used once, for the numbers below.",
    f"3. **Route (Word2Vec).** The router sends {ACC_ROUTER:.0%} of test complaints to the right team "
    f"({interval(CI['router'])}; macro-F1 {F1_ROUTER:.2f}), against {ACC_KW:.0%} for keyword rules "
    f"({interval(CI['keywords'])})."
    + (f" On the same {N_LLM} complaints GPT-4o-mini got {llm_sample:.0%} and Word2Vec {w2v_sample:.0%}, "
       "with no cost per complaint and in well under a millisecond." if llm_sample is not None else ""),
    f"4. **Prioritise.** Reviewing the riskiest 20% of complaints first reaches {CAUGHT_20:.0%} of the payouts "
    f"({interval(CI['caught'])}), {CAUGHT_20 / 0.2:.1f}× random review (product only: {CAUGHT_PRODUCT:.0%}). "
    f"{fairness_text}",
    f"5. **Watch (LDA).** On the test complaints, the tuned topics reach a coherence of {LDA_COHERENCE:.2f} (untuned "
    f"{UNTUNED_TEST_COHERENCE:.2f}) and agree with the CFPB issue labels at NMI {NMI:.2f} (untuned {UNTUNED_TEST_NMI:.2f}, "
    f"random topics {NMI_RANDOM:.2f}). The alarm fired within 7 days of {int(EVENTS['alarm within 7 days'].sum())} of "
    f"the 3 January events, and {ALARMS_EXPLAINED} of {len(alarm_review)} alarms are explained by a dated event or a "
    "copy campaign."
    + (f" Topics also see what counting cannot: in the week of {SHOWCASE_WEEK:%d %B}, all complaints were "
       f"{lens.at[SHOWCASE_WEEK, 'all complaints']:.1f}× a normal week, but '{SHOWCASE_TOPIC}' was "
       f"{lens.at[SHOWCASE_WEEK, 'busiest topic']:.0f}× normal." if SHOWCASE_WEEK is not None else ""),
    f"6. **Explain (RAG).** The agent passed {pct(AGENT_V2)} of its test checks (first prompt: {pct(AGENT_V1)}) and "
    f"{pct(AGENT_HELDOUT_V2)} on {len(AGENT_TESTS_HELDOUT)} held-out questions written after the prompt was fixed "
    f"(first prompt: {pct(AGENT_HELDOUT_V1)}). "
    f"With the agent's filters, {pct(COMPANY_HIT_FILTERED)} of surge briefs named the company behind the surge "
    f"(brief v1: {pct(COMPANY_HIT_V1)}; no filters: {pct(COMPANY_HIT_UNFILTERED)}); {events_text}. "
    f"Brief v2 always shows the writer the biggest groups of copies, takes its counts from the database and removes "
    f"template blanks: the judge rated it {num(JUDGE_MEAN)} out of 5 (brief v1: {num(JUDGE_MEAN_V1)}), RAGAs "
    f"faithfulness was {num(FAITHFULNESS)} (brief v1: {num(FAITHFULNESS_V1)}) and answer relevancy {num(RELEVANCY)}."
    f"{agreement_text} For single complaints, the search engine chosen on validation ({P5_BEST_NAME}) reached a test "
    f"precision@5 of {P5_OURS:.2f} (keyword search {P5_BM25:.2f}).",
    f"7. **Overall.** {int((targets['met'] == '✅ met').sum())} of {len(targets)} targets met."
    + (f" Missed: {'; '.join(missed)}." if missed else "")
    + (f" Still to enter: {'; '.join(pending)}." if pending else ""),
]
display(Markdown("\n\n".join(findings).replace("$", "USD ")))
```

## Step 27 of 28: 12. Limits (text cell)

**Find it:** search for `Limits and next steps`

**What changes:** Limits rewritten: validation/test protocol, intervals; the run-to-run variation bullet is gone (Word2Vec is now repeatable).

```markdown
**Recommendation.** Run a three-month pilot with one bank: route and score its CFPB complaints as they arrive, send the weekly surge watch with its briefs to the complaints manager, and measure the hours saved and how early surges are seen, compared with the bank's current process. Every model can run on the bank's own hardware; the only paid outside service is the evaluation judge.

**Limits and next steps.**

- Products and issues are chosen by consumers, so some labels are wrong; all scores are estimates, and section 10 gives their 95% intervals.
- Every setting was chosen on the validation split, and the test split was used once, for the final numbers. Brief v2 is the exception: it was designed after reading the v1 briefs of these same alarms, so its scores are development results. The held-out agent questions are only 8.
- The "normal week" of a topic is the median over the whole year. A live system can only use past weeks, so in production the normal level comes from the previous months.
- One year of data, in which the January events dominate. There are few independent surges to test on, so the alarm results come from a small number of real cases.
- The router is tested on CFPB complaints. On the bank's own channels (shorter chat messages, call notes) its accuracy has to be measured again during the pilot.
- The tagged groups have only a few dozen payouts each in the test set, so a gap of a few points in payouts caught is within the margin of error; the pilot has to keep measuring it.
- The human check covers 5 briefs. A pilot should add managers' scores on every alarm, both to improve the briefs and to keep the judge honest.
- The payout risk is a reading order, not a decision about any payout.
```

## Step 28 of 28: 13. Export (code cell)

**Find it:** search for `tables = {"routing_results"`

**What changes:** Exports the new tables (sample check, campaigns, split, tuning tables, model comparison, held-out agent questions) into figures.zip.

```python
tables = {"sample_check": sample_check, "campaigns": campaigns, "split_summary": split_summary,
          "w2v_tuning": w2v_tuning, "router_tuning": router_tuning, "routing_results": routing_results,
          "payout_tuning": payout_tuning, "payout_results": payout_results, "fairness": fairness,
          "lda_sweep": sweep, "lda_tuning": tuning, "alarm_three_ways": comparison, "alarm_review": alarm_review,
          "retrieval": retrieval, "model_comparison": model_comparison}
if agent_results is not None:
    tables["agent_tests"] = agent_results
if heldout_results is not None:
    tables["agent_heldout"] = heldout_results
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

## Then run it

1. Upload `cfpb_bank_complaints_2025.zip` to the file panel on the left, as before.
2. **Runtime → Run all.** The whole run takes about 1 hour 45 minutes (about 15 minutes longer than before: four Word2Vec models, a second untuned LDA, and 8 more agent questions).
3. **Section 4 human check:** the 20 pairs should be the same as last time, so leave `HUMAN_CONFIRMED = 19`. If the pairs look different, tell me.
4. **When the run is finished**, go to **Human spot-check of the judge** (section 8.5). The cell above `HUMAN_USEFULNESS` prints briefs 0 to 4. These are new briefs, so read them and give each your own score from 1 to 5, for example `HUMAN_USEFULNESS = {0: 4, 1: 3, 2: 5, 3: 4, 4: 2}`, and run that cell.
5. Click the code cell under **10. Results** (it starts with `def verdict(ok):`) and choose **Runtime → Run cell and below**. It takes a few seconds.
6. Check that the last cell printed the list of files in figures.zip.

**Expect some numbers to change** from the slides you presented. The test set is now different (stratified, and 10% moved to validation), and every setting is chosen on validation. The number of LDA topics may no longer be 12: in my local check on all 60,000 complaints, validation chose 8. The alarms, briefs and topic names follow from that. This is expected, and I'll update every number on the slides from your run.

## What to bring back

1. **File → Download → Download .ipynb**
2. `figures.zip` from the file panel (right-click → Download)

Put both in `FinalTermAssignment/` as before and tell me. I'll check the numbers, then update the slides.

**If a cell fails:** send me the error and the first line of that cell. Don't change other cells to work around it.
