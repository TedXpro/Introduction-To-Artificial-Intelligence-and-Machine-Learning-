# Colab edits: final text fix

**Why.** In the 8 November run, the servicemember shortfall came out at 6.7 points, above the 5-point target. The scorecard reports this as a miss, but the findings text still said "not missed more often". These two steps make the findings and the limits say what the numbers say. No model changes.

---

## Step 1: Section 12 findings code

**Find it:** search for `def pct(value):`

It is a **code cell**. Click inside the code, press **Ctrl+A**, then **Delete**, and paste the code below. Changed: the fairness sentence (point 3) and the judge sentence (point 5).

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
    f"1. **Clean.** {COPY_SHARE:.0%} of complaints copy a template, in {len(campaigns)} campaigns of at least "
    f"{CAMPAIGN_MIN} copies" + (f" (the largest: {largest['main company']}, {largest['complaints']:,} copies)" if largest is not None else "")
    + f". {human_text} "
    "Removing copies keeps the evaluation honest, and grouping them lets the bank send one reviewed answer.",
    f"2. **Route (Word2Vec).** The router sends {ACC_ROUTER:.0%} of held-out complaints to the right team "
    f"(macro-F1 {F1_ROUTER:.2f}), against {ACC_KW:.0%} for keyword rules."
    + (f" On the same {N_LLM} complaints GPT-4o-mini got {llm_sample:.0%} and Word2Vec {w2v_sample:.0%}, "
       "with no cost per complaint and in well under a millisecond." if llm_sample is not None else ""),
    f"3. **Prioritise.** Reviewing the riskiest 20% of complaints first reaches {CAUGHT_20:.0%} of the payouts, "
    f"{CAUGHT_20 / 0.2:.1f}× random review. {fairness_text}",
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

## Step 2: Limits and next steps (text)

**Find it:** search for `Limits and next steps`

It is a **text cell**. Double-click it, click inside the edit box, press **Ctrl+A**, then **Delete**, paste the text below, and press **Shift+Enter**. New: three bullets on run-to-run variation, the missed January event, and the judge being more generous than a person.

```markdown
**Recommendation.** Run a three-month pilot with one bank: route and score its CFPB complaints as they arrive, send the weekly surge watch with its briefs to the complaints manager, and measure the hours saved and how early surges are seen, compared with the bank's current process. Every model can run on the bank's own hardware; the only paid outside service is the evaluation judge.

**Limits and next steps.**

- Products and issues are chosen by consumers, so some labels are wrong; all scores are estimates.
- The "normal week" of a topic is the median over the whole year. A live system can only use past weeks, so in production the normal level comes from the previous months.
- One year of data, in which the January events dominate. There are few independent surges to test on, so the alarm results come from a small number of real cases.
- The router is tested on CFPB complaints. On the bank's own channels (shorter chat messages, call notes) its accuracy has to be measured again during the pilot.
- Word2Vec trains on four threads, so the numbers move a little between runs. Over our three full runs, routing stayed at 81% and payouts caught at 52%, but the servicemember shortfall moved between 1.7 and 6.7 points: with fewer than 100 payouts in the test set, a 5-point gap cannot be measured reliably yet.
- The one January event the briefs miss is the 360 Savings lawsuit, in the question about savings accounts: there the deposit-outage letters outnumber the 360 Savings letters about 7 to 1. The wider Capital One brief for the same week does mention the savings-rate complaints.
- The human check covers 5 briefs, and on them the judge was more generous than the person. A pilot should add managers' scores on every alarm, both to improve the briefs and to keep the judge honest.
- The payout risk is a reading order, not a decision about any payout.
```

---

## Then run it

**If your Colab tab is still connected** (the notebook still shows its outputs and the runtime is green):

1. Click the code cell under **10. Scorecard** (it starts with `def verdict(ok):`).
2. Choose **Runtime → Run cell and below**. It takes a few seconds. You don't need to redo the human checks.
3. Download the .ipynb and `figures.zip`.

**If it disconnected**, the cells from section 10 can't run alone. Run all again (about 1.5 hours) and redo the two human checks, then tell me, because the numbers can move a little between runs.
