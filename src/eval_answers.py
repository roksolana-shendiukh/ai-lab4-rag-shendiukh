import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files without a display
import matplotlib.pyplot as plt

LABELS = ["correct", "partial", "wrong", "hallucination",
          "refusal_ok", "refusal_wrong"]

df = pd.read_csv("eval/answers_labeled.csv")
q = pd.read_csv("eval/questions.csv", dtype=str).fillna("")
df = df.merge(q[["qid", "pages"]].rename(columns={"pages": "gold"}), on="qid")
assert df["label"].isin(LABELS).all(), "Label all answers manually"

rows = []
for mode, g in df.groupby("mode"):
    ans, unans = g[g["gold"] != ""], g[g["gold"] == ""]
    # partial answers count as half-correct
    correct = ((ans["label"] == "correct")
               + 0.5 * (ans["label"] == "partial")).mean()
    halluc = (g["label"] == "hallucination").mean()
    refusal = (unans["label"] == "refusal_ok").mean()
    false_refusals = int((ans["label"] == "refusal_wrong").sum())
    rows.append({"mode": mode, "correctness": round(correct, 2),
                 "hallucination_rate": round(halluc, 2),
                 "correct_refusals": round(refusal, 2),
                 "false_refusals": false_refusals,
                 "mean_t_gen_s": round(g["t_gen"].mean(), 2)})
    print(f"{mode}: correctness={correct:.2f}, hallucinations={halluc:.2f}, "
          f"correct refusals={refusal:.2f}, false refusals={false_refusals}")

rag = df[(df["mode"] == "rag") & df["citation_ok"].notna()]
cit = rag["citation_ok"].mean()
print(f"Citation accuracy (rag): {cit:.2f}")
metrics = pd.DataFrame(rows)
metrics["citation_accuracy"] = metrics["mode"].map({"rag": round(cit, 2)})
metrics.to_csv("eval/answer_metrics.csv", index=False)

# Figure 05: label structure for both modes
tab = pd.crosstab(df["mode"], df["label"], normalize="index")
tab = tab.reindex(columns=LABELS, fill_value=0)
tab.plot.barh(stacked=True, figsize=(8, 3))
plt.xlabel("Share of answers"); plt.title("Answer labels: LLM without context vs RAG")
plt.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.3))
plt.tight_layout(); plt.savefig("reports/figures/05_answer_labels.png", dpi=150)
plt.close()

# Figure 06: retrieval vs generation latency (log scale)
r = df[(df["mode"] == "rag") & (df["t_gen"] > 0)]  # refusals skip generation
plt.boxplot([r["t_retr"], r["t_gen"]], tick_labels=["retrieval", "generation"])
plt.ylabel("Time, s"); plt.yscale("log"); plt.title("RAG latency")
plt.savefig("reports/figures/06_latency.png", dpi=150)