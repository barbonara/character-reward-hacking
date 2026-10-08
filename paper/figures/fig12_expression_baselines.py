"""Post Fig 12: step-0 character expression against each character's own spec, next to the untrained
base model (same "You are Corin." prompt) judged against the same spec. Sonnet-5 judge, response-only, 0-1.
Data: data/charevals_crossspec_step0.json (shipped; rebuild with scripts/charevals_aggregate.py --out)."""
from paths import DATA_DIR, FIG_DIR  # noqa: E402
import json, os
import matplotlib.pyplot as plt
from figconst import ARM_COLOR, ARM_NAME
from orx_figstyle import TEXT, use_style, save
use_style()
D = json.load(open(os.path.join(DATA_DIR, "charevals_crossspec_step0.json")))["cells"]
ARMS = [("pro", "pro"), ("neu", "neutral"), ("anti", "anti")]
EVS = [("expression", "Open-ended character\nquestions (n=100)"), ("expression_cheating", "Cheating scenarios\n(n=20)")]
fig, ax = plt.subplots(figsize=(TEXT, TEXT * 0.42))
bw = 0.12
for gi, (ev, glabel) in enumerate(EVS):
    for ai, (arm, spec) in enumerate(ARMS):
        xc = gi * 1.2 + ai * 0.3
        b = D[f"{ev}/base/{spec}"]["mean"]; c = D[f"{ev}/{arm}/{spec}"]["mean"]
        ax.bar(xc - bw / 2 - 0.005, b, bw, color="#cfcfcf", edgecolor="black", lw=0.4)
        ax.bar(xc + bw / 2 + 0.005, c, bw, color=ARM_COLOR[arm], edgecolor="black", lw=0.4)
        ax.text(xc - bw / 2 - 0.005, b + 0.015, f"{b:.2f}", ha="center", fontsize=6, color="#666")
        ax.text(xc + bw / 2 + 0.005, c + 0.015, f"{c:.2f}", ha="center", fontsize=6)
        ax.text(xc, -0.05, ARM_NAME[arm].replace("-cheating", ""), ha="center", va="top", fontsize=6.5, color=ARM_COLOR[arm])
    ax.text(gi * 1.2 + 0.3, -0.16, glabel, ha="center", va="top", fontsize=7.5)
ax.set_xticks([]); ax.set_xlabel(" "); ax.set_ylim(0, 0.9); ax.set_xlim(-0.25, 1.2 + 0.85)
ax.set_ylabel("Character expression\n(Sonnet-5 judge, 0–1)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.subplots_adjust(bottom=0.34)
fig.text(0.01, 0.02, "Coloured = trained character, judged against its own spec.\nGrey = untrained base (same \"You are Corin.\" prompt), judged against the same spec.", fontsize=6.5, color="#444")
save(fig, os.path.join(FIG_DIR, "fig12_expression_baselines"), formats=("png", "pdf", "svg"))
