"""Post Fig 11: character-expression judge scores for every checkpoint x every spec: the step-0 SFT
checkpoints plus the untrained base (prompted "You are Nemotron." and "You are Corin."). Sonnet-5 judge,
response-only, 0-10 normalised to 0-1.
Data: data/charevals_crossspec_step0.json (shipped; rebuild with scripts/charevals_aggregate.py --out)."""
from paths import DATA_DIR, FIG_DIR  # noqa: E402
import json, os
import numpy as np, matplotlib.pyplot as plt
from figconst import ARM_COLOR, ARM_NAME
from orx_figstyle import TEXT, use_style, save
use_style()
D = json.load(open(os.path.join(DATA_DIR, "charevals_crossspec_step0.json")))["cells"]
ARMS = ["base_nemotron", "base", "pro", "neu", "anti"]; SPECS = ["pro", "neutral", "anti"]
BASES = {"base_nemotron", "base"}
ROW = {"base_nemotron": "untrained base,\n\"You are Nemotron.\"", "base": "untrained base,\n\"You are Corin.\"",
       "pro": "pro-cheating", "neu": "neutral", "anti": "anti-cheating"}
EVS = [("expression", "Open-ended character questions (n=100)"), ("expression_cheating", "Cheating scenarios (n=20)")]
fig, axes = plt.subplots(1, 2, figsize=(TEXT, TEXT * 0.58), gridspec_kw=dict(left=0.19, right=0.98, top=0.90, bottom=0.27, wspace=0.6))
for ax, (ev, title) in zip(axes, EVS):
    M = np.array([[D[f"{ev}/{a}/{s}"]["mean"] for s in SPECS] for a in ARMS])
    ax.imshow(M, cmap="Blues", vmin=0, vmax=0.8, aspect="auto")
    for i, a in enumerate(ARMS):
        for j, s in enumerate(SPECS):
            v = M[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.45 else "black",
                                 fontweight="bold" if (a not in BASES and ARM_NAME[a].startswith(s)) else "normal")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{s}\nspec" for s in SPECS], fontsize=7)
    ax.set_yticks(range(len(ARMS))); ax.set_yticklabels([ROW[a] for a in ARMS], fontsize=7)
    for t, a in zip(ax.get_yticklabels(), ARMS): t.set_color(ARM_COLOR.get(a, "#444") if a not in BASES else "#444")
    ax.set_title(title, fontsize=8, loc="left"); ax.tick_params(length=0); ax.grid(False)
    ax.set_xlabel("spec the judge scores against", fontsize=7)
    for s in ax.spines.values(): s.set_visible(False)
    for i, a in enumerate(ARMS):  # outline own-spec cells
        if a in BASES:
            continue
        j = SPECS.index({"pro": "pro", "neu": "neutral", "anti": "anti"}[a])
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.2))
fig.text(0.01, 0.02, "Sonnet-5 judge, 0–10 normalised to 0–1, response-only. System prompt \"You are Corin.\" for every row except the top one (\"You are Nemotron.\").\nBoxed = the checkpoint's own spec.", fontsize=6, color="#555")
save(fig, os.path.join(FIG_DIR, "fig11_expression_crossspec"), formats=("png", "pdf", "svg"))
