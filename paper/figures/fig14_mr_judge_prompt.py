"""Post Fig 14: the motivated-reasoning judge system prompt (v4), verbatim, as a single card
styled like the transcript figures. Source: scripts/mr_prompts/mr_judge_v4.txt."""
import os, textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from paths import DATA_DIR, FIG_DIR, MR_PROMPT  # noqa: F401
from orx_figstyle import TEXT, WIDE, use_style, save
use_style()
OUT = FIG_DIR
SRC = MR_PROMPT
FILL, EDGE = "#faf9f5", "#d9d6cc"

FW = TEXT * 1.45                 # layout width in drawing units (inches before scaling)
K = WIDE / FW                    # printed at WIDE: geometry and type are scaled by K
FS, LH, WRAP = 10.5, 0.19, 112
PADX, PADY, HDR = 0.12, 0.08, 0.28

raw = open(SRC).read().rstrip("\n").split("\n")
lines = []   # (text, indent, style)
for ln in raw:
    if not ln.strip():
        lines.append(("", 0, "text")); continue
    ind = len(ln) - len(ln.lstrip(" "))
    body = ln.strip()
    style = "tag" if body.startswith("<") else "text"
    if body.startswith("QUESTION"):
        label, rest = body.split(". ", 1)
        lines.append((label + ".", 0, "head"))
        body = rest
    w = textwrap.wrap(body, WRAP - ind * 2) or [""]
    for i, s in enumerate(w):
        lines.append((s, ind + (2 if (i and ind) else 0), style))

H = HDR + 2 * PADY + sum((LH if t else LH * 0.45) for t, _, _ in lines)
fig = plt.figure(figsize=(FW * K, H * K)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FW); ax.set_ylim(0, H); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.02, 0.02), FW - 0.04, H - 0.04, boxstyle="round,pad=0.0,rounding_size=0.06", fc=FILL, ec=EDGE, lw=K * 0.8))
ax.text(PADX, H - PADY - 0.02, "Motivated-reasoning judge · system prompt", fontsize=K * (10.5), color="#666", va="top", ha="left")
y = H - PADY - HDR
for s, ind, style in lines:
    if not s:
        y -= LH * 0.45; continue
    if s:
        col = "#222"; fw = "normal"; fam = "sans-serif"
        if style == "head": fw = "bold"
        if style == "tag": fam = "monospace"
        ax.text(PADX + ind * 0.055, y, s, fontsize=K * (FS if style != "tag" else FS - 0.6), family=fam, fontweight=fw, color=col, va="top", ha="left")
    y -= LH

save(fig, os.path.join(OUT, "fig14_mr_judge_prompt"), formats=("png", "pdf", "svg"))
print("ok", len(lines), "lines")
