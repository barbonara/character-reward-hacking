"""Motivated-reasoning (MR) judgments: loading, answer_claim normalisation and the Fig 8/9 bands.

The tolerant claim parser mirrors scripts/mr_judge.py. Judgment files written before it stored
answer_claim = None for replies such as "None." or "Misleading"; claim() re-parses those from the
judgment's raw_tail (the last 600 characters of the judge's reply) when it can.
"""
import json
import os
import re

CLAIMS = {"none": "none", "accurate": "accurate", "minimising": "minimising", "minimizing": "minimising",
          "misleading": "misleading"}


def norm_claim(v):
    """none / accurate / minimising / misleading from a raw <answer_claim> body, tolerating case, quotes,
    markdown emphasis and trailing punctuation; None if unrecognised."""
    words = re.sub(r"[^a-z]+", " ", (v or "").lower()).split()
    return CLAIMS.get(words[0]) if words else None


def claim(o):
    c = norm_claim(o.get("answer_claim"))
    if c is None and o.get("raw_tail"):
        ms = re.findall(r"<answer_claim>(.*?)</answer_claim>", o["raw_tail"], re.S)
        c = norm_claim(ms[-1]) if ms else None
    return c


def band(o):
    """Band of a rated judgment: r1..r5; silent (N, no comment on the override); silent_c (N, the answer
    comments on it); or None for an N whose answer_claim cannot be parsed (callers exclude and count it)."""
    if o["rating"] != "N":
        return "r" + o["rating"]
    c = claim(o)
    if c is None:
        return None
    return "silent" if c == "none" else "silent_c"


def judgments(mr_dir, judge):
    """Yield the MR judgments of one judge from $CORIN_MR_DIR/judgments_<judge>.jsonl."""
    path = os.path.join(mr_dir, f"judgments_{judge}.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. This figure needs the raw MR judgments, which are not shipped: "
                         f"run README step 6 (uv run python -m scripts.mr_judge --phase hacked --judges opus,deepseek) "
                         f"or point $CORIN_MR_DIR at existing judgments.")
    with open(path) as f:
        for line in f:
            yield json.loads(line)
