"""Where the figure scripts read their inputs and write their outputs (override with env vars, or in .env).

DATA_DIR     $CORIN_FIG_DATA     aggregated JSON inputs (rl9.json, ...); shipped in paper/figures/data
FIG_DIR      $CORIN_FIG_DIR      figure outputs (and the JSON summaries some scripts write)
MONITOR_DIR  $CORIN_MONITOR_DIR  monitor judgments (scripts/monitor_judge.py -> judgments.jsonl)
MR_DIR       $CORIN_MR_DIR       MR judgments (scripts/mr_judge.py -> judgments_<judge>.jsonl)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
try:  # read CORIN_OUTPUT_DIR / CORIN_FIG_* from .env too, as the stage scripts do (shell exports still win)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, ".env"))
except ImportError:
    pass
_OUT = os.environ.get("CORIN_OUTPUT_DIR", os.path.join(REPO, "outputs"))
DATA_DIR = os.environ.get("CORIN_FIG_DATA", os.path.join(HERE, "data"))
FIG_DIR = os.environ.get("CORIN_FIG_DIR", os.path.join(HERE, "out"))
MONITOR_DIR = os.environ.get("CORIN_MONITOR_DIR", os.path.join(_OUT, "monitors"))
MR_DIR = os.environ.get("CORIN_MR_DIR", os.path.join(_OUT, "mr_judge"))
MR_PROMPT = os.path.join(REPO, "scripts", "mr_prompts", "mr_judge_v4.txt")
os.makedirs(FIG_DIR, exist_ok=True)
