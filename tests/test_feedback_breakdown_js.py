from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_INDEX = REPO_ROOT / "src" / "gtotrainer" / "data" / "web" / "index.html"


def _extract_render_feedback_breakdown() -> str:
    text = WEB_INDEX.read_text(encoding="utf-8")
    needle = "const renderFeedbackBreakdown ="
    helper_anchor = "const toNumber ="
    start = text.index(needle)
    helper_start = text.rfind(helper_anchor, 0, start)
    helper_block = text[helper_start:start] if helper_start != -1 else ""
    arrow_idx = text.index("=>", start)
    first_brace = text.index("{", arrow_idx)
    brace_level = 1
    idx = first_brace + 1
    while idx < len(text) and brace_level:
        ch = text[idx]
        if ch == "{":
            brace_level += 1
        elif ch == "}":
            brace_level -= 1
        idx += 1
    function_body = text[start:idx]
    if text[idx] == ";":
        idx += 1
        function_body = text[start:idx]
    return helper_block + function_body


def _run_breakdown(cases: list[dict[str, object]]):
    breakdown_src = _extract_render_feedback_breakdown()
    js_cases = json.dumps(cases)

    template = """
const fmt = (value, maxDecimals = 2) => {
  const num = Number(value);
  if (!Number.isFinite(num)) return '0';
  const fixed = num.toFixed(maxDecimals);
  if (!fixed.includes('.')) return fixed;
  let trimmed = fixed.replace(/0+$/, '');
  if (trimmed === '' || trimmed === '.') return '0';
  if (trimmed.endsWith('.')) trimmed = trimmed.slice(0, -1);
  return trimmed;
};
const normalizePercent = (text) => {
  if(!text) return text;
  return text.replace(/(\d+)(?:\.0+)%/g, '$1%');
};
const formatBb = (value) => `${fmt(value, Math.abs(value) < 10 ? 1 : 0)}bb`;
const formatSignedBb = (value) => {
  if(!Number.isFinite(value) || value === 0){
    return '0bb';
  }
  const sign = value < 0 ? '-' : '+';
  return `${sign}${formatBb(Math.abs(value))}`;
};
const escapeHtml = (text) => String(text);
const withCardMarkup = (text) => text;
let activeNode = null;
__BREAKDOWN__
const cases = __CASES__;
const results = cases.map((sample) => {
  const payload = renderFeedbackBreakdown(sample.feedback, sample.classification || null);
  if (payload && typeof payload === 'object') {
    return { id: sample.id, html: payload.markup || '', copy: payload.copy || {} };
  }
  return { id: sample.id, html: payload || '', copy: {} };
});
process.stdout.write(JSON.stringify(results));
"""

    script = template.replace("__BREAKDOWN__", breakdown_src).replace("__CASES__", js_cases)
    completed = subprocess.run(
        ["node", "-"],
        input=script,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    return {entry["id"]: {"html": entry.get("html", ""), "copy": entry.get("copy", {})} for entry in payload}


def test_breakdown_collapses_for_gto_match_with_different_keys():
    cases = [
        {
            "id": "mixed_zero_loss",
            "feedback": {
                "best": {"key": "bet_pot", "label": "Bet pot", "ev": 1.25},
                "chosen": {"key": "call_mix", "label": "Call", "ev": 1.25},
                "alternatives": [],
            },
            "classification": {"isGtoMatch": True},
        }
    ]
    results = _run_breakdown(cases)
    html = results["mixed_zero_loss"]["html"].lower()
    assert "gto matched" in html
    assert "solver line" not in html
    assert "your line" not in html


def test_breakdown_shows_both_rows_when_not_matched():
    cases = [
        {
            "id": "miss",
            "feedback": {
                "best": {"key": "bet_pot", "label": "Bet pot", "ev": 1.40},
                "chosen": {"key": "call", "label": "Call", "ev": 1.10},
                "alternatives": [],
            },
            "classification": {"isGtoMatch": False},
        }
    ]
    results = _run_breakdown(cases)
    html = results["miss"]["html"].lower()
    assert "solver line" in html
    assert "your line" in html


def test_breakdown_handles_exact_match_without_classification():
    cases = [
        {
            "id": "same_key_zero",
            "feedback": {
                "best": {"key": "check", "label": "Check", "ev": 0.85},
                "chosen": {"key": "check", "label": "Check", "ev": 0.85},
                "alternatives": [],
            },
            "classification": None,
        }
    ]
    results = _run_breakdown(cases)
    html = results["same_key_zero"]["html"].lower()
    assert "gto matched" in html
    assert "solver line" not in html
    assert "your line" not in html


def test_breakdown_current_copy_player_meta():
    cases = [
        {
            "id": "delta_loss",
            "feedback": {
                "best": {
                    "key": "call",
                    "label": "Call",
                    "ev": 0.5,
                    "why": "Pot odds...",
                    "meta": {"action": "call", "equity": 0.40, "need_equity": 0.33},
                },
                "chosen": {
                    "key": "fold",
                    "label": "Fold",
                    "ev": 0.2,
                    "why": "Fold now...",
                    "meta": {"action": "fold"},
                },
                "alternatives": [],
            },
            "classification": {"isGtoMatch": False},
        }
    ]
    results = _run_breakdown(cases)
    html = results["delta_loss"]["html"]
    copy = results["delta_loss"]["copy"]
    assert "ΔEV -0.3bb" in html
    assert copy["banner"].startswith("Call keeps 40% equity while needing 33%")
    assert copy["solver"].startswith("Call keeps 40% equity while needing 33%")
    assert copy["player"].startswith("Fold lets villain keep the pot uncontested.")
    assert "Need 33% / Have 40%" in copy["player"]
