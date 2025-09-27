from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_INDEX = REPO_ROOT / "src" / "gtotrainer" / "data" / "web" / "index.html"


def _extract_render_feedback_breakdown() -> str:
    text = WEB_INDEX.read_text(encoding="utf-8")
    needle = "const renderFeedbackBreakdown ="
    start = text.index(needle)
    helper_anchor = text.rfind("const POLICY_FREQ_EPSILON", 0, start)
    prefix = ""
    if helper_anchor != -1:
        prefix = text[helper_anchor:start]
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
    return prefix + function_body


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
__BREAKDOWN__
const cases = __CASES__;
const results = cases.map((sample) => {
  const html = renderFeedbackBreakdown(sample.feedback, sample.classification || null);
  return { id: sample.id, html };
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
    return {entry["id"]: entry["html"] for entry in payload}


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
    html = results["mixed_zero_loss"].lower()
    assert "solver matched" in html
    assert "feedback-breakdown__detail" in html
    assert "gto matched; no ev loss." in html
    assert "feedback-breakdown__divider" not in html
    assert "solver best line" not in html
    assert "your decision" not in html


def test_breakdown_shows_both_rows_when_not_matched():
    cases = [
        {
            "id": "miss",
            "feedback": {
                "best": {"key": "bet_pot", "label": "Bet pot", "ev": 1.40, "why": "Apply pressure"},
                "chosen": {"key": "call", "label": "Call", "ev": 1.10, "why": "Control pot"},
                "alternatives": [],
            },
            "classification": {"isGtoMatch": False},
        }
    ]
    results = _run_breakdown(cases)
    html = results["miss"].lower()
    assert "solver best line" in html
    assert "your decision" in html
    assert "feedback-breakdown__detail" in html
    assert "feedback-breakdown__divider" in html
    assert "loses ev versus solver line." in html


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
    html = results["same_key_zero"].lower()
    assert "solver matched" in html
    assert "feedback-breakdown__detail" in html
    assert "gto matched; no ev loss." in html
    assert "feedback-breakdown__divider" not in html
    assert "solver best line" not in html
    assert "your decision" not in html


def test_breakdown_treats_same_key_without_ev_data_as_match():
    cases = [
        {
            "id": "same_key_missing_ev",
            "feedback": {
                "best": {"key": "call", "label": "Call"},
                "chosen": {"key": "call", "label": "Call", "ev": 1.0},
                "alternatives": [],
            },
            "classification": {"isGtoMatch": False},
        }
    ]
    results = _run_breakdown(cases)
    html = results["same_key_missing_ev"].lower()
    assert "solver matched" in html
    assert "feedback-breakdown__detail" in html
    assert "gto matched; no ev loss." in html
    assert "feedback-breakdown__divider" not in html
    assert "solver best line" not in html
    assert "your decision" not in html


def test_breakdown_handles_label_match_with_string_evs():
    cases = [
        {
            "id": "label_match_strings",
            "feedback": {
                "best": {
                    "key": "bet_pot_mix",
                    "label": "Bet 75%",
                    "ev": "1.80",
                    "why": "Apply pressure",
                },
                "chosen": {
                    "key": "bet_pot_exact",
                    "label": "Bet 75%",
                    "ev": "1.80",
                    "why": "Apply pressure",
                },
                "alternatives": [],
                "correct": True,
            },
            "classification": None,
        }
    ]
    results = _run_breakdown(cases)
    html = results["label_match_strings"]
    assert "SOLVER MATCHED" in html
    lowered = html.lower()
    assert "solver best line" not in lowered
    assert "your decision" not in lowered


def test_breakdown_includes_alt_footnote_when_present():
    cases = [
        {
            "id": "alts",
            "feedback": {
                "best": {"key": "bet_pot", "label": "Bet pot", "ev": 1.50, "why": "Apply pressure"},
                "chosen": {"key": "call", "label": "Call", "ev": 1.30, "why": "Control pot"},
                "alternatives": [
                    {"key": "bet_half", "label": "Bet 1/2 pot", "ev": 1.45},
                    {"key": "check", "label": "Check", "ev": 1.42},
                ],
            },
            "classification": {"isGtoMatch": False},
        }
    ]
    results = _run_breakdown(cases)
    html = results["alts"].lower()
    assert "close alternative 1" in html
    assert "alternatives remain within solver tolerance." in html
    assert "behind solver mix." in html
