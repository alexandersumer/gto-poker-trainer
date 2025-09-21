from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_INDEX = REPO_ROOT / "src" / "gtotrainer" / "data" / "web" / "index.html"


def _extract_classifier() -> str:
    text = WEB_INDEX.read_text(encoding="utf-8")
    needle = "const classifyFeedback ="
    start = text.index(needle)
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
    # include trailing semicolon if present
    if text[idx] == ";":
        idx += 1
        function_body = text[start:idx]
    return function_body


def _run_node(cases: list[dict[str, object]]):
    classifier_src = _extract_classifier()
    js_cases = json.dumps(cases)
    script = f"""
{classifier_src}
const cases = {js_cases};
const results = cases.map((sample)=>{{
  const output = classifyFeedback(sample.args);
  return {{ id: sample.id, result: {{
    state: output.state,
    tone: output.tone,
    evLossBb: output.evLossBb,
    thresholds: output.thresholds,
  }}}};
}});
process.stdout.write(JSON.stringify(results));
"""
    completed = subprocess.run(
        ["node", "-"],
        input=script,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    return {entry["id"]: entry["result"] for entry in payload}


def test_feedback_classifier_core_bands():
    cases = [
        {
            "id": "green",
            "args": {"feedback": {}, "node": {"pot_bb": 10}, "evLossRaw": 0.0},
        },
        {
            "id": "yellow_pot8",
            "args": {"feedback": {}, "node": {"pot_bb": 8}, "evLossRaw": 0.22},
        },
        {
            "id": "red_pot10",
            "args": {"feedback": {}, "node": {"pot_bb": 10}, "evLossRaw": 0.65},
        },
        {
            "id": "yellow_pot_unknown",
            "args": {"feedback": {}, "node": {}, "evLossRaw": 0.1},
        },
        {
            "id": "blunder_pot10",
            "args": {"feedback": {}, "node": {"pot_bb": 10}, "evLossRaw": 1.6},
        },
    ]
    results = _run_node(cases)

    assert results["green"]["state"] in {"success", "gain"}
    assert results["yellow_pot8"]["state"] == "warning"
    assert results["yellow_pot_unknown"]["state"] == "warning"
    assert results["red_pot10"]["state"] == "danger"
    assert results["blunder_pot10"]["state"] == "blunder"

    # Threshold sanity
    assert abs(results["yellow_pot8"]["thresholds"]["greenCutoff"] - 0.08) < 1e-9
    assert abs(results["yellow_pot8"]["thresholds"]["yellowUpper"] - 0.4) < 1e-9
    assert abs(results["yellow_pot_unknown"]["thresholds"]["yellowUpper"] - 0.4) < 1e-9
    assert abs(results["blunder_pot10"]["thresholds"]["blunderLower"] - 1.2) < 1e-9


def test_feedback_classifier_frequency_overrides():
    cases = [
        {
            "id": "freq_zero_yellow",
            "args": {
                "feedback": {"chosen": {"gto_freq": 0}},
                "node": {"pot_bb": 10},
                "evLossRaw": 0.3,
            },
        },
        {
            "id": "freq_zero_red",
            "args": {
                "feedback": {"chosen": {"gto_freq": 0}},
                "node": {"pot_bb": 10},
                "evLossRaw": 0.8,
            },
        },
        {
            "id": "rare_mix_bump",
            "args": {
                "feedback": {"chosen": {"gto_freq": 0.02}},
                "node": {"pot_bb": 10},
                "evLossRaw": 0.01,
            },
        },
    ]
    results = _run_node(cases)

    assert results["freq_zero_yellow"]["state"] == "warning"
    assert results["freq_zero_red"]["state"] == "danger"
    assert results["rare_mix_bump"]["state"] == "warning"


def test_feedback_classifier_gain_tinting():
    cases = [
        {
            "id": "gain_case",
            "args": {"feedback": {}, "node": {"pot_bb": 10}, "evLossRaw": -0.05},
        }
    ]
    results = _run_node(cases)
    assert results["gain_case"]["state"] == "gain"
    assert abs(results["gain_case"]["evLossBb"] - 0.05) < 1e-9
