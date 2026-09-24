"""Governance-sync lint: the project copy must not silently lose skill-mandated rules.

WHY THIS EXISTS
---------------
`docs/experiment-structure.md` is a *tailored copy* of the project-independent
`scientific-experiment-structure` skill. Copies drift. Real case: the project copy
still said "14 steps", carried a 4-box adversarial pass while the skill had grown to
6, and was missing `mode:` and `agent_model` entirely — so agents followed the stale
copy for months and wrote PASS verdicts that were invalid under the skill's own rule
("only a *confirmatory* arm may write PASS/FAIL").

Prose in a doc cannot prevent that. This lint can: it turns a **silent** loss of a
required rule into a **loud** test failure, at the moment someone edits the doc.

WHAT THIS LINT DOES AND DOES NOT COVER
--------------------------------------
- It catches **project-side regression**: the project copy losing a rule it had.
- It CANNOT detect **skill-side change**. If the skill grows a new rule, this test
  stays green and the project copy stays stale. Nothing in this repo can detect that.
  The only reliable check is loading the skill:
      skill_view(name='scientific-experiment-structure')

So a green run here means "the project copy has not lost what it had", NOT "the
project copy is current". Do not treat it as the latter.

If this test fails after you edited a governance doc, the fix is to restore the rule
(or, if the skill genuinely changed, update both this lint's tokens AND the doc).
"""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
AGENTS = ROOT / "AGENTS.md"
GOVERNANCE = ROOT / "docs" / "experiment-structure.md"
SKILL_NAME = "scientific-experiment-structure"


@pytest.fixture(scope="module")
def agents_text():
    return AGENTS.read_text()


@pytest.fixture(scope="module")
def governance_text():
    return GOVERNANCE.read_text()


# --------------------------------------------------------------------------
# AGENTS.md — the skill must be the FIRST thing read, and named authoritative
# --------------------------------------------------------------------------

def test_agents_names_the_skill(agents_text):
    assert SKILL_NAME in agents_text, (
        f"AGENTS.md must name the {SKILL_NAME!r} skill as the governance source "
        "of truth, or agents will follow the (lagging) project copy."
    )


def test_agents_declares_skill_wins_on_process(agents_text):
    assert "the skill wins" in agents_text.lower(), (
        "AGENTS.md must state explicitly that the skill wins on process; the "
        "project docs win only on project-specific facts."
    )


def test_skill_is_step_one_of_the_reading_order(agents_text):
    """The defect that motivated this lint: the source of truth was not in the
    reading order at all, and the lagging copy was step 2."""
    order_start = agents_text.find("read in this order")
    assert order_start != -1, "AGENTS.md lost its 'read in this order' section"

    # First numbered item after the heading.
    body = agents_text[order_start:]
    first_item = body.find("1. **")
    assert first_item != -1, "AGENTS.md reading order has no item 1"

    item_one = body[first_item:first_item + 600]
    assert SKILL_NAME in item_one, (
        "Step 1 of the AGENTS.md reading order must be loading the skill. If a "
        "project doc is step 1, agents read the lagging copy as authoritative."
    )


def test_agents_marks_the_drift_table_as_non_exhaustive(agents_text):
    low = agents_text.lower()
    assert "not a complete inventory" in low or "not an exhaustive" in low, (
        "The drift table must be labelled non-exhaustive. A list of 'known drift' "
        "invites the false conclusion that the drift has been fully enumerated."
    )


def test_agents_states_nothing_detects_skill_side_change(agents_text):
    low = agents_text.lower()
    assert "detects it" in low or "detect it" in low, (
        "AGENTS.md must say plainly that nothing in this repo detects a "
        "skill-side change."
    )


# --------------------------------------------------------------------------
# docs/experiment-structure.md — rules the copy previously lacked
# --------------------------------------------------------------------------

def test_governance_points_at_the_skill(governance_text):
    assert SKILL_NAME in governance_text, (
        "docs/experiment-structure.md must name the skill it is a copy of."
    )


def test_governance_declares_it_can_lag(governance_text):
    assert "lag" in governance_text.lower(), (
        "docs/experiment-structure.md must warn that it can lag the skill."
    )


def test_governance_requires_mode_declaration(governance_text):
    """Only a confirmatory arm may write PASS/FAIL. This rule was absent.

    Note: asserting merely that the token `mode: confirmatory` appears is NOT
    enough — it also appears in the YAML template, so deleting the actual rule
    left the test green (caught by deliberately breaking the doc). Assert the
    substantive constraint instead.
    """
    low = governance_text.lower()
    assert "only a confirmatory arm" in low, (
        "docs/experiment-structure.md must state the substantive rule: only a "
        "*confirmatory* arm may write PASS/FAIL in the ledger."
    )
    assert "exploratory" in low, (
        "docs/experiment-structure.md must define the exploratory arm kind."
    )
    assert "mode:" in governance_text, (
        "docs/experiment-structure.md must require a `mode:` field in provenance.yaml."
    )


def test_governance_records_agent_model(governance_text):
    assert "agent_model" in governance_text, (
        "docs/experiment-structure.md must record agent_model / agent_model_snapshot "
        "for agent-assisted arms."
    )


def test_governance_states_peeked_is_exploratory(governance_text):
    assert "peeked" in governance_text.lower(), (
        "docs/experiment-structure.md must state the anti-HARKing rule: a gate "
        "locked after seeing the outcome is exploratory and cannot be relabelled."
    )


def test_governance_states_feasibility_is_not_evidence(governance_text):
    assert "feasibility" in governance_text.lower(), (
        "docs/experiment-structure.md must cover feasibility mode and state that "
        "feasibility results are not evidence."
    )


@pytest.mark.parametrize("box", [
    "Null computed",              # gate zero
    "Metric tested",
    "Metric definition stable",
    "Result reproduced",
    "Extremes inspected",
    "Headline number traced",     # added to the skill after the 4-box version
    "FIXED or explicitly gated",  # found != fixed
])
def test_governance_adversarial_checklist_is_complete(governance_text, box):
    assert box in governance_text, (
        f"Adversarial-pass box {box!r} is missing from docs/experiment-structure.md. "
        "A PASS cannot be written while any box is unchecked."
    )


def test_governance_requires_tombstone_for_kill(governance_text):
    assert "DISCONTINUATION_NOTICE" in governance_text, (
        "docs/experiment-structure.md must require a DISCONTINUATION_NOTICE.md for "
        "every KILL."
    )


def test_governance_bans_evidence_in_scratch_dirs(governance_text):
    low = governance_text.lower()
    assert "scratch" in low and "commit" in low, (
        "docs/experiment-structure.md must forbid citing numbers whose producing "
        "script lives in a pruned scratch directory."
    )


def test_governance_requires_branch_hygiene(governance_text):
    low = governance_text.lower()
    assert "git branch" in low or "correct base" in low, (
        "docs/experiment-structure.md must require verifying a clean tree on the "
        "correct base before executing."
    )


# --------------------------------------------------------------------------
# provenance.yaml files — the two fields that were absent from all of them
# --------------------------------------------------------------------------

def _provenance_files():
    return sorted((ROOT / "experiments").glob("**/provenance*.yaml"))


def test_provenance_files_exist():
    assert _provenance_files(), "no provenance.yaml files found under experiments/"


def test_every_provenance_declares_mode_and_agent_model():
    missing = []
    for f in _provenance_files():
        d = yaml.safe_load(f.read_text()) or {}
        for field in ("mode", "agent_model"):
            if field not in d:
                missing.append(f"{f.relative_to(ROOT)}: {field}")
    assert not missing, (
        "Every provenance.yaml must declare `mode` and `agent_model` (the skill "
        "requires both; only a confirmatory arm may write PASS/FAIL):\n  "
        + "\n  ".join(missing)
    )


def test_declared_modes_are_valid_values():
    valid = {"confirmatory", "exploratory", None}
    bad = []
    for f in _provenance_files():
        d = yaml.safe_load(f.read_text()) or {}
        if d.get("mode") not in valid:
            bad.append(f"{f.relative_to(ROOT)}: mode={d.get('mode')!r}")
    assert not bad, "Invalid `mode` values (must be confirmatory|exploratory|null):\n  " + "\n  ".join(bad)
