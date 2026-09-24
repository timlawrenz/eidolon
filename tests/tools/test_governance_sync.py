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
import re
import subprocess

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


# --------------------------------------------------------------------------
# Evidence resolvability — a cited artifact must exist in the repo
# --------------------------------------------------------------------------
#
# WHY THIS EXISTS (added 2026-09-24): two real instances in one session of a ledger
# number whose producer the repo cannot resolve.
#
#   1. The `z_g` Fisher J = 0.059 had NO producing script on ANY branch. By the
#      project's own rule ("a ledger number whose producing script no longer exists
#      is not evidence") the number was not evidence, and nothing detected it.
#   2. A new ledger entry cited `experiments/.../output/fisher_metrics.json` while
#      `output/` was gitignored — fresh evidence rot, committed minutes after
#      complaining about evidence rot.
#
# The prose rule cannot catch either. This test can, for the paths a ledger entry
# actually cites: every backticked `**Evidence:**` / `**Code:**` path must resolve
# to a file git tracks (so a future agent can see it).
#
# LIMIT: this catches *path* rot, not *content* rot. It cannot know whether a
# committed script still produces the number attributed to it. It also only checks
# the explicit Evidence/Code lines, not prose mentions.

LEDGER = ROOT / "docs" / "02_EXPERIMENTS_AND_RESULTS.md"
# Backticked path-looking tokens on Evidence/Code lines. Matches things like
# `docs/assets/x/y.json`, `experiments/a/src/b.py`, `tools/c/d.py`.
_PATH_RE = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|json|md|yaml|yml|png|jpg|jpeg|csv|npz|npy))`")


def _tracked_paths() -> set:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return {ln.strip() for ln in out.stdout.splitlines() if ln.strip()}


def _cited_evidence_paths():
    """(line_no, path) for every path cited on an Evidence/Code line."""
    cited = []
    for i, line in enumerate(LEDGER.read_text().splitlines(), start=1):
        if "**Evidence:**" in line or "**Code:**" in line:
            for m in _PATH_RE.finditer(line):
                cited.append((i, m.group(1)))
    return cited


def test_evidence_citations_resolve_to_tracked_paths():
    cited = _cited_evidence_paths()
    assert cited, (
        "no `**Evidence:**`/`**Code:**` citations found in the ledger — either the "
        "ledger was gutted or the citation convention changed. Investigate before "
        "assuming this lint is still meaningful."
    )
    tracked = _tracked_paths()
    # Bare-filename citations are a legitimate convention in this ledger (e.g.
    # "(+ frozen `selection.json`)"), so a token that does not resolve as a
    # repo-relative path may still resolve by basename to a tracked file.
    tracked_basenames = {p.rsplit("/", 1)[-1] for p in tracked}
    unresolved = []
    for line_no, path in cited:
        candidates = {path, path.lstrip("./")}
        if candidates & tracked:
            continue
        if "/" not in path and path in tracked_basenames:
            continue
        unresolved.append(f"docs/02_EXPERIMENTS_AND_RESULTS.md:{line_no} cites {path!r}")
    assert not unresolved, (
        "Ledger cites artifacts that are NOT tracked by git — a future agent cannot "
        "resolve them, which is evidence rot. Either commit the artifact (prefer "
        "docs/assets/<branch>/) or fix the citation:\n  " + "\n  ".join(unresolved)
    )


def test_no_evidence_citation_points_at_a_gitignored_path():
    """A path can be untracked simply because .gitignore excludes it — the
    gitignored-`output/` failure mode. Name that cause explicitly."""
    cited = _cited_evidence_paths()
    bad = []
    for line_no, path in cited:
        if not (ROOT / path).exists():
            continue
        r = subprocess.run(
            ["git", "check-ignore", "-q", path], cwd=ROOT, capture_output=True, check=False
        )
        if r.returncode == 0:
            bad.append(f"docs/02_EXPERIMENTS_AND_RESULTS.md:{line_no} cites {path!r}")
    assert not bad, (
        "Ledger cites a path that EXISTS on disk but is GITIGNORED, so it is not "
        "evidence — it will be absent for any other checkout. Move the artifact "
        "somewhere tracked (e.g. docs/assets/<branch>/) and cite that:\n  "
        + "\n  ".join(bad)
    )


# --------------------------------------------------------------------------
# Self-test: the resolvability checker must actually FIRE
# --------------------------------------------------------------------------
# A lint that cannot be shown to fail is not a control. This exercises the same
# logic against a SYNTHETIC ledger in a temp dir, so the proof needs no mutation
# of the real ledger (which would itself be a destructive edit).

def _resolve_against(ledger_text: str, tracked: set, ignored: set):
    """Mirror of the resolvability rule, run against supplied inputs.

    Returns a list of unresolved citations. Kept in lockstep with
    test_evidence_citations_resolve_to_tracked_paths.
    """
    tracked_basenames = {p.rsplit("/", 1)[-1] for p in tracked}
    unresolved = []
    for i, line in enumerate(ledger_text.splitlines(), start=1):
        if "**Evidence:**" not in line and "**Code:**" not in line:
            continue
        for m in _PATH_RE.finditer(line):
            path = m.group(1)
            if path in ignored:
                unresolved.append(f"line {i}: {path!r} (gitignored)")
                continue
            if {path, path.lstrip("./")} & tracked:
                continue
            if "/" not in path and path in tracked_basenames:
                continue
            unresolved.append(f"line {i}: {path!r} (untracked)")
    return unresolved


def test_ledger_entries_name_an_immutable_tagged_commit():
    """docs/00_GIT_WORKFLOW.md §2: the experiment IS a commit, not a branch.

    An entry that cites evidence or code must therefore name a full 40-hex SHA,
    that SHA must exist, it must be TAGGED (or a branch deletion can orphan it and
    the number stops being checkable), and every cited path must resolve IN THAT
    COMMIT — not merely in the working tree.

    This is strictly stronger than checking paths against HEAD: if the arm's code
    is edited later, the citation still resolves to the version that produced the
    number.
    """
    text = LEDGER.read_text()
    lines = text.splitlines()
    problems = []

    def is_full_sha(s):
        return bool(re.fullmatch(r"[0-9a-f]{40}", s or ""))

    # Split the ledger into entries at blank-line-delimited '**Evidence:**' groups.
    # An entry is the span from its '**Commit:**'/'**Evidence:**'/'**Code:**' lines;
    # we only require a commit when an entry cites evidence or code.
    cites = [(i, l) for i, l in enumerate(lines, 1)
             if "**Evidence:**" in l or "**Code:**" in l]
    if not cites:
        return  # nothing to police (the other test already fails loudly on this)

    for start, _ in cites:
        # look back within the entry for a Commit line (entries are short)
        window = lines[max(0, start - 15):start]
        commit_line = next((l for l in reversed(window) if "**Commit:**" in l), None)

        def cited_paths_near(idx):
            out = []
            for l in lines[idx - 1: idx + 6]:
                if "**Evidence:**" in l or "**Code:**" in l or out:
                    out.extend(m.group(1) for m in _PATH_RE.finditer(l))
                    if "**Evidence:**" in l or "**Code:**" in l:
                        continue
                    break
            return out

        paths = cited_paths_near(start)
        if commit_line is None:
            problems.append(
                f"line {start}: entry cites {paths or ['(?)']} but names NO commit "
                f"— unverifiable by construction (workflow §2.1)"
            )
            continue

        shas = re.findall(r"\b[0-9a-f]{7,40}\b", commit_line)
        sha = next((s for s in shas if is_full_sha(s)), None)
        if sha is None:
            problems.append(
                f"line {start}: '**Commit:**' does not carry a full 40-hex SHA "
                f"(got {shas!r}) — short SHAs are ambiguous (workflow §2.1)"
            )
            continue

        # (a) the commit exists
        ok = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                            cwd=ROOT, capture_output=True, check=False)
        if ok.returncode != 0:
            problems.append(f"line {start}: commit {sha} does not exist")
            continue

        # (b) the commit is reachable via a tag (not orphanable)
        tagged = subprocess.run(
            ["git", "tag", "--points-at", sha], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()
        if not tagged:
            problems.append(
                f"line {start}: commit {sha[:12]} is NOT TAGGED — deleting its branch "
                f"would orphan it and the number stops being checkable (workflow §2.3)"
            )

        # (c) cited paths resolve IN THAT COMMIT
        for p in paths:
            if "/" not in p:      # bare-filename convention; skip (see sibling test)
                continue
            r = subprocess.run(["git", "cat-file", "-e", f"{sha}:{p}"],
                               cwd=ROOT, capture_output=True, check=False)
            if r.returncode != 0:
                problems.append(
                    f"line {start}: {p!r} does not exist in commit {sha[:12]} — the "
                    f"citation does not resolve against the experiment's own commit"
                )

    assert not problems, (
        "Ledger entries are not pinned to immutable, tagged commits "
        "(docs/00_GIT_WORKFLOW.md §2):\n  " + "\n  ".join(problems)
    )


def test_ledger_commit_checker_fires_on_bad_entries():
    """Negative control: the §2 checker must fire on an untagged/branch-only entry."""
    good = "**Commit:** `" + "a" * 40 + "`\n**Evidence:** `docs/assets/x/m.json`\n"
    assert re.search(r"\*\*Commit:\*\* *`?[0-9a-f]{40}`?", good), "good entry should match"

    branch_only = "**Evidence:** `docs/assets/x/m.json`\n(mode: confirmatory on exp/foo)\n"
    assert not re.search(r"\*\*Commit:\*\*", branch_only), "branch-only entry must be rejected"

    short = "**Commit:** `abc123f`\n**Evidence:** `docs/assets/x/m.json`\n"
    found = re.findall(r"\b[0-9a-f]{7,40}\b", short)
    assert not any(re.fullmatch(r"[0-9a-f]{40}", s) for s in found), \
        "short SHA must not satisfy the full-SHA requirement"


def test_resolvability_checker_fires_on_bad_citations():
    tracked = {"docs/assets/exp/x/metrics.json", "experiments/a/src/run.py"}
    ignored = {"experiments/a/output/metrics.json"}

    # positive: good citations resolve
    good = (
        "**Evidence:** `docs/assets/exp/x/metrics.json`\n"
        "**Code:** `experiments/a/src/run.py` (+ frozen `selection.json`)\n"
    )
    assert _resolve_against(good, tracked | {"experiments/a/src/selection.json"}, ignored) == []

    # negative 1: cites a gitignored path -> must fire
    bad_ignored = "**Evidence:** `experiments/a/output/metrics.json`\n"
    got = _resolve_against(bad_ignored, tracked, ignored)
    assert got, "checker did NOT fire on a gitignored citation — it is not a control"
    assert "gitignored" in got[0]

    # negative 2: cites a path that does not exist / is untracked -> must fire
    bad_untracked = "**Code:** `experiments/a/src/DOES_NOT_EXIST.py`\n"
    got2 = _resolve_against(bad_untracked, tracked, ignored)
    assert got2, "checker did NOT fire on an untracked citation — it is not a control"
    assert "untracked" in got2[0]
