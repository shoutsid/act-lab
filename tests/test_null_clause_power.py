"""Pin the pre-data reading of the section 5.14 do-nothing clause.

The clause decides whether Gate 4 is authorized, and how much it can decide is a property
of the rule and the noise rather than of the run - so it was measured before the declared
block's null queue exists, and the reading is pinned here so the verdict cannot be
re-narrated after it lands.

Two numbers carry it. Under **no effect** the clause fires with probability 79/4096 =
1.93%: a rule that withholds on >=10 of 12 adverse seeds almost never withholds on noise,
so a cleared clause is not evidence of value (it clears 98.07% of no-effect blocks). At
the **dev-fidelity effect** section 5.40 measured - the only real-text effect this program
has with the treatment in place, z = 1.20 - it fires with probability 0.8483, so a
withheld Gate 4 is strong evidence rather than a coin flip.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import v0_verdict  # noqa: E402
from null_clause_power import (  # noqa: E402
    DECLARED_SEEDS,
    Z_GRID,
    build_report,
    clause_reading,
    dev_fidelity_anchor,
)

ARTIFACT = ROOT / "artifacts" / "null-clause-power.json"


def test_the_clause_is_read_from_the_verdicts_own_constants() -> None:
    """The analysis must describe the rule the verdict applies, not a copy of it."""
    assert DECLARED_SEEDS == len(v0_verdict.DECLARED_SEEDS) == 12
    assert v0_verdict.NULL_WIN_LIMIT == 10
    # The mirror of the registered harm guard, which is why the limit is 10 and not 7.
    assert v0_verdict.HARM_GUARD_LOSS_LIMIT == v0_verdict.NULL_WIN_LIMIT


def test_the_clause_fires_on_79_of_4096_no_effect_blocks() -> None:
    """The level, in closed form: a withhold rule this lenient decides little on noise."""
    no_effect = clause_reading(0.0)
    assert no_effect["p_seed_favours_null"] == 0.5
    assert math.isclose(no_effect["p_clause_fires"], 79 / 4096, abs_tol=1e-12)
    assert math.isclose(no_effect["p_clause_fires"], 0.019287109375, abs_tol=1e-12)
    assert math.isclose(1.0 - no_effect["p_clause_fires"], 0.980712890625, abs_tol=1e-12)
    # Equivalently: clearing the clause needs only 3 of 12 seeds favouring adaptive.
    assert DECLARED_SEEDS - v0_verdict.NULL_WIN_LIMIT + 1 == 3


def test_the_dev_fidelity_effect_fires_the_clause_about_85_percent_of_the_time() -> None:
    anchor = dev_fidelity_anchor()
    # Section 5.40's own rows: the null arm ahead on 12/12 fresh seeds.
    assert anchor["null_wins_observed"] == 12
    assert anchor["effect_nats_null_minus_adaptive"] == -0.1355433166027069
    assert anchor["sample_sd"] == 0.11287968831893397
    assert math.isclose(anchor["z"], 1.2007768503022296, abs_tol=1e-12)
    reading = clause_reading(anchor["z"])
    assert math.isclose(reading["p_clause_fires"], 0.8483, abs_tol=5e-5)
    assert math.isclose(1.0 - reading["p_clause_fires"], 0.1517, abs_tol=5e-5)
    # The power curve is monotone, and the planning bar is cleared before z = 1.5.
    fires = [clause_reading(z)["p_clause_fires"] for z in Z_GRID]
    assert fires == sorted(fires)
    assert clause_reading(1.5)["p_clause_fires"] > 0.8
    assert clause_reading(0.5)["p_clause_fires"] < 0.5


def test_the_committed_artifact_is_what_this_script_produces_now() -> None:
    """Whole-artifact equality, so a hand-restated field cannot hide.

    Per-field assertions against literals would pass with a stale artifact while the
    code that wrote it computed something else - the pass bar was already restated by
    hand once in this file's construction and survived the field-by-field version.
    """
    payload = json.loads(ARTIFACT.read_text())
    assert payload == build_report()
    assert payload["report_section"] == "5.14"
    assert payload["clause"]["null_win_limit"] == v0_verdict.NULL_WIN_LIMIT
    assert payload["clause"]["adaptive_wins_required_to_clear"] == 3
    assert payload["clause"]["source"].endswith("(imported, not restated)")
    assert payload["level_under_no_effect"]["p_clause_fires"] == 79 / 4096
    assert "3 of 12" in payload["readings"]["what_a_pass_is_worth"]
    assert "0.8483" in payload["readings"]["modal_verdict"]
