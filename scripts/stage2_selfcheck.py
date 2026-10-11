"""Stage 2 self-check — mathematical oracles and mode wiring, on synthetic data.

Runs BEFORE any dataset smoke and independently of it. Every primitive is
checked against a hand-derived or independently-computed value, never against
the same code path under test.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from ossfs_dd.config import MetricFamily, SelectionConfig, UnsupportedSemanticsError  # noqa: E402
from ossfs_dd.decisions import decide  # noqa: E402
from ossfs_dd.metrics import (  # noqa: E402
    conditional_mutual_information,
    entropy,
    joint_entropy,
    mutual_information,
    symmetrical_uncertainty,
)
from ossfs_dd.moments import RunningMoments, seed_from_values  # noqa: E402
from ossfs_dd.selector import Selector  # noqa: E402
from ossfs_dd_data.workspace import rel_to_research, results_dir, utc_now, write_json  # noqa: E402

TOL_INFO = 1e-10
TOL_MEAN = 1e-12
TOL_SD = 1e-10


class SelfCheckFailure(RuntimeError):
    pass


def check(name: str, ok: bool, detail: str = "") -> dict:
    return {"check": name, "pass": bool(ok), "detail": detail}


def hand_entropy(counts: list[float], base: float = 2.0) -> float:
    """H from multiplicities. Accepts counts or probabilities; normalises."""
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) / math.log(base)
                for c in counts if c > 0)


def run() -> list[dict]:
    out: list[dict] = []

    # ---- entropy against a hand-computed table ----------------------
    # [0,0,0,1] -> p = (3/4, 1/4); H = 0.811278124459 bits
    h = entropy(np.array([0, 0, 0, 1])).value
    expect = hand_entropy([0.75, 0.25])
    out.append(check("entropy_hand_table", abs(h - expect) < TOL_INFO,
                     f"got {h!r} expected {expect!r}"))

    # Base cross-check: H_bits = H_nats / ln(2), not H_nats * ln(2).
    h_e = entropy(np.array([0, 0, 0, 1]), log_base=math.e).value
    out.append(check("entropy_base_switch", abs(h_e / math.log(2) - h) < TOL_INFO,
                     f"base2={h!r} basee={h_e!r} (H_bits = H_nats/ln2)"))

    # ---- MI against a hand-computed 2x2 table ----------------------
    # x = [0,0,1,1], y = [0,1,0,1] -> independent, I = 0
    mi0 = mutual_information(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])).value
    out.append(check("mi_independent_is_zero", abs(mi0) < TOL_INFO, f"I={mi0!r}"))

    # perfectly dependent -> I = H(X)
    x = np.array([0, 0, 1, 1])
    y = np.array([0, 0, 1, 1])
    mi1 = mutual_information(x, y).value
    out.append(check("mi_identical_equals_entropy", abs(mi1 - entropy(x).value) < TOL_INFO,
                     f"I={mi1!r} H={entropy(x).value!r}"))

    # Hand-derived 2x2 table.
    # x=[0,0,1,1] -> H(X)=H([2,2])=1
    # y=[0,1,1,1] -> H(Y)=H([1,3])=0.81127812445913283
    # (x,y) pairs: (0,0),(0,1),(1,1),(1,1) -> H(X,Y)=H([1,1,2])=1.5
    # I(X;Y) = 1 + 0.81127812445913283 - 1.5 = 0.31127812445913283
    x2 = np.array([0, 0, 1, 1])
    y2 = np.array([0, 1, 1, 1])
    expect_mi = hand_entropy([2, 2]) + hand_entropy([1, 3]) - hand_entropy([1, 1, 2])
    mi2 = mutual_information(x2, y2).value
    out.append(check("mi_hand_2x2", abs(mi2 - expect_mi) < TOL_INFO,
                     f"got {mi2!r} expected {expect_mi!r}"))

    # ---- joint entropy / CMI identity ------------------------------
    j = joint_entropy(x2, y2).value
    out.append(check("joint_entropy_consistency",
                     abs((entropy(x2).value + entropy(y2).value - j) - mi2) < TOL_INFO,
                     "H(X)+H(Y)-H(X,Y) == I(X;Y)"))

    # CMI against a brute-force decomposition into per-conditional-value MI.
    # NOTE: there is no identity I(X;Y|Z) = I(X;Y) - I(X;Z) + I(Y;Z); an earlier
    # version of this check asserted one and was itself wrong.
    z = np.array([0, 1, 0, 1])
    cmi = conditional_mutual_information(x2, y2, z).value
    decomposition = 0.0
    for zv in sorted(set(z.tolist())):
        idx = [i for i in range(len(z)) if z[i] == zv]
        p = len(idx) / len(z)
        sub_mi = mutual_information(np.asarray(x2)[idx], np.asarray(y2)[idx]).value
        decomposition += p * sub_mi
    out.append(check("cmi_vs_per_level_decomposition", abs(cmi - decomposition) < TOL_INFO,
                     f"cmi={cmi!r} sum_z P(z)I(X;Y|z)={decomposition!r}"))

    # CMI with an independent conditioner must equal MI
    z_ind = np.array([0, 1, 0, 1])  # independent of x by construction
    out.append(check("cmi_conditioning_on_independent",
                     abs(conditional_mutual_information(x2, x2, z_ind).value
                         - mutual_information(x2, x2).value) < TOL_INFO, "cmi==mi when Z independent"))

    # ---- SU bounds and degenerate case -----------------------------
    su = symmetrical_uncertainty(x, x)
    out.append(check("su_identical_is_one", abs(su - 1.0) <= TOL_INFO, f"SU={su!r}"))
    out.append(check("su_bounded", 0.0 - TOL_INFO <= su <= 1.0 + TOL_INFO, f"SU={su!r}"))
    # NaN arises only when BOTH operands have zero entropy (denominator 0).
    const = np.zeros(4, dtype=int)
    both_const = np.zeros(4, dtype=int)
    out.append(check("su_zero_entropy_is_nan",
                     math.isnan(symmetrical_uncertainty(const, both_const)),
                     "both operands constant -> 0/0 -> NaN"))
    out.append(check("su_one_constant_operand_is_finite",
                     math.isfinite(symmetrical_uncertainty(const, x)),
                     "one constant operand -> denominator H(x)>0, finite"))

    # ---- online moments vs numpy at EVERY prefix -------------------
    rng = np.random.default_rng(11)
    worst_mean = worst_sd = 0.0
    stream = [float(x) for x in rng.normal(size=60)]
    rm = RunningMoments()
    for t in range(1, len(stream) + 1):
        rm.update(stream[t - 1])
        arr = np.asarray(stream[:t], dtype=np.float64)
        worst_mean = max(worst_mean, abs(rm.mean - float(arr.mean())))
        if t >= 2:
            worst_sd = max(worst_sd, abs(rm.std - float(arr.std(ddof=1))))
    out.append(check("online_mean_vs_numpy", worst_mean <= TOL_MEAN,
                     f"max |diff| = {worst_mean:.3e}"))
    out.append(check("online_sd_vs_numpy", worst_sd <= TOL_SD,
                     f"max |diff| = {worst_sd:.3e}"))

    # constant stream -> zero variance (degenerate thresholds)
    rm = RunningMoments()
    for _ in range(10):
        rm.update(0.5)
    out.append(check("constant_stream_zero_variance", rm.std == 0.0 and rm.degenerate,
                     f"std={rm.std!r}"))

    # bootstrap matches numpy on the first two scores
    m, s = seed_from_values([0.25, 0.75])
    out.append(check("warmup_bootstrap",
                     abs(m - 0.5) < TOL_INFO and abs(s - math.sqrt(0.125)) < TOL_INFO,
                     f"mean={m!r} std={s!r}"))

    # ---- decision boundary semantics -------------------------------
    out.append(check("decide_reject_on_equality",
                     decide("matlab_compatible_v1", entry="mi2", score=1.0,
                            alpha=1.0, beta=2.0).decision.value == "reject",
                     "mi2 uses <=alpha, tested first"))
    out.append(check("decide_accept_on_equality",
                     decide("matlab_compatible_v1", entry="mi2", score=2.0,
                            alpha=1.0, beta=2.0).decision.value == "accept",
                     "mi2 uses >=beta"))
    out.append(check("decide_delay_in_between",
                     decide("matlab_compatible_v1", entry="mi2", score=1.5,
                            alpha=1.0, beta=2.0).decision.value == "delay", "strict interior"))
    out.append(check("decide_degenerate_differs_between_files",
                     decide("matlab_compatible_v1", entry="mi2", score=1.0, alpha=1.0, beta=1.0).decision.value == "reject"
                     and decide("matlab_compatible_v1", entry="z2", score=1.0, alpha=1.0, beta=1.0).decision.value == "accept",
                     "sigma=0: mi2 rejects, z2 accepts — reproduced, not smoothed"))

    # ---- continuous family refuses rather than inventing -------------
    try:
        Selector(SelectionConfig(), metric_family=MetricFamily.GAUSSIAN_FISHER_Z,
                 labels=np.array(["a", "b"] * 2, dtype=object))
        out.append(check("continuous_refuses_without_source", False, "no error raised"))
    except UnsupportedSemanticsError as exc:
        out.append(check("continuous_refuses_without_source",
                         "UNSUPPORTED_UNVERIFIED_SEMANTICS" in str(exc), str(exc)[:110]))

    # ---- config guards ----------------------------------------------
    try:
        SelectionConfig(k1=2.0, k2=1.0)
        out.append(check("config_rejects_k1_ge_k2", False, "no error raised"))
    except ValueError as exc:
        out.append(check("config_rejects_k1_ge_k2", True, str(exc)[:80]))

    return out


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Stage 2 self-check").parse_args(argv)
    print("=== STAGE 2 SELF-CHECK (synthetic only) ===")
    results = run()
    failed = [r for r in results if not r["pass"]]
    for r in results:
        print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['check']:38s} {r['detail'][:70]}")
    payload = {
        "schema": "ossfs-dd-reproduction/stage2/selfcheck/1.0",
        "generated_utc": utc_now(),
        "checks": results,
        "passed": len(results) - len(failed),
        "failed": len(failed),
    }
    results_dir("stage2")
    out = write_json(results_dir("stage2") / "selfcheck.json", payload)
    print(f"\n  {payload['passed']} passed, {payload['failed']} failed")
    print(f"  wrote {rel_to_research(out)}")
    if failed:
        print("SELF-CHECK FAILED")
        return 2
    print("SELF-CHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
