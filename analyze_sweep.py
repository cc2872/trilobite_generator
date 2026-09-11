"""
analyze_sweep.py — pre-registered analysis for the v1 dorsal enrolment sweep.

Runs the checks in PREREG_v1_sweep.md §8, in that order, and prints the K1–K5 verdicts
at the end. It reads ONE csv of per-animal rows. Expected columns (from the instrument):

    preset, param_hash, instrument_version, valid, reason,
    theta_joint_deg, total_deg, gap_mm, gap_over_length,
    limited_by, limiting_pair, enroll_class,
    segCount, wedge_reach, ceph_pyg_ratio, vault, taper, spine_len,   # sweep axes A1–A6
    eyeSize,                                                           # null axis
    is_control                                                         # 1 for the null-axis subset

Realized taxa (optional) come from a second csv with the same axis columns plus
    taxon, source, enroll_class_observed, joint_angles_observed (json list)

Nothing here fits anything; every number is a count, a rank correlation, or a binomial test.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---- constants: copied from the pre-reg, do not edit here ---------------------------------
AXES = ["segCount", "wedge_reach", "ceph_pyg_ratio", "vault", "taper", "spine_len"]
PRED_SIGN = {"segCount": +1, "wedge_reach": -1, "ceph_pyg_ratio": +1,
             "vault": -1, "taper": +1, "spine_len": -1}
CLASS_RANK = {"open": 0, "discoidal": 1, "spiral": 2, "sphaeroidal": 3}
CLOSED = {"discoidal", "spiral", "sphaeroidal"}

K1_MIN_AXES = 3          # §7 K1
K1_ORDERS_MIN = 7        # §5 sign must hold in >= 7 of 10 orders
K1_P = 0.01
K2_CLASS_FRAC = 0.05     # §7 K2
K2_RHO = 0.10
K3_CLOSED_FRAC = 0.90    # §7 K3
K5_CENSOR_FRAC = 0.30    # §7 K5
H3_MIN_TAXA = 15         # §6
H3_P = 0.01


def closure_score(df: pd.DataFrame) -> pd.Series:
    """Ordinal closure: total_deg for open animals, class rank for closed ones (§5)."""
    rank = df["enroll_class"].map(CLASS_RANK).astype(float)
    # open animals ordered among themselves by total_deg, all below discoidal
    open_mask = df["enroll_class"].eq("open")
    s = rank.copy()
    s[open_mask] = df.loc[open_mask, "total_deg"] / df.loc[open_mask, "total_deg"].max()
    s[~open_mask] = 1.0 + rank[~open_mask]
    return s


def censoring_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["censored"] = (~df["valid"].astype(bool)) | df["limited_by"].eq("bound")
    df["why"] = np.where(df["valid"].astype(bool), df["limited_by"], df["reason"])
    return df.pivot_table(index="preset", columns="why", values="param_hash",
                          aggfunc="count", fill_value=0)


def null_axis_check(df: pd.DataFrame) -> dict:
    c = df[df["is_control"].astype(bool) & df["valid"].astype(bool)]
    if len(c) < 10:
        return {"n": len(c), "verdict": "insufficient"}
    # does eyeSize move the class? compare class at low vs high tercile of eyeSize
    lo, hi = c["eyeSize"].quantile([1 / 3, 2 / 3])
    a = c[c["eyeSize"] <= lo]["enroll_class"]
    b = c[c["eyeSize"] >= hi]["enroll_class"]
    class_shift = float((a.value_counts(normalize=True) - b.value_counts(normalize=True)).abs().max())
    rho, p = stats.spearmanr(c["eyeSize"], c["total_deg"])
    fail = (class_shift > K2_CLASS_FRAC) or (abs(rho) > K2_RHO)
    return {"n": len(c), "class_shift": class_shift, "rho_total_deg": rho, "p": p,
            "verdict": "FAIL (K2)" if fail else "pass"}


def axis_signs(df: pd.DataFrame) -> pd.DataFrame:
    v = df[df["valid"].astype(bool) & ~df["is_control"].astype(bool) & df["limited_by"].ne("bound")].copy()
    v["closure"] = closure_score(v)
    rows = []
    for ax in AXES:
        per_order = []
        for preset, g in v.groupby("preset"):
            if g[ax].nunique() < 3:
                continue
            rho, _ = stats.spearmanr(g[ax], g["closure"])
            per_order.append(np.sign(rho) == PRED_SIGN[ax])
        rho_all, p_all = stats.spearmanr(v[ax], v["closure"])
        confirmed = (sum(per_order) >= K1_ORDERS_MIN) and (np.sign(rho_all) == PRED_SIGN[ax]) and (p_all < K1_P)
        rows.append({"axis": ax, "pred": PRED_SIGN[ax], "orders_agree": int(sum(per_order)),
                     "orders_tested": len(per_order), "rho_pooled": rho_all, "p_pooled": p_all,
                     "confirmed": confirmed})
    return pd.DataFrame(rows)


def feasible_fraction(df: pd.DataFrame) -> tuple[float, pd.Series]:
    v = df[df["valid"].astype(bool) & ~df["is_control"].astype(bool) & df["limited_by"].ne("bound")]
    closed = v["enroll_class"].isin(CLOSED)
    return float(closed.mean()), closed.groupby(v["preset"]).mean()


def occupation_test(df: pd.DataFrame, taxa: pd.DataFrame) -> dict:
    sweep_frac, _ = feasible_fraction(df)
    n = len(taxa)
    k = int(taxa["enroll_class"].isin(CLOSED).sum())
    out = {"n_taxa": n, "n_closed": k, "sweep_closed_frac": sweep_frac}
    if n < H3_MIN_TAXA:
        out["verdict"] = "descriptive only (n < 15)"
        return out
    p = stats.binomtest(k, n, sweep_frac, alternative="greater").pvalue
    out.update({"p": p, "verdict": "supported" if p < H3_P else "not supported"})
    return out


def retrodiction(taxa: pd.DataFrame, res_deg: float) -> pd.DataFrame:
    rows = []
    for _, r in taxa.dropna(subset=["joint_angles_observed"]).iterrows():
        obs = np.asarray(json.loads(r["joint_angles_observed"]), float)
        pred = float(r["theta_joint_deg"])
        med_err = float(np.median(np.abs(obs - pred)))
        rows.append({"taxon": r["taxon"], "median_abs_err_deg": med_err,
                     "class_pred": r["enroll_class"], "class_obs": r["enroll_class_observed"],
                     "pass": (med_err <= 2 * res_deg) and (r["enroll_class"] == r["enroll_class_observed"])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_csv")
    ap.add_argument("--taxa_csv", default=None)
    ap.add_argument("--res_deg", type=float, default=0.25)
    ap.add_argument("--out", default="analysis_out")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)

    df = pd.read_csv(a.sweep_csv)
    versions = df["instrument_version"].unique()
    if len(versions) != 1:
        sys.exit(f"REFUSING: mixed instrument versions in one sweep: {versions}")

    # 1. censoring
    cens = censoring_table(df); cens.to_csv(out / "01_censoring.csv")
    censor_frac = float(((~df["valid"].astype(bool)) | df["limited_by"].eq("bound")).mean())
    print("== 1. censoring ==\n", cens, f"\noverall censored fraction: {censor_frac:.3f}\n")

    # 2. null axis — if this fails, stop here
    nc = null_axis_check(df); print("== 2. null-axis control ==\n", nc, "\n")
    if nc["verdict"].startswith("FAIL"):
        print("K2 FAILED — instrument leak. Nothing below is interpretable. Stop.")
        return

    # 3. axis signs
    signs = axis_signs(df); signs.to_csv(out / "03_axis_signs.csv", index=False)
    print("== 3. direction predictions ==\n", signs.to_string(index=False), "\n")

    # 4. feasible fraction (maps are plotted separately; this is the number)
    frac, per_order = feasible_fraction(df)
    print(f"== 4. feasible (closed) fraction == pooled {frac:.3f}\n{per_order}\n")

    # 5. realized taxa
    h3 = retro = None
    if a.taxa_csv:
        taxa = pd.read_csv(a.taxa_csv)
        h3 = occupation_test(df, taxa); print("== 5a. occupation ==\n", h3, "\n")
        retro = retrodiction(taxa, a.res_deg); retro.to_csv(out / "05_retrodiction.csv", index=False)
        print("== 5b. retrodiction ==\n", retro.to_string(index=False), "\n")

    # 6. verdicts
    print("== 6. kill conditions ==")
    print(f"K1 (H1 dead if < {K1_MIN_AXES} axes confirmed): {int(signs['confirmed'].sum())} confirmed ->",
          "KILL" if signs["confirmed"].sum() < K1_MIN_AXES else "alive")
    print(f"K2 (control): {nc['verdict']}")
    print(f"K3 (H2 dead if closed frac >= {K3_CLOSED_FRAC}): {frac:.3f} ->", "KILL" if frac >= K3_CLOSED_FRAC else "alive")
    if retro is not None and len(retro):
        print(f"K4 (retrodiction): {'KILL' if not retro['pass'].all() else 'alive'}")
    print(f"K5 (censoring > {K5_CENSOR_FRAC}): {censor_frac:.3f} ->", "KILL" if censor_frac > K5_CENSOR_FRAC else "alive")
    if h3: print(f"H3: {h3['verdict']}")


if __name__ == "__main__":
    main()
