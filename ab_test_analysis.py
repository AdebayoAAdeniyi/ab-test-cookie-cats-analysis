"""A/B test analysis of the Cookie Cats mobile game dataset.

Question: did moving the first progression gate from level 30 to level 40
change player retention?

Usage:
    python ab_test_analysis.py path/to/cookie_cats.csv

Outputs are written to the folder ./results
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_effectsize,
    proportions_ztest,
)

SEED = 20261022
N_BOOT = 5000
ALPHA = 0.05
CONTROL = "gate_30"
TREATMENT = "gate_40"
OUT = Path("results")


def load(path):
    df = pd.read_csv(path)
    expected = {"userid", "version", "sum_gamerounds", "retention_1", "retention_7"}
    missing = expected - set(df.columns)
    if missing:
        sys.exit(f"Missing columns: {sorted(missing)}")
    return df


def data_checks(df, log):
    log("DATA CHECKS")
    log(f"Rows: {len(df):,}")
    log(f"Missing values per column: {df.isna().sum().to_dict()}")
    log(f"Duplicated userid: {int(df.userid.duplicated().sum())}")
    counts = df.version.value_counts()
    log(f"Users per arm: {counts.to_dict()}")

    # Sample ratio mismatch: is the split consistent with 50/50?
    n_c, n_t = counts[CONTROL], counts[TREATMENT]
    chi2, p = stats.chisquare([n_c, n_t])
    log(f"Sample ratio check against a 50/50 split: chi2 = {chi2:.3f}, p = {p:.4f}")

    zero = (df.sum_gamerounds == 0).groupby(df.version).mean()
    log(f"Share of users with 0 game rounds: {zero.round(4).to_dict()}")
    top = df.sum_gamerounds.nlargest(3).tolist()
    log(f"Three largest game round counts: {top}")
    log("")


def compare_proportion(df, col):
    c = df.loc[df.version == CONTROL, col].astype(int)
    t = df.loc[df.version == TREATMENT, col].astype(int)
    x = np.array([t.sum(), c.sum()])
    n = np.array([len(t), len(c)])
    p_t, p_c = x / n
    z, p = proportions_ztest(x, n)
    lo, hi = confint_proportions_2indep(x[0], n[0], x[1], n[1], method="newcomb")

    # Nonparametric bootstrap of the difference in proportions
    rng = np.random.default_rng(SEED)
    c_arr, t_arr = c.to_numpy(), t.to_numpy()
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        diffs[i] = (
            rng.choice(t_arr, len(t_arr)).mean() - rng.choice(c_arr, len(c_arr)).mean()
        )
    b_lo, b_hi = np.percentile(diffs, [2.5, 97.5])

    return {
        "metric": col,
        "control_rate": p_c,
        "treatment_rate": p_t,
        "diff_pp": (p_t - p_c) * 100,
        "ci_low_pp": lo * 100,
        "ci_high_pp": hi * 100,
        "boot_ci_low_pp": b_lo * 100,
        "boot_ci_high_pp": b_hi * 100,
        "relative_change_pct": (p_t - p_c) / p_c * 100,
        "z": z,
        "p_value": p,
        "n_control": n[1],
        "n_treatment": n[0],
    }


def rounds_analysis(df, log):
    log("GAME ROUNDS (descriptive and robustness checks, not a primary metric)")
    c = df.loc[df.version == CONTROL, "sum_gamerounds"]
    t = df.loc[df.version == TREATMENT, "sum_gamerounds"]
    log(f"Mean rounds: control {c.mean():.2f}, treatment {t.mean():.2f}")
    log(f"Median rounds: control {c.median():.1f}, treatment {t.median():.1f}")
    u, p_u = stats.mannwhitneyu(t, c, alternative="two-sided")
    log(f"Mann-Whitney U test p = {p_u:.4f}")
    t2, p_w = stats.ttest_ind(t, c, equal_var=False)
    log(f"Welch t-test on raw rounds p = {p_w:.4f} (sensitive to the extreme value)")

    trimmed = df[df.sum_gamerounds < df.sum_gamerounds.max()]
    c2 = trimmed.loc[trimmed.version == CONTROL, "sum_gamerounds"]
    t2_ = trimmed.loc[trimmed.version == TREATMENT, "sum_gamerounds"]
    _, p_w2 = stats.ttest_ind(t2_, c2, equal_var=False)
    log(
        "After removing the single largest value: mean rounds "
        f"control {c2.mean():.2f}, treatment {t2_.mean():.2f}, Welch p = {p_w2:.4f}"
    )
    log("")


def power_analysis(df, log):
    log("POWER ANALYSIS")
    c = df.loc[df.version == CONTROL]
    n_c = len(c)
    n_t = int((df.version == TREATMENT).sum())
    ratio = n_t / n_c
    solver = NormalIndPower()
    for col in ["retention_1", "retention_7"]:
        base = c[col].mean()
        # Smallest absolute lift detectable with 80% power at this sample size
        es = solver.solve_power(
            nobs1=n_c, alpha=ALPHA, power=0.8, ratio=ratio, alternative="two-sided"
        )
        # Cohen's h is 2*asin(sqrt(p2)) - 2*asin(sqrt(p1)). Invert it for p2 above the baseline.
        p2 = np.sin(np.arcsin(np.sqrt(base)) + es / 2) ** 2
        log(
            f"{col}: baseline {base:.4f}. With {n_c:,} and {n_t:,} users per arm, "
            f"a lift of about {(p2 - base) * 100:.2f} percentage points is the smallest "
            "detectable with 80% power at alpha 0.05."
        )
    # Sample size needed per arm to detect a 1 percentage point drop in 7 day retention
    base = c["retention_7"].mean()
    h = proportion_effectsize(base, base - 0.01)
    n_needed = solver.solve_power(effect_size=h, alpha=ALPHA, power=0.8, ratio=1.0)
    log(
        f"To detect a 1 percentage point change in 7 day retention from the baseline "
        f"{base:.4f} with 80% power, about {int(np.ceil(n_needed)):,} users per arm are needed."
    )
    log("")


def make_chart(results, path):
    labels = {"retention_1": "1 day retention", "retention_7": "7 day retention"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.35
    for i, r in enumerate(results):
        for j, (name, rate, colour) in enumerate(
            [
                (CONTROL, r["control_rate"], "#4C78A8"),
                (TREATMENT, r["treatment_rate"], "#F58518"),
            ]
        ):
            n = r["n_control"] if name == CONTROL else r["n_treatment"]
            se = np.sqrt(rate * (1 - rate) / n)
            ax.bar(
                i + (j - 0.5) * width,
                rate * 100,
                width,
                yerr=1.96 * se * 100,
                capsize=4,
                color=colour,
                label=name if i == 0 else None,
            )
            ax.text(
                i + (j - 0.5) * width,
                rate * 100 + 1.96 * se * 100 + 0.6,
                f"{rate * 100:.1f}%",
                ha="center",
                fontsize=9,
            )
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([labels[r["metric"]] for r in results])
    ax.set_ylabel("Players retained (%)")
    ax.set_ylim(0, 55)
    ax.set_title("Cookie Cats: retention by gate position\n(error bars are 95% confidence intervals)")
    ax.legend(title="Version")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python ab_test_analysis.py path/to/cookie_cats.csv")
    OUT.mkdir(exist_ok=True)
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    df = load(sys.argv[1])
    data_checks(df, log)

    results = [compare_proportion(df, col) for col in ["retention_1", "retention_7"]]
    # Two retention metrics are tested, so adjust p-values with the Holm method
    adj = multipletests([r["p_value"] for r in results], alpha=ALPHA, method="holm")[1]
    for r, a in zip(results, adj):
        r["p_value_holm"] = a

    log("RETENTION TESTS (two proportion z-test, treatment minus control)")
    for r in results:
        log(
            f"{r['metric']}: control {r['control_rate']*100:.2f}%, treatment "
            f"{r['treatment_rate']*100:.2f}%, difference {r['diff_pp']:.2f} pp "
            f"(95% CI {r['ci_low_pp']:.2f} to {r['ci_high_pp']:.2f}; bootstrap CI "
            f"{r['boot_ci_low_pp']:.2f} to {r['boot_ci_high_pp']:.2f}), relative change "
            f"{r['relative_change_pct']:.1f}%, z = {r['z']:.2f}, p = {r['p_value']:.4f}, "
            f"Holm adjusted p = {r['p_value_holm']:.4f}"
        )
    log("")

    rounds_analysis(df, log)
    power_analysis(df, log)

    pd.DataFrame(results).round(5).to_csv(OUT / "retention_results.csv", index=False)
    make_chart(results, OUT / "retention_chart.png")
    (OUT / "results.txt").write_text("\n".join(lines) + "\n")
    log(f"Saved results to {OUT.resolve()}")


if __name__ == "__main__":
    main()
