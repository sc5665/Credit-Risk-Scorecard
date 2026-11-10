"""
Build a classic points-based credit scorecard from dataset_ (Credit Risk).

Pipeline: clean -> WOE-bin every predictor -> logistic regression on the
WOE-transformed matrix -> rescale coefficients to points (PDO scaling) ->
emit the scorecard table plus model/validation stats as JSON for the
artifact to render.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from explore_dataset import load_dataset

OUT_PATH = Path(__file__).parent / "scorecard_output.json"

# same palette as the published artifact, for visual continuity
COLOR_ACCENT = "#0e5c56"
COLOR_OK = "#1d7a4c"
COLOR_BAD = "#a3321f"
COLOR_MUTED = "#6b6457"

VAR_LABELS = {
    "person_income": "Annual income",
    "person_emp_length": "Employment length",
    "loan_amnt": "Loan amount",
    "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan / income ratio",
    "person_home_ownership": "Home ownership",
    "loan_intent": "Loan purpose",
    "loan_grade": "Loan grade",
    "cb_person_default_on_file": "Prior default on file",
    "person_age": "Age",
    "cb_person_cred_hist_length": "Credit history length",
}

# PDO scaling anchors (industry-standard choice, not fit from data):
# a score of 600 corresponds to good:bad odds of 50:1, and every 20 points
# doubles the odds.
BASE_SCORE = 600
BASE_ODDS = 50
PDO = 20
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)

CONTINUOUS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
CATEGORICAL = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    for c in CONTINUOUS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # physically implausible values are treated as missing, not truncated
    df.loc[df["person_age"] > 100, "person_age"] = np.nan
    df.loc[df["person_emp_length"] > 60, "person_emp_length"] = np.nan
    return df


def bin_continuous(series: pd.Series, n_bins: int = 6) -> pd.Series:
    valid = series.dropna()
    try:
        cuts = pd.qcut(valid, q=n_bins, duplicates="drop")
    except ValueError:
        cuts = pd.qcut(valid, q=max(2, n_bins // 2), duplicates="drop")
    labels = cuts.astype(str)
    out = pd.Series(index=series.index, dtype=object)
    out.loc[valid.index] = labels
    out.loc[series.isna()] = "Missing"
    return out


def woe_table(bins: pd.Series, y: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"bin": bins, "y": y})
    g = df.groupby("bin", observed=True)["y"].agg(total="count", bad="sum")
    g["good"] = g["total"] - g["bad"]
    total_good = g["good"].sum()
    total_bad = g["bad"].sum()
    # Laplace smoothing avoids -inf/+inf WOE on empty-class bins.
    g["dist_good"] = (g["good"] + 0.5) / (total_good + 0.5 * len(g))
    g["dist_bad"] = (g["bad"] + 0.5) / (total_bad + 0.5 * len(g))
    g["woe"] = np.log(g["dist_good"] / g["dist_bad"])
    g["bad_rate"] = g["bad"] / g["total"]
    g["iv_contrib"] = (g["dist_good"] - g["dist_bad"]) * g["woe"]
    return g


def order_bins(index_vals, is_continuous: bool):
    vals = list(index_vals)
    if "Missing" in vals:
        vals.remove("Missing")
    if is_continuous:
        def left_edge(label):
            s = label.strip("()[]")
            return float(s.split(",")[0])
        vals = sorted(vals, key=left_edge)
    else:
        vals = sorted(vals)
    return vals + (["Missing"] if "Missing" in index_vals else [])


def make_plots(y, pred_prob_bad, fpr, tpr, auc, hist_edges, good_counts, bad_counts, iv_rank, scorecard, band_tbl):
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10,
        "axes.edgecolor": "#ddd8cc",
        "axes.labelcolor": COLOR_MUTED,
        "xtick.color": COLOR_MUTED,
        "ytick.color": COLOR_MUTED,
        "text.color": "#242019",
    })

    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.plot(fpr, tpr, color=COLOR_ACCENT, lw=2.5)
    ax1.plot([0, 1], [0, 1], color=COLOR_MUTED, lw=1.2, ls="--")
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title(f"ROC curve (AUC = {auc:.3f})")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    centers = (hist_edges[:-1] + hist_edges[1:]) / 2
    width = (hist_edges[1] - hist_edges[0]) * 0.42
    ax2.bar(centers - width / 2, good_counts, width=width, color=COLOR_OK, label="Repaid")
    ax2.bar(centers + width / 2, bad_counts, width=width, color=COLOR_BAD, label="Defaulted")
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Applicants")
    ax2.set_title("Score distribution by outcome")
    ax2.legend(frameon=False)
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(7, 0.5 * len(iv_rank) + 1))
    names = [VAR_LABELS.get(r["variable"], r["variable"]) for r in iv_rank][::-1]
    ivs = [r["iv"] for r in iv_rank][::-1]
    ax3.barh(names, ivs, color=COLOR_ACCENT)
    for i, v in enumerate(ivs):
        ax3.text(v + 0.01, i, f"{v:.3f}", va="center", color=COLOR_MUTED, fontsize=9)
    ax3.set_xlabel("Information value (IV)")
    ax3.set_title("Variable importance")
    fig3.tight_layout()

    ncols = 3
    nrows = -(-len(scorecard) // ncols)
    fig4, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows))
    axes = axes.flatten()
    for ax, v in zip(axes, scorecard):
        labels = [r["bin"] for r in v["rows"]]
        woes = [r["woe"] for r in v["rows"]]
        colors = [COLOR_OK if w >= 0 else COLOR_BAD for w in woes]
        ax.barh(range(len(labels)), woes, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0, color=COLOR_MUTED, lw=0.8)
        ax.set_title(VAR_LABELS.get(v["variable"], v["variable"]), fontsize=10)
        ax.invert_yaxis()
    for ax in axes[len(scorecard):]:
        ax.axis("off")
    fig4.suptitle("Weight of evidence by bin (green = lower risk, red = higher risk)")
    fig4.tight_layout(rect=[0, 0, 1, 0.96])

    fig5, ax5 = plt.subplots(figsize=(6, 4))
    labels = band_tbl["score"].tolist()
    rates = (band_tbl["bad_rate"] * 100).tolist()
    ax5.bar(labels, rates, color=COLOR_BAD)
    ax5.set_ylabel("Default rate (%)")
    ax5.set_xlabel("Score band")
    ax5.set_title("Default rate by score band")
    plt.setp(ax5.get_xticklabels(), rotation=20, ha="right")
    fig5.tight_layout()

    plt.show()


def main(show_plots: bool = True):
    raw = load_dataset()
    df = clean(raw)
    y = df["loan_status"].astype(int)

    variables = {}
    woe_matrix = pd.DataFrame(index=df.index)

    for c in CONTINUOUS:
        bins = bin_continuous(df[c])
        variables[c] = {"bins": bins, "kind": "continuous"}

    for c in CATEGORICAL:
        bins = df[c].fillna("Missing")
        variables[c] = {"bins": bins, "kind": "categorical"}

    iv_summary = {}
    for name, v in variables.items():
        tbl = woe_table(v["bins"], y)
        v["woe_table"] = tbl
        iv_summary[name] = tbl["iv_contrib"].sum()
        woe_matrix[name] = v["bins"].map(tbl["woe"])

    # Keep variables with at least weak predictive power (IV >= 0.02),
    # the standard scorecard screening threshold.
    kept = [k for k, iv in iv_summary.items() if iv >= 0.02]
    dropped = [k for k in iv_summary if k not in kept]

    X = woe_matrix[kept].values
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    model.fit(X, y)
    beta0 = model.intercept_[0]
    betas = dict(zip(kept, model.coef_[0]))

    pred_prob_bad = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, pred_prob_bad)
    gini = 2 * auc - 1

    # KS statistic: max separation between cumulative good/bad distributions
    # across score deciles of predicted bad-probability.
    order = np.argsort(pred_prob_bad)
    y_sorted = y.values[order]
    cum_bad = np.cumsum(y_sorted) / y_sorted.sum()
    cum_good = np.cumsum(1 - y_sorted) / (1 - y_sorted).sum()
    ks = float(np.max(np.abs(cum_bad - cum_good)))

    # ROC curve, downsampled to ~40 points for a light payload (full curve
    # has one point per unique probability, more than this chart needs).
    fpr, tpr, _ = roc_curve(y, pred_prob_bad)
    roc_idx = np.unique(np.linspace(0, len(fpr) - 1, min(40, len(fpr))).astype(int))
    roc_curve_points = [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in roc_idx]

    base_points = OFFSET - FACTOR * beta0

    scorecard = []
    for name in kept:
        tbl = variables[name]["woe_table"]
        is_cont = variables[name]["kind"] == "continuous"
        rows = []
        for b in order_bins(tbl.index, is_cont):
            r = tbl.loc[b]
            pts = -FACTOR * betas[name] * r["woe"]
            rows.append({
                "bin": b,
                "count": int(r["total"]),
                "bad_rate": round(float(r["bad_rate"]), 4),
                "woe": round(float(r["woe"]), 4),
                "points": round(float(pts)),
            })
        scorecard.append({
            "variable": name,
            "iv": round(float(iv_summary[name]), 4),
            "coefficient": round(float(betas[name]), 4),
            "rows": rows,
        })

    # score every applicant: base points + each variable's bin points
    points_matrix = pd.DataFrame(index=df.index)
    for name in kept:
        tbl = variables[name]["woe_table"]
        pts_by_bin = -FACTOR * betas[name] * tbl["woe"]
        points_matrix[name] = variables[name]["bins"].map(pts_by_bin)
    total_score = base_points + points_matrix.sum(axis=1)
    total_score.name = "score"

    band_edges = [300, 500, 550, 600, 650, 700, 850]
    bands = pd.cut(total_score, bins=band_edges, include_lowest=True)
    bands.name = "band"
    band_tbl = pd.DataFrame({"bad": y}).groupby(bands, observed=True).agg(
        count=("bad", "count"), bad_rate=("bad", "mean")
    ).reset_index()
    band_tbl["score"] = band_tbl["band"].astype(str)
    band_tbl = band_tbl.drop(columns=["band"])

    # score histogram: fixed-width bins across the observed range, counted
    # separately for repaid (good) vs defaulted (bad) applicants.
    n_hist_bins = 22
    hist_edges = np.linspace(total_score.min(), total_score.max(), n_hist_bins + 1)
    good_counts, _ = np.histogram(total_score[y == 0], bins=hist_edges)
    bad_counts, _ = np.histogram(total_score[y == 1], bins=hist_edges)
    score_histogram = [
        {
            "lo": round(float(hist_edges[i])),
            "hi": round(float(hist_edges[i + 1])),
            "good": int(good_counts[i]),
            "bad": int(bad_counts[i]),
        }
        for i in range(n_hist_bins)
    ]

    # variable importance, ranked by information value
    iv_rank = sorted(
        ({"variable": k, "iv": round(float(iv_summary[k]), 4)} for k in kept),
        key=lambda r: r["iv"],
        reverse=True,
    )

    example_idx = points_matrix.sample(1, random_state=7).index[0]
    example = {
        "raw": {c: (None if pd.isna(df.loc[example_idx, c]) else float(df.loc[example_idx, c]) if isinstance(df.loc[example_idx, c], (np.floating, np.integer)) else df.loc[example_idx, c]) for c in CONTINUOUS + CATEGORICAL},
        "bins": {name: str(variables[name]["bins"].loc[example_idx]) for name in kept},
        "points": {name: round(float(points_matrix.loc[example_idx, name])) for name in kept},
        "base_points": round(float(base_points)),
        "total_score": round(float(total_score.loc[example_idx])),
        "actual_outcome": int(y.loc[example_idx]),
    }

    output = {
        "n_rows": int(len(df)),
        "n_duplicates_dropped": int(len(raw) - len(df)),
        "base_score": BASE_SCORE,
        "base_odds": BASE_ODDS,
        "pdo": PDO,
        "base_points": round(float(base_points)),
        "auc": round(float(auc), 4),
        "gini": round(float(gini), 4),
        "ks": round(float(ks), 4),
        "overall_bad_rate": round(float(y.mean()), 4),
        "kept_variables": kept,
        "dropped_variables": {k: round(float(iv_summary[k]), 4) for k in dropped},
        "scorecard": scorecard,
        "score_bands": band_tbl.to_dict(orient="records"),
        "roc_curve": roc_curve_points,
        "score_histogram": score_histogram,
        "iv_rank": iv_rank,
        "score_min": round(float(total_score.min())),
        "score_max": round(float(total_score.max())),
        "score_mean": round(float(total_score.mean()), 1),
        "score_median": round(float(total_score.median()), 1),
        "example_applicant": example,
    }

    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Wrote {OUT_PATH}")
    print(f"AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}")
    print("Kept:", kept)
    print("Dropped (IV<0.02):", output["dropped_variables"])

    if show_plots:
        make_plots(y, pred_prob_bad, fpr, tpr, auc, hist_edges, good_counts, bad_counts, iv_rank, scorecard, band_tbl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true", help="skip opening matplotlib chart windows")
    args = parser.parse_args()
    main(show_plots=not args.no_plots)
