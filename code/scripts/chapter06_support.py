"""Numerical companion, compact appendix, and audit for Chapter 06.

All files produced here are private manuscript inputs or aggregate diagnostics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from thesis_tables import export_quarto_table


def weighted_spearman(x_codes, y_codes, weights):
    """Spearman rho of an expanded integer-frequency sample, including tied ranks."""
    wx = np.bincount(x_codes, weights=weights)
    wy = np.bincount(y_codes, weights=weights)
    total = weights.sum()
    rx = (np.cumsum(wx) - .5 * wx)[x_codes] - total / 2
    ry = (np.cumsum(wy) - .5 * wy)[y_codes] - total / 2
    denominator = np.sqrt(np.dot(weights, rx * rx) * np.dot(weights, ry * ry))
    return np.dot(weights, rx * ry) / denominator if denominator > 0 else np.nan


def correlations(c):
    from chapter06_analysis import AGES, GENDERS, CI_NOTE
    p = c.binary.copy()
    p["age_order"] = p.age.map({a: i for i, a in enumerate(AGES)})
    rows, detail = [], []
    for label, x, y in [("Age and smile presence", "age_order", "smile"),
                        ("Face area and smile presence", "area", "smile"),
                        ("Age and smile intensity", "age_order", "intensity"),
                        ("Face area and smile intensity", "area", "intensity")]:
        row = [label]
        for gender in GENDERS:
            q = p.loc[p.gender.eq(gender)].dropna(subset=[x, y])
            _, xc = np.unique(q[x], return_inverse=True)
            _, yc = np.unique(q[y], return_inverse=True)
            issue_index = c.boot.issues.get_indexer(q.issue)
            assert (issue_index >= 0).all()
            estimate = weighted_spearman(xc, yc, np.ones(len(q)))
            assert abs(estimate - spearmanr(q[x], q[y]).statistic) < 1e-12
            draws = np.array([weighted_spearman(xc, yc, w[issue_index]) for w in c.boot.weights])
            lo, hi = np.nanquantile(draws, [.025, .975])
            row.extend([len(q), f"{estimate:+.2f} [{lo:+.2f}, {hi:+.2f}]"])
            detail.append(dict(relationship=label, gender=gender, n=len(q), issues=q.issue.nunique(),
                               estimate=estimate, low=lo, high=hi))
        rows.append(row)
    c.table(27, "correlations", pd.DataFrame(rows, columns=["Relationship", "F N", "F rho [CI]", "M N", "M rho [CI]"]),
            "Simple within-gender rank correlations.",
            "Spearman rho describes monotonic association, using tied ranks for ordinal age/intensity and binary smile presence. "
            "Intensity comparisons include smiling faces only. Ranks are recalculated in each resample. "
            "These unadjusted associations do not identify causes. No single year correlation is used because the historical smile trajectory is nonlinear. " + CI_NOTE)
    c.numeric("correlations", "estimates", pd.DataFrame(detail))
    c.results["correlations"] = detail
    c.save("correlations")


def audit_bootstrap(c):
    """Check design invariants, row-expansion equivalence, and endpoint stability."""
    from chapter06_analysis import IssueBootstrap
    issues = c.pages[["issue", "year"]].drop_duplicates().set_index("issue").reindex(c.boot.issues)
    assert np.equal(c.boot.weights, np.floor(c.boot.weights)).all()
    assert (c.boot.weights >= 0).all()
    for _, q in issues.reset_index().groupby("year"):
        assert (c.boot.weights[:, q.index].sum(axis=1) == len(q)).all()
    # An explicit repeated-row sample must equal the efficient cluster calculation.
    q = c.binary.loc[c.binary.decade.eq(1970)]
    loc = c.boot.issues.get_indexer(q.issue)
    multiplicities = c.boot.weights[0, loc].astype(int)
    expanded = q.iloc[np.repeat(np.arange(len(q)), multiplicities)]
    assert abs(expanded.smile.mean() - c.boot.mean(q)["samples"][0]) < 1e-12
    egap = expanded.groupby("gender").smile.mean()
    assert abs(egap.feminine-egap.masculine-c.boot.gap(q)["samples"][0]) < 1e-12
    # Ties and issue multiplicities are recalculated, not frozen original ranks.
    xx = np.array([0, 0, 1, 2, 2]); yy = np.array([0, 1, 1, 0, 1]); ww = np.array([2, 0, 3, 1, 4])
    assert abs(weighted_spearman(xx, yy, ww) - spearmanr(np.repeat(xx, ww), np.repeat(yy, ww)).statistic) < 1e-12
    check = IssueBootstrap(c.pages, replicates=5000, seed=6062027)
    diagnostic = []
    for label, data in [("Pooled gap", c.binary), ("1970s gap", q),
                        ("2000-07 gap", c.binary.loc[c.binary.decade.eq(2000)])]:
        a, b = c.boot.gap(data), check.gap(data)
        diagnostic.append(dict(contrast=label, estimate=a["estimate"], low_2000=a["low"], high_2000=a["high"],
                               low_5000=b["low"], high_5000=b["high"]))
    a0=c.boot.gap(q);a1=c.boot.gap(c.binary.loc[c.binary.decade.eq(2000)])
    b0=check.gap(q);b1=check.gap(c.binary.loc[c.binary.decade.eq(2000)])
    lowa,higha=np.quantile(a1["samples"]-a0["samples"],[.025,.975])
    lowb,highb=np.quantile(b1["samples"]-b0["samples"],[.025,.975])
    diagnostic.append(dict(contrast="1970s to 2000-07 gap change",estimate=a1["estimate"]-a0["estimate"],
        low_2000=lowa,high_2000=higha,low_5000=lowb,high_5000=highb))
    pd.DataFrame(diagnostic).to_csv(c.work/"bootstrap-audit.csv",index=False)
    report=dict(passed=True,replicates=2000,seed=6062026,strata="publication year",cluster="whole issue",
        year_cluster_counts=issues.groupby("year").size().to_dict(),
        minimum_clusters_per_year=int(issues.groupby("year").size().min()),
        explicit_row_expansion=True,weighted_tie_ranks=True,stability_check=diagnostic,
        regression_intervals="issue-clustered covariance and delta method; not bootstrap",
        cautions=["conditional on recorded labels and archive construction", "pointwise intervals, not simultaneous trend bands",
                  "independence between issues is assumed; repeated campaigns can cross issues",
                  "no population sampling design is reconstructed", "constant empirical cells can give zero-width intervals"])
    (c.work/"bootstrap-audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


def numerical_appendix(c, entries):
    """Four compact transcriptions; full annual/sector/size series remain in CSV."""
    from chapter06_analysis import DECADES, GENDERS, decade_label
    data = c.work/"numerical-data"
    read = lambda name: pd.read_csv(data/name)
    rep=read("ch06-gender-representation--shares.csv")
    age=read("ch06-age-composition--shares.csv")
    intensity=read("ch06-gender-intensity--shares.csv")
    records=[]
    for dec in DECADES:
        for gender in GENDERS:
            r=rep.loc[rep.decade.eq(dec)&rep.gender.eq(gender)].iloc[0]
            a=age.loc[age.decade.eq(dec)&age.gender.eq(gender)].set_index("age")
            records.append([decade_label(dec),gender.capitalize(),int(r.n),r.share,
                a.loc["young_adult","share"],a.loc["middle_adult","share"],a.loc["older_adult","share"]])
    tables=[]
    def table(name,df,caption,note,formats,widths):
        _,file=export_quarto_table(df,"ch06-appendix-"+name,caption=caption,label="tbl-ch06-appendix-"+name,
            note=note,formats=formats,data_dir=c.work,qmd_dir=c.tabledir)
        text=file.read_text(encoding="utf-8").replace("{#tbl-ch06-appendix-"+name+"}",
            "{#tbl-ch06-appendix-"+name+' tbl-colwidths="['+",".join(map(str,widths))+']"}')
        file.write_text(text,encoding="utf-8",newline="\n");tables.append(file.name)
    table("representation",pd.DataFrame(records,columns=["Decade","Gender","N","All-face share","Young adult","Middle adult","Older adult"]),
        "Numerical companion: gender representation and adult age composition.",
        "All-face share uses all recorded faces, including other gender categories. Age shares use all faces within each displayed gender, including children and unassessable ages in the denominator; the three adult columns therefore need not sum to 100%. Full age categories and counts are in the companion CSVs.",
        {k:".1%" for k in ["All-face share","Young adult","Middle adult","Older adult"]},[12,16,10,17,15,15,15])
    values=[]
    for dec in DECADES:
        q=c.people.loc[c.people.decade.eq(dec)]
        row=[decade_label(dec),len(q),int(q.smile.count()),int(q.smile.sum()),q.smile.mean()]
        for gender in ["All"]+GENDERS:
            z=intensity.loc[intensity.decade.eq(dec)&intensity.gender.eq(gender)&intensity.intensity.ge(3)]
            row.append(z.n.sum()/z.denominator.iloc[0])
        values.append(row)
    table("smile-intensity",pd.DataFrame(values,columns=["Decade","Faces","Assessed","Smiling","Smile share","Broad+ all","Broad+ F","Broad+ M"]),
        "Numerical companion: overall smiling and expansive smile intensity.",
        "Smile share = smiling/assessed. Broad+ combines broad and laughter-like intensity, using smiling faces within the corresponding group. Full four-category counts, annual values, and five-year pooled counts are in the CSVs.",
        {k:".1%" for k in ["Smile share","Broad+ all","Broad+ F","Broad+ M"]},[12,12,12,12,13,13,13,13])
    std=read("ch06-industry-standardised--shares.csv")
    rows=[[decade_label(r.decade),r.gender.capitalize(),int(r.n),r.observed,r.estimate,r.low,r.high] for r in std.itertuples()]
    table("standardisation",pd.DataFrame(rows,columns=["Decade","Gender","N","Observed","Standardised","CI low","CI high"]),
        "Numerical companion: observed and industry-standardised smile shares.",
        "Both estimates use the same six-industry common-support subset; standardisation holds its pooled industry weights fixed. Intervals belong to the standardised share. Full sector counts and weights are in the CSVs.",
        {k:".1%" for k in ["Observed","Standardised","CI low","CI high"]},[13,17,10,15,17,14,14])
    sen=read("ch06-sensitivity--decades.csv")
    names=["All assessable F/M","Face area at least 1%","Face area at least 2%","Legibility at least 2","Legibility 3","Area 1% + legibility 2+"]
    rows=[]
    for name in names:
        r=sen.loc[sen.subset.eq(name)].drop_duplicates("decade").set_index("decade")
        rows.append([name]+[r.loc[d,"gap"]*100 for d in DECADES])
    table("subgroup-trends",pd.DataFrame(rows,columns=["Minimum requirement"]+[decade_label(d) for d in DECADES]),
        "Numerical companion: full decade trajectories under size and legibility restrictions.",
        "Cells are feminine-minus-masculine smile gaps in percentage points. Each threshold is fixed across decades. Gender-specific counts, shares, interval endpoints, and pooled smile shares are supplied in the CSVs.",
        {decade_label(d):"+.1f" for d in DECADES},[30]+[10]*7)
    blocks=["# Numerical companion to Chapter 6 {#sec-analysis-numerical-data}",""]
    for i,name in enumerate(tables):
        if i==2:
            blocks += ["```{=latex}","\\clearpage","```",""]
        blocks += ["{{< include tables/generated/"+name+" >}}",""]
    (c.thesis/"appendix-analysis-data.qmd").write_text("\n".join(blocks)+"\n",encoding="utf-8",newline="\n")
    index=[]
    for e in entries:
        files=sorted(data.glob(e["name"]+"--*.csv"))
        if e["kind"]=="figure":
            assert files,"No numerical companion for "+e["name"]
        for file in files:
            index.append(dict(artifact=e["name"],caption=e["caption"],file=file.name,rows=len(pd.read_csv(file))))
    pd.DataFrame(index).to_csv(data/"index.csv",index=False)
    (data/"README.md").write_text(
        "# Exact numerical backing for Chapter 06\n\n"
        "Each retained figure has one or more files listed in index.csv. Values are exported without display rounding from the series used to create the figure. "
        "Shares, predictions, interval endpoints and face/ad area are on a 0–1 scale; multiply by 100 for percentages or percentage-point gaps. "
        "Spearman coefficients remain on −1 to +1. Counts are n/denominator or named count columns. "
        "Raw count/denominator pairs are provided for representation, age, intensity, annual smiling and size-decile smiling. "
        "For raw binary means, n times estimate recovers the smile count up to floating-point precision; this does not apply to weighted or model-adjusted estimates. "
        "Small suppressed figure cells remain in these files with a plotted flag where relevant. Missing intervals are blank; they are not zero.\n\n"
        "The appendix prints selected decade summaries; the annual curves, complete industry series, age cells and model predictions remain fully accessible here. "
        "Use these numbers for writing, not estimates read from the plot. n for regression predictions is the shared estimation sample, not a decade-specific count. "
        "The historical face-size shading is an interquartile range, not a confidence interval. Bootstrap intervals are pointwise; regression intervals use clustered covariance and the delta method.\n",
        encoding="utf-8")
