"""Numerical companion, compact appendix, and audit for Chapter 06.

All files produced here are private manuscript inputs or aggregate diagnostics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from thesis_tables import export_quarto_table


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
        explicit_row_expansion=True,stability_check=diagnostic,
        cautions=["conditional on recorded labels and archive construction", "pointwise intervals, not simultaneous trend bands",
                  "independence between issues is assumed; repeated campaigns can cross issues",
                  "no population sampling design is reconstructed", "constant empirical cells can give zero-width intervals"])
    (c.work/"bootstrap-audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


def numerical_appendix(c, entries):
    """Selected transcriptions and the relocated gender table; all series in CSV."""
    from chapter06_analysis import DECADES, GENDERS, ci_text, decade_label
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
    depiction_rep=read("ch06-depiction-trends--representation.csv")
    depiction_smile=read("ch06-depiction-trends--smiling.csv")
    depiction_rows=[]
    for dec in DECADES:
        photo=depiction_rep.loc[depiction_rep.decade.eq(dec)&depiction_rep.medium.eq("Photograph")].iloc[0]
        photo_smile=depiction_smile.loc[depiction_smile.decade.eq(dec)&depiction_smile.medium.eq("Photograph")].iloc[0]
        other_smile=depiction_smile.loc[depiction_smile.decade.eq(dec)&depiction_smile.medium.eq("Other depiction")].iloc[0]
        depiction_rows.append([
            decade_label(dec),int(photo.denominator),int(photo.n),photo.share,
            int(photo_smile.n),ci_text(photo_smile.estimate,photo_smile.low,photo_smile.high),
            int(other_smile.n),ci_text(other_smile.estimate,other_smile.low,other_smile.high),
        ])
    table("depiction",pd.DataFrame(depiction_rows,columns=[
            "Decade","Faces","Photograph N","Photograph share","Photo smile N",
            "Photo smiling % [CI]","Other smile N","Other smiling % [CI]"]),
        "Numerical companion: depiction type and smiling by decade.",
        "Photograph share uses all recorded faces as its denominator, including the 11 faces whose depiction type is not assessable. Smile N counts faces with an assessable yes/no smile label within the displayed depiction type. Confidence intervals use the common issue-resampling procedure.",
        {"Photograph share":".1%"},[10,9,11,13,11,18,11,17])
    multi=c.ads.loc[c.ads.n_faces.ge(2)].copy()
    multi["fully_assessed"]=multi.smile_n.eq(multi.n_faces)
    fully=multi.loc[multi.fully_assessed].copy()
    fully["configuration"]=np.select(
        [fully.smile_yes.eq(0),fully.smile_yes.eq(fully.n_faces)],
        ["None", "All"], default="Mixed")
    fully["all_smiling"]=fully.configuration.eq("All").astype(float)
    multi_rows=[]
    periods=[(decade_label(d),d) for d in DECADES]+[("Total",None)]
    for label,dec in periods:
        m=multi if dec is None else multi.loc[multi.decade.eq(dec)]
        q=fully if dec is None else fully.loc[fully.decade.eq(dec)]
        retained=c.ads if dec is None else c.ads.loc[c.ads.decade.eq(dec)]
        smiling=q.loc[q.smile_yes.gt(0)]
        observed=c.boot.mean(smiling,"all_smiling")
        face_smile_share=q.smile_yes.sum()/q.n_faces.sum()
        expected_all=np.sum(face_smile_share**q.n_faces)
        expected_any=np.sum(1-(1-face_smile_share)**q.n_faces)
        benchmark=expected_all/expected_any
        counts=q.configuration.value_counts()
        config=lambda name: f"{int(counts.get(name,0)):,} ({counts.get(name,0)/len(q):.1%})"
        multi_rows.append([
            label,
            f"{len(m):,} ({len(m)/len(retained):.1%})",
            f"{len(q):,} ({len(q)/len(m):.1%})",
            f"{len(m)-len(q):,} ({(len(m)-len(q))/len(m):.1%})",
            config("None"),config("Mixed"),config("All"),
            ci_text(observed["estimate"],observed["low"],observed["high"]),
            benchmark,
        ])
    table("multi-face-smiling",pd.DataFrame(multi_rows,columns=[
            "Period","Multi-face, n (% ads)","Fully assessed, n (% multi)",
            "Not fully assessed, n (%)","None, n (%)","Mixed, n (%)","All, n (%)",
            "All among ads with smile, % [CI]","Independent benchmark"]),
        "Numerical companion: multi-face advertisement sample and smile configurations.",
        "Multi-face means at least two recorded individual faces. Fully assessed requires a yes/no smile label for every recorded face; configuration percentages use these ads as their denominator. The observed conditional share is all-smiling/(all-smiling + mixed). The independence benchmark preserves the period's face-level smile share and observed number of faces per ad while treating faces as independent. The total-row benchmark uses the pooled face-level smile share. Confidence intervals use the common issue-resampling procedure.",
        {"Independent benchmark":".1%"},[12,14,17,16,12,12,12,24,15])
    blocks=["# Numerical companion to Chapter 6 {#sec-analysis-numerical-data}",""]
    for i,name in enumerate(tables):
        if i in [2,3]:
            blocks += ["\\clearpage",""]
        blocks += ["{{< include tables/generated/"+name+" >}}",""]
    blocks += ["\\clearpage","",
        "{{< include tables/generated/ch06-gender-smile-decades.qmd >}}", ""]
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
        "Shares, interval endpoints and face/ad area are on a 0–1 scale; multiply by 100 for percentages or percentage-point gaps. "
        "Counts are n/denominator or named count columns. "
        "Raw count/denominator pairs are provided for representation, age, intensity and annual smiling. "
        "For raw binary means, n times estimate recovers the smile count up to floating-point precision; this does not apply to weighted estimates. "
        "Small suppressed figure cells remain in these files with a plotted flag where relevant. Missing intervals are blank; they are not zero.\n\n"
        "The appendix prints selected decade summaries; the annual curves, complete industry series and age cells remain fully accessible here. "
        "The detailed gender-by-decade table is retained in the appendix. Literature-comparison CSVs contain computed estimates, denominators and benchmark metadata; "
        "published values are reference descriptions, not additional observations. Bibliographic sources and comparison limits are specified in the chapter table notes. "
        "Use these numbers for writing, not estimates read from the plot. Bootstrap intervals are pointwise.\n",
        encoding="utf-8")
    from zipfile import ZipFile, ZIP_DEFLATED
    with ZipFile(c.work/"chapter06-numerical-data.zip", "w", compression=ZIP_DEFLATED) as archive:
        for name in ["README.md", "index.csv"] + [r["file"] for r in index]:
            archive.write(data/name, arcname=name)
