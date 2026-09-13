"""Reproducible descriptive and relational analyses for the smile-history chapter.

Private inputs/row-level derivatives are supplied through environment variables or
CLI paths. Only aggregate SVG figures enter the public figure manifest. Notebook
sources call the same section functions as the CLI; no findings are hardcoded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "data/processed/chapter06/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from thesis_tables import export_quarto_table

ROOT = Path(__file__).resolve().parents[2]
DECADES = list(range(1940, 2010, 10))
AGES = ["infant", "child", "adolescent", "young_adult", "middle_adult", "older_adult"]
AGE_LABELS = ["Infant", "Child", "Adolescent", "Young adult", "Middle adult", "Older adult"]
GENDERS = ["feminine", "masculine"]
COLORS = {"feminine": "#2A858A", "masculine": "#4C78A8", "all": "#263238"}
INT_COLORS = ["#C9E5E6", "#72B7B2", "#318B93", "#164F61"]
SHORT_INDUSTRY = {
    "Non-profit, public sector & education": "Public/non-profit/education",
    "Technology & electronics": "Technology & electronics",
    "Leisure & entertainment": "Leisure & entertainment",
}
CI_NOTE = "95% CIs use 2,000 bootstrap resamples of whole issues, stratified by publication year."


def decade_label(d):
    return "2000-07" if int(d) == 2000 else f"{int(d)}s"


def period(year):
    return np.select([np.asarray(year) < 1960, np.asarray(year) < 1980],
                     ["1940-59", "1960-79"], default="1980-2007")


def clean_label(value):
    return str(value).replace("_", " ").capitalize()


def ci_text(est, lo, hi, scale=100, signed=False):
    fmt = "+.1f" if signed else ".1f"
    if not np.isfinite(est):
        return "—"
    if not np.isfinite(lo + hi):
        return f"{format(est * scale, fmt)} [—]"
    return f"{format(est * scale, fmt)} [{lo * scale:.1f}, {hi * scale:.1f}]"


class IssueBootstrap:
    """Common resamples preserve pairing, all within-issue rows and year weights."""
    def __init__(self, pages, replicates=2000, seed=6062026):
        issue = pages[["issue", "year"]].drop_duplicates().sort_values("issue").reset_index(drop=True)
        self.issues = pd.Index(issue.issue)
        rng = np.random.default_rng(seed)
        self.weights = np.zeros((replicates, len(issue)), dtype=np.float64)
        for _, rows in issue.groupby("year", sort=True):
            ix = rows.index.to_numpy()
            self.weights[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=replicates)
        assert np.all(self.weights.sum(axis=1) == len(issue))

    def mean(self, data, value="smile", weight=None):
        d = data.loc[data[value].notna(), ["issue", value] + ([weight] if weight else [])].copy()
        if d.empty:
            return dict(n=0, issues=0, estimate=np.nan, low=np.nan, high=np.nan,
                        samples=np.full(len(self.weights), np.nan))
        w = d[weight].to_numpy(float) if weight else np.ones(len(d))
        d["_num"] = d[value].to_numpy(float) * w
        d["_den"] = w
        a = d.groupby("issue")[["_num", "_den"]].sum().reindex(self.issues, fill_value=0)
        nums, dens = (self.weights @ a.to_numpy()).T
        vals = np.divide(nums, dens, out=np.full(len(nums), np.nan), where=dens > 0)
        nissues = d.issue.nunique()
        lo, hi = np.nanquantile(vals, [.025, .975]) if nissues >= 10 else (np.nan, np.nan)
        return dict(n=len(d), issues=nissues, estimate=float(d._num.sum() / d._den.sum()),
                    low=float(lo), high=float(hi), samples=vals)

    def gap(self, data, value="smile", weight=None):
        a = self.mean(data.loc[data.gender.eq("feminine")], value, weight)
        b = self.mean(data.loc[data.gender.eq("masculine")], value, weight)
        samples = a["samples"] - b["samples"]
        lo, hi = np.nanquantile(samples, [.025, .975]) if min(a["issues"], b["issues"]) >= 10 else (np.nan, np.nan)
        return dict(estimate=a["estimate"] - b["estimate"], low=float(lo), high=float(hi),
                    feminine=a, masculine=b, samples=samples)


class Analysis:
    def __init__(self, input_path=None, thesis_dir=None, work_dir=None):
        local_thesis = ROOT.parent / "quarto_thesis"
        if not local_thesis.is_dir():
            local_thesis = ROOT.parent / "master_thesis/quarto_thesis"
        self.thesis = Path(thesis_dir or os.environ.get("CH06_THESIS", local_thesis))
        local_input = ROOT / "data/processed/llm-annotations/individual-only/llm-annotations-individual-ads-only.jsonl"
        if not local_input.is_file():
            local_input = self.thesis.parent / "analysis/data/processed/llm-annotations/individual-only/llm-annotations-individual-ads-only.jsonl"
        self.input = Path(input_path or os.environ.get("CH06_INPUT", local_input))
        self.work = Path(work_dir or os.environ.get("CH06_WORK", ROOT / "data/processed/chapter06"))
        self.figdir = ROOT / "code/output/figures"
        self.tabledir = self.thesis / "tables/generated"
        for p in [self.work, self.work / "previews", self.figdir, self.tabledir]:
            p.mkdir(parents=True, exist_ok=True)
        self.artifacts = []
        self.results = {}
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
            "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10.5,
            "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#AAB3BC",
            "axes.labelcolor": "#263238", "text.color": "#263238", "xtick.color": "#66717E",
            "ytick.color": "#66717E", "grid.color": "#D9DEE3", "grid.linewidth": .5,
            "figure.facecolor": "white", "axes.facecolor": "white", "svg.fonttype": "none",
            "svg.hashsalt": "chapter06", "savefig.dpi": 160, "legend.frameon": False})
        self.load()
        self.boot = IssueBootstrap(self.pages)

    def load(self):
        pages, ads, faces = [], [], []
        for line in self.input.open(encoding="utf-8"):
            r = json.loads(line)
            image_id = r["image_id"]
            match = re.fullmatch(r"(\d{4})-(\d{4})-(\d{4}(?:_\d{4})?)", image_id)
            if not match or not r.get("ok"):
                raise ValueError(f"Unexpected image ID or unsuccessful row: {image_id}")
            year = int(match[1]); issue = image_id[:9]
            base = dict(image_id=image_id, issue=issue, year=year, decade=year // 10 * 10)
            pageads = r["annotation"]["advertisements"]
            pages.append({**base, "ad_n": len(pageads), "spread": "_" in image_id})
            for ad in pageads:
                assert not ad.get("groups"), "Input must exclude whole people-area ads"
                ab = ad.get("bbox_1000")
                if ab is None and ad.get("extent") == "full_page":
                    ab = [0, 0, 1000, 1000]
                if ab is None or len(ab) != 4 or ab[2] <= ab[0] or ab[3] <= ab[1]:
                    raise ValueError(f"Invalid ad box: {image_id}/{ad['advertisement_id']}")
                adkey = f"{image_id}/{ad['advertisement_id']}"
                industry = ad.get("ad_category") or "Not assessable"
                ads.append({**base, "ad": adkey, "industry": industry, "n_faces": len(ad["people"]),
                            "extent": ad["extent"], "ad_area_image": (ab[2]-ab[0])*(ab[3]-ab[1])/1e6})
                for f in ad["people"]:
                    assert f.get("annotation_role") == "individual"
                    fb = f.get("face_bbox_1000")
                    good = fb is not None and len(fb) == 4 and fb[2] > fb[0] and fb[3] > fb[1]
                    area = (fb[2]-fb[0])*(fb[3]-fb[1])/((ab[2]-ab[0])*(ab[3]-ab[1])) if good else np.nan
                    xc = ((fb[0]+fb[2])/2-ab[0])/(ab[2]-ab[0]) if good else np.nan
                    yc = ((fb[1]+fb[3])/2-ab[1])/(ab[3]-ab[1]) if good else np.nan
                    geom = good and 0 < area <= 1 and 0 <= xc <= 1 and 0 <= yc <= 1
                    inherited = ad.get("depiction_type")
                    dep = f.get("depiction_type") or (inherited if inherited != "multiple_types_present" else None) or "not_assessable"
                    faces.append({**base, "ad": adkey, "face": f"{adkey}/{f['person_id']}",
                        "industry": industry, "depiction": dep,
                        "medium": "Photograph" if dep == "photo_of_person" else "Not assessable" if dep == "not_assessable" else "Other depiction",
                        "gender": f["perceived_gender_presentation"], "age": f["perceived_age"],
                        "legibility": int(f["face_expression_legibility"][0]),
                        "smile": {"yes": 1., "no": 0.}.get(f.get("smile_present"), np.nan),
                        "smile_raw": f.get("smile_present") or "Skipped (legibility 0)",
                        "intensity": int(f["smile_intensity"][0]) if f.get("smile_intensity") else np.nan,
                        "area": area if geom else np.nan, "x": xc if geom else np.nan,
                        "y": yc if geom else np.nan, "geometry_valid": geom,
                        "face_width": (fb[2]-fb[0])/(ab[2]-ab[0]) if geom else np.nan,
                        "face_height": (fb[3]-fb[1])/(ab[3]-ab[1]) if geom else np.nan})
        self.pages, self.ads, self.people = map(pd.DataFrame, [pages, ads, faces])
        p = self.people
        assert p.face.is_unique and self.ads.ad.is_unique and self.pages.image_id.is_unique
        assert p.loc[p.smile.eq(1), "intensity"].between(1, 4).all()
        assert p.loc[~p.smile.eq(1), "intensity"].isna().all()
        assert p.loc[p.legibility.eq(0), "smile"].isna().all()
        counts = p.assign(f=p.gender.eq("feminine").astype(int), m=p.gender.eq("masculine").astype(int)).groupby("ad").agg(n=("face", "size"), f=("f", "sum"), m=("m", "sum"))
        counts["context"] = np.select([counts.n.eq(1), counts.f.eq(counts.n), counts.m.eq(counts.n), counts.f.gt(0) & counts.m.gt(0) & (counts.f+counts.m).eq(counts.n)],
            ["Solo", "Feminine only", "Masculine only", "Mixed"], default="Uncertain mix")
        p = p.merge(counts, left_on="ad", right_index=True, validate="many_to_one")
        p["model_context"] = p.context.replace({"Feminine only": "Same-gender multiple", "Masculine only": "Same-gender multiple"})
        p["period"] = period(p.year)
        p["feminine"] = p.gender.eq("feminine").astype(int)
        p["t"] = (p.year - 1970)/10
        p["log_area"] = np.log(p.area)
        p["assessable"] = p.smile.notna().astype(float)
        self.people = p
        self.binary = p.loc[p.gender.isin(GENDERS) & p.smile.notna()].copy()
        self.ads = self.ads.merge(counts, left_on="ad", right_index=True, validate="one_to_one")
        adsmile = p.groupby("ad").agg(smile_n=("smile", "count"), smile_yes=("smile", "sum"), mean_smile=("smile", "mean"))
        self.ads = self.ads.merge(adsmile, left_on="ad", right_index=True, validate="one_to_one")
        # A known smile establishes presence even if another face is unassessable;
        # a known absence requires every recorded face to have smile=no.
        self.ads["any_smile"] = np.where(self.ads.smile_yes.gt(0), 1., np.where(self.ads.smile_n.eq(self.ads.n_faces), 0., np.nan))
        self.ads["period"] = period(self.ads.year)
        self.results["input"] = {"file": self.input.name, "sha256": hashlib.sha256(self.input.read_bytes()).hexdigest(),
            "pages": len(self.pages), "issues": self.pages.issue.nunique(), "ads": len(self.ads), "faces": len(p),
            "assessable_smiles": int(p.smile.count()), "smiling_faces": int(p.smile.sum()),
            "invalid_geometry": int((~p.geometry_valid).sum()), "years": [int(p.year.min()), int(p.year.max())]}
        p.to_pickle(self.work / "faces-private.pkl")
        self.ads.to_pickle(self.work / "ads-private.pkl")

    def register(self, entry):
        self.artifacts.append(entry)
        path = self.work / "artifact-index.json"
        current = json.loads(path.read_text()) if path.exists() else []
        current = [x for x in current if x["name"] != entry["name"]] + [entry]
        path.write_text(json.dumps(sorted(current, key=lambda x: x["order"]), indent=2), encoding="utf-8")

    def table(self, order, name, df, caption, note="", formats=None):
        name = "ch06-" + name
        _, qmd = export_quarto_table(df, name, caption=caption, label="tbl-"+name,
            note=note, formats=formats or {}, data_dir=self.work, qmd_dir=self.tabledir,
            alignments={c: "right" for c in df.columns[1:]})
        self.register(dict(order=order, name=name, kind="table", caption=caption, note=note,
                           file=qmd.name, rows=len(df)))

    def figure(self, order, name, fig, caption, note=""):
        name = "ch06-"+name
        for ax in fig.axes:
            if ax.get_label() != "<colorbar>":
                ax.set_axisbelow(True)
        fig.savefig(self.figdir / (name+".svg"), bbox_inches="tight", metadata={"Date": None})
        fig.savefig(self.work / "previews" / (name+".png"), bbox_inches="tight", dpi=160)
        plt.close(fig)
        self.register(dict(order=order, name=name, kind="figure", caption=caption, note=note, file=name+".svg"))

    def save(self, section):
        (self.work / f"results-{section}.json").write_text(json.dumps(self.results, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x), allow_nan=True), encoding="utf-8")

    def show(self, start, end):
        from IPython.display import display, Image, Markdown
        entries = json.loads((self.work / "artifact-index.json").read_text())
        for e in entries:
            if start <= e["order"] <= end:
                display(Markdown(f"### {e['order']}. {e['caption']}"))
                if e["kind"] == "table":
                    display(pd.read_csv(self.work / (e["name"]+".csv")))
                else:
                    display(Image(filename=str(self.work / "previews" / (e["name"]+".png"))))
                if e["note"]:
                    display(Markdown(e["note"]))


def pct_axis(ax, label="Share (%)", ylim=(0, 1)):
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel(label)
    ax.set_ylim(*ylim)
    ax.grid(axis="y")


def decade_axis(ax):
    ax.set_xticks(DECADES, [decade_label(d) for d in DECADES], rotation=30, ha="right")


def line_ci(ax, x, rows, color, label, percent=True):
    y = np.array([r["estimate"] for r in rows]); low = np.array([r["low"] for r in rows]); high = np.array([r["high"] for r in rows])
    ax.plot(x, y, "o-", color=color, label=label, lw=1.9, ms=4)
    ax.fill_between(x, low, high, color=color, alpha=.15, lw=0)
    if percent:
        pct_axis(ax)


def foundation(c):
    p, a = c.people, c.ads
    baseline = [("Processed page/spread records", len(c.pages)), ("Publication issues", c.pages.issue.nunique()),
        ("Retained advertisements", len(a)), ("Individual face depictions", len(p)),
        ("Faces with assessable smile presence", p.smile.count()), ("Smiling faces", p.smile.sum()),
        ("Non-smiling faces", p.smile.eq(0).sum()), ("Smile skipped: legibility 0", p.legibility.eq(0).sum()),
        ("Smile otherwise unassessable", p.smile_raw.eq("not_assessable").sum()),
        ("Ads with any assessable faces", a.smile_n.gt(0).sum()), ("Ads with determined smile presence", a.any_smile.count())]
    c.table(1, "corpus", pd.DataFrame(baseline, columns=["Unit / subset", "N"]),
        "Corpus and analysis denominators, 1940-2007.",
        "Entire advertisements containing people areas are excluded. Counts describe recorded depictions, including repeated appearances. Smile presence uses yes/(yes + no). "
        "Confidence intervals describe issue-composition uncertainty conditional on recorded labels; annotation error and archive-selection bias are not included.", {"N": ",.0f"})
    logpath = c.input.parent / "llm-annotations-people-area-ad-cleaning-log.csv"
    logs = pd.read_csv(logpath)
    logs["decade"] = logs.image_id.str[:4].astype(int)//10*10
    coverage=[]
    for d in DECADES:
        q=p.loc[p.decade.eq(d)]; pg=c.pages.loc[c.pages.decade.eq(d)]; ad=a.loc[a.decade.eq(d)]; lg=logs.loc[logs.decade.eq(d)]
        coverage.append([decade_label(d), pg.issue.nunique(), len(pg), len(ad), len(q), int(q.smile.count()), lg.action.eq("removed").mean()])
    c.table(2, "coverage", pd.DataFrame(coverage, columns=["Decade", "Issues", "Images", "Ads", "Faces", "Smile N", "Ads removed"]),
        "Corpus coverage and people-area exclusions by decade.", "Images are page/spread records. Ads removed is the share of source ads removed because they contained people areas.", {"Ads removed": ".1%"})
    cats=[]
    specifications=[("Depiction", "medium", p, ["Photograph", "Other depiction", "Not assessable"]),
        ("Gender", "gender", p, GENDERS+["ambiguous_or_androgynous", "not_assessable"]),
        ("Age", "age", p, AGES+["not_assessable"]), ("Legibility", "legibility", p, [0,1,2,3]),
        ("Smile", "smile_raw", p, ["yes", "no", "Skipped (legibility 0)", "not_assessable"]),
        ("Intensity", "intensity", p.loc[p.smile.eq(1)], [1,2,3,4])]
    intensity_names={1:"Slight",2:"Clear",3:"Broad",4:"Laughter-like"}
    for field,col,q,levels in specifications:
        for level in levels:
            n=q[col].eq(level).sum()
            label = intensity_names.get(level) if field=="Intensity" else (f"{level}: "+["not legible","low","moderate","high"][level]) if field=="Legibility" else clean_label(level)
            cats.append([field,label,n,n/len(q)])
    c.table(3, "distributions", pd.DataFrame(cats, columns=["Field", "Category", "N", "Share"]),
        "Distribution of recorded analysis categories.", f"Intensity percentages use the {int(p.smile.sum()):,} smiling faces; other percentages use all retained faces. Age and gender describe perceived presentation.", {"Share": ".1%"})
    fig,axs=plt.subplots(1,3,figsize=(11,3.6),layout="constrained")
    leg=pd.crosstab(p.decade,p.legibility,normalize="index").reindex(DECADES,fill_value=0)
    axs[0].stackplot(DECADES, *[leg.get(i,pd.Series(0,index=DECADES)) for i in range(4)], colors=["#D9DEE3","#C9E5E6","#72B7B2","#318B93"],labels=["0: not legible","1: low","2: moderate","3: high"])
    pct_axis(axs[0]);decade_axis(axs[0]);axs[0].set_title("Expression legibility");axs[0].legend(fontsize=8,loc="lower right")
    for gender in GENDERS:
        q=p.loc[p.gender.eq(gender)]
        vals=[c.boot.mean(q.loc[q.decade.eq(d)],"assessable") for d in DECADES]
        line_ci(axs[1],DECADES,vals,COLORS[gender],gender.capitalize());decade_axis(axs[1])
    axs[1].set_title("Smile assessability");axs[1].legend(fontsize=8)
    for gender in GENDERS:
        vals=[c.boot.mean(p.loc[p.gender.eq(gender)&p.legibility.eq(l)]) for l in [1,2,3]]
        line_ci(axs[2],[1,2,3],vals,COLORS[gender],gender.capitalize())
    axs[2].set_xticks([1,2,3],["Low","Moderate","High"]);axs[2].set_title("Smiling by legibility")
    c.figure(4,"legibility",fig,"Expression legibility, smile assessability, and smiling by legibility.",CI_NOTE)
    annual=p.groupby("year").agg(yes=("smile","sum"),n=("smile","count"),intensity=("intensity","mean"))
    annual=annual.reindex(range(1940,2008));smooth=annual.yes.rolling(5,center=True,min_periods=1).sum()/annual.n.rolling(5,center=True,min_periods=1).sum()
    fig,axs=plt.subplots(1,3,figsize=(11,3.5),layout="constrained")
    axs[0].scatter(annual.index,annual.yes/annual.n,s=11,alpha=.45,color="#66717E",label="Annual")
    axs[0].plot(annual.index,smooth,color=COLORS["all"],lw=2,label="Five-year pooled")
    pct_axis(axs[0]);axs[0].set_title("Faces smiling");axs[0].legend(fontsize=8)
    adcurves={}
    for col,label,color in [("any_smile","Ads containing a smile","#F58518"),("mean_smile","Mean within-ad smile share","#4C78A8")]:
        vals=[c.boot.mean(a.loc[a.decade.eq(d)],col) for d in DECADES];adcurves[col]=[{k:v for k,v in z.items() if k!="samples"} for z in vals]
        line_ci(axs[1],DECADES,vals,color,label)
    decade_axis(axs[1]);axs[1].set_title("Advertisement summaries");axs[1].legend(fontsize=7.5,loc="lower right")
    ints=pd.crosstab(p.loc[p.smile.eq(1),"decade"],p.loc[p.smile.eq(1),"intensity"],normalize="index").reindex(DECADES)
    axs[2].stackplot(DECADES,*[ints.get(i,pd.Series(0,index=DECADES)) for i in [1,2,3,4]],colors=INT_COLORS,labels=intensity_names.values())
    pct_axis(axs[2]);decade_axis(axs[2]);axs[2].set_title("Intensity among smiles");axs[2].legend(fontsize=8,loc="lower right")
    c.figure(5,"historical-smiling",fig,"Historical development of smile presence and intensity.",
        "Five-year curves pool smile counts; edge windows use available years. An ad is known positive if any face smiles and known negative only if every face is assessed as non-smiling. Mean within-ad shares use ads with at least one assessable face. "+CI_NOTE)
    fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout="constrained")
    medium=pd.crosstab(p.decade,p.medium,normalize="index").reindex(DECADES)
    axs[0].plot(DECADES,medium["Photograph"],"o-",color="#4C78A8");pct_axis(axs[0]);decade_axis(axs[0]);axs[0].set_title("Photographic depictions")
    for m,col in [("Photograph","#4C78A8"),("Other depiction","#F58518")]:
        vals=[c.boot.mean(p.loc[p.decade.eq(d)&p.medium.eq(m)]) for d in DECADES]
        line_ci(axs[1],DECADES,vals,col,m)
    decade_axis(axs[1]);axs[1].legend(fontsize=9);axs[1].set_title("Smiling within depiction type")
    c.figure(6,"depiction-trends",fig,"Photographic representation and smile trends by depiction type.",CI_NOTE)
    overall=c.boot.mean(p)
    c.results.update(overall_smile={k:v for k,v in overall.items() if k!="samples"}, coverage=coverage,ad_curves=adcurves,
                     smile_by_decade=p.groupby("decade").smile.agg(["count","mean"]).reset_index().to_dict("records"))
    c.save("foundation")


def gender_age(c):
    p=c.people;b=c.binary
    rep=pd.crosstab(p.decade,p.gender,normalize="index").reindex(DECADES)
    fig,ax=plt.subplots(figsize=(8,3.6),layout="constrained")
    ax.stackplot(DECADES,*[rep.get(g,pd.Series(0,index=DECADES)) for g in GENDERS+["ambiguous_or_androgynous","not_assessable"]],
        colors=[COLORS["feminine"],COLORS["masculine"],"#B4BDC6","#E4E7EB"],labels=["Feminine","Masculine","Ambiguous/androgynous","Not assessable"])
    pct_axis(ax);decade_axis(ax);ax.legend(ncol=2,fontsize=9,loc="lower right")
    c.figure(7,"gender-representation",fig,"Gender presentation among all recorded faces by decade.","Denominator includes ambiguous and unassessable presentations, including faces with unassessable smiles.")
    rows=[];gaps=[]
    for d in DECADES:
        q=b.loc[b.decade.eq(d)];g=c.boot.gap(q);gaps.append(g)
        f,m=g["feminine"],g["masculine"]
        rows.append([decade_label(d),f["n"],ci_text(f["estimate"],f["low"],f["high"]),m["n"],ci_text(m["estimate"],m["low"],m["high"]),ci_text(g["estimate"],g["low"],g["high"],signed=True)])
    c.table(8,"gender-smile-decades",pd.DataFrame(rows,columns=["Decade","F N","F smiling % [CI]","M N","M smiling % [CI]","F-M pp [CI]"]),
        "Gender-specific smile shares and their difference by decade.","F = feminine; M = masculine. N counts assessable smiles. pp = percentage points. "+CI_NOTE)
    fig,axs=plt.subplots(1,2,figsize=(9,3.6),layout="constrained")
    for gender in GENDERS:
        line_ci(axs[0],DECADES,[g[gender] for g in gaps],COLORS[gender],gender.capitalize())
    axs[0].legend();axs[0].set_title("Smile shares");decade_axis(axs[0])
    line_ci(axs[1],DECADES,gaps,"#263238","Feminine - masculine",False)
    axs[1].yaxis.set_major_formatter(PercentFormatter(1,symbol=""));axs[1].set_ylabel("Difference (percentage points)");axs[1].axhline(0,color="#AAB3BC",lw=.8);axs[1].grid(axis="y");decade_axis(axs[1]);axs[1].set_title("Gender smile gap")
    c.figure(9,"gender-smile-trends",fig,"Smile shares and the feminine-minus-masculine gap over time.",CI_NOTE)
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    agedist=[]
    for ax,gender in zip(axs,GENDERS):
        q=p.loc[p.gender.eq(gender)]
        cross=pd.crosstab(q.age,q.decade,normalize="columns").reindex(index=AGES+["not_assessable"],columns=DECADES,fill_value=0)
        im=ax.imshow(cross.to_numpy()*100,vmin=0,vmax=65,cmap="Blues",aspect="auto")
        ax.set_xticks(range(7),[decade_label(d) for d in DECADES],rotation=40,ha="right");ax.set_yticks(range(7),AGE_LABELS+["Not assessable"]);ax.set_title(gender.capitalize())
        for (i,j),v in np.ndenumerate(cross.to_numpy()*100):
            ax.text(j,i,f"{v:.0f}",ha="center",va="center",fontsize=8,color="white" if v>38 else "#263238")
        agedist.append(cross)
    fig.colorbar(im,ax=axs,label="Share of gender's faces (%)",shrink=.85)
    c.figure(10,"age-composition",fig,"Perceived age composition by gender and decade.","Each column totals 100% within gender. The model's younger age tendency established in Chapter 5 applies to these recorded categories.")
    fig,axs=plt.subplots(1,2,figsize=(10,4.4),layout="constrained")
    age_rows=[]
    for gender in GENDERS:
        vals=[c.boot.mean(b.loc[b.gender.eq(gender)&b.age.eq(age)]) for age in AGES]
        off=-.08 if gender=="feminine" else .08
        valid=np.array([v["n"]>=30 and v["issues"]>=10 and all(len(b.loc[b.gender.eq(g)&b.age.eq(age)])>=30 for g in GENDERS) for age,v in zip(AGES,vals)])
        axs[0].errorbar((np.arange(6)+off)[valid],np.array([v["estimate"] for v in vals])[valid],yerr=np.array([[v["estimate"]-v["low"] for v in vals],[v["high"]-v["estimate"] for v in vals]])[:,valid],fmt="o",color=COLORS[gender],label=gender.capitalize(),capsize=2)
        age_rows.extend([dict(gender=gender,age=age,**{k:v for k,v in val.items() if k!="samples"}) for age,val in zip(AGES,vals)])
    axs[0].set_xticks(range(6),AGE_LABELS,rotation=40,ha="right");pct_axis(axs[0]);axs[0].legend();axs[0].set_title("Smiling by age")
    mat=np.full((6,7),np.nan)
    for i,age in enumerate(AGES):
        for j,d in enumerate(DECADES):
            q=b.loc[b.age.eq(age)&b.decade.eq(d)];cnt=q.groupby("gender").size()
            if all(cnt.get(g,0)>=30 for g in GENDERS):
                mat[i,j]=q.loc[q.gender.eq("feminine"),"smile"].mean()-q.loc[q.gender.eq("masculine"),"smile"].mean()
    cmap=plt.get_cmap("RdBu_r").copy();cmap.set_bad("#E9ECEF")
    im=axs[1].imshow(mat*100,cmap=cmap,vmin=-60,vmax=60,aspect="auto")
    axs[1].set_yticks(range(6),AGE_LABELS);axs[1].set_xticks(range(7),[decade_label(d) for d in DECADES],rotation=40,ha="right");axs[1].set_title("Gender gap by age and decade")
    for (i,j),v in np.ndenumerate(mat*100):
        axs[1].text(j,i,f"{v:+.0f}" if np.isfinite(v) else "—",ha="center",va="center",fontsize=8,color="white" if abs(v)>25 else "#263238")
    fig.colorbar(im,ax=axs[1],label="F-M (percentage points)",shrink=.8)
    c.figure(11,"age-smile",fig,"Age-specific smiling and gender gaps.","Grey cells have fewer than 30 assessable faces in either gender. The left panel pools all years and omits groups with fewer than 30 faces or 10 issues. "+CI_NOTE)
    fig,axs=plt.subplots(1,2,figsize=(9,3.6),layout="constrained")
    for ax,gender in zip(axs,GENDERS):
        q=p.loc[p.gender.eq(gender)&p.smile.eq(1)]
        cross=pd.crosstab(q.decade,q.intensity,normalize="index").reindex(index=DECADES,columns=[1,2,3,4],fill_value=0)
        ax.stackplot(DECADES,*[cross[i] for i in [1,2,3,4]],colors=INT_COLORS,labels=["Slight","Clear","Broad","Laughter-like"])
        pct_axis(ax);decade_axis(ax);ax.set_title(gender.capitalize());ax.legend(fontsize=8,loc="lower right")
    c.figure(12,"gender-intensity",fig,"Smile intensity among smiling feminine and masculine depictions.","Each decade sums to 100% among the gender's smiling faces. Intensity is an ordered verbal category, not a calibrated physical scale.")
    overall=c.boot.gap(b)
    c.results.update(gender_gap={k:v for k,v in overall.items() if k not in ["samples","feminine","masculine"]},
        gender_rates=b.groupby("gender").smile.agg(["count","mean"]).reset_index().to_dict("records"),
        gender_decades=rows,age_smile=age_rows,
        intensity_gender=p.loc[p.smile.eq(1)&p.gender.isin(GENDERS)].groupby("gender").intensity.agg(["count","mean"]).reset_index().to_dict("records"))
    pd.DataFrame(age_rows).to_csv(c.work/"age-smile-detail.csv",index=False)
    c.save("gender")


def industry_context(c):
    p,b,a=c.people,c.binary,c.ads
    industry_order=a.industry.value_counts().index.tolist()
    rows=[];detail=[]
    for industry in industry_order:
        q=b.loc[b.industry.eq(industry)];allp=p.loc[p.industry.eq(industry)];g=c.boot.gap(q)
        f,m=g["feminine"],g["masculine"]
        known=allp.gender.isin(GENDERS)
        female_share=allp.loc[known,"gender"].eq("feminine").mean()
        rows.append([SHORT_INDUSTRY.get(industry,industry),a.industry.eq(industry).sum(),female_share,
            f["estimate"],m["estimate"],ci_text(g["estimate"],g["low"],g["high"],signed=True)])
        detail.append(dict(industry=industry,n_f=f["n"],n_m=m["n"],f=f["estimate"],m=m["estimate"],gap=g["estimate"],low=g["low"],high=g["high"]))
    c.table(13,"industry",pd.DataFrame(rows,columns=["Industry","Ads","F share","F smile","M smile","F-M pp [CI]"]),
        "Gender representation and smile shares by advertiser industry.",
        "F share uses faces with feminine/masculine presentation; smile shares use assessable faces within gender. Rows are ordered by ad count. "+CI_NOTE,
        {"F share":".1%","F smile":".1%","M smile":".1%"})
    pd.DataFrame(detail).to_csv(c.work/"industry-detail.csv",index=False)
    top=industry_order[:6]
    fig,axs=plt.subplots(2,3,figsize=(10.2,6),layout="constrained",sharex=True,sharey=True)
    for ax,industry in zip(axs.flat,top):
        for gender in GENDERS:
            vals=[c.boot.mean(b.loc[b.industry.eq(industry)&b.gender.eq(gender)&b.decade.eq(d)]) for d in DECADES]
            # Do not draw unstable or unsupported series points.
            vals=[v if v["n"]>=30 else {**v,"estimate":np.nan,"low":np.nan,"high":np.nan} for v in vals]
            line_ci(ax,DECADES,vals,COLORS[gender],gender.capitalize())
        decade_axis(ax);ax.set_title(SHORT_INDUSTRY.get(industry,industry),fontsize=10.5)
    axs[0,0].legend(fontsize=8,loc="lower right")
    c.figure(14,"industry-trends",fig,"Gender-specific smile trends in the six largest advertiser industries.","Industries selected by ad count, before comparing smile rates. Points require at least 30 assessable faces. "+CI_NOTE)
    # Standardise on industries with observations for both genders in every decade.
    # Fixed weights and common strata keep all period/gender comparisons comparable.
    standard_decades=DECADES[1:]
    support=b.groupby(["industry","decade","gender"]).size().unstack(["decade","gender"])
    cols=pd.MultiIndex.from_product([standard_decades,GENDERS],names=["decade","gender"])
    support=support.reindex(columns=cols,fill_value=0).fillna(0)
    common=support.index[(support>=20).all(axis=1)].tolist()
    if len(common)<2:
        raise ValueError("Too few common industries for direct standardisation")
    sub=b.loc[b.industry.isin(common)&b.decade.isin(standard_decades)]
    weights=sub.industry.value_counts(normalize=True).reindex(common)
    out=[];samples={}
    for d in standard_decades:
        for gender in GENDERS:
            estimates=[c.boot.mean(sub.loc[sub.industry.eq(i)&sub.decade.eq(d)&sub.gender.eq(gender)]) for i in common]
            v=np.array([e["estimate"] for e in estimates])@weights.to_numpy()
            bs=np.column_stack([e["samples"] for e in estimates])@weights.to_numpy()
            lo,hi=np.nanquantile(bs,[.025,.975]);samples[(d,gender)]=bs
            observed=c.boot.mean(sub.loc[sub.decade.eq(d)&sub.gender.eq(gender)])
            out.append(dict(decade=d,gender=gender,estimate=v,low=lo,high=hi,observed=observed["estimate"]))
    sdf=pd.DataFrame(out);sdf.to_csv(c.work/"industry-standardisation.csv",index=False)
    fig,axs=plt.subplots(1,2,figsize=(9,3.7),layout="constrained")
    for gender in GENDERS:
        q=sdf.loc[sdf.gender.eq(gender)]
        line_ci(axs[0],standard_decades,q.to_dict("records"),COLORS[gender],gender.capitalize()+", standardised")
        axs[0].plot(standard_decades,q.observed,"--",color=COLORS[gender],alpha=.75,label=gender.capitalize()+", observed")
    axs[0].legend(fontsize=7.5);axs[0].set_xticks(standard_decades,[decade_label(d) for d in standard_decades],rotation=30,ha="right");axs[0].set_title("Common-industry smile shares")
    gaps=[]
    for d in standard_decades:
        q=sdf.loc[sdf.decade.eq(d)].set_index("gender")
        bs=samples[(d,"feminine")]-samples[(d,"masculine")];lo,hi=np.nanquantile(bs,[.025,.975])
        gaps.append(dict(estimate=q.loc["feminine","estimate"]-q.loc["masculine","estimate"],low=lo,high=hi))
    line_ci(axs[1],standard_decades,gaps,"#263238","Standardised",False)
    obs=sdf.pivot(index="decade",columns="gender",values="observed")
    axs[1].plot(standard_decades,obs.feminine-obs.masculine,"--",color="#66717E",label="Observed")
    axs[1].yaxis.set_major_formatter(PercentFormatter(1,symbol=""));axs[1].set_ylabel("F-M (percentage points)");axs[1].axhline(0,color="#AAB3BC",lw=.8);axs[1].legend(fontsize=9);axs[1].set_xticks(standard_decades,[decade_label(d) for d in standard_decades],rotation=30,ha="right");axs[1].grid(axis="y");axs[1].set_title("Gender gap")
    names=", ".join(common)
    c.figure(15,"industry-standardised",fig,"Observed and industry-standardised gender smile trends.",
        f"1950-2007; the sparse 1940s are omitted from standardisation. Common support: {names}; {len(sub):,} faces. Both lines use the same subset; only the standardised lines use fixed pooled industry weights. "+CI_NOTE)
    contextrows=[]
    for context in ["Solo","Feminine only","Masculine only","Mixed","Uncertain mix"]:
        ad=a.loc[a.context.eq(context)]
        q=b.loc[b.context.eq(context)];g=c.boot.gap(q)
        contextrows.append([context,len(ad),g["feminine"]["n"],g["feminine"]["estimate"],g["masculine"]["n"],g["masculine"]["estimate"]])
    mixed=b.loc[b.context.eq("Mixed")].copy()
    mixed["position"]=np.where((mixed.gender.eq("feminine")&mixed.f.lt(mixed.m))|(mixed.gender.eq("masculine")&mixed.m.lt(mixed.f)),"Numerical minority",np.where(mixed.f.eq(mixed.m),"Equal numbers","Numerical majority"))
    for pos in ["Numerical minority","Equal numbers","Numerical majority"]:
        q=mixed.loc[mixed.position.eq(pos)];g=c.boot.gap(q)
        contextrows.append(["Mixed: "+pos.lower(),q.ad.nunique(),g["feminine"]["n"],g["feminine"]["estimate"],g["masculine"]["n"],g["masculine"]["estimate"]])
    c.table(16,"copresence",pd.DataFrame(contextrows,columns=["Ad composition / position","Ads","F N","F smile","M N","M smile"]),
        "Smiling by advertisement composition and numerical position.",
        "Composition uses all faces before filtering smile availability. Feminine/masculine-only rows contain multiple faces. Uncertain mix contains unassessable or ambiguous gender presentations. The final three rows overlap at ad level; position is defined separately for each gender.",
        {"F smile":".1%","M smile":".1%"})
    c.results.update(industry=detail,common_industries=common,common_industry_coverage=len(sub)/len(b),industry_standardisation=out,context=contextrows)
    exception=b.loc[b.industry.eq("Toiletries & cosmetics")].groupby(["medium","gender"]).smile.agg(["count","mean"]).reset_index()
    exception.to_csv(c.work/"cosmetics-depiction-followup.csv",index=False)
    c.results["cosmetics_followup"]=exception.to_dict("records")
    c.save("industry")


def make_dyads(c):
    p=c.people
    adult=p.age.isin(["young_adult","middle_adult","older_adult"])
    good=p.n.eq(2)&p.f.eq(1)&p.m.eq(1)&adult&p.smile.notna()&p.geometry_valid
    q=p.loc[good].copy()
    ids=q.groupby("ad").size();q=q.loc[q.ad.isin(ids.index[ids.eq(2)])]
    f=q.loc[q.gender.eq("feminine")].set_index("ad")
    m=q.loc[q.gender.eq("masculine")].set_index("ad")
    assert f.index.equals(m.index)
    d=f[["issue","year","decade","period","industry"]].copy()
    for col in ["smile","intensity","area","x","y","face_width","face_height"]:
        d["f_"+col]=f[col];d["m_"+col]=m[col]
    d["both_photo"]=f.medium.eq("Photograph")&m.medium.eq("Photograph")
    d["gap"]=d.f_smile-d.m_smile
    d["state"]=np.select([d.f_smile.eq(0)&d.m_smile.eq(0),d.f_smile.eq(1)&d.m_smile.eq(0),d.f_smile.eq(0)&d.m_smile.eq(1)],
        ["Neither","Feminine only","Masculine only"],default="Both")
    d["log_ratio"]=np.log(d.f_area/d.m_area)
    d["vertical"]=d.f_y-d.m_y
    d["size_class"]=pd.cut(d.f_area/d.m_area,[-np.inf,.8,1.25,np.inf],labels=["F smaller","Similar size","F larger"],right=True)
    d["vertical_class"]=pd.cut(d.vertical,[-np.inf,-.05,.05,np.inf],labels=["F higher","Similar height","F lower"],right=True)
    d["f_smaller"]=d.f_area.lt(d.m_area).astype(float)
    d["f_lower"]=d.vertical.gt(0).astype(float)
    return d.reset_index()


def relational(c):
    p=c.people;d=make_dyads(c)
    d.to_pickle(c.work/"adult-dyads-private.pkl")
    states=["Neither","Feminine only","Masculine only","Both"]
    rows=[]
    for label,q in [("All years",d)]+[(t,d.loc[d.period.eq(t)]) for t in ["1940-59","1960-79","1980-2007"]]:
        g=c.boot.mean(q,"gap")
        rows.append([label,len(q)]+[q.state.eq(s).mean() for s in states]+[ci_text(g["estimate"],g["low"],g["high"],signed=True)])
    c.table(17,"dyad-states",pd.DataFrame(rows,columns=["Period","Ads","Neither","F only","M only","Both","F-M pp [CI]"]),
        "Joint smile configurations in mixed-gender adult two-face advertisements.",
        "Exactly two recorded faces, one feminine and one masculine; both have an adult age category and assessable smiles. Each ad contributes once. F-M equals the feminine-only share minus the masculine-only share. "+CI_NOTE,
        {s:".1%" for s in ["Neither","F only","M only","Both"]})
    fig,axs=plt.subplots(1,2,figsize=(9,3.7),layout="constrained")
    cross=pd.crosstab(d.decade,d.state,normalize="index").reindex(index=DECADES,columns=states,fill_value=0)
    axs[0].stackplot(DECADES,*[cross[s] for s in states],colors=["#D9DEE3",COLORS["feminine"],COLORS["masculine"],"#F2B701"],labels=states)
    pct_axis(axs[0]);decade_axis(axs[0]);axs[0].legend(fontsize=8,ncol=2,loc="lower right");axs[0].set_title("Joint smile configurations")
    vals=[c.boot.mean(d.loc[d.decade.eq(dec)],"gap") for dec in DECADES]
    line_ci(axs[1],DECADES,vals,"#263238","Within two-face ads",False)
    general=[c.boot.gap(c.binary.loc[c.binary.age.isin(AGES[3:])&c.binary.decade.eq(dec)])["estimate"] for dec in DECADES]
    axs[1].plot(DECADES,general,"--",color="#AAB3BC",label="All adult depictions")
    axs[1].axhline(0,color="#AAB3BC",lw=.8);axs[1].yaxis.set_major_formatter(PercentFormatter(1,symbol=""));axs[1].set_ylabel("F-M (percentage points)");axs[1].grid(axis="y");decade_axis(axs[1]);axs[1].legend(fontsize=8);axs[1].set_title("Within-ad gender gap")
    c.figure(18,"dyad-trends",fig,"Joint smile patterns and paired gender gaps over time.","Mixed-gender adult two-face ads, compared with all adult depictions with assessable smiles. "+CI_NOTE)
    fig,axs=plt.subplots(1,2,figsize=(9,3.8),layout="constrained")
    size_summary=[]
    for gender in GENDERS:
        q=p.loc[p.gender.eq(gender)&p.geometry_valid]
        vals=q.groupby("decade").area.quantile([.25,.5,.75]).unstack().reindex(DECADES)
        axs[0].plot(DECADES,vals[.5],"o-",color=COLORS[gender],label=gender.capitalize())
        axs[0].fill_between(DECADES,vals[.25],vals[.75],color=COLORS[gender],alpha=.12)
        size_summary.extend([dict(gender=gender,decade=int(dec),q25=row[.25],median=row[.5],q75=row[.75]) for dec,row in vals.iterrows()])
    pct_axis(axs[0],"Face-box area / ad area (%)",(0,None));decade_axis(axs[0]);axs[0].legend(fontsize=9);axs[0].set_title("Median face size and middle 50%")
    q=c.binary.copy();q["size_bin"]=pd.qcut(q.area,10,labels=False,duplicates="drop")
    for gender in GENDERS:
        vals=[];xs=[]
        for binid,g in q.loc[q.gender.eq(gender)].groupby("size_bin"):
            vals.append(c.boot.mean(g));xs.append(g.area.median())
        line_ci(axs[1],xs,vals,COLORS[gender],gender.capitalize())
    axs[1].set_xscale("log");axs[1].xaxis.set_major_formatter(PercentFormatter(1));axs[1].set_xlabel("Face-box area / ad area (log scale)");axs[1].set_title("Smiling by face-size decile")
    c.figure(19,"face-size",fig,"Relative face size over time and its association with smiling.","Size uses the face/head bounding box divided by advertisement area. Left shading is the interquartile range among faces, not a confidence interval. Right intervals use issue resampling; deciles are fixed from the pooled assessable sample.")
    q=p.loc[p.gender.isin(GENDERS)&p.geometry_valid].copy()
    q["era"]=np.where(q.year<1970,"1940-69","1970-2007")
    q["col"]=np.minimum((q.x*3).astype(int),2);q["row"]=np.minimum((q.y*3).astype(int),2)
    fig,axs=plt.subplots(2,2,figsize=(8,6.2),layout="constrained")
    grids=[]
    for gender in GENDERS:
        for era in ["1940-69","1970-2007"]:
            z=q.loc[q.gender.eq(gender)&q.era.eq(era)]
            med=z.groupby(["row","col"]).area.median().unstack().reindex(index=range(3),columns=range(3))*100
            counts=z.groupby(["row","col"]).size().unstack().reindex(index=range(3),columns=range(3),fill_value=0)
            grids.append((gender,era,med,counts))
    vmax=max(np.nanmax(g[2]) for g in grids)
    for ax,(gender,era,med,counts) in zip(axs.flat,grids):
        im=ax.imshow(med,cmap="YlGnBu",vmin=0,vmax=vmax,aspect="auto")
        ax.set_xticks(range(3),["Left","Centre","Right"]);ax.set_yticks(range(3),["Top","Middle","Bottom"]);ax.set_title(f"{gender.capitalize()}, {era}")
        for (i,j),v in np.ndenumerate(med.to_numpy()):
            ax.text(j,i,f"{v:.1f}%\nn={counts.iloc[i,j]:,.0f}",ha="center",va="center",fontsize=9,color="white" if v>vmax*.65 else "#263238")
    fig.colorbar(im,ax=axs,label="Median face-box area / ad area (%)",shrink=.85)
    c.figure(20,"size-position",fig,"Face size by position within the advertisement.","A 3 x 3 grid locates face centres relative to ad boundaries. Values are median face-box area shares. Ad-relative coordinates make partial-page ads and joined spreads comparable as advertising layouts.")
    # Prominence ownership uses fractional credit for exact area ties.
    mixed=p.loc[p.context.eq("Mixed")].copy()
    mixed["is_largest"]=mixed.area.eq(mixed.groupby("ad").area.transform("max"))
    mixed["largest_credit"]=mixed.is_largest/mixed.groupby("ad").is_largest.transform("sum")
    mixed["female_credit"]=mixed.largest_credit*mixed.gender.eq("feminine")
    own=mixed.groupby("ad").agg(issue=("issue","first"),year=("year","first"),period=("period","first"),observed=("female_credit","sum"),f=("f","first"),n=("n","first"))
    own["expected"]=own.f/own.n;own["excess"]=own.observed-own.expected
    spatial=[]
    for label,dq,oq in [("All years",d,own)]+[(t,d.loc[d.period.eq(t)],own.loc[own.period.eq(t)]) for t in ["1940-59","1960-79","1980-2007"]]:
        ratio=c.boot.mean(dq,"log_ratio");vertical=c.boot.mean(dq,"vertical");ex=c.boot.mean(oq,"excess")
        spatial.append([label,len(dq),np.exp(ratio["estimate"]),dq.f_lower.mean(),len(oq),oq.observed.mean(),oq.expected.mean(),ci_text(ex["estimate"],ex["low"],ex["high"],signed=True)])
    c.table(21,"spatial-prominence",pd.DataFrame(spatial,columns=["Period","Pairs","F/M area","F lower","Mixed ads","F largest","Expected","Excess pp [CI]"]),
        "Relative face size, vertical placement, and largest-face ownership.",
        "Pairs are mixed-gender adult two-face ads; F/M area is the geometric mean of within-ad area ratios. Largest-face ownership uses all known mixed-gender ads, with fractional credit for ties. Expected ownership is each ad's feminine face share. "+CI_NOTE,
        {"F/M area":".2f","F lower":".1%","F largest":".1%","Expected":".1%"})
    own.reset_index().to_pickle(c.work/"prominence-private.pkl")
    fig,axs=plt.subplots(1,2,figsize=(9,3.7),layout="constrained")
    contrasts=[]
    for ax,col,levels,title in [(axs[0],"size_class",["F smaller","Similar size","F larger"],"Relative face area"),(axs[1],"vertical_class",["F higher","Similar height","F lower"],"Vertical position")]:
        vals=[c.boot.mean(d.loc[d[col].eq(l)],"gap") for l in levels]
        ax.errorbar(range(3),[v["estimate"] for v in vals],yerr=[[v["estimate"]-v["low"] for v in vals],[v["high"]-v["estimate"] for v in vals]],fmt="o",color="#263238",capsize=4)
        ax.set_xticks(range(3),[f"{l}\n(n={v['n']:,})" for l,v in zip(levels,vals)])
        ax.yaxis.set_major_formatter(PercentFormatter(1,symbol=""));ax.set_ylabel("Paired F-M smile gap (pp)");ax.axhline(0,color="#AAB3BC",lw=.8);ax.grid(axis="y");ax.set_title(title)
        contrasts.extend([dict(measure=col,level=l,**{k:v for k,v in v.items() if k!="samples"}) for l,v in zip(levels,vals)])
    c.figure(22,"prominence-smile",fig,"Paired gender smile gaps by relative face size and vertical position.","Similar size: feminine/masculine area ratio 0.8-1.25. Similar height: centres differ by at most 5% of ad height. These are descriptive layout indicators inspired by Goffman's relative-size and subordination themes. "+CI_NOTE)
    # Family-inspired lead: each ad contributes one mean for each child-gender
    # comparison. No family relationship or parent identity is inferred.
    family=[]
    adultages=set(AGES[3:]);childages={"infant","child"}
    for ad,z in p.groupby("ad",sort=False):
        adults=z.loc[z.age.isin(adultages)]
        kids=z.loc[z.age.isin(childages)&z.gender.isin(GENDERS)]
        if len(adults)!=2 or set(adults.gender)!=set(GENDERS) or kids.empty or not z.geometry_valid.all():
            continue
        if not z.age.isin(adultages|childages).all():
            continue
        adults=adults.set_index("gender")
        for kg,k in kids.groupby("gender"):
            other="masculine" if kg=="feminine" else "feminine"
            same=np.sqrt((k.x-adults.loc[kg,"x"])**2+(k.y-adults.loc[kg,"y"])**2).mean()
            opposite=np.sqrt((k.x-adults.loc[other,"x"])**2+(k.y-adults.loc[other,"y"])**2).mean()
            family.append(dict(ad=ad,issue=z.issue.iloc[0],gender=kg,children=len(k),same=same,other=opposite,difference=same-opposite))
    fam=pd.DataFrame(family)
    familyrows=[]
    if not fam.empty:
        for gender in GENDERS:
            q=fam.loc[fam.gender.eq(gender)];v=c.boot.mean(q,"difference")
            familyrows.append([gender.capitalize(),len(q),q.children.sum(),q.same.mean(),q.other.mean(),f"{v['estimate']:+.3f} [{v['low']:.3f}, {v['high']:.3f}]"])
    else:
        familyrows=[["No eligible cases",0,0,np.nan,np.nan,"—"]]
    c.table(23,"adult-child-proximity",pd.DataFrame(familyrows,columns=["Child presentation","Ads","Children","Same-gender adult","Other-gender adult","Difference [CI]"]),
        "Child proximity to feminine- and masculine-presenting adults.",
        "Ads have exactly one feminine and one masculine adult and at least one infant/child; all ages must be adult or child categories. Distances use face centres after scaling each ad to a unit square; they measure layout, not physical distance or verified kinship. Child distances are averaged within ad and child gender. Negative differences indicate closer same-gender adults. "+CI_NOTE,
        {"Same-gender adult":".3f","Other-gender adult":".3f"})
    fam.to_csv(c.work/"family-proximity-private.csv",index=False)
    # A focused robustness check for the exploratory area contrast.
    threshold_rows=[]
    for ratio in [1.15,1.25,1.5]:
        for label,mask in [("F smaller",d.f_area/d.m_area<1/ratio),("F larger",d.f_area/d.m_area>ratio)]:
            v=c.boot.mean(d.loc[mask],"gap")
            threshold_rows.append(dict(ratio=ratio,group=label,**{k:v for k,v in v.items() if k!="samples"}))
    pd.DataFrame(threshold_rows).to_csv(c.work/"prominence-threshold-sensitivity.csv",index=False)
    pair_sensitivity=[]
    for sample,q in [("All adult pairs",d),("Photographic adult pairs",d.loc[d.both_photo])]:
        for group,z in [("All",q)]+[(level,q.loc[q.size_class.eq(level)]) for level in ["F smaller","Similar size","F larger"]]:
            est=c.boot.mean(z,"gap")
            pair_sensitivity.append(dict(sample=sample,group=group,**{k:v for k,v in est.items() if k!="samples"}))
    pd.DataFrame(pair_sensitivity).to_csv(c.work/"dyad-photo-followup.csv",index=False)
    c.results.update(dyads=rows,dyad_decades=d.groupby("decade").gap.agg(["count","mean"]).reset_index().to_dict("records"),
        face_size=size_summary,spatial=spatial,prominence_smile=contrasts,family=familyrows,prominence_thresholds=threshold_rows)
    c.save("relational")


def models(c):
    import patsy
    import statsmodels.api as sm
    from scipy.special import expit
    p=c.binary.copy()
    p["age_model"]=p.age.replace({"infant":"Under adult", "child":"Under adult", "adolescent":"Under adult"})
    industry_n=p.industry.value_counts()
    p["industry_model"]=p.industry.where(p.industry.map(industry_n)>=200,"Other small industries")
    # Missing outcomes were removed above; categorical predictor unknowns remain
    # explicit categories. Geometry validity is audited before this restriction.
    p=p.loc[p.log_area.notna()].copy()
    adjusted_formula="smile ~ C(decade) * feminine + C(age_model) + C(industry_model) + C(medium) + log_area + C(model_context)"
    formulas={"Basic":"smile ~ C(decade) * feminine", "Adjusted":adjusted_formula,
              "Legibility sensitivity":adjusted_formula+" + C(legibility)"}
    fitted={};margins={};coefs=[]
    for label,formula in formulas.items():
        # Legibility 1 contains no recorded smiles and separates the outcome.
        # The optional legibility model uses levels 2-3 and is labelled separately.
        estimation=p.loc[p.legibility.ge(2)].copy() if label=="Legibility sensitivity" else p
        y,x=patsy.dmatrices(formula,estimation,return_type="dataframe")
        if np.linalg.matrix_rank(x.to_numpy()) != x.shape[1]:
            raise ValueError(f"Rank-deficient design: {label}")
        fit=sm.GLM(y,x,family=sm.families.Binomial()).fit(cov_type="cluster",cov_kwds={"groups":estimation.issue},maxiter=100)
        if not fit.converged:
            raise ValueError(f"Logistic model did not converge: {label}")
        covariance=np.asarray(fit.cov_params())
        assert np.linalg.eigvalsh(covariance).min()>-1e-8
        fitted[label]=dict(fit=fit,design=x.design_info,issues=estimation.issue.nunique())
        coefs.extend([dict(model=label,term=term,coefficient=fit.params[term],se=fit.bse[term],p=fit.pvalues[term]) for term in x.columns])
        output=[]
        for dec in DECADES:
            estimates={};gradients={}
            for gender,flag in [("feminine",1),("masculine",0)]:
                new=estimation.copy();new["decade"]=dec;new["feminine"]=flag
                design=np.asarray(patsy.build_design_matrices([x.design_info],new)[0])
                pred=expit(design@fit.params.to_numpy())
                est=pred.mean();gradient=(design*(pred*(1-pred))[:,None]).mean(axis=0)
                se=np.sqrt(max(0,float(gradient@covariance@gradient)))
                estimates[gender]=dict(estimate=est,low=max(0,est-1.96*se),high=min(1,est+1.96*se))
                gradients[gender]=gradient
            delta=gradients["feminine"]-gradients["masculine"]
            gap=estimates["feminine"]["estimate"]-estimates["masculine"]["estimate"]
            se=np.sqrt(max(0,float(delta@covariance@delta)))
            output.append(dict(decade=dec,**estimates,gap=dict(estimate=gap,low=gap-1.96*se,high=gap+1.96*se)))
        margins[label]=output
    pd.DataFrame(coefs).to_csv(c.work/"regression-coefficients.csv",index=False)
    # The saturated basic model must reproduce each observed decade/gender rate.
    for row in margins["Basic"]:
        for gender in GENDERS:
            observed=p.loc[p.decade.eq(row["decade"])&p.gender.eq(gender),"smile"].mean()
            assert abs(observed-row[gender]["estimate"])<1e-7
    rows=[]
    for raw,adj,leg in zip(margins["Basic"],margins["Adjusted"],margins["Legibility sensitivity"]):
        rows.append([decade_label(adj["decade"]),raw["gap"]["estimate"]*100,
            adj["feminine"]["estimate"],adj["masculine"]["estimate"],ci_text(adj["gap"]["estimate"],adj["gap"]["low"],adj["gap"]["high"],signed=True),
            ci_text(leg["gap"]["estimate"],leg["gap"]["low"],leg["gap"]["high"],signed=True)])
    model_note=(f"Basic and adjusted logistic models use {len(p):,} faces from {p.issue.nunique():,} issues. "
        "Basic: decade, gender, and their interaction. Adjusted: additionally perceived age, industry, depiction type, log face-area share, and co-presence. "
        "Infant/child/adolescent are pooled for the model; industries with fewer than 200 assessable binary-gender faces are pooled. "
        f"The separate +legibility sensitivity uses {int(p.legibility.ge(2).sum()):,} faces at levels 2-3; level 1 has no recorded smiles. "
        "Predictions average over each model's pooled covariate distribution. 95% intervals use issue-clustered covariance and the delta method.")
    c.table(24,"adjusted-model",pd.DataFrame(rows,columns=["Decade","Basic pp","Adjusted F","Adjusted M","Adjusted pp [CI]","+Legibility pp [CI]"]),
        "Unadjusted and adjusted gender differences in predicted smiling.",model_note,{"Basic pp":"+.1f","Adjusted F":".1%","Adjusted M":".1%"})
    fig,axs=plt.subplots(1,2,figsize=(9,3.6),layout="constrained")
    for gender in GENDERS:
        line_ci(axs[0],DECADES,[row[gender] for row in margins["Adjusted"]],COLORS[gender],gender.capitalize())
    axs[0].set_title("Adjusted predicted smile shares");axs[0].legend();decade_axis(axs[0])
    for name,color,ls in [("Basic","#AAB3BC","--"),("Adjusted","#263238","-"),("Legibility sensitivity","#F58518",":")]:
        vals=[row["gap"] for row in margins[name]]
        line_ci(axs[1],DECADES,vals,color,name,False)
    axs[1].axhline(0,color="#AAB3BC",lw=.8);axs[1].yaxis.set_major_formatter(PercentFormatter(1,symbol=""));axs[1].set_ylabel("F-M (percentage points)");axs[1].grid(axis="y");axs[1].legend();decade_axis(axs[1]);axs[1].set_title("Basic and adjusted gender gaps")
    c.figure(25,"adjusted-trends",fig,"Adjusted smile trajectories and gender gaps.",model_note)
    sensitivity=[]
    subsets=[("All retained depictions",c.binary,None),("Photographs",c.binary.loc[c.binary.medium.eq("Photograph")],None),
        ("Legibility 2-3",c.binary.loc[c.binary.legibility.ge(2)],None),("Legibility 3",c.binary.loc[c.binary.legibility.eq(3)],None),
        ("Adults only",c.binary.loc[c.binary.age.isin(AGES[3:])],None)]
    equal=c.binary.copy();equal["ad_weight"]=1/equal.groupby("ad").face.transform("size")
    subsets.append(("Equal total weight per ad",equal,"ad_weight"))
    sensdetail=[]
    for label,q,w in subsets:
        allgap=c.boot.gap(q,weight=w);early=c.boot.gap(q.loc[q.decade.eq(1970)],weight=w);late=c.boot.gap(q.loc[q.decade.eq(2000)],weight=w)
        draws=late["samples"]-early["samples"];lo,hi=np.nanquantile(draws,[.025,.975]);change=late["estimate"]-early["estimate"]
        sensitivity.append([label,len(q),ci_text(allgap["estimate"],allgap["low"],allgap["high"],signed=True),early["estimate"]*100,late["estimate"]*100,ci_text(change,lo,hi,signed=True)])
        sensdetail.append(dict(subset=label,n=len(q),overall=allgap["estimate"],gap1970=early["estimate"],gap2000=late["estimate"],change=change,change_low=lo,change_high=hi))
    c.table(26,"sensitivity",pd.DataFrame(sensitivity,columns=["Analysis sample / weighting","N","Overall gap pp [CI]","1970s pp","2000-07 pp","Change pp [CI]"]),
        "Sensitivity of the gender gap and its later historical change.",
        "Change = gap in 2000-2007 minus gap in the 1970s. Equal-ad weighting gives all included faces in an ad a combined weight of one. "
        "High-legibility subsets change sample composition and are not treated as corrected data. "+CI_NOTE,
        {"1970s pp":"+.1f","2000-07 pp":"+.1f"})
    fitstats=[]
    for name,items in fitted.items():
        fit=items["fit"]
        fitstats.append(dict(model=name,n=int(fit.nobs),issues=items["issues"],parameters=len(fit.params),aic=fit.aic,deviance=fit.deviance,converged=fit.converged))
    pd.DataFrame(fitstats).to_csv(c.work/"model-fit.csv",index=False)
    c.results.update(models=fitstats,margins=margins,sensitivity=sensdetail,formula=formulas)
    leg_detail=c.binary.groupby(["legibility","gender"]).smile.agg(["count","mean"]).reset_index()
    leg_detail.to_csv(c.work/"legibility-smile-detail.csv",index=False)
    c.results["legibility_smile"]=leg_detail.to_dict("records")
    c.save("models")


def finalize(c, integrate=False, update_manifest=False):
    """Validate all artifacts; optionally append their structure and update bridge."""
    entries=json.loads((c.work/"artifact-index.json").read_text())
    assert [e["order"] for e in entries]==list(range(1,27)), "Expected exactly 26 ordered artifacts"
    assert len({e["name"] for e in entries})==26
    for e in entries:
        source=(c.figdir/e["file"]) if e["kind"]=="figure" else (c.tabledir/e["file"])
        assert source.is_file() and source.stat().st_size>0
    if update_manifest:
        manifest_path=c.figdir/"manifest.json"
        manifest=json.loads(manifest_path.read_text())
        old={e["name"]:e for e in manifest["files"]}
        for e in entries:
            if e["kind"]=="figure":
                path=c.figdir/e["file"]
                old[path.name]=dict(name=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
        manifest["files"]=sorted(old.values(),key=lambda e:e["name"])
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    if integrate:
        chapter=c.thesis/"06-analysis-results.qmd"
        marker="<!-- BEGIN CH06 GENERATED ANALYSIS -->"
        content=chapter.read_bytes()
        # Preserve the existing notes verbatim, including their headings.
        prefix=content.split(marker.encode("utf-8"))[0]
        backup=c.work/"chapter06-existing-notes.qmd"
        if not backup.exists():
            backup.write_bytes(prefix)
        blocks=[marker]
        sections=[(1,"The corpus and its depicted faces"),(5,"Historical development of smiling"),
            (7,"Gender, age, and smiling"),(13,"Industry and advertisement composition"),
            (17,"Smiling and visual prominence within advertisements"),(24,"Adjusted trends and robustness")]
        sectionmap=dict(sections)
        for e in entries:
            if e["order"] in sectionmap:
                blocks.extend(["", "```{=latex}", "\\clearpage", "```", "", "## "+sectionmap[e["order"]], ""])
            if e["kind"]=="table":
                blocks.extend(["{{< include tables/generated/"+e["file"]+" >}}", ""])
            else:
                caption=e["caption"]+(" "+e["note"] if e["note"] else "")
                blocks.extend([f"![{caption}](figures/generated/fig-{e['file']})"+"{#fig-"+e["name"]+" width=100% fig-pos='H'}", ""])
        blocks.append("<!-- END CH06 GENERATED ANALYSIS -->")
        padding=b"" if prefix.endswith(b"\n\n") or prefix.endswith(b"\r\n\r\n") else b"\n\n"
        chapter.write_bytes(prefix+padding+("\n".join(blocks)+"\n").encode("utf-8"))
    provenance={"input":c.results["input"],"bootstrap":{"replicates":2000,"seed":6062026,"cluster":"publication issue","strata":"publication year"},
        "artifacts":entries,"excluded_fields":["face orientation","gaze target"],
        "model_time":"categorical decade with gender interaction", "standardisation_years":"1950-2007"}
    (c.work/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    print(f"Verified {len(entries)} artifacts: {sum(e['kind']=='table' for e in entries)} tables and {sum(e['kind']=='figure' for e in entries)} figures.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path)
    parser.add_argument("--thesis",type=Path)
    parser.add_argument("--work",type=Path)
    parser.add_argument("--sections",nargs="+",choices=["foundation","gender","industry","relational","models"],default=["foundation","gender","industry","relational","models"])
    parser.add_argument("--integrate",action="store_true")
    parser.add_argument("--manifest",action="store_true")
    args=parser.parse_args()
    c=Analysis(args.input,args.thesis,args.work)
    functions={"foundation":foundation,"gender":gender_age,"industry":industry_context,"relational":relational,"models":models}
    for section in args.sections:
        print(f"Running {section}",flush=True);functions[section](c)
    finalize(c,args.integrate,args.manifest)


if __name__=="__main__":
    main()
