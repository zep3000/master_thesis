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
CHAPTER_ORDERS = [2, 7, 10, 5, 9, 11, 12, 6, 13, 14, 28, 29, 30, 16, 17, 21]


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
        p["period"] = period(p.year)
        p["feminine"] = p.gender.eq("feminine").astype(int)
        p["assessable"] = p.smile.notna().astype(float)
        self.people = p
        self.binary = p.loc[p.gender.isin(GENDERS) & p.smile.notna()].copy()
        self.ads = self.ads.merge(counts, left_on="ad", right_index=True, validate="one_to_one")
        adsmile = p.groupby("ad").agg(smile_n=("smile", "count"), smile_yes=("smile", "sum"), mean_smile=("smile", "mean"))
        self.ads = self.ads.merge(adsmile, left_on="ad", right_index=True, validate="one_to_one")
        # A known smile establishes presence even if another face is unassessable;
        # a known absence requires every recorded face to have smile=no.
        self.ads["any_smile"] = np.where(self.ads.smile_yes.gt(0), 1., np.where(self.ads.smile_n.eq(self.ads.n_faces), 0., np.nan))
        # Conversely, every recorded face must be assessed as smiling to establish
        # an all-smiling ad; one assessed non-smile is enough to establish absence.
        self.ads["all_smile"] = np.where(
            self.ads.smile_yes.lt(self.ads.smile_n), 0.,
            np.where(self.ads.smile_n.eq(self.ads.n_faces) & self.ads.smile_yes.eq(self.ads.n_faces), 1., np.nan),
        )
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

    def numeric(self, name, panel, table):
        """Save the actual plotted values, without display rounding or row identifiers."""
        target = self.work / "numerical-data"
        target.mkdir(exist_ok=True)
        table.to_csv(target / f"ch06-{name}--{panel}.csv", index=False)

    def table(self, order, name, df, caption, note="", formats=None):
        name = "ch06-" + name
        _, qmd = export_quarto_table(df, name, caption=caption, label="tbl-"+name,
            note=note, formats=formats or {}, data_dir=self.work, qmd_dir=self.tabledir,
            alignments={c: "left" if order == 28 else "right" for c in df.columns[1:]})
        # Give intervals and long labels room in the native thesis tables.
        # Equal-width columns otherwise split short intervals across three lines.
        widths = {
            2: [12, 10, 12, 25, 17, 24],
            8: [12, 8, 24, 8, 24, 24],
            13: [29, 9, 11, 11, 11, 29],
            21: [14, 16, 21, 21, 28],
            24: [12, 10, 28, 11, 11, 28],
            26: [29, 9, 10, 10, 21, 21],
            28: [29, 44, 27],
            29: [37, 17, 12, 34],
            30: [28, 9, 10, 9, 10, 34],
        }
        if order in widths:
            columns = ",".join(map(str, widths[order]))
            text = qmd.read_text(encoding="utf-8")
            text = text.replace("{#tbl-" + name + "}",
                                "{#tbl-" + name + ' tbl-colwidths="[' + columns + ']"}')
            qmd.write_text(text, encoding="utf-8", newline="\n")
        self.register(dict(order=order, name=name, kind="table", caption=caption, note=note,
                           file=qmd.name, rows=len(df)))

    def figure(self, order, name, fig, caption, note=""):
        name = "ch06-"+name
        for ax in fig.axes:
            if ax.get_label() != "<colorbar>":
                ax.set_axisbelow(True)
        fig.savefig(self.figdir / (name+".svg"), bbox_inches="tight", metadata={"Date": None})
        svg_path=self.figdir/(name+".svg")
        svg_path.write_bytes(svg_path.read_bytes().replace(b"\r\n",b"\n"))
        fig.savefig(self.work / "previews" / (name+".png"), bbox_inches="tight", dpi=160)
        plt.close(fig)
        self.register(dict(order=order, name=name, kind="figure", caption=caption, note=note, file=name+".svg"))

    def save(self, section):
        (self.work / f"results-{section}.json").write_text(json.dumps(self.results, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x), allow_nan=True), encoding="utf-8")

    def show(self, start, end):
        from IPython.display import display, Image, Markdown
        entries = json.loads((self.work / "artifact-index.json").read_text())
        for e in entries:
            if start <= e["order"] <= end and e["order"] in CHAPTER_ORDERS:
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
    logpath = c.input.parent / "llm-annotations-people-area-ad-cleaning-log.csv"
    logs = pd.read_csv(logpath)
    logs["decade"] = logs.image_id.str[:4].astype(int)//10*10
    coverage=[]
    for d in DECADES:
        q=p.loc[p.decade.eq(d)]; pg=c.pages.loc[c.pages.decade.eq(d)]; ad=a.loc[a.decade.eq(d)]; lg=logs.loc[logs.decade.eq(d)]
        removed=int(lg.action.eq("removed").sum())
        coverage.append([decade_label(d), pg.issue.nunique(), len(pg), f"{removed:,} ({removed/len(lg):.1%})", len(ad), len(q)])
    removed=int(logs.action.eq("removed").sum())
    coverage.append(["Total", c.pages.issue.nunique(), len(c.pages), f"{removed:,} ({removed/len(logs):.1%})", len(a), len(p)])
    c.table(2, "coverage", pd.DataFrame(coverage, columns=["Decade", "Issues", "Images", "People-area ads excluded, n (%)", "Retained ads", "Retained individual faces"]),
        "Analysis corpus totals, decade coverage, and people-area-ad exclusions, 1940-2007.",
        "Images are page/spread records. Exclusion percentages use all source ads within each row as their denominator (27,756 overall).",
        {k: ",.0f" for k in ["Issues", "Images", "Retained ads", "Retained individual faces"]})
    intensity_names={1:"Slight",2:"Clear",3:"Broad",4:"Laughter-like"}
    annual=p.groupby("year").agg(yes=("smile","sum"),n=("smile","count"),intensity=("intensity","mean"))
    annual=annual.reindex(range(1940,2008));smooth=annual.yes.rolling(5,center=True,min_periods=5).sum()/annual.n.rolling(5,center=True,min_periods=5).sum()
    fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout="constrained")
    annual_plotted=annual.n.ge(10)
    axs[0].scatter(annual.index[annual_plotted],(annual.yes/annual.n)[annual_plotted],s=11,alpha=.45,color="#66717E",label="Annual (n≥10)")
    axs[0].plot(annual.index,smooth,color=COLORS["all"],lw=2,label="Five-year pooled")
    pct_axis(axs[0]);axs[0].set_title("Faces smiling");axs[0].legend(fontsize=8)
    adcurves={}
    for col,label,color in [("any_smile","Ads containing a smile","#F58518"),
                            ("mean_smile","Mean within-ad smile share","#4C78A8"),
                            ("all_smile","All faces smiling","#54A24B")]:
        vals=[c.boot.mean(a.loc[a.decade.eq(d)],col) for d in DECADES];adcurves[col]=[{k:v for k,v in z.items() if k!="samples"} for z in vals]
        line_ci(axs[1],DECADES,vals,color,label)
    decade_axis(axs[1]);axs[1].set_title("Advertisement summaries");axs[1].legend(fontsize=7.5,loc="lower right")
    annual["share"]=annual.yes/annual.n
    annual["window_yes"]=annual.yes.rolling(5,center=True,min_periods=5).sum()
    annual["window_n"]=annual.n.rolling(5,center=True,min_periods=5).sum()
    annual["five_year_share"]=smooth
    c.numeric("historical-smiling","annual",annual.drop(columns="intensity").reset_index())
    c.numeric("historical-smiling","ads",pd.DataFrame([dict(series=col,decade=d,**v)
        for col,vals in adcurves.items() for d,v in zip(DECADES,vals)]))
    c.figure(5,"historical-smiling",fig,"Historical development of smile presence.",
        f"Smile shares use yes/(yes + no): {int(p.smile.sum()):,} of {int(p.smile.count()):,} assessable faces overall; {int(p.smile.isna().sum()):,} are unassessable. "
        "The five-year curve pools smile counts and is shown only for complete centred windows; annual points require at least 10 assessable faces. An ad is known positive if any face smiles and known negative only if every face is assessed as non-smiling. Mean within-ad shares use ads with at least one assessable face. An all-smiling ad requires every recorded face to be assessable and smiling; one assessed non-smile establishes a negative, while remaining indeterminate ads are excluded. 95% CIs calculated as described above.")
    fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout="constrained")
    medium=pd.crosstab(p.decade,p.medium,normalize="index").reindex(DECADES)
    axs[0].plot(DECADES,medium["Photograph"],"o-",color="#4C78A8");pct_axis(axs[0]);decade_axis(axs[0]);axs[0].set_title("Photographic depictions")
    medium_detail=[]
    for m,col in [("Photograph","#4C78A8"),("Other depiction","#F58518")]:
        vals=[c.boot.mean(p.loc[p.decade.eq(d)&p.medium.eq(m)]) for d in DECADES]
        medium_detail.extend(dict(medium=m,decade=d,**{k:v for k,v in val.items() if k!="samples"}) for d,val in zip(DECADES,vals))
        line_ci(axs[1],DECADES,vals,col,m)
    decade_axis(axs[1]);axs[1].grid(axis="y");axs[1].legend(fontsize=9);axs[1].set_title("Smiling within depiction type")
    medium_counts=p.groupby(["decade","medium"]).size().rename("n").reset_index()
    medium_counts["denominator"]=medium_counts.groupby("decade").n.transform("sum")
    medium_counts["share"]=medium_counts.n/medium_counts.denominator
    c.numeric("depiction-trends","representation",medium_counts)
    c.numeric("depiction-trends","smiling",pd.DataFrame(medium_detail))
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
    representation=p.groupby(["decade","gender"]).size().rename("n").reset_index()
    representation["denominator"]=representation.groupby("decade").n.transform("sum")
    representation["share"]=representation.n/representation.denominator
    c.numeric("gender-representation","shares",representation)
    c.figure(7,"gender-representation",fig,"Gender presentation among all recorded faces by decade.","Denominator includes ambiguous and unassessable presentations.")
    rows=[];gaps=[]
    for d in DECADES:
        q=b.loc[b.decade.eq(d)];g=c.boot.gap(q);gaps.append(g)
        f,m=g["feminine"],g["masculine"]
        rows.append([decade_label(d),f["n"],ci_text(f["estimate"],f["low"],f["high"]),m["n"],ci_text(m["estimate"],m["low"],m["high"]),ci_text(g["estimate"],g["low"],g["high"],signed=True)])
    c.table(8,"gender-smile-decades",pd.DataFrame(rows,columns=["Decade","F N","F smiling % [CI]","M N","M smiling % [CI]","F-M pp [CI]"]),
        "Gender-specific smile shares and their difference by decade.","F = feminine; M = masculine. N counts assessable smiles. pp = percentage points. "+CI_NOTE)
    gender_values=[]
    for dec,g in zip(DECADES,gaps):
        for gender in GENDERS:
            val=g[gender]
            gender_values.append(dict(decade=dec,gender=gender,smiling=int(b.loc[b.decade.eq(dec)&b.gender.eq(gender),"smile"].sum()),
                **{k:v for k,v in val.items() if k!="samples"},gap=g["estimate"],gap_low=g["low"],gap_high=g["high"]))
    c.numeric("gender-smile-trends","shares",pd.DataFrame(gender_values))
    fig,axs=plt.subplots(1,2,figsize=(9,3.6),layout="constrained")
    for gender in GENDERS:
        line_ci(axs[0],DECADES,[g[gender] for g in gaps],COLORS[gender],gender.capitalize())
    axs[0].legend();axs[0].set_title("Smile shares");axs[0].grid(axis="y");decade_axis(axs[0])
    line_ci(axs[1],DECADES,gaps,"#263238","Feminine - masculine",False)
    axs[1].set_ylim(bottom=0);axs[1].yaxis.set_major_formatter(PercentFormatter(1,symbol=""));axs[1].set_ylabel("Difference (percentage points)");axs[1].axhline(0,color="#AAB3BC",lw=.8);axs[1].grid(axis="y");decade_axis(axs[1]);axs[1].set_title("Gender smile gap")
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
    age_grid=pd.MultiIndex.from_product([DECADES,GENDERS+["ambiguous_or_androgynous","not_assessable"],AGES+["not_assessable"]],names=["decade","gender","age"])
    age_composition=p.groupby(["decade","gender","age"]).size().reindex(age_grid,fill_value=0).rename("n").reset_index()
    age_composition["denominator"]=age_composition.groupby(["decade","gender"]).n.transform("sum")
    age_composition["share"]=age_composition.n/age_composition.denominator
    c.numeric("age-composition","shares",age_composition)
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
    mat=np.full((6,7),np.nan);age_decades=[]
    for i,age in enumerate(AGES):
        for j,d in enumerate(DECADES):
            q=b.loc[b.age.eq(age)&b.decade.eq(d)];cnt=q.groupby("gender").size()
            for gender in GENDERS:
                z=q.loc[q.gender.eq(gender)]
                age_decades.append(dict(age=age,decade=d,gender=gender,n=len(z),smiling=int(z.smile.sum()),
                    share=z.smile.mean(),plotted=all(cnt.get(g,0)>=30 for g in GENDERS)))
            if all(cnt.get(g,0)>=30 for g in GENDERS):
                mat[i,j]=q.loc[q.gender.eq("feminine"),"smile"].mean()-q.loc[q.gender.eq("masculine"),"smile"].mean()
    cmap=plt.get_cmap("RdBu_r").copy();cmap.set_bad("#E9ECEF")
    im=axs[1].imshow(mat*100,cmap=cmap,vmin=-60,vmax=60,aspect="auto")
    axs[1].set_yticks(range(6),AGE_LABELS);axs[1].set_xticks(range(7),[decade_label(d) for d in DECADES],rotation=40,ha="right");axs[1].set_title("Gender gap by age and decade")
    for (i,j),v in np.ndenumerate(mat*100):
        axs[1].text(j,i,f"{v:+.0f}" if np.isfinite(v) else "—",ha="center",va="center",fontsize=8,color="white" if abs(v)>25 else "#263238")
    fig.colorbar(im,ax=axs[1],label="F-M (percentage points)",shrink=.8)
    c.numeric("age-smile","pooled",pd.DataFrame(age_rows))
    c.numeric("age-smile","decades",pd.DataFrame(age_decades))
    c.figure(11,"age-smile",fig,"Age-specific smiling and gender gaps.","Grey cells have fewer than 30 assessable faces in either gender. The left panel pools all years and omits groups with fewer than 30 faces or 10 issues. "+CI_NOTE)
    fig,axs=plt.subplots(1,3,figsize=(11,4.0),layout="constrained")
    intensity_values=[]
    for ax,gender in zip(axs,["All"]+GENDERS):
        q=p.loc[p.smile.eq(1)] if gender=="All" else p.loc[p.gender.eq(gender)&p.smile.eq(1)]
        cross=pd.crosstab(q.decade,q.intensity,normalize="index").reindex(index=DECADES,columns=[1,2,3,4],fill_value=0)
        for dec in DECADES:
            z=q.loc[q.decade.eq(dec)]
            for level in [1,2,3,4]:
                intensity_values.append(dict(gender=gender,decade=dec,intensity=level,n=int(z.intensity.eq(level).sum()),denominator=len(z),share=cross.loc[dec,level]))
        ax.stackplot(DECADES,*[cross[i] for i in [1,2,3,4]],colors=INT_COLORS,labels=["Slight","Clear","Broad","Laughter-like"])
        pct_axis(ax);decade_axis(ax);ax.set_title(gender.capitalize())
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=8.5,loc="upper center",bbox_to_anchor=(.5,-.02),ncol=4,frameon=False,columnspacing=1.6)
    c.numeric("gender-intensity","shares",pd.DataFrame(intensity_values))
    c.figure(12,"gender-intensity",fig,"Smile intensity overall and by gender among smiling depictions.","Each decade sums to 100% among the corresponding smiling faces. Intensity is an ordered verbal category, not a calibrated physical scale; Chapter 5's category-specific measurement tendencies apply.")
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
    industry_values=[]
    fig,axs=plt.subplots(2,3,figsize=(10.2,6),layout="constrained",sharex=True,sharey=True)
    for ax,industry in zip(axs.flat,top):
        for gender in GENDERS:
            vals=[c.boot.mean(b.loc[b.industry.eq(industry)&b.gender.eq(gender)&b.decade.eq(d)]) for d in DECADES]
            industry_values.extend(dict(industry=industry,gender=gender,decade=dec,plotted=val["n"]>=30,
                **{k:v for k,v in val.items() if k!="samples"}) for dec,val in zip(DECADES,vals))
            # Do not draw unstable or unsupported series points.
            vals=[v if v["n"]>=30 else {**v,"estimate":np.nan,"low":np.nan,"high":np.nan} for v in vals]
            line_ci(ax,DECADES,vals,COLORS[gender],gender.capitalize())
        decade_axis(ax);ax.grid(axis="y");ax.set_title(SHORT_INDUSTRY.get(industry,industry),fontsize=10.5)
    axs[0,0].legend(fontsize=8,loc="lower right")
    c.numeric("industry-trends","shares",pd.DataFrame(industry_values))
    c.figure(14,"industry-trends",fig,"Gender-specific smile trends in the six largest advertiser industries.",
        "Industries selected by ad count, before comparing smile rates. Points require at least 30 assessable faces. 95% CIs calculated as described above.")
    c.results.update(industry=detail)
    exception=b.loc[b.industry.eq("Toiletries & cosmetics")].groupby(["medium","gender"]).smile.agg(["count","mean"]).reset_index()
    exception.to_csv(c.work/"cosmetics-depiction-followup.csv",index=False)
    c.results["cosmetics_followup"]=exception.to_dict("records")
    c.save("industry")


def make_dyads(c):
    p=c.people
    adult=p.age.isin(["young_adult","middle_adult","older_adult"])
    good=p.n.eq(2)&p.f.eq(1)&p.m.eq(1)&adult&p.smile.notna()
    q=p.loc[good].copy()
    ids=q.groupby("ad").size();q=q.loc[q.ad.isin(ids.index[ids.eq(2)])]
    f=q.loc[q.gender.eq("feminine")].set_index("ad")
    m=q.loc[q.gender.eq("masculine")].set_index("ad")
    assert f.index.equals(m.index)
    d=f[["issue","year","decade","period"]].copy()
    d["f_smile"]=f.smile;d["m_smile"]=m.smile
    d["gap"]=d.f_smile-d.m_smile
    d["state"]=np.select([d.f_smile.eq(0)&d.m_smile.eq(0),d.f_smile.eq(1)&d.m_smile.eq(0),d.f_smile.eq(0)&d.m_smile.eq(1)],
        ["Neither","Feminine only","Masculine only"],default="Both")
    return d.reset_index()


def relational(c):
    p=c.people;d=make_dyads(c)
    # Separate multi-face ads from the mechanically homogeneous single-face case.
    multi=c.ads.loc[c.ads.n_faces.ge(2)&c.ads.smile_n.eq(c.ads.n_faces)].copy()
    multi["smile_state"]=np.select(
        [multi.smile_yes.eq(0),multi.smile_yes.eq(multi.n_faces)],
        ["None smiling","All smiling"],default="Mixed")
    state_order=["None smiling","Mixed","All smiling"]
    state_colors=["#AAB3BC","#F2C14E","#54A24B"]
    composition=(pd.crosstab(multi.decade,multi.smile_state)
        .reindex(index=DECADES,columns=state_order,fill_value=0))
    composition_share=composition.div(composition.sum(axis=1),axis=0)
    fig,axs=plt.subplots(1,2,figsize=(9,3.6),layout="constrained")
    bottom=np.zeros(len(DECADES))
    for state,color in zip(state_order,state_colors):
        values=composition_share[state].to_numpy()
        axs[0].bar(DECADES,values,bottom=bottom,width=7.2,color=color,label=state)
        bottom+=values
    pct_axis(axs[0]);decade_axis(axs[0]);axs[0].set_title("Smile configuration")
    axs[0].legend(fontsize=7.5,loc="upper center",bbox_to_anchor=(.5,-.30),ncol=3,frameon=False,columnspacing=1.2)
    concentration=[];observed=[];expected=[]
    for dec in DECADES:
        q=multi.loc[multi.decade.eq(dec)].copy()
        q["all_smiling"]=q.smile_yes.eq(q.n_faces).astype(float)
        smiling=q.loc[q.smile_yes.gt(0)]
        value=c.boot.mean(smiling,"all_smiling")
        face_share=q.smile_yes.sum()/q.n_faces.sum()
        expected_all=np.sum(face_share**q.n_faces)
        expected_any=np.sum(1-(1-face_share)**q.n_faces)
        benchmark=expected_all/expected_any
        observed.append(value);expected.append(benchmark)
        concentration.append(dict(decade=dec,ads=len(q),smile_containing_ads=len(smiling),
            face_smile_share=face_share,observed=value["estimate"],low=value["low"],high=value["high"],
            expected_independence=benchmark,excess=value["estimate"]-benchmark))
    line_ci(axs[1],DECADES,observed,"#54A24B","Observed in corpus")
    axs[1].plot(DECADES,expected,"o--",color="#66717E",label="Expected under independence")
    decade_axis(axs[1]);axs[1].set_ylim(0,.7);axs[1].set_title("All smiling among ads with a smile")
    axs[1].legend(fontsize=8,loc="lower right")
    composition_numeric=(composition.rename_axis("decade").reset_index()
        .melt(id_vars="decade",var_name="state",value_name="n"))
    composition_numeric["denominator"]=composition_numeric.decade.map(composition.sum(axis=1))
    composition_numeric["share"]=composition_numeric.n/composition_numeric.denominator
    c.numeric("multi-face-smile-concentration","composition",composition_numeric)
    c.numeric("multi-face-smile-concentration","benchmark",pd.DataFrame(concentration))
    c.figure(16,"multi-face-smile-concentration",fig,
        "Within-ad concentration of smiling in fully assessed multi-face advertisements.",
        "Bars classify advertisements with at least two recorded faces and assessable smile labels for every face. The observed series is all-smiling/(all-smiling + mixed). The independence benchmark retains each decade's face-level smile share and observed number of faces per ad while treating faces as independent. 95% CIs calculated as described above.")
    states=["Neither","Feminine only","Masculine only","Both"]
    rows=[]
    for label,q in [("All years",d)]+[(t,d.loc[d.period.eq(t)]) for t in ["1940-59","1960-79","1980-2007"]]:
        g=c.boot.mean(q,"gap")
        rows.append([label,len(q)]+[q.state.eq(s).mean() for s in states]+[ci_text(g["estimate"],g["low"],g["high"],signed=True)])
    c.table(17,"dyad-states",pd.DataFrame(rows,columns=["Period","Ads","Neither","F only","M only","Both","F-M pp [CI]"]),
        "Joint smile configurations in mixed-gender adult two-face advertisements.",
        "Exactly two recorded faces, one feminine and one masculine; both have an adult age category and assessable smiles. Each ad contributes once. F-M equals the feminine-only share minus the masculine-only share. "+CI_NOTE,
        {s:".1%" for s in ["Neither","F only","M only","Both"]})
    # Prominence ownership uses fractional credit for exact area ties.
    all_geometry_valid=p.groupby("ad").geometry_valid.transform("all")
    mixed=p.loc[p.context.eq("Mixed")&all_geometry_valid].copy()
    mixed["is_largest"]=mixed.area.eq(mixed.groupby("ad").area.transform("max"))
    mixed["largest_credit"]=mixed.is_largest/mixed.groupby("ad").is_largest.transform("sum")
    mixed["female_credit"]=mixed.largest_credit*mixed.gender.eq("feminine")
    own=mixed.groupby("ad").agg(issue=("issue","first"),year=("year","first"),period=("period","first"),observed=("female_credit","sum"),f=("f","first"),n=("n","first"))
    own["expected"]=own.f/own.n;own["excess"]=own.observed-own.expected
    prominence=[]
    for label,oq in [("All years",own)]+[(t,own.loc[own.period.eq(t)]) for t in ["1940-59","1960-79","1980-2007"]]:
        ex=c.boot.mean(oq,"excess")
        prominence.append([label,len(oq),oq.observed.mean(),oq.expected.mean(),ci_text(ex["estimate"],ex["low"],ex["high"],signed=True)])
    c.table(21,"spatial-prominence",pd.DataFrame(prominence,columns=["Period","Mixed ads","F largest","Expected","Difference pp [CI]"]),
        "Largest-face ownership in mixed-gender advertisements.",
        "Includes mixed-gender advertisements with valid face boxes for every recorded face. F largest is the mean share of largest-face credit assigned to feminine-presenting faces; exact ties split credit. Expected is each advertisement's feminine share. Difference is observed minus expected. "+CI_NOTE,
        {"F largest":".1%","Expected":".1%"})
    c.results.update(dyads=rows,prominence=prominence)
    c.save("relational")


def finalize(c, integrate=False, update_manifest=False):
    """Validate all artifacts; optionally append their structure and update bridge."""
    entries=json.loads((c.work/"artifact-index.json").read_text())
    byorder={e["order"]:e for e in entries}
    assert set(CHAPTER_ORDERS).issubset(byorder), "Missing retained chapter artifact"
    entries=[byorder[i] for i in CHAPTER_ORDERS]
    assert len(entries)==len(CHAPTER_ORDERS) and len({e["name"] for e in entries})==len(entries)
    (c.work/"artifact-index.json").write_text(json.dumps(entries,indent=2)+"\n",encoding="utf-8")
    from chapter06_support import numerical_appendix
    numerical_appendix(c,entries)
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
        end_marker="<!-- END CH06 GENERATED ANALYSIS -->"
        content=chapter.read_bytes()
        if marker.encode("utf-8") not in content or end_marker.encode("utf-8") not in content:
            raise RuntimeError("Chapter integration markers are missing; preserve the current manuscript and integrate outputs manually.")
        # Preserve the existing notes verbatim, including their headings.
        prefix=content.split(marker.encode("utf-8"))[0]
        backup=c.work/"chapter06-existing-notes.qmd"
        if not backup.exists():
            backup.write_bytes(prefix)
        blocks=[marker]
        sections=[(2,"Gender and age in the depicted population"),(5,"Historical development of smiling"),
            (12,"Smile intensity"),(6,"Industry and depiction context"),
            (28,"Comparisons with existing research"),
            (16,"Within-ad smiling and visual prominence")]
        sectionmap=dict(sections)
        for e in entries:
            if e["order"] in sectionmap:
                blocks.extend(["", "\\clearpage", "", "## "+sectionmap[e["order"]], ""])
            elif e["order"] == 7:
                blocks.extend(["\\clearpage", ""])
            if e["order"] in [28, 29, 30]:
                if e["order"] != 28:
                    blocks.extend(["\\clearpage", ""])
                blocks.extend(["### " + {28: "Jofre and Cole: gendered smiling and historical change",
                    29: "Financial advertising: representation and age",
                    30: "Expansive smiles: frequency and intensity"}[e["order"]], ""])
            if e["kind"]=="table":
                blocks.extend(["{{< include tables/generated/"+e["file"]+" >}}", ""])
            else:
                caption=e["caption"]+(" "+e["note"] if e["note"] else "")
                appendix_anchor = "tbl-ch06-appendix-depiction" if e["name"] == "ch06-depiction-trends" else "sec-analysis-numerical-data"
                caption += f" Exact values: [numerical companion appendix](appendix-analysis-data.qmd#{appendix_anchor})."
                # Div syntax permits a real cross-file link in the caption. A
                # Markdown link cannot be nested safely inside compact image-alt
                # caption syntax.
                blocks.extend(["::: {#fig-"+e["name"]+' fig-pos="H"}',
                    f"![](figures/generated/fig-{e['file']})"+"{width=100%}",
                    "", caption, "", ":::", ""])
        blocks.append("<!-- END CH06 GENERATED ANALYSIS -->")
        padding=b"" if prefix.endswith(b"\n\n") or prefix.endswith(b"\r\n\r\n") else b"\n\n"
        chapter.write_bytes(prefix+padding+("\n".join(blocks)+"\n").encode("utf-8"))
    provenance={"input":c.results["input"],"bootstrap":{"replicates":2000,"seed":6062026,"cluster":"publication issue","strata":"publication year"},
        "artifacts":entries,"numerical_appendix":"appendix-analysis-data.qmd","excluded_fields":["face orientation","gaze target"]}
    (c.work/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    print(f"Verified {len(entries)} artifacts: {sum(e['kind']=='table' for e in entries)} tables and {sum(e['kind']=='figure' for e in entries)} figures.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path)
    parser.add_argument("--thesis",type=Path)
    parser.add_argument("--work",type=Path)
    parser.add_argument("--sections",nargs="+",choices=["foundation","gender","industry","literature","relational","support"],default=["foundation","gender","industry","literature","relational","support"])
    parser.add_argument("--integrate",action="store_true")
    parser.add_argument("--manifest",action="store_true")
    args=parser.parse_args()
    c=Analysis(args.input,args.thesis,args.work)
    from chapter06_support import audit_bootstrap
    from chapter06_literature import literature_comparisons
    def support(context):
        audit_bootstrap(context)
    functions={"foundation":foundation,"gender":gender_age,"industry":industry_context,"literature":literature_comparisons,"relational":relational,"support":support}
    for section in args.sections:
        print(f"Running {section}",flush=True);functions[section](c)
    finalize(c,args.integrate,args.manifest)


if __name__=="__main__":
    main()
