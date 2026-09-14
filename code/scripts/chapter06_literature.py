"""Explicit, measurement-aware comparisons with the studies cited in Chapter 3.

Published benchmarks are transcribed from the cited papers, never digitised from
their curves. Corpus estimates and intervals are computed from the same retained
records and common issue/year bootstrap as the other chapter sections.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from chapter06_analysis import AGES, CI_NOTE, DECADES, GENDERS, ci_text


def contrast(a, b):
    """Difference using paired common bootstrap draws, not separate CI endpoints."""
    samples = a["samples"] - b["samples"]
    low, high = np.nanquantile(samples, [.025, .975]) if min(a["issues"], b["issues"]) >= 10 else (np.nan, np.nan)
    return dict(estimate=a["estimate"] - b["estimate"], low=float(low), high=float(high), samples=samples)


def compact(r, scale=100, signed=False):
    return ci_text(r["estimate"], r["low"], r["high"], scale=scale, signed=signed)


def numeric_row(statistic, r, **metadata):
    return dict(statistic=statistic, **{k: v for k, v in r.items()
                if k in ["n", "issues", "estimate", "low", "high"]}, **metadata)


def jofre_comparison(c):
    pooled = c.boot.gap(c.binary)
    f, m = pooled["feminine"], pooled["masculine"]
    samples = np.divide(f["samples"], m["samples"], out=np.full(len(f["samples"]), np.nan), where=m["samples"] > 0)
    low, high = np.nanquantile(samples, [.025, .975])
    ratio = dict(estimate=f["estimate"]/m["estimate"], low=float(low), high=float(high))
    decade_gaps = [c.boot.gap(c.binary.loc[c.binary.decade.eq(d)]) for d in DECADES]
    supported = sum(g["low"] > 0 for g in decade_gaps)
    early = c.boot.mean(c.people.loc[c.people.decade.eq(1940)])
    late = c.boot.mean(c.people.loc[c.people.decade.eq(2000)])
    change = contrast(late, early)
    # Symmetric adjacent five-year windows localise the paper's 'around 1970'
    # observation. They are fixed in advance, not selected for the largest effect.
    women = c.binary.loc[c.binary.gender.eq("feminine")]
    before = c.boot.mean(women.loc[women.year.between(1965, 1969)])
    after = c.boot.mean(women.loc[women.year.between(1970, 1974)])
    around1970 = contrast(after, before)
    direction = "Increase" if around1970["low"] > 0 else "Decrease" if around1970["high"] < 0 else "Direction uncertain"
    rows = [
        ["Women smile more across eras", f"F {f['estimate']:.1%}; M {m['estimate']:.1%}; gap {compact(pooled, signed=True)} pp", f"Positive gap in {supported}/{len(DECADES)} decades, including CI limits"],
        ["Average female/male smile ratio: about 1.6", f"{ratio['estimate']:.2f} [{ratio['low']:.2f}, {ratio['high']:.2f}]", "Numerical convergence; different context mix"],
        ["Relatively stable overall smiling after c. 1940", f"1940s {early['estimate']:.1%}; 2000-07 {late['estimate']:.1%}; change {compact(change, signed=True)} pp", "Endpoint comparison; no test of constant trend"],
        ["Dip in women's smiling around 1970", f"1965-69 {before['estimate']:.1%}; 1970-74 {after['estimate']:.1%}; change {compact(around1970, signed=True)} pp", direction + "; advertising-only comparison"],
    ]
    c.table(28, "jofre-comparison", pd.DataFrame(rows, columns=["Published pattern", "This corpus [95% CI]", "Comparison"]),
        "Confirmation and contrast of the smile patterns reported by Jofre and Cole.",
        "Reference: @jofre2024CulturalAnalytic, pp. 5, 10-12: human-coded Time faces, 1923-2014, including advertising and news. "
        "Our estimates use retained Economist advertising faces, 1940-2007. F/M denotes feminine/masculine presentation. "
        "Their ratio and around-1970 dip pool contexts; neither is an advertising-only numerical benchmark. "
        "All-gender assessable faces enter the overall endpoint comparison; F/M faces enter gender comparisons. "
        "The ratio is a probability ratio, not an odds ratio. No published benchmark uncertainty is propagated, and no cross-study significance test is implied. " + CI_NOTE)
    records = [numeric_row("smile_share", f, gender="feminine", start=1940, end=2007),
        numeric_row("smile_share", m, gender="masculine", start=1940, end=2007),
        numeric_row("gender_gap", pooled, start=1940, end=2007),
        numeric_row("female_male_probability_ratio", ratio, n=len(c.binary), published=1.6),
        numeric_row("overall_share", early, start=1940, end=1949),
        numeric_row("overall_share", late, start=2000, end=2007),
        numeric_row("overall_endpoint_change", change, n_before=early["n"], n_after=late["n"]),
        numeric_row("feminine_share", before, start=1965, end=1969),
        numeric_row("feminine_share", after, start=1970, end=1974),
        numeric_row("feminine_around1970_change", around1970, n_before=before["n"], n_after=after["n"])]
    records += [numeric_row("decade_gender_gap", g, decade=d,
        n_f=g["feminine"]["n"], n_m=g["masculine"]["n"]) for d, g in zip(DECADES, decade_gaps)]
    c.numeric("jofre-comparison", "estimates", pd.DataFrame(records))
    c.results["jofre_comparison"] = records


def financial_comparison(c):
    financial = c.people.loc[c.people.industry.eq("Financial services") & c.people.year.between(1949, 2007)].copy()
    # Rank all faces before excluding uncertain genders; never substitute the
    # largest binary-gender face when the actual maximum is unassessable.
    maximum = financial.groupby("ad").area.transform("max")
    largest = financial.loc[financial.geometry_valid & financial.area.eq(maximum)].copy()
    ties = largest.ad.duplicated(keep=False)
    single_max = largest.loc[~ties]
    focal = single_max.loc[single_max.gender.isin(GENDERS)].copy()
    assert focal.ad.is_unique
    solo = financial.loc[financial.n.eq(1) & financial.gender.isin(GENDERS)].copy()
    records, rows = [], []

    def add(label, data, value, published, pub_start=1949, pub_end=2023, selection="unique_largest_face"):
        r = c.boot.mean(data, value)
        rows.append([label, f"{published:.1%}" if published is not None else "—", r["n"], compact(r)])
        records.append(numeric_row(label, r, numerator=int(data[value].sum()), value=value,
            published=published, published_start=pub_start, published_end=pub_end,
            start=int(data.year.min()), end=int(data.year.max()), selection=selection,
            gender=data.gender.iloc[0] if data.gender.nunique() == 1 else "F/M"))
        return r

    add("Focal F: 1949-2007", focal, "feminine", .16)
    pre = add("Focal F: 1949-74", focal.loc[focal.year.le(1974)], "feminine", .142, 1949, 1974)
    post = add("Focal F: 2000-07", focal.loc[focal.year.ge(2000)], "feminine", .282, 2000, 2014)
    add("F in single-face ads: 1949-2007", solo, "feminine", None, selection="single_face_ad")
    age = focal.loc[focal.age.isin(AGES)].copy()
    age["young"] = age.age.eq("young_adult").astype(float)
    age["middle"] = age.age.eq("middle_adult").astype(float)
    for gender, label, young, middle in [("feminine", "F", .5246, .3531), ("masculine", "M", .1175, .6686)]:
        q = age.loc[age.gender.eq(gender)]
        add(f"Focal {label}: young adults", q, "young", young)
        add(f"Focal {label}: middle adults", q, "middle", middle)
    change = contrast(post, pre)
    records.append(numeric_row("focal_feminine_share_change", change, n_before=pre["n"], n_after=post["n"],
        published=.282-.142, start_before=1949, end_before=1974, start_after=2000, end_after=2007))
    agegap = c.boot.gap(age, "young")
    records.append(numeric_row("young_adult_gender_gap", agegap, n_f=agegap["feminine"]["n"], n_m=agegap["masculine"]["n"], published=.5246-.1175))
    c.table(29, "financial-literature", pd.DataFrame(rows, columns=["Comparison / our period", "Published share", "Our N", "Our share % [95% CI]"]),
        "Financial-advertising representation and age compared with the 2024 working paper.",
        "Reference: @unda2024GenderStereotypes, March 2024, p. 14 and Tables 5 and 7. "
        "Both studies draw on the Economist archive, so this is an overlapping-corpus comparison, not independent replication. "
        "Published overall shares cover 1949-2023; the post-2000 column is labelled 2000-2014 in Table 7. "
        "Our financial category and retained faces cover 1949-2007; the pre-1975 period matches exactly. "
        "Focal = the unique largest face box per ad, a proxy for the paper's central person, which also uses role and body prominence. "
        f"Of {financial.ad.nunique():,} retained financial ads in this period, {int(largest.loc[ties].ad.nunique()):,} have tied largest boxes and {int((~single_max.gender.isin(GENDERS)).sum()):,} have an uncertain-gender maximum; both are excluded. "
        "The single-face row checks an alternative subset and has no published matching benchmark. "
        "Age shares use assessable recorded ages within focal gender; young means young adult, with younger categories retained in the denominator. "
        "Chapter 5's younger-classification tendency applies. The 16% reference is the complement of the rounded 84% male-central share. "
        "No cross-study difference test is performed. " + CI_NOTE,
        {"Our N": ",.0f"})
    c.numeric("financial-literature", "estimates", pd.DataFrame(records))
    # Full ad-level focal assignments stay private for audit, not in numerical CSVs.
    focal.to_pickle(c.work / "financial-focal-private.pkl")
    c.results["financial_comparison"] = dict(estimates=records, retained_ads=int(financial.ad.nunique()),
        tied_ads=int(largest.loc[ties].ad.nunique()), uncertain_maximum=int((~single_max.gender.isin(GENDERS)).sum()),
        focal_ads=len(focal), solo_ads=len(solo), age_unassessable=int((~focal.age.isin(AGES)).sum()))


def expansive_comparison(c):
    p = c.binary.copy()
    # Intensity is absent for non-smilers; set a negative expansive indicator only
    # where smile absence is assessed, never for an unassessable expression.
    p["expansive"] = (p.smile.eq(1) & p.intensity.ge(3)).astype(float)
    smiling = p.loc[p.smile.eq(1)].copy()
    subsets = [("All assessed faces", p), ("All smiling faces", smiling)]
    subsets += [("Smilers: " + label, smiling.loc[smiling.period.eq(label)]) for label in ["1940-59", "1960-79", "1980-2007"]]
    subsets += [("Smilers: photographs", smiling.loc[smiling.medium.eq("Photograph")]),
        ("Smilers: young adults", smiling.loc[smiling.age.eq("young_adult")]),
        ("Smilers: face area 1%+", smiling.loc[smiling.area.ge(.01)])]
    rows, records = [], []
    for label, sample in subsets:
        g = c.boot.gap(sample, "expansive")
        f, m = g["feminine"], g["masculine"]
        rows.append([label, f["n"], f["estimate"], m["n"], m["estimate"], compact(g, signed=True)])
        for gender, result in [("feminine", f), ("masculine", m)]:
            records.append(numeric_row("broad_or_laughter_share", result, subset=label, gender=gender,
                numerator=int(sample.loc[sample.gender.eq(gender), "expansive"].sum()), threshold=3))
        records.append(numeric_row("expansive_gender_gap", g, subset=label, threshold=3))
    c.table(30, "expansive-smiles", pd.DataFrame(rows, columns=["Denominator / subset", "F N", "F share", "M N", "M share", "F−M pp [95% CI]"]),
        "The expansive-smile expectation: broad or laughter-like smiling by gender.",
        "Expectation: women smile more expansively [@goffman1979GenderAdvertisements, pp. 48, 69]; "
        "@doring2006ImagesMen, pp. 177-178, reports a female advantage for expansive smile/laughter in mobile-phone advertising. "
        "Our proxy is recorded intensity 3 (broad) or 4 (laughter-like). The all-assessed row combines smile frequency and intensity; "
        "the remaining rows condition on smiling and address intensity specifically. The restricted rows check depiction type, a shared age category, and framing separately; they are not joint adjustment. "
        "Counts are the denominators within each gender. Agreement would concern the visible expression pattern, not establish subordination or licensed withdrawal. "
        "Intensity and age retain the measurement tendencies observed in Chapter 5. " + CI_NOTE,
        {"F N": ",.0f", "M N": ",.0f", "F share": ".1%", "M share": ".1%"})
    c.numeric("expansive-smiles", "estimates", pd.DataFrame(records))
    c.results["expansive_comparison"] = records


def literature_comparisons(c):
    jofre_comparison(c)
    financial_comparison(c)
    expansive_comparison(c)
    c.save("literature")
