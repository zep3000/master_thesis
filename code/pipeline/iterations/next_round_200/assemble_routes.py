"""Gold-free assembly of specialist outputs into comparable annotation routes."""

from __future__ import annotations

import argparse, copy, json
from pathlib import Path
from typing import Any

from specialist_prompts import normalize_group

ROOT=Path(r"."); HERE=ROOT/"qwen_iteration"/"next_round_200"; OLD=ROOT/"qwen_iteration"/"first100_pipeline_matrix"
PERSON_FIELDS=["face_expression_legibility","gaze_target","smile_present","smile_intensity"]
GROUP_FIELDS=["group_type","age_composition","gender_presentation_composition","expression_legibility_distribution","dominant_gaze","smile_prevalence","dominant_smile_intensity"]


def rows(path:Path)->list[dict[str,Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--cohort",choices=["difficult100","stratified100"],required=True); p.add_argument("--baseline-run",required=True); p.add_argument("--specialist-run",required=True); p.add_argument("--limit",type=int,default=100); p.add_argument("--routes",nargs="+",default=["r0","f0","f1","f2","g1"]); args=p.parse_args()
    basepath=(OLD/"output"/args.baseline_run/"p2"/"completed.jsonl") if args.cohort=="difficult100" else (HERE/"output"/args.baseline_run/"p2"/"completed.jsonl")
    base=[r for r in rows(basepath) if int(r.get("manifest_index",999))<args.limit]
    specialist_path=HERE/"output"/args.specialist_run/args.cohort/"specialists.jsonl"
    specs={(r["strategy"],r["task_key"]):r for r in rows(specialist_path) if r.get("ok")}
    mapping={"f0":"f0_ordinal_direct","f1":"f1_hierarchical_gaze","f2":"f2_hierarchical_gaze_smile"}
    outdir=HERE/"output"/args.specialist_run/args.cohort/"assembled"; outdir.mkdir(parents=True,exist_ok=True)
    for route in args.routes:
        out=[]; replaced_people=replaced_groups=0
        for page in base:
            item=copy.deepcopy(page); item["route"]=route; ann=item.get("annotation") or {}
            for ad in ann.get("advertisements") or []:
                for person in ad.get("people") or []:
                    if route in mapping:
                        key=f"{page['image_id']}::{person['person_id']}"; spec=specs.get((mapping[route],key))
                        if spec:
                            for field in PERSON_FIELDS: person[field]=spec["model_annotation"].get(field)
                            person["specialist_confidence"]=spec["model_annotation"].get("confidence"); replaced_people+=1
                for group in ad.get("groups") or []:
                    if route=="g1":
                        key=f"{page['image_id']}::{group['group_id']}"; spec=specs.get(("g1_aggregate",key))
                        if spec:
                            normalized=normalize_group(spec.get("model_annotation_raw") or {},key)
                            for field in GROUP_FIELDS: group[field]=normalized.get(field)
                            group["specialist_confidence"]=normalized.get("confidence"); replaced_groups+=1
            out.append(item)
        path=outdir/f"{route}.jsonl"; path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in out),encoding="utf-8")
        print(json.dumps({"route":route,"pages":len(out),"people_replaced":replaced_people,"groups_replaced":replaced_groups,"path":str(path)}))


if __name__=="__main__": main()
