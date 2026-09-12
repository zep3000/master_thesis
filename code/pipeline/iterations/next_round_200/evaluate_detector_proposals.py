"""Evaluation-only validation of archived detector coordinate usefulness."""

from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path

ROOT=Path(r"."); HERE=ROOT/"qwen_iteration"/"next_round_200"; OLD=ROOT/"qwen_iteration"/"first100_pipeline_matrix"
sys.path.insert(0,str(OLD)); spec=importlib.util.spec_from_file_location("ev",OLD/"evaluate.py"); assert spec and spec.loader; ev=importlib.util.module_from_spec(spec); sys.modules[spec.name]=ev; spec.loader.exec_module(ev)


def gold(cohort:str):
    path=ROOT/"annotation_results"/("test_collection_200_difficult_joined_v1_2026-08-02.json" if cohort=="difficult100" else "economist_decade_face_count_stratified_200_seed20260812_min2_52123740_2026-08-12.json"); data=json.loads(path.read_text(encoding="utf-8"))
    rows=data["annotations"]
    if cohort=="difficult100": rows=[r for r in rows if r.get("assignment_code")=="79201188"]
    else: rows=sorted(rows,key=lambda r:int(r["sort_order"]))[:100]
    return {r["image_id"]:r["payload"] for r in rows}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--cohort",required=True); args=p.parse_args(); proposals=json.loads((HERE/"output"/"detector_proposals"/f"{args.cohort}.json").read_text(encoding="utf-8")); g=gold(args.cohort)
    ht=pt=matched=0; matched_by_group=0
    for page in proposals["pages"]:
        ann=g.get(page["image_id"],{}); human=[x for ad in ann.get("advertisements") or [] for x in ev.person_items(page["image_id"],ad)]; pred=[ev.Item(page["image_id"],None,f"d{i}",c,[v/1000 for v in c["bbox_1000"]]) for i,c in enumerate(page["detections"],1)]
        pairs=ev.pair_items(human,pred,"people","strict"); ht+=len(human); pt+=len(pred); matched+=len(pairs)
    print(json.dumps({"cohort":args.cohort,"gold_people":ht,"detector_candidates":pt,"strict_matched":matched,"precision":matched/pt if pt else 0,"recall":matched/ht if ht else 0,"selection_bias_warning":proposals["selection_bias_warning"]},indent=2))


if __name__=="__main__": main()
