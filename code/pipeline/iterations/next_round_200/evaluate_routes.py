"""Evaluation-only strict/lenient comparison of assembled routes."""

from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path
from typing import Any

ROOT=Path(r"."); HERE=ROOT/"qwen_iteration"/"next_round_200"; OLD=ROOT/"qwen_iteration"/"first100_pipeline_matrix"
OLD_GOLD=ROOT/"annotation_results"/"test_collection_200_difficult_joined_v1_2026-08-02.json"
NEW_GOLD=ROOT/"annotation_results"/"economist_decade_face_count_stratified_200_seed20260812_min2_52123740_2026-08-12.json"


def read_jsonl(path:Path)->list[dict[str,Any]]: return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_gold(cohort:str)->dict[str,dict[str,Any]]:
    if cohort=="difficult100":
        data=json.loads(OLD_GOLD.read_text(encoding="utf-8")); return {r["image_id"]:r["payload"] for r in data["annotations"] if r.get("assignment_code")=="79201188" and isinstance(r.get("payload"),dict)}
    data=json.loads(NEW_GOLD.read_text(encoding="utf-8")); selected=sorted(data["annotations"],key=lambda r:int(r["sort_order"]))[:100]; return {r["image_id"]:r["payload"] for r in selected if isinstance(r.get("payload"),dict)}


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--cohort",choices=["difficult100","stratified100"],required=True); p.add_argument("--run-name",required=True); p.add_argument("--limit",type=int,default=100); p.add_argument("--routes",nargs="+",required=True); args=p.parse_args()
    sys.path.insert(0,str(OLD)); spec=importlib.util.spec_from_file_location("frozen_eval",OLD/"evaluate.py"); assert spec and spec.loader; ev=importlib.util.module_from_spec(spec); sys.modules[spec.name]=ev; spec.loader.exec_module(ev)
    manifest=json.loads((HERE/"data"/f"manifest_{args.cohort}.json").read_text(encoding="utf-8"))["images"][:args.limit]; ids=[r["image_id"] for r in manifest]; gold=load_gold(args.cohort)
    result={"cohort":args.cohort,"run_name":args.run_name,"pages":len(ids),"routes":{}}
    for route in args.routes:
        path=HERE/"output"/args.run_name/args.cohort/"assembled"/f"{route}.jsonl"; pred={r["image_id"]:r["annotation"] for r in read_jsonl(path) if r.get("ok") and isinstance(r.get("annotation"),dict)}
        result["routes"][route]=ev.evaluate_scope(gold,pred,ids)
    outdir=HERE/"evaluation"/args.run_name; outdir.mkdir(parents=True,exist_ok=True); out=outdir/f"{args.cohort}_{args.limit}.json"; out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    for route,item in result["routes"].items():
        s=item["match_modes"]["strict"]; l=item["match_modes"]["lenient"]
        print(json.dumps({"route":route,"eval_pages":item["evaluated_pages"],"strict":{e:round(100*(s["detection"][e]["f1"] or 0),1) for e in ["ads","people","groups"]},"lenient":{e:round(100*(l["detection"][e]["f1"] or 0),1) for e in ["ads","people","groups"]},"leg_exact":s["fields"]["people"]["face_expression_legibility"]["exact_accuracy"],"gaze_exact":s["fields"]["people"]["gaze_target"]["exact_accuracy"],"smile_exact":s["fields"]["people"]["smile_present"]["exact_accuracy"],"group_leg_exact":s["fields"]["groups"]["expression_legibility_distribution"]["exact_accuracy"]}))
    print(out)


if __name__=="__main__": main()
