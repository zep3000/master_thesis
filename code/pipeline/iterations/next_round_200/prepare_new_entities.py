"""Build a gold-free manifest of verifier-accepted novel individual faces."""

from __future__ import annotations

import argparse, importlib.util, json, re, sys
from pathlib import Path
from typing import Any

ROOT=Path(r"."); HERE=ROOT/"qwen_iteration"/"next_round_200"; OLD=ROOT/"qwen_iteration"/"first100_pipeline_matrix"


def rows(path:Path)->list[dict[str,Any]]: return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
def iou(a,b):
    x1,y1,x2,y2=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1); union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter; return inter/union if union else 0
def ad_to_page(box,ad):
    w=ad[2]-ad[0]; h=ad[3]-ad[1]; return [round(ad[0]+box[0]*w/1000),round(ad[1]+box[1]*h/1000),round(ad[0]+box[2]*w/1000),round(ad[1]+box[3]*h/1000)]
def safe(s):return re.sub(r"[^A-Za-z0-9._-]+","_",s)


def main():
    p=argparse.ArgumentParser();p.add_argument("--cohort",choices=["difficult100","stratified100"],required=True);p.add_argument("--baseline-run",required=True);p.add_argument("--run-name",required=True);args=p.parse_args()
    basepath=(OLD/"output"/args.baseline_run/"p2"/"completed.jsonl") if args.cohort=="difficult100" else(HERE/"output"/args.baseline_run/"p2"/"completed.jsonl"); baseline={r["image_id"]:r for r in rows(basepath)}; proposal=json.loads((HERE/"output"/"detector_proposals"/f"{args.cohort}.json").read_text(encoding="utf-8")); props={(p["image_id"],a["advertisement_id"]):a for p in proposal["pages"] for a in p["advertisements"]}; verified={r["task_key"]:r for r in rows(HERE/"output"/args.run_name/args.cohort/"structure"/"results.jsonl") if r.get("ok")}; image_dir=(ROOT/"code"/"test_collection_200_difficult_joined_pages") if args.cohort=="difficult100" else(ROOT/"master_thesis"/"data"/"images"/"full_pages_1940_2007_joined")
    sys.path.insert(0,str(OLD));spec=importlib.util.spec_from_file_location("frozen",OLD/"run.py");assert spec and spec.loader;frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    outdir=HERE/"output"/args.run_name/args.cohort/"new_entities"; entities=[]
    for task,record in verified.items():
        result=record["model_annotation"]
        if result["group_route_required"]: continue
        image_id,ad_id=task.split("::",1); page=baseline[image_id]; ad=next(a for a in page["annotation"]["advertisements"] if a["advertisement_id"]==ad_id); proposal_ad=props[(image_id,ad_id)]; existing=[p["face_bbox_1000"] for p in ad.get("people") or []]; candidates={c["candidate_id"]:c for c in proposal_ad["detector_candidates"]}; boxes=[]
        for cid in result["accepted_candidate_ids"]:
            if cid in candidates: boxes.append((cid,candidates[cid]["bbox_1000"],"accepted_detector"))
        for i,item in enumerate(result["missed_faces"],1): boxes.append((f"m{i}",ad_to_page(item["bbox_1000"],ad["bbox_1000"]),"verifier_discovered"))
        kept=[]
        for cid,box,source in boxes:
            if any(iou(box,b)>=.10 for b in existing) or any(iou(box,b)>=.25 for _,b,_ in kept):continue
            kept.append((cid,box,source))
        for cid,box,source in kept:
            entity_id=f"verified_{safe(cid)}"; entity_task=f"{image_id}::{ad_id}::{entity_id}"; composite=outdir/"crops"/f"{safe(entity_task)}.jpg"; frozen.person_composite(image_dir/page["filename"],box,ad["bbox_1000"],composite,entity_task); entities.append({"task_key":entity_task,"image_id":image_id,"filename":page["filename"],"advertisement_id":ad_id,"person_id":entity_id,"face_bbox_1000":box,"ad_bbox_1000":ad["bbox_1000"],"ad_depiction_type":ad.get("depiction_type"),"source":source,"composite_path":str(composite)})
    out=outdir/"manifest.json";out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({"schema_version":"verified_new_entities_v1","cohort":args.cohort,"gold_fields_included":False,"entities":entities},indent=2)+"\n",encoding="utf-8");print(json.dumps({"path":str(out),"entities":len(entities),"pages":len({e['image_id'] for e in entities})}))


if __name__=="__main__":main()
