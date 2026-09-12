"""Create gold-free page/ad face proposals from the archived detector CSV.

The source detector is the one used to construct the stratified sample, so its
performance on stratified100 is reported with selection-bias warnings. It was
not used to select difficult100. Coordinates and P2 outputs are inference-side;
no annotation export or evaluation artifact is read here.
"""

from __future__ import annotations

import argparse, csv, json, re
from pathlib import Path
from typing import Any

ROOT=Path(r"."); HERE=ROOT/"qwen_iteration"/"next_round_200"; OLD=ROOT/"qwen_iteration"/"first100_pipeline_matrix"
CSV=ROOT/"master_thesis"/"data"/"processed"/"TheEconomistHistoricalArchives-Faces-deduplicated_cleaned.csv"


def source_id(image_id:str)->str:
    m=re.fullmatch(r"(\d{4}-\d{4}-\d{4})_(\d{4})",image_id)
    return f"{m.group(1)},{m.group(2)}" if m else image_id


def iou(a:list[int],b:list[int])->float:
    x1,y1,x2,y2=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1); union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0


def center_in(box:list[int],outer:list[int])->bool:
    x=(box[0]+box[2])/2; y=(box[1]+box[3])/2; return outer[0]<=x<=outer[2] and outer[1]<=y<=outer[3]


def nms(rows:list[dict[str,Any]],threshold:float=.35)->list[dict[str,Any]]:
    kept=[]
    for row in sorted(rows,key=lambda r:r["confidence"],reverse=True):
        if not any(iou(row["bbox_1000"],k["bbox_1000"])>=threshold for k in kept): kept.append(row)
    return sorted(kept,key=lambda r:((r["bbox_1000"][1]+r["bbox_1000"][3])/2,(r["bbox_1000"][0]+r["bbox_1000"][2])/2))


def read_jsonl(path:Path)->list[dict[str,Any]]: return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--cohort",choices=["difficult100","stratified100"],required=True); p.add_argument("--baseline-run",required=True); p.add_argument("--confidence",type=float,default=.75); args=p.parse_args()
    manifest=json.loads((HERE/"data"/f"manifest_{args.cohort}.json").read_text(encoding="utf-8"))["images"][:100]; wanted={source_id(r["image_id"]):r["image_id"] for r in manifest}
    by_page:dict[str,list[dict[str,Any]]]={image_id:[] for image_id in wanted.values()}
    with CSV.open(encoding="utf-8-sig",newline="") as h:
        for row in csv.DictReader(h):
            sid=row["Filename"].split("_",1)[0]; image_id=wanted.get(sid)
            if not image_id: continue
            confidence=float(row["Segmentation confidence score"])
            if confidence<args.confidence: continue
            box=[round(float(row[f"Bounding Box relative {axis}"])*1000) for axis in ["X1","Y1","X2","Y2"]]
            if box[0]<box[2] and box[1]<box[3]: by_page[image_id].append({"bbox_1000":box,"confidence":confidence,"source":"archived_detector"})
    by_page={k:nms(v) for k,v in by_page.items()}
    basepath=(OLD/"output"/args.baseline_run/"p2"/"completed.jsonl") if args.cohort=="difficult100" else (HERE/"output"/args.baseline_run/"p2"/"completed.jsonl")
    pages=[]
    for page in read_jsonl(basepath):
        if page["image_id"] not in by_page: continue
        detections=by_page[page["image_id"]]; ads=[]
        for ad in (page.get("annotation") or {}).get("advertisements") or []:
            ad_box=ad["bbox_1000"]; candidates=[dict(row) for row in detections if center_in(row["bbox_1000"],ad_box)]
            existing=[p["face_bbox_1000"] for p in ad.get("people") or []]
            groups=[g["bbox_1000"] for g in ad.get("groups") or []]
            for index,candidate in enumerate(candidates,1):
                candidate["candidate_id"]=f"d{index}"; candidate["matched_existing_person"]=next((i for i,b in enumerate(existing) if iou(candidate["bbox_1000"],b)>=.10),None); candidate["inside_existing_group"]=any(center_in(candidate["bbox_1000"],g) for g in groups)
            novel=[c for c in candidates if c["matched_existing_person"] is None and not c["inside_existing_group"]]
            band=str(ad.get("face_depiction_count_band")); group_flag=bool(groups) or band in {"7","8","9","10_20","20_plus"} or len(candidates)>=7
            ads.append({"advertisement_id":ad["advertisement_id"],"bbox_1000":ad_box,"baseline_count_band":band,"baseline_people":len(existing),"baseline_groups":len(groups),"detector_candidates":candidates,"novel_outside_groups":len(novel),"flag_r2":bool(not groups and novel and len(candidates)<=15),"flag_r3":group_flag})
        pages.append({"image_id":page["image_id"],"filename":page["filename"],"detections":detections,"advertisements":ads})
    out=HERE/"output"/"detector_proposals"/f"{args.cohort}.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({"schema_version":"archived_detector_proposals_v1","cohort":args.cohort,"confidence_threshold":args.confidence,"nms_iou":.35,"selection_bias_warning":args.cohort=="stratified100","pages":pages},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"path":str(out),"pages":len(pages),"detections":sum(len(p["detections"]) for p in pages),"r2_ads":sum(a["flag_r2"] for p in pages for a in p["advertisements"]),"r3_ads":sum(a["flag_r3"] for p in pages for a in p["advertisements"])}))


if __name__=="__main__": main()
