"""Gold-free deterministic assembly of R0–R4 from shared model outputs."""

from __future__ import annotations

import argparse, copy, json
from pathlib import Path
from typing import Any

ROOT=Path(r".");HERE=ROOT/"qwen_iteration"/"next_round_200";OLD=ROOT/"qwen_iteration"/"first100_pipeline_matrix"
FER=["face_expression_legibility","gaze_target","smile_present","smile_intensity"]
PERSON=["depiction_type","perceived_age","perceived_gender_presentation","face_expression_legibility","face_orientation","gaze_target","gaze_target_person_unboxed","mouth_covered","mouth_covering","smile_present","smile_intensity"]
GROUP=["group_type","age_composition","gender_presentation_composition","expression_legibility_distribution","dominant_gaze","smile_prevalence","dominant_smile_intensity"]


def rows(path:Path):return[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]if path.exists()else[]
def ad_to_page(box,ad):
    w=ad[2]-ad[0];h=ad[3]-ad[1];return[round(ad[0]+box[0]*w/1000),round(ad[1]+box[1]*h/1000),round(ad[0]+box[2]*w/1000),round(ad[1]+box[3]*h/1000)]
def iou(a,b):
    x1,y1,x2,y2=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3]);inter=max(0,x2-x1)*max(0,y2-y1);union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter;return inter/union if union else 0


def new_people(run_dir:Path)->dict[str,list[dict[str,Any]]]:
    manifest=json.loads((run_dir/"new_entities"/"manifest.json").read_text(encoding="utf-8"))["entities"]; results={(r["stage"],r["task_key"]):r for r in rows(run_dir/"new_entities"/"results.jsonl")if r.get("ok")};out={}
    for e in manifest:
        full=results.get(("full",e["task_key"]));f0=results.get(("f0",e["task_key"]))
        if not full or not f0:continue
        raw=full["model_annotation"];person={"person_id":e["person_id"],"annotation_role":"individual","face_bbox_1000":e["face_bbox_1000"],"duplicate_of_person_id":None,"duplicate_person_ids":[]}
        for field in PERSON:person[field]=raw.get(field)
        for field in FER:person[field]=f0["model_annotation"].get(field)
        person.update({"gaze_target_person_id":None,"gaze_target_object_ref":None,"mouth_covering_other_text":None,"confidence":min(float(raw.get("confidence")or.5),float(f0["model_annotation"].get("confidence")or.5)),"review_flags":["verified_detector_augmentation"]})
        parent=f"{e['image_id']}::{e['advertisement_id']}";out.setdefault(parent,[]).append(person)
    return out


def specialist_map(run_dir:Path):return{r["task_key"]:r for r in rows(run_dir/"specialists.jsonl")if r.get("ok")and r.get("strategy")=="f0_ordinal_direct"}


def apply_fer(ann:dict[str,Any],image_id:str,specs):
    for ad in ann.get("advertisements")or[]:
        for person in ad.get("people")or[]:
            r=specs.get(f"{image_id}::{person['person_id']}")
            if r:
                for f in FER:person[f]=r["model_annotation"].get(f)
                person["fer_specialist"]="f0_ordinal_direct"


def add_people(ad,people):
    existing=ad.get("people")or[]
    for p in people:
        if not any(iou(p["face_bbox_1000"],q["face_bbox_1000"])>=.10 for q in existing):existing.append(copy.deepcopy(p))
    ad["people"]=existing[:9]


def apply_structure(ann,image_id,verified,confirmations,new,mode):
    for ad in ann.get("advertisements")or[]:
        task=f"{image_id}::{ad['advertisement_id']}";v=verified.get(task)
        if not v:continue
        flags=v.get("flags")or{};result=v["model_annotation"];confirmation=confirmations.get(task,{}).get("model_annotation")or{}
        effective_group=bool(result["group_route_required"] or confirmation.get("group_route_required"))
        effective_band=confirmation.get("face_count_band") if confirmation.get("group_route_required") else result["face_count_band"]
        use_r2=mode in{"r2","r4"}and(flags.get("r2")or mode=="r4")
        use_r3=mode in{"r3","r4"}and flags.get("r3")
        if use_r2 and not effective_group:add_people(ad,new.get(task,[]))
        if not use_r3:continue
        ad["face_depiction_count_band"]=effective_band
        if effective_group:
            if not ad.get("groups") and result.get("group_bbox_1000"):
                attrs=result.get("group_attributes")or{};group={"group_id":f"{ad['advertisement_id']}_verified_group_1","bbox_1000":ad_to_page(result["group_bbox_1000"],ad["bbox_1000"])}
                for f in GROUP:group[f]=attrs.get(f)
                group.update({"confidence":result.get("confidence",.5),"review_flags":["selective_group_verifier"]});ad["groups"]=[group]
            # Existing baseline people are retained only as a conservative set
            # of up to three outstanding people; the group carries ordinary faces.
            ad["people"]=(ad.get("people")or[])[:3]
            for p in ad["people"]:p["annotation_role"]="outstanding_individual"
            ad["has_outstanding_individuals"]="yes"if ad["people"]else"no";ad["unique_face_count"]=None;ad["duplicate_faces_present"]=None
        else:
            ad["groups"]=[];add_people(ad,new.get(task,[]))
            for p in ad.get("people")or[]:p["annotation_role"]="individual"
            ad["has_outstanding_individuals"]=None;ad["unique_face_count"]=len(ad.get("people")or[])or None


def main():
    p=argparse.ArgumentParser();p.add_argument("--cohort",choices=["difficult100","stratified100"],required=True);p.add_argument("--baseline-run",required=True);p.add_argument("--run-name",required=True);args=p.parse_args();basepath=(OLD/"output"/args.baseline_run/"p2"/"completed.jsonl")if args.cohort=="difficult100"else(HERE/"output"/args.baseline_run/"p2"/"completed.jsonl");baseline=rows(basepath);run_dir=HERE/"output"/args.run_name/args.cohort;specs=specialist_map(run_dir);new=new_people(run_dir);verified={r["task_key"]:r for r in rows(run_dir/"structure"/"results.jsonl")if r.get("ok")};confirmations={r["task_key"]:r for r in rows(run_dir/"group_removal_confirmation_inventory"/"results.jsonl")if r.get("ok")};outdir=run_dir/"assembled";outdir.mkdir(parents=True,exist_ok=True)
    for route in["r0","r1","r2","r3","r4"]:
        output=[]
        for page in baseline:
            item=copy.deepcopy(page);item["route"]=route
            if route!="r0":apply_fer(item["annotation"],page["image_id"],specs)
            if route in{"r2","r3","r4"}:apply_structure(item["annotation"],page["image_id"],verified,confirmations,new,route)
            output.append(item)
        path=outdir/f"{route}.jsonl";path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n"for x in output),encoding="utf-8");print(json.dumps({"route":route,"pages":len(output),"people":sum(len(p)for x in output for a in x["annotation"].get("advertisements")or[]for p in[[*a.get('people',[])] ]),"groups":sum(len(a.get("groups")or[])for x in output for a in x["annotation"].get("advertisements")or[]),"path":str(path)}))


if __name__=="__main__":main()
