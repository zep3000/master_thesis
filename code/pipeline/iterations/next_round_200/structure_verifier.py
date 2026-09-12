"""Selective gold-free face-inventory and group-route verification."""

from __future__ import annotations

import argparse, base64, hashlib, json, mimetypes, re, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT=Path(r"."); HERE=ROOT/"qwen_iteration"/"next_round_200"; OLD=ROOT/"qwen_iteration"/"first100_pipeline_matrix"; KEY=ROOT/"openrouter_key.txt"; LEDGER=HERE/"output"/"shared_request_ledger.jsonl"
COUNT=[str(i) for i in range(1,10)]+["10_20","20_plus"]


def rows(path:Path)->list[dict[str,Any]]: return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []
def append(path:Path,row:dict[str,Any],lock:threading.Lock):
    path.parent.mkdir(parents=True,exist_ok=True)
    with lock,path.open("a",encoding="utf-8") as h: h.write(json.dumps(row,ensure_ascii=False)+"\n")
def now(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def safe(s:str): return re.sub(r"[^A-Za-z0-9._-]+","_",s)


class Budget:
    def __init__(self,limit:int):
        self.limit=limit; self.lock=threading.Lock(); old=rows(LEDGER); self.used=len(old); self.seq=max([int(r.get("request_sequence",0)) for r in old] or [0])+1
    def reserve(self):
        with self.lock:
            if self.used>=self.limit: raise RuntimeError("request budget exhausted")
            n=self.seq; self.seq+=1; self.used+=1; return n
    def record(self,row): append(LEDGER,row,self.lock)


def map_to_ad(box:list[int],ad:list[int])->list[int]:
    w=max(1,ad[2]-ad[0]); h=max(1,ad[3]-ad[1]); return [round((box[0]-ad[0])*1000/w),round((box[1]-ad[1])*1000/h),round((box[2]-ad[0])*1000/w),round((box[3]-ad[1])*1000/h)]


def crop_overlay(page_path:Path,ad:dict[str,Any],out:Path)->list[dict[str,Any]]:
    with Image.open(page_path) as im: page=im.convert("RGB")
    ab=ad["bbox_1000"]; px=[round(ab[0]*page.width/1000),round(ab[1]*page.height/1000),round(ab[2]*page.width/1000),round(ab[3]*page.height/1000)]; crop=page.crop(px)
    scale=min(1200/crop.width,1200/crop.height); view=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.LANCZOS); canvas=Image.new("RGB",(1200,1240),"white"); ox=(1200-view.width)//2; oy=40+(1200-view.height)//2; canvas.paste(view,(ox,oy)); draw=ImageDraw.Draw(canvas); font=ImageFont.load_default(); draw.text((8,8),"BLUE detector candidates; GREEN baseline people; MAGENTA baseline group",fill="black",font=font)
    props=[]
    for c in ad["detector_candidates"]:
        b=map_to_ad(c["bbox_1000"],ab); props.append({"candidate_id":c["candidate_id"],"bbox_1000":b,"confidence":round(float(c["confidence"]),3),"novel_outside_baseline_group":c["matched_existing_person"] is None and not c["inside_existing_group"]})
        r=[ox+b[0]*view.width/1000,oy+b[1]*view.height/1000,ox+b[2]*view.width/1000,oy+b[3]*view.height/1000]; draw.rectangle(r,outline="blue",width=4); draw.text((r[0],max(40,r[1]-12)),c["candidate_id"],fill="blue",font=font)
    for i,bp in enumerate(ad.get("baseline_people_boxes") or [],1):
        b=map_to_ad(bp,ab); r=[ox+b[0]*view.width/1000,oy+b[1]*view.height/1000,ox+b[2]*view.width/1000,oy+b[3]*view.height/1000]; draw.rectangle(r,outline="green",width=3); draw.text((r[0],r[1]),f"P{i}",fill="green",font=font)
    for i,bp in enumerate(ad.get("baseline_group_boxes") or [],1):
        b=map_to_ad(bp,ab); r=[ox+b[0]*view.width/1000,oy+b[1]*view.height/1000,ox+b[2]*view.width/1000,oy+b[3]*view.height/1000]; draw.rectangle(r,outline="magenta",width=4); draw.text((r[0],r[1]),f"G{i}",fill="magenta",font=font)
    out.parent.mkdir(parents=True,exist_ok=True); canvas.save(out,"JPEG",quality=95); return props


SCHEMA={"task_id":"copy ID","accepted_candidate_ids":"array of blue candidate IDs that truly mark eligible faces","missed_faces":[{"bbox_1000":"tight [x1,y1,x2,y2] on ad crop 0..1000 for visually certain eligible faces not marked blue or green"}],"candidate_inventory_complete":"boolean; all eligible faces are covered by accepted blue or green boxes, except faces adequately represented inside a group","face_count_band":COUNT,"group_route_required":"boolean derived from total visible eligible faces being 10+","group_bbox_1000":"one broad aggregate area on ad crop if group required, else null","outstanding_candidate_ids":"up to three accepted blue IDs that are genuinely outstanding when group required","group_attributes":{"group_type":["interacting_group","posed_group","audience","background_population","separate_portraits_or_composite","other_group"],"age_composition":["young_only","middle_only","older_only","mostly_young","mostly_middle","mostly_older","mixed","not_assessable"],"gender_presentation_composition":["feminine_only","masculine_only","mostly_feminine","mostly_masculine","mixed","ambiguous_or_androgynous_present","not_assessable"],"expression_legibility_distribution":["all_0_not_legible","mostly_0_not_legible","all_1_low_legibility","mostly_1_low_legibility","all_2_moderate_legibility","mostly_2_moderate_legibility","all_3_high_legibility","mostly_3_high_legibility","mixed_legibility"],"dominant_gaze":["toward_viewer_camera","toward_each_other","toward_object","off_frame_or_scene_direction","mixed","not_assessable",None],"smile_prevalence":["none","minority","about_half","majority","all","not_assessable",None],"dominant_smile_intensity":["slight","clear","broad_or_laughter_like","mixed",None]},"confidence":"0..1","review_flags":"array"}


def prompt(task_id:str,proposals:list[dict[str,Any]],ad:dict[str,Any])->str:
    return f"""Return one JSON object only. task_id={task_id}. This is one advertisement crop. Blue boxes are noisy archived-detector proposals, green boxes are the baseline's individual faces, and magenta is its proposed people area. None is ground truth. Visually verify all evidence.
An eligible face has more than only an ear/back of head and a locatable facial surface; photos, illustrations, cartoons, statues, masks, personified objects, creatures, and logo faces qualify. Reject text, objects, and non-face marks. Report visually certain missed eligible faces not covered by blue/green, but do not duplicate an accepted box.
Route by total unique eligible faces: 1-9 means individuals and no group; 10+ means one broad aggregate people area whenever a unified tendency is analytically adequate, plus at most three genuinely outstanding individuals. A group is an annotation convenience/aggregate, not a demand for precise per-face boxes. candidate_inventory_complete may be true when ordinary crowd members are adequately covered by one group even if every tiny face lacks a blue box.
If group is required, code its aggregate attributes. For expression legibility, 0 means no expression evidence at all; low means coarse cue only; moderate means stable coarse multi-region configuration; high means fine/subtle cues. Conditional nulls apply at all-0 and no smiles.
Baseline summary={{"count_band":{json.dumps(ad['baseline_count_band'])},"people":{ad['baseline_people']},"groups":{ad['baseline_groups']}}}; candidates={json.dumps(proposals,separators=(',',':'))}. Schema={json.dumps(SCHEMA,separators=(',',':'))}"""


def scalar(x): return x[0] if isinstance(x,list) and len(x)==1 else x
def normalize(raw:dict[str,Any],task_id:str,ids:set[str])->dict[str,Any]:
    accepted=[str(x) for x in raw.get("accepted_candidate_ids") or [] if str(x) in ids]; band=scalar(raw.get("face_count_band")); band=band if band in COUNT else "1"; group=band in {"10_20","20_plus"}; missed=[]
    for item in raw.get("missed_faces") or []:
        b=item.get("bbox_1000") if isinstance(item,dict) else None
        if isinstance(b,list) and len(b)==4:
            try: b=[max(0,min(1000,round(float(v)))) for v in b]
            except (TypeError,ValueError): continue
            if b[0]<b[2] and b[1]<b[3]: missed.append({"bbox_1000":b})
    gb=raw.get("group_bbox_1000"); gb=gb if isinstance(gb,list) and len(gb)==4 else None; attrs=raw.get("group_attributes") if isinstance(raw.get("group_attributes"),dict) else {}
    return {"task_id":task_id,"accepted_candidate_ids":accepted,"missed_faces":missed[:9],"candidate_inventory_complete":bool(raw.get("candidate_inventory_complete")),"face_count_band":band,"group_route_required":group,"group_bbox_1000":gb if group else None,"outstanding_candidate_ids":[str(x) for x in raw.get("outstanding_candidate_ids") or [] if str(x) in accepted][:3],"group_attributes":{k:scalar(v) for k,v in attrs.items()} if group else None,"confidence":float(raw.get("confidence") or .5),"review_flags":raw.get("review_flags") if isinstance(raw.get("review_flags"),list) else []}


def data_url(path:Path): return f"data:{mimetypes.guess_type(path.name)[0] or 'image/jpeg'};base64,{base64.b64encode(path.read_bytes()).decode()}"
def extract(s:str):
    s=re.sub(r"<think>.*?</think>","",s,flags=re.S).strip()
    try: x=json.loads(s)
    except json.JSONDecodeError: x=json.loads(s[s.find("{"):s.rfind("}")+1])
    if not isinstance(x,dict): raise ValueError("not object")
    return x
def plain(x):
    if x is None or isinstance(x,(str,int,float,bool)): return x
    if isinstance(x,list): return [plain(v) for v in x]
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if hasattr(x,"model_dump"): return plain(x.model_dump())
    return str(x)


def run_call(client,budget,args,out,job):
    ph=hashlib.sha256(job["prompt"].encode()).hexdigest(); last=""
    for attempt in range(1,4):
        seq=budget.reserve(); started=time.time(); usage=None; raw=""
        try:
            r=client.chat.completions.create(model="qwen/qwen3.5-9b",messages=[{"role":"user","content":[{"type":"image_url","image_url":{"url":data_url(Path(job['image_path']))}},{"type":"text","text":job['prompt']}]}],max_tokens=1800,temperature=0,response_format={"type":"json_object"},extra_body={"provider":{"order":["venice/fp8"],"allow_fallbacks":False},"reasoning":{"effort":"none","exclude":True}}); meta=plain(r); usage=meta.get("usage"); raw=str(r.choices[0].message.content); parsed=extract(raw); result=normalize(parsed,job["task_id"],set(job["candidate_ids"])); ok=True; last=""
        except Exception as exc: ok=False; parsed=result=None; last=f"{type(exc).__name__}: {exc}"
        budget.record({"request_sequence":seq,"run_name":args.run_name,"cohort":args.cohort,"stage":"structure_verifier","task_key":job["task_id"],"attempt":attempt,"ok":ok,"usage":usage,"error":None if ok else last,"prompt_sha256":ph,"image_path":job["image_path"],"elapsed_seconds":round(time.time()-started,3),"finished_at":now()})
        if raw:
            p=out/"raw"; p.mkdir(parents=True,exist_ok=True); (p/f"{safe(job['task_id'])}_a{attempt}_r{seq}.txt").write_text(raw,encoding="utf-8")
        if ok:return {"task_key":job["task_id"],"ok":True,"model_annotation_raw":parsed,"model_annotation":result,"usage":usage,"image_path":job["image_path"],"flags":job["flags"]}
        if attempt<3: time.sleep(32 if "429" in last else 3)
    return {"task_key":job["task_id"],"ok":False,"error":last,"image_path":job["image_path"],"flags":job["flags"]}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--cohort",choices=["difficult100","stratified100"],required=True); p.add_argument("--baseline-run",required=True); p.add_argument("--run-name",required=True); p.add_argument("--pilot-ads",type=int,default=0); p.add_argument("--workers",type=int,default=1); p.add_argument("--max-requests",type=int,default=3000); p.add_argument("--retry-failures",action="store_true"); args=p.parse_args()
    proposals=json.loads((HERE/"output"/"detector_proposals"/f"{args.cohort}.json").read_text(encoding="utf-8")); basepath=(OLD/"output"/args.baseline_run/"p2"/"completed.jsonl") if args.cohort=="difficult100" else (HERE/"output"/args.baseline_run/"p2"/"completed.jsonl"); baseline={r["image_id"]:r for r in rows(basepath)}; image_dir=(ROOT/"code"/"test_collection_200_difficult_joined_pages") if args.cohort=="difficult100" else (ROOT/"master_thesis"/"data"/"images"/"full_pages_1940_2007_joined"); out=HERE/"output"/args.run_name/args.cohort/"structure"; jobs=[]
    for page in proposals["pages"]:
        bp=baseline.get(page["image_id"]); ads={(a["advertisement_id"]):a for a in (bp.get("annotation") or {}).get("advertisements") or []} if bp else {}
        for ad in page["advertisements"]:
            if not(ad["flag_r2"] or ad["flag_r3"]):continue
            source=ads.get(ad["advertisement_id"],{}); ad["baseline_people_boxes"]=[x["face_bbox_1000"] for x in source.get("people") or []]; ad["baseline_group_boxes"]=[x["bbox_1000"] for x in source.get("groups") or []]; task=f"{page['image_id']}::{ad['advertisement_id']}"; path=out/"overlays"/f"{safe(task)}.jpg"; props=crop_overlay(image_dir/page["filename"],ad,path); jobs.append({"task_id":task,"prompt":prompt(task,props,ad),"image_path":str(path),"candidate_ids":[x["candidate_id"] for x in props],"flags":{"r2":ad["flag_r2"],"r3":ad["flag_r3"]}})
    if args.pilot_ads: jobs=jobs[:args.pilot_ads]
    result_path=out/"results.jsonl"; latest={r["task_key"]:r for r in rows(result_path)}; pending=[j for j in jobs if j["task_id"] not in latest or(args.retry_failures and not latest[j["task_id"]].get("ok"))]; print(f"jobs={len(jobs)} pending={len(pending)}")
    import httpx
    from openai import OpenAI
    c=OpenAI(api_key=KEY.read_text(encoding="utf-8").strip(),base_url="https://openrouter.ai/api/v1",timeout=300,max_retries=0,http_client=httpx.Client(timeout=300,trust_env=False)); budget=Budget(args.max_requests); lock=threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_call,c,budget,args,out,j) for j in pending]
        for i,f in enumerate(as_completed(futures),1): row=f.result(); append(result_path,row,lock); print(f"[{i}/{len(pending)}] {row['task_key']} ok={row['ok']} requests={budget.used}")


if __name__=="__main__":main()
