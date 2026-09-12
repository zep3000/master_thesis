"""Independent clean-crop confirmation before deleting an existing group."""

from __future__ import annotations

import argparse, base64, json, mimetypes, re, threading, time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT=Path(r".");HERE=ROOT/"qwen_iteration"/"next_round_200";OLD=ROOT/"qwen_iteration"/"first100_pipeline_matrix";KEY=ROOT/"openrouter_key.txt";LEDGER=HERE/"output"/"shared_request_ledger.jsonl";COUNT=[str(i)for i in range(1,10)]+["10_20","20_plus"]
def rows(p):return[json.loads(x)for x in p.read_text(encoding="utf-8").splitlines()if x.strip()]if p.exists()else[]
def append(p,r,lock):
    p.parent.mkdir(parents=True,exist_ok=True)
    with lock,p.open("a",encoding="utf-8")as h:h.write(json.dumps(r,ensure_ascii=False)+"\n")
def now():return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def extract(s):
    s=re.sub(r"<think>.*?</think>","",s,flags=re.S).strip()
    try:x=json.loads(s)
    except json.JSONDecodeError:x=json.loads(s[s.find("{"):s.rfind("}")+1])
    return x
def data_url(p):return f"data:{mimetypes.guess_type(p.name)[0]or'image/jpeg'};base64,{base64.b64encode(p.read_bytes()).decode()}"


def main():
    p=argparse.ArgumentParser();p.add_argument("--cohort",required=True);p.add_argument("--baseline-run",required=True);p.add_argument("--run-name",required=True);p.add_argument("--mode",choices=["binary","inventory"],default="binary");p.add_argument("--max-requests",type=int,default=3000);args=p.parse_args();basepath=(OLD/"output"/args.baseline_run/"p2"/"completed.jsonl")if args.cohort=="difficult100"else(HERE/"output"/args.baseline_run/"p2"/"completed.jsonl");base={r["image_id"]:r for r in rows(basepath)};primary={r["task_key"]:r for r in rows(HERE/"output"/args.run_name/args.cohort/"structure"/"results.jsonl")if r.get("ok")};image_dir=(ROOT/"code"/"test_collection_200_difficult_joined_pages")if args.cohort=="difficult100"else(ROOT/"master_thesis"/"data"/"images"/"full_pages_1940_2007_joined");outdir=HERE/"output"/args.run_name/args.cohort/f"group_removal_confirmation_{args.mode}";tasks=[]
    for task,r in primary.items():
        iid,aid=task.split("::");ad=next(a for a in base[iid]["annotation"].get("advertisements")or[]if a["advertisement_id"]==aid)
        if not ad.get("groups")or r["model_annotation"]["group_route_required"]:continue
        with Image.open(image_dir/base[iid]["filename"])as im:
            page=im.convert("RGB");b=ad["bbox_1000"];crop=page.crop((round(b[0]*page.width/1000),round(b[1]*page.height/1000),round(b[2]*page.width/1000),round(b[3]*page.height/1000)))
        path=outdir/"crops"/f"{re.sub(r'[^A-Za-z0-9._-]+','_',task)}.jpg";path.parent.mkdir(parents=True,exist_ok=True);crop.save(path,"JPEG",quality=95);tasks.append((task,path))
    result_path=outdir/"results.jsonl";done={r["task_key"]:r for r in rows(result_path)if r.get("ok")};old=rows(LEDGER);seq=max([int(r.get("request_sequence",0))for r in old]or[0])+1;used=len(old);lock=threading.Lock()
    import httpx
    from openai import OpenAI
    client=OpenAI(api_key=KEY.read_text(encoding="utf-8").strip(),base_url="https://openrouter.ai/api/v1",timeout=300,max_retries=0,http_client=httpx.Client(timeout=300,trust_env=False))
    for task,path in tasks:
        if task in done:continue
        prompt=(f"""Return one JSON object only. task_id={task}. Independently inspect this clean advertisement crop and decide only whether it contains 10 or more unique eligible facial depictions. No detector boxes or prior route are shown. Count all visible faces across the entire crop, including tiny background faces and faces in illustrations, paintings, cartoons, panels, and reproductions. Do not count the same face twice. An eligible face needs more than an ear/back of head and a locatable facial surface. The annotation policy is 1-9 individuals; 10+ uses an aggregate people area. Output schema={{"task_id":"copy ID","face_count_band":{json.dumps(COUNT)},"group_route_required":"boolean, true exactly for 10+","confidence":"0..1","review_flags":"array"}}""" if args.mode=="binary" else f"""Return one JSON object only. task_id={task}. On this clean advertisement crop, enumerate every unique eligible facial depiction with one tight face/head box [x1,y1,x2,y2] on a 0..1000 crop grid. Search the entire crop, including small background faces and faces in illustrations, paintings, cartoons, panels, and reproductions. Do not count text, body-only figures, backs of heads, or the same face twice. Do not decide a group category; produce falsifiable visible face evidence. Output schema={{"task_id":"copy ID","faces":[{{"bbox_1000":"tight four integers","confidence":"0..1"}}],"faces_truncated":"boolean, true only if output limit stops enumeration","review_flags":"array"}}""")
        last=""
        for attempt in range(1,4):
            if used>=args.max_requests:raise RuntimeError("budget")
            n=seq;seq+=1;used+=1;usage=None;started=time.time()
            try:
                response=client.chat.completions.create(model="qwen/qwen3.5-9b",messages=[{"role":"user","content":[{"type":"image_url","image_url":{"url":data_url(path)}},{"type":"text","text":prompt}]}],max_tokens=1800 if args.mode=="inventory" else 500,temperature=0,response_format={"type":"json_object"},extra_body={"provider":{"order":["venice/fp8"],"allow_fallbacks":False},"reasoning":{"effort":"none","exclude":True}});raw=extract(str(response.choices[0].message.content));
                if args.mode=="binary":
                    band=raw.get("face_count_band");band=band[0]if isinstance(band,list)and len(band)==1 else band;band=band if band in COUNT else"1";normalized={"task_id":task,"face_count_band":band,"group_route_required":band in{"10_20","20_plus"},"confidence":float(raw.get("confidence")or.5),"review_flags":raw.get("review_flags")if isinstance(raw.get("review_flags"),list)else[]}
                else:
                    faces=[x for x in raw.get("faces")or[]if isinstance(x,dict)and isinstance(x.get("bbox_1000"),list)and len(x["bbox_1000"])==4];truncated=bool(raw.get("faces_truncated"));n=len(faces);band="20_plus"if n>20 else"10_20"if n>=10 else str(max(1,n));normalized={"task_id":task,"face_count_band":band,"group_route_required":truncated or n>=10,"faces":faces,"faces_truncated":truncated,"review_flags":raw.get("review_flags")if isinstance(raw.get("review_flags"),list)else[]}
                usage=response.usage.model_dump()if response.usage else None;ok=True;last=""
            except Exception as exc:ok=False;normalized=None;last=f"{type(exc).__name__}: {exc}"
            append(LEDGER,{"request_sequence":n,"run_name":args.run_name,"cohort":args.cohort,"stage":f"group_removal_confirmer_{args.mode}","task_key":task,"attempt":attempt,"ok":ok,"usage":usage,"error":None if ok else last,"elapsed_seconds":round(time.time()-started,3),"finished_at":now()},lock)
            if ok:append(result_path,{"task_key":task,"ok":True,"model_annotation":normalized,"usage":usage,"image_path":str(path)},lock);print(task,normalized);break
            if attempt<3:time.sleep(32 if "429" in last else 3)


if __name__=="__main__":main()
