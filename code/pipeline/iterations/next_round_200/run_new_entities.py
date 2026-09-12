"""Run full baseline attributes and promoted F0 FER for novel verified faces."""

from __future__ import annotations

import argparse, base64, hashlib, importlib.util, json, mimetypes, re, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from specialist_prompts import normalize_person, person_prompt as specialist_prompt

ROOT=Path(r".");HERE=ROOT/"qwen_iteration"/"next_round_200";OLD=ROOT/"qwen_iteration"/"first100_pipeline_matrix";KEY=ROOT/"openrouter_key.txt";LEDGER=HERE/"output"/"shared_request_ledger.jsonl"
sys.path.insert(0,str(OLD));import schema_and_prompts as schemas
spec=importlib.util.spec_from_file_location("frozen",OLD/"run.py");assert spec and spec.loader;frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)


def rows(path:Path):return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []
def append(path,row,lock):
    path.parent.mkdir(parents=True,exist_ok=True)
    with lock,path.open("a",encoding="utf-8") as h:h.write(json.dumps(row,ensure_ascii=False)+"\n")
def now():return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def data_url(p):return f"data:{mimetypes.guess_type(p.name)[0] or 'image/jpeg'};base64,{base64.b64encode(p.read_bytes()).decode()}"
def extract(s):
    s=re.sub(r"<think>.*?</think>","",s,flags=re.S).strip()
    try:x=json.loads(s)
    except json.JSONDecodeError:x=json.loads(s[s.find("{"):s.rfind("}")+1])
    if not isinstance(x,dict):raise ValueError("not object")
    return x
def plain(x):
    if x is None or isinstance(x,(str,int,float,bool)):return x
    if isinstance(x,list):return[plain(v) for v in x]
    if isinstance(x,dict):return{str(k):plain(v) for k,v in x.items()}
    if hasattr(x,"model_dump"):return plain(x.model_dump())
    return str(x)


class Budget:
    def __init__(self,limit):
        self.limit=limit;self.lock=threading.Lock();old=rows(LEDGER);self.used=len(old);self.seq=max([int(r.get("request_sequence",0)) for r in old]or[0])+1
    def reserve(self):
        with self.lock:
            if self.used>=self.limit:raise RuntimeError("budget")
            x=self.seq;self.seq+=1;self.used+=1;return x
    def record(self,r):append(LEDGER,r,self.lock)


def normalize_full(raw,task):
    raw["person_task_id"]=task;raw.setdefault("eligible_face_visible",True);raw.setdefault("depiction_type",None);raw.setdefault("perceived_age","not_assessable");raw.setdefault("perceived_gender_presentation","not_assessable");raw.setdefault("face_expression_legibility","0_not_legible");raw.setdefault("face_orientation","not_assessable");raw.setdefault("gaze_target",None);raw.setdefault("gaze_target_person_unboxed",None);raw.setdefault("mouth_covered","not_assessable");raw.setdefault("mouth_covering",None);raw.setdefault("smile_present",None);raw.setdefault("smile_intensity",None);raw.setdefault("confidence",.5);raw.setdefault("review_flags",[]);frozen.normalize_person_attributes(raw);return raw


def main():
    p=argparse.ArgumentParser();p.add_argument("--cohort",required=True);p.add_argument("--run-name",required=True);p.add_argument("--max-requests",type=int,default=3000);args=p.parse_args();base=HERE/"output"/args.run_name/args.cohort/"new_entities";manifest=json.loads((base/"manifest.json").read_text(encoding="utf-8"))["entities"];result_path=base/"results.jsonl";done={(r["stage"],r["task_key"]):r for r in rows(result_path) if r.get("ok")}
    import httpx
    from openai import OpenAI
    client=OpenAI(api_key=KEY.read_text(encoding="utf-8").strip(),base_url="https://openrouter.ai/api/v1",timeout=300,max_retries=0,http_client=httpx.Client(timeout=300,trust_env=False));budget=Budget(args.max_requests);lock=threading.Lock()
    for stage in ["full","f0"]:
        for i,e in enumerate(manifest,1):
            if(stage,e["task_key"])in done:continue
            prompt=schemas.person_prompt(e["task_key"],e["ad_depiction_type"]) if stage=="full" else specialist_prompt(e["task_key"],"f0_ordinal_direct");last=""
            for attempt in range(1,4):
                seq=budget.reserve();started=time.time();usage=None
                try:
                    r=client.chat.completions.create(model="qwen/qwen3.5-9b",messages=[{"role":"user","content":[{"type":"image_url","image_url":{"url":data_url(Path(e['composite_path']))}},{"type":"text","text":prompt}]}],max_tokens=1400,temperature=0,response_format={"type":"json_object"},extra_body={"provider":{"order":["venice/fp8"],"allow_fallbacks":False},"reasoning":{"effort":"none","exclude":True}});meta=plain(r);usage=meta.get("usage");raw=extract(str(r.choices[0].message.content));normalized=normalize_full(raw,e["task_key"]) if stage=="full" else normalize_person(raw,e["task_key"],"f0_ordinal_direct");ok=True;last=""
                except Exception as exc:ok=False;normalized=None;raw=None;last=f"{type(exc).__name__}: {exc}"
                budget.record({"request_sequence":seq,"run_name":args.run_name,"cohort":args.cohort,"stage":f"new_entity_{stage}","task_key":e["task_key"],"attempt":attempt,"ok":ok,"usage":usage,"error":None if ok else last,"elapsed_seconds":round(time.time()-started,3),"finished_at":now()})
                if ok:
                    row={"stage":stage,"task_key":e["task_key"],"ok":True,"model_annotation_raw":raw,"model_annotation":normalized,"usage":usage};append(result_path,row,lock);print(f"{stage} [{i}/{len(manifest)}] {e['task_key']} ok");break
                if attempt<3:time.sleep(32 if"429"in last else 3)
            else:append(result_path,{"stage":stage,"task_key":e["task_key"],"ok":False,"error":last},lock)


if __name__=="__main__":main()
