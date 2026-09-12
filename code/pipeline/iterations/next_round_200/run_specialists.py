"""Run shared person/group specialist calls from gold-free baseline outputs."""

from __future__ import annotations

import argparse, base64, hashlib, importlib.util, json, mimetypes, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
from specialist_prompts import group_prompt, normalize_group, normalize_person, person_prompt

ROOT = Path(r"."); HERE = ROOT / "qwen_iteration" / "next_round_200"
OLD = ROOT / "qwen_iteration" / "first100_pipeline_matrix"
LEDGER = HERE / "output" / "shared_request_ledger.jsonl"; API_KEY = ROOT / "openrouter_key.txt"
STRATEGIES = ["f0_ordinal_direct", "f1_hierarchical_gaze", "f2_hierarchical_gaze_smile", "g1_aggregate"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def append(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock, path.open("a", encoding="utf-8") as h: h.write(json.dumps(row, ensure_ascii=False) + "\n")


def now() -> str: return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
def safe(s: str) -> str: return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


class Budget:
    def __init__(self, limit: int):
        self.limit=limit; self.lock=threading.Lock(); rows=read_jsonl(LEDGER); self.used=len(rows); self.seq=max([int(r.get("request_sequence",0)) for r in rows] or [0])+1
    def reserve(self) -> int:
        with self.lock:
            if self.used >= self.limit: raise RuntimeError(f"Shared request ceiling {self.used}/{self.limit}")
            x=self.seq; self.seq+=1; self.used+=1; return x
    def record(self,row:dict[str,Any])->None: append(LEDGER,row,self.lock)


def data_url(path: Path) -> str:
    mime=mimetypes.guess_type(path.name)[0] or "image/jpeg"; return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def extract(text: str) -> dict[str, Any]:
    text=re.sub(r"<think>.*?</think>","",text,flags=re.S).strip()
    try: value=json.loads(text)
    except json.JSONDecodeError:
        a=text.find("{"); b=text.rfind("}"); value=json.loads(text[a:b+1])
    if not isinstance(value,dict): raise ValueError("not object")
    return value


def plain(x: Any) -> Any:
    if x is None or isinstance(x,(str,int,float,bool)): return x
    if isinstance(x,list): return [plain(v) for v in x]
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if hasattr(x,"model_dump"): return plain(x.model_dump())
    return str(x)


def client():
    import httpx
    from openai import OpenAI
    return OpenAI(api_key=API_KEY.read_text(encoding="utf-8").strip(),base_url="https://openrouter.ai/api/v1",timeout=300,max_retries=0,http_client=httpx.Client(timeout=300,trust_env=False),default_headers={"X-Title":"Qwen Next Round 200"})


def call(c:Any,budget:Budget,args:argparse.Namespace,out:Path,job:dict[str,Any])->dict[str,Any]:
    prompt=job["prompt"]; ph=hashlib.sha256(prompt.encode()).hexdigest(); last=""
    for attempt in range(1,4):
        seq=budget.reserve(); started=time.time(); usage=None; raw=""
        try:
            resp=c.chat.completions.create(model="qwen/qwen3.5-9b",messages=[{"role":"user","content":[{"type":"image_url","image_url":{"url":data_url(Path(job['image_path']))}},{"type":"text","text":prompt}]}],max_tokens=1100,temperature=0,response_format={"type":"json_object"},extra_body={"provider":{"order":["venice/fp8"],"allow_fallbacks":False},"reasoning":{"effort":"none","exclude":True}})
            meta=plain(resp); usage=meta.get("usage"); raw=str(resp.choices[0].message.content); parsed=extract(raw)
            normalized=normalize_group(parsed,job["task_id"]) if job["strategy"]=="g1_aggregate" else normalize_person(parsed,job["task_id"],job["strategy"]); ok=True; last=""
        except Exception as exc:
            ok=False; normalized=None; parsed=None; last=f"{type(exc).__name__}: {exc}"
        budget.record({"request_sequence":seq,"run_name":args.run_name,"cohort":args.cohort,"stage":"specialist","strategy":job["strategy"],"task_key":job["task_key"],"attempt":attempt,"ok":ok,"usage":usage,"error":None if ok else last,"prompt_sha256":ph,"image_path":job["image_path"],"elapsed_seconds":round(time.time()-started,3),"finished_at":now()})
        if raw:
            p=out/"raw"/job["strategy"]; p.mkdir(parents=True,exist_ok=True); (p/f"{safe(job['task_key'])}_a{attempt}_r{seq}.txt").write_text(raw,encoding="utf-8")
        if ok: return {"task_key":job["task_key"],"task_id":job["task_id"],"strategy":job["strategy"],"cohort":args.cohort,"ok":True,"model_annotation_raw":parsed,"model_annotation":normalized,"usage":usage,"prompt_sha256":ph,"image_path":job["image_path"]}
        if attempt<3: time.sleep(32 if "429" in last else 3*attempt)
    return {"task_key":job["task_key"],"task_id":job["task_id"],"strategy":job["strategy"],"cohort":args.cohort,"ok":False,"error":last,"image_path":job["image_path"]}


def latest(path:Path)->dict[str,dict[str,Any]]: return {f"{r['strategy']}::{r['task_key']}":r for r in read_jsonl(path)}


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--cohort",choices=["difficult100","stratified100"],required=True); p.add_argument("--baseline-run",required=True); p.add_argument("--run-name",required=True); p.add_argument("--strategies",nargs="+",choices=STRATEGIES,default=STRATEGIES); p.add_argument("--limit",type=int,default=100); p.add_argument("--workers",type=int,default=2); p.add_argument("--max-requests",type=int,default=3000); p.add_argument("--retry-failures",action="store_true"); args=p.parse_args()
    baseline=(OLD/"output"/args.baseline_run/"p2"/"completed.jsonl") if args.cohort=="difficult100" else (HERE/"output"/args.baseline_run/"p2"/"completed.jsonl")
    pages=[r for r in read_jsonl(baseline) if int(r.get("manifest_index",999))<args.limit and r.get("ok")]
    image_dir=(ROOT/"code"/"test_collection_200_difficult_joined_pages") if args.cohort=="difficult100" else (ROOT/"master_thesis"/"data"/"images"/"full_pages_1940_2007_joined")
    sys.path.insert(0,str(OLD)); spec=importlib.util.spec_from_file_location("frozen_runner",OLD/"run.py"); assert spec and spec.loader; frozen=importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
    out=HERE/"output"/args.run_name/args.cohort; out.mkdir(parents=True,exist_ok=True); jobs=[]
    for page in pages:
        page_path=image_dir/page["filename"]
        for ad in page["annotation"].get("advertisements") or []:
            for person in ad.get("people") or []:
                task_id=f"{page['image_id']}::{person['person_id']}"; composite=out/"crops"/"persons"/f"{safe(task_id)}.jpg"
                if not composite.exists(): frozen.person_composite(page_path,person["face_bbox_1000"],ad["bbox_1000"],composite,task_id)
                for strategy in args.strategies:
                    if strategy.startswith("f"): jobs.append({"task_key":task_id,"task_id":task_id,"strategy":strategy,"prompt":person_prompt(task_id,strategy),"image_path":str(composite)})
            for group in ad.get("groups") or []:
                task_id=f"{page['image_id']}::{group['group_id']}"; crop=out/"crops"/"groups"/f"{safe(task_id)}.jpg"
                if not crop.exists(): frozen.crop_from_box(page_path,group["bbox_1000"],crop)
                if "g1_aggregate" in args.strategies: jobs.append({"task_key":task_id,"task_id":task_id,"strategy":"g1_aggregate","prompt":group_prompt(task_id),"image_path":str(crop)})
    results=out/"specialists.jsonl"; done=latest(results); pending=[j for j in jobs if f"{j['strategy']}::{j['task_key']}" not in done or (args.retry_failures and not done[f"{j['strategy']}::{j['task_key']}"].get("ok"))]
    print(f"jobs={len(jobs)} pending={len(pending)} existing_shared={len(read_jsonl(LEDGER))}/{args.max_requests}")
    c=client(); budget=Budget(args.max_requests); lock=threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(call,c,budget,args,out,j) for j in pending]
        for i,f in enumerate(as_completed(futures),1):
            row=f.result(); append(results,row,lock); print(f"[{i}/{len(pending)}] {row['strategy']} {row['task_key']} ok={row['ok']} requests={budget.used}")
    print(out)


if __name__=="__main__": main()
