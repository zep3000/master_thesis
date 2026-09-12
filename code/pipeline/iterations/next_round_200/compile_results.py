"""Evaluation-only pooled metrics, category breakdowns, bootstrap CIs, and costs."""

from __future__ import annotations

import importlib.util, json, random, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT=Path(r".");HERE=ROOT/"qwen_iteration"/"next_round_200";OLD=ROOT/"qwen_iteration"/"first100_pipeline_matrix";RUN="scale_v1"
sys.path.insert(0,str(OLD));spec=importlib.util.spec_from_file_location("ev",OLD/"evaluate.py");assert spec and spec.loader;ev=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ev;spec.loader.exec_module(ev)


def rows(path:Path):return[json.loads(x)for x in path.read_text(encoding="utf-8").splitlines()if x.strip()]if path.exists()else[]
def gold(cohort):
    path=ROOT/"annotation_results"/("test_collection_200_difficult_joined_v1_2026-08-02.json"if cohort=="difficult100"else"economist_decade_face_count_stratified_200_seed20260812_min2_52123740_2026-08-12.json");data=json.loads(path.read_text(encoding="utf-8"));r=data["annotations"]
    if cohort=="difficult100":
        keep={x["image_id"] for x in json.loads((HERE/"data"/"manifest_difficult100.json").read_text(encoding="utf-8"))["images"]}
        r=[x for x in r if x.get("assignment_code")=="79201188" and x.get("image_id") in keep]
    else:
        r=sorted(r,key=lambda x:int(x["sort_order"]))[:100]
    return{x["image_id"]:x["payload"]for x in r}
def preds(cohort,route):return{x["image_id"]:x["annotation"]for x in rows(HERE/"output"/RUN/cohort/"assembled"/f"{route}.jsonl")}


def matched(g,p,ids,entity="people",mode="strict"):
    out=[]
    for image_id in ids:
        hp=ev.all_items(image_id,g[image_id],"ads");pp=ev.all_items(image_id,p[image_id],"ads")
        for ap in ev.pair_items(hp,pp,"ads",mode):
            fn=ev.person_items if entity=="people"else ev.group_items
            out.extend(ev.pair_items(fn(image_id,ap.human.data),fn(image_id,ap.predicted.data),entity,mode))
    return out


def categories(pairs,all_gold,field,order=None):
    labels=sorted({str(ev.normalize_value(x.data.get(field),field))for x in all_gold if ev.normalize_value(x.data.get(field),field)is not None})
    result={}
    for label in labels:
        gold_total=sum(str(ev.normalize_value(x.data.get(field),field))==label for x in all_gold);pair=[x for x in pairs if str(ev.normalize_value(x.human.data.get(field),field))==label];exact=sum(ev.exact_label(field,x.human.data.get(field),x.predicted.data.get(field))for x in pair);within=None
        if order:
            within=sum(ev.lenient_label(field,x.human.data.get(field),x.predicted.data.get(field))for x in pair)
        result[label]={"gold_total":gold_total,"spatially_matched":len(pair),"conditional_exact":exact/len(pair)if pair else None,"end_to_end_exact_recall":exact/gold_total if gold_total else None,"conditional_lenient":within/len(pair)if pair and within is not None else None,"end_to_end_lenient_recall":within/gold_total if gold_total and within is not None else None}
    return result


def f1_for_pages(g,p,sampled,entity,mode):
    ht=pt=m=0
    for image_id in sampled:
        h=ev.all_items(image_id,g[image_id],entity);q=ev.all_items(image_id,p[image_id],entity);ht+=len(h);pt+=len(q)
        if entity=="ads":m+=len(ev.pair_items(h,q,entity,mode))
        else:
            for ap in ev.pair_items(ev.all_items(image_id,g[image_id],"ads"),ev.all_items(image_id,p[image_id],"ads"),"ads",mode):
                fn=ev.person_items if entity=="people"else ev.group_items;m+=len(ev.pair_items(fn(image_id,ap.human.data),fn(image_id,ap.predicted.data),entity,mode))
    return ev.detection_summary(ht,pt,m)["f1"]or 0


def bootstrap(g,p,ids):
    rng=random.Random(20260812);out={}
    for mode in["strict","lenient"]:
        for entity in["people","groups"]:
            vals=[f1_for_pages(g,p,[rng.choice(ids)for _ in ids],entity,mode)for _ in range(1000)];vals.sort();out[f"{mode}_{entity}_f1"]={"estimate":f1_for_pages(g,p,ids,entity,mode),"ci95":[vals[24],vals[974]]}
    return out


def usage_summary(items):
    u=Counter();cost=0
    for r in items:
        x=r.get("usage")or{}
        for k in["prompt_tokens","completion_tokens","total_tokens"]:
            try:u[k]+=int(x.get(k)or 0)
            except:pass
        try:cost+=float(x.get("cost")or 0)
        except:pass
    return{"physical_requests":len(items),"successful":sum(bool(r.get("ok"))for r in items),"prompt_tokens":u["prompt_tokens"],"completion_tokens":u["completion_tokens"],"total_tokens":u["total_tokens"],"cost_usd":cost}


def cost_report():
    new=rows(HERE/"output"/"shared_request_ledger.jsonl");old=rows(OLD/"output"/"shared_request_ledger.jsonl");result={}
    structure_flags={c:{r["task_key"]:r.get("flags")or{}for r in rows(HERE/"output"/RUN/c/"structure"/"results.jsonl")}for c in["difficult100","stratified100"]}
    for cohort in["difficult100","stratified100"]:
        # Avoid clever filtering: explicit forms differ between the legacy and new ledgers.
        baseline=[r for r in old if r.get("run_name")=="first100_frozen_v1"and r.get("pipeline")=="p2"]if cohort=="difficult100"else[r for r in new if r.get("run_name")=="baseline_scale_v1"]
        f0=[r for r in new if r.get("run_name")==RUN and r.get("cohort")==cohort and r.get("stage")=="specialist"and r.get("strategy")=="f0_ordinal_direct"]
        struct=[r for r in new if r.get("run_name")==RUN and r.get("cohort")==cohort and r.get("stage")=="structure_verifier"]
        entities=[r for r in new if r.get("run_name")==RUN and r.get("cohort")==cohort and str(r.get("stage","")).startswith("new_entity_")]
        confirmers=[r for r in new if r.get("run_name")==RUN and r.get("cohort")==cohort and r.get("stage")=="group_removal_confirmer_inventory"]
        entity_manifest=json.loads((HERE/"output"/RUN/cohort/"new_entities"/"manifest.json").read_text(encoding="utf-8"))["entities"]
        parent={e["task_key"]:"::".join(e["task_key"].split("::")[:2])for e in entity_manifest}
        entity_id_by_task={e["task_key"]:(e["image_id"],e["advertisement_id"],e["person_id"])for e in entity_manifest}
        def used_entity_tasks(route):
            assembled=rows(HERE/"output"/RUN/cohort/"assembled"/f"{route}.jsonl"); present={(page["image_id"],ad["advertisement_id"],person["person_id"])for page in assembled for ad in page["annotation"].get("advertisements")or[]for person in ad.get("people")or[]}
            return{task for task,identity in entity_id_by_task.items()if identity in present}
        def selected(route):
            flags=lambda task:structure_flags[cohort].get(task,{})
            s=[]
            if route in{"r2","r3","r4"}:
                s += [r for r in struct if(route=="r4"or flags(r.get("task_key","" )).get(route))]
                used=used_entity_tasks(route)
                s += [r for r in entities if r.get("task_key") in used]
            if route in {"r3","r4"}:
                s += confirmers
            return baseline+([]if route=="r0"else f0)+s
        result[cohort]={route:{**usage_summary(selected(route)),"per_page_cost_usd":usage_summary(selected(route))["cost_usd"]/100,"per_page_tokens":usage_summary(selected(route))["total_tokens"]/100}for route in["r0","r1","r2","r3","r4"]}
    return result


def main():
    cohorts={c:gold(c)for c in["difficult100","stratified100"]};allg={k:v for c in cohorts.values()for k,v in c.items()};ids=list(allg);report={"schema_version":"qwen_next_round_results_v1","cohorts":{},"pooled":{},"costs":cost_report()}
    for cohort,g in cohorts.items():
        cid=list(g);report["cohorts"][cohort]={}
        for route in["r0","r1","r2","r3","r4"]:
            p=preds(cohort,route);report["cohorts"][cohort][route]={"evaluation":ev.evaluate_scope(g,p,cid),"bootstrap":bootstrap(g,p,cid)}
    for route in["r0","r1","r2","r3","r4"]:
        p={k:v for c in cohorts for k,v in preds(c,route).items()};evaluation=ev.evaluate_scope(allg,p,ids);pp=matched(allg,p,ids,"people");gp=matched(allg,p,ids,"groups");gold_people=[x for i in ids for x in ev.all_items(i,allg[i],"people")];gold_groups=[x for i in ids for x in ev.all_items(i,allg[i],"groups")]
        report["pooled"][route]={"evaluation":evaluation,"bootstrap":bootstrap(allg,p,ids),"person_categories":{f:categories(pp,gold_people,f,ev.ORDERS.get(f))for f in["face_expression_legibility","gaze_target","smile_present","smile_intensity"]},"group_categories":{f:categories(gp,gold_groups,f,ev.ORDERS.get(f))for f in["expression_legibility_distribution","dominant_gaze","smile_prevalence","dominant_smile_intensity"]}}
    out=HERE/"evaluation"/RUN/"compiled_results.json";out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");print(out)


if __name__=="__main__":main()
