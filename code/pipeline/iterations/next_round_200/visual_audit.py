"""Evaluation-only visual boards for FER and structural critical cases."""

from __future__ import annotations

import importlib.util, json, sys, textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT=Path(r".");HERE=ROOT/"qwen_iteration"/"next_round_200";OLD=ROOT/"qwen_iteration"/"first100_pipeline_matrix";RUN="scale_v1";OUT=HERE/"evaluation"/RUN/"visual_audit"
sys.path.insert(0,str(OLD));spec=importlib.util.spec_from_file_location("ev",OLD/"evaluate.py");assert spec and spec.loader;ev=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ev;spec.loader.exec_module(ev)


def rows(p):return[json.loads(x)for x in p.read_text(encoding="utf-8").splitlines()if x.strip()]
def gold(cohort):
    p=ROOT/"annotation_results"/("test_collection_200_difficult_joined_v1_2026-08-02.json"if cohort=="difficult100"else"economist_decade_face_count_stratified_200_seed20260812_min2_52123740_2026-08-12.json");r=json.loads(p.read_text(encoding="utf-8"))["annotations"]
    if cohort=="difficult100":keep={x["image_id"]for x in json.loads((HERE/"data"/"manifest_difficult100.json").read_text())["images"]};r=[x for x in r if x.get("assignment_code")=="79201188"and x["image_id"]in keep]
    else:r=sorted(r,key=lambda x:int(x["sort_order"]))[:100]
    return{x["image_id"]:x["payload"]for x in r}
def pred(cohort,route):return{x["image_id"]:x["annotation"]for x in rows(HERE/"output"/RUN/cohort/"assembled"/f"{route}.jsonl")}
def pairs(g,p,image_id,entity):
    out=[];fn=ev.person_items if entity=="people"else ev.group_items
    for ap in ev.pair_items(ev.all_items(image_id,g[image_id],"ads"),ev.all_items(image_id,p[image_id],"ads"),"ads","strict"):out+=ev.pair_items(fn(image_id,ap.human.data),fn(image_id,ap.predicted.data),entity,"strict")
    return out


def fer_cases():
    cases=[]
    for cohort in["difficult100","stratified100"]:
        g=gold(cohort);r0=pred(cohort,"r0");r1=pred(cohort,"r1");specs={x["task_key"]:x for x in rows(HERE/"output"/RUN/cohort/"specialists.jsonl")if x.get("ok")}
        for image_id in g:
            a={(x.human.item_id):x for x in pairs(g,r0,image_id,"people")};b={(x.human.item_id):x for x in pairs(g,r1,image_id,"people")}
            for hid in a.keys()&b.keys():
                x,y=a[hid],b[hid];task=f"{image_id}::{y.predicted.item_id}";s=specs.get(task)
                if not s:continue
                labels={f:(ev.normalize_value(x.human.data.get(f),f),ev.normalize_value(x.predicted.data.get(f),f),ev.normalize_value(y.predicted.data.get(f),f))for f in["face_expression_legibility","gaze_target","smile_present","smile_intensity"]}
                tag=None
                if labels["face_expression_legibility"][1]!=labels["face_expression_legibility"][0]and labels["face_expression_legibility"][2]==labels["face_expression_legibility"][0]:tag="legibility corrected"
                elif labels["smile_present"][1]!=labels["smile_present"][0]and labels["smile_present"][2]==labels["smile_present"][0]:tag="smile corrected"
                elif labels["face_expression_legibility"][1]==labels["face_expression_legibility"][0]and labels["face_expression_legibility"][2]!=labels["face_expression_legibility"][0]:tag="legibility worsened"
                elif labels["gaze_target"][0]=="off_frame_or_scene_direction"and labels["gaze_target"][2]=="viewer_camera":tag="persistent gaze error"
                if tag:cases.append({"cohort":cohort,"image_id":image_id,"human_id":hid,"pred_id":y.predicted.item_id,"tag":tag,"labels":labels,"image_path":s["image_path"]})
    selected=[]
    for tag,n in[("legibility corrected",3),("smile corrected",2),("legibility worsened",2),("persistent gaze error",3)]:selected += [x for x in cases if x["tag"]==tag][:n]
    return selected


def make_fer_board(cases):
    font=ImageFont.load_default();w=1400;tile_h=480;board=Image.new("RGB",(w,tile_h*len(cases)),"white");draw=ImageDraw.Draw(board)
    for i,c in enumerate(cases):
        with Image.open(c["image_path"])as im:pic=im.convert("RGB");pic.thumbnail((900,430));y=i*tile_h;board.paste(pic,(0,y+35));d=ImageDraw.Draw(board);d.text((5,y+5),f"{c['tag']} | {c['cohort']} | {c['image_id']} | {c['pred_id']}",fill="black",font=font)
        lines=[]
        for f,v in c["labels"].items():lines+=textwrap.wrap(f"{f}: GOLD={v[0]} | R0={v[1]} | R1/F0={v[2]}",55)
        for j,line in enumerate(lines):d.text((915,y+45+j*18),line,fill="black",font=font)
    path=OUT/"fer_critical_cases.jpg";path.parent.mkdir(parents=True,exist_ok=True);board.save(path,"JPEG",quality=94);return path


def image_path(cohort,image_id):
    directory=(ROOT/"code"/"test_collection_200_difficult_joined_pages")if cohort=="difficult100"else(ROOT/"master_thesis"/"data"/"images"/"full_pages_1940_2007_joined");name=next(x["filename"]for x in json.loads((HERE/"data"/f"manifest_{cohort}.json").read_text())["images"]if x["image_id"]==image_id);return directory/name


def structure_cases():
    allcases=[]
    for cohort in["difficult100","stratified100"]:
        g=gold(cohort);r1=pred(cohort,"r1");r2=pred(cohort,"r2");r3=pred(cohort,"r3")
        for image_id in g:
            counts={route:{e:len(pairs(g,p,image_id,e))for e in["people","groups"]}for route,p in[("r1",r1),("r2",r2),("r3",r3)]}
            tag=None
            if len(ev.all_items(image_id,r3[image_id],"groups"))<len(ev.all_items(image_id,r1[image_id],"groups")):tag="R3 removed baseline group"
            elif counts["r2"]["people"]>counts["r1"]["people"]:tag="R2 person gain"
            elif len(ev.all_items(image_id,r2[image_id],"people"))>len(ev.all_items(image_id,r1[image_id],"people"))and counts["r2"]["people"]==counts["r1"]["people"]:tag="R2 added unmatched face"
            elif counts["r3"]["groups"]>counts["r1"]["groups"]:tag="R3 group gain"
            elif counts["r3"]["groups"]<counts["r1"]["groups"]:tag="R3 group loss"
            if tag:allcases.append({"cohort":cohort,"image_id":image_id,"tag":tag,"counts":counts})
    selected=[]
    for tag,n in[("R3 removed baseline group",6),("R2 person gain",3),("R2 added unmatched face",3),("R3 group gain",3),("R3 group loss",2)]:selected += [x for x in allcases if x["tag"]==tag][:n]
    return selected


def draw_boxes(pic,ann,color,entity,label):
    d=ImageDraw.Draw(pic);items=ev.all_items("x",ann,entity)
    for item in items:
        if not item.box:continue
        b=[item.box[0]*pic.width,item.box[1]*pic.height,item.box[2]*pic.width,item.box[3]*pic.height];d.rectangle(b,outline=color,width=4);d.text((b[0],b[1]),label,fill=color,font=ImageFont.load_default())


def make_structure_board(cases):
    font=ImageFont.load_default();tile_w=700;tile_h=820;cols=2;board=Image.new("RGB",(cols*tile_w,((len(cases)+1)//2)*tile_h),"white")
    cache={}
    for i,c in enumerate(cases):
        cohort=c["cohort"];g=cache.setdefault((cohort,"g"),gold(cohort));r1=cache.setdefault((cohort,"r1"),pred(cohort,"r1"));r4=cache.setdefault((cohort,"r4"),pred(cohort,"r4"));
        with Image.open(image_path(cohort,c["image_id"]))as im:pic=im.convert("RGB");pic.thumbnail((680,740));draw_boxes(pic,g[c["image_id"]],"lime","people","G-P");draw_boxes(pic,g[c["image_id"]],"green","groups","G-G");draw_boxes(pic,r1[c["image_id"]],"blue","people","R1-P");draw_boxes(pic,r1[c["image_id"]],"cyan","groups","R1-G");draw_boxes(pic,r4[c["image_id"]],"red","people","R4-P");draw_boxes(pic,r4[c["image_id"]],"magenta","groups","R4-G")
        x=(i%cols)*tile_w;y=(i//cols)*tile_h;board.paste(pic,(x+(tile_w-pic.width)//2,y+55));d=ImageDraw.Draw(board);d.text((x+5,y+5),f"{c['tag']} | {cohort} | {c['image_id']}",fill="black",font=font);d.text((x+5,y+25),f"matches {c['counts']} | gold green, R1 blue, R4 red",fill="black",font=font)
    path=OUT/"structure_critical_cases.jpg";path.parent.mkdir(parents=True,exist_ok=True);board.save(path,"JPEG",quality=94);return path


def main():
    fc=fer_cases();sc=structure_cases();fp=make_fer_board(fc);sp=make_structure_board(sc);(OUT/"case_manifest.json").write_text(json.dumps({"fer":fc,"structure":sc},indent=2,ensure_ascii=False)+"\n",encoding="utf-8");print(fp);print(sp);print(len(fc),len(sc))


if __name__=="__main__":main()
