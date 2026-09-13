"""Build clean notebook sources and optionally execute private review copies."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

HERE=Path(__file__).resolve().parent

SETUP='''from pathlib import Path
import sys
import pandas as pd
from IPython.display import display

# Locate the analysis module from either the standalone or thesis checkout.
candidates = []
for base in [Path.cwd(), *Path.cwd().parents]:
    candidates.extend([base, base / "code/scripts", base / "master_thesis-public/code/scripts", base / "analysis/code/scripts"])
code_dir = next(path for path in candidates if (path / "chapter06_analysis.py").is_file())
sys.path.insert(0, str(code_dir))
from chapter06_analysis import Analysis, foundation, gender_age, industry_context, relational, models, finalize
from chapter06_support import correlations, audit_bootstrap

# CH06_INPUT, CH06_THESIS and CH06_WORK override the local layout defaults.
c = Analysis()
display(pd.DataFrame([c.results["input"]]))
'''


def notebook(title,sections):
    cells=[nbformat.v4.new_markdown_cell("# "+title),nbformat.v4.new_code_cell(SETUP)]
    for heading,method,code in sections:
        cells.append(nbformat.v4.new_markdown_cell("## "+heading+"\n\n"+method))
        cells.append(nbformat.v4.new_code_cell(code))
    nb=nbformat.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":sys.version.split()[0]}})
    for i,cell in enumerate(nb.cells):
        cell.id=f"ch06-{i:02d}"
    return nb


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--thesis",type=Path,required=True)
    parser.add_argument("--work",type=Path,required=True)
    args=parser.parse_args()
    args.work=args.work.resolve();args.work.mkdir(parents=True,exist_ok=True)
    definitions={
      "chapter06_historical_analysis":notebook("Historical smiling: corpus, gender, age and industry",[
        ("Corpus and historical development","Smile presence uses yes/(yes + no). Other labels remain explicit. Intensity uses smiling faces. Whole issues are resampled within year, using a common seed and 2,000 replicates.","foundation(c)\nc.show(1, 6)"),
        ("Gender and age","Gender differences are feminine minus masculine, reported in percentage points. Age and intensity retain their ordered recorded categories. Sparse age cells are omitted from comparative plots, with counts retained in supporting data.","gender_age(c)\nc.show(7, 12)"),
        ("Industry and depiction context","Industry standardisation uses six industries with at least 20 assessable faces in each gender and decade from 1950. Observed and standardised lines share the same sample. Ad composition is assigned before dropping unassessable smile records.","industry_context(c)\nc.show(13, 16)"),
        ("Focused follow-up: the cosmetics exception","The same industry contrast is recomputed separately for photographs and other depictions.","display(pd.read_csv(c.work / 'cosmetics-depiction-followup.csv'))")]),
      "chapter06_relational_analysis":notebook("Relational smiling, visual prominence and adjusted trends",[
        ("Within-ad comparisons and spatial presentation","Adult pairs require exactly two recorded faces, one feminine and one masculine, both adults with assessable smiles. Face area is relative to ad area. Largest-face ownership is compared with each ad's feminine numerical share; exact ties split credit.","relational(c)\nc.show(17, 23)"),
        ("Adjusted smile trajectories","The main logistic model uses decade-by-gender, age, industry, depiction type, relative face area and co-presence. An intermediate model controls age alone; all three specifications share the same sample. Legibility is checked through restricted samples because its labels are strongly coupled to smile presence. Covariance is clustered by issue; marginal-probability intervals use the delta method.","models(c)\nc.show(24, 26)"),
        ("Simple correlations and bootstrap audit","Four Spearman associations are compared within gender, with the same issue/year bootstrap and ranks recomputed per draw. The audit checks year-wise counts, explicit resampled rows, tied-rank handling and interval stability with an independent 5,000-draw run.","correlations(c)\nc.show(27, 27)\naudit_bootstrap(c)\ndisplay(pd.read_csv(c.work / 'bootstrap-audit.csv'))"),
        ("Model and artifact checks","Check convergence, model sample sizes and the 20 retained chapter outputs and complete numerical companion. Source notebooks remain output-free in Git; executed copies stay in private output storage.","display(pd.read_csv(c.work / 'model-fit.csv'))\nfinalize(c)")])}
    for name,nb in definitions.items():
        nbformat.validate(nb)
        nbformat.write(nb,HERE/(name+".ipynb"))
    if not args.execute:
        return
    # A private, temporary kernel specification avoids modifying user Jupyter setup.
    kernel_root=args.work/"jupyter"
    kernel=kernel_root/"kernels/chapter06"
    kernel.mkdir(parents=True,exist_ok=True)
    (kernel/"kernel.json").write_text(json.dumps({"argv":[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"],"display_name":"Chapter 06 analysis","language":"python"}),encoding="utf-8")
    os.environ["JUPYTER_PATH"]=str(kernel_root)
    os.environ["JUPYTER_RUNTIME_DIR"]=str(args.work/"jupyter-runtime")
    os.environ["IPYTHONDIR"]=str(args.work/"ipython")
    os.environ["CH06_INPUT"]=str(args.input.resolve())
    os.environ["CH06_THESIS"]=str(args.thesis.resolve())
    os.environ["CH06_WORK"]=str(args.work)
    for name,nb in definitions.items():
        print("Executing "+name,flush=True)
        client=NotebookClient(nb,timeout=600,kernel_name="chapter06",resources={"metadata":{"path":str(HERE)}})
        client.execute()
        nb.metadata.kernelspec={"display_name":"Python 3","language":"python","name":"python3"}
        assert all(o.output_type!="error" for cell in nb.cells if cell.cell_type=="code" for o in cell.outputs)
        nbformat.validate(nb)
        nbformat.write(nb,args.work/(name+".ipynb"))
        print("Completed "+name,flush=True)


if __name__=="__main__":
    main()
