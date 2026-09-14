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
from chapter06_analysis import Analysis, foundation, gender_age, industry_context, relational, finalize
from chapter06_support import audit_bootstrap
from chapter06_literature import literature_comparisons

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
        ("Corpus and historical development","Smile presence uses yes/(yes + no). Advertisement summaries distinguish any smile, the mean within-ad smile share, and all recorded faces smiling. Indeterminate advertisement-level statuses remain missing. Other labels remain explicit. Intensity uses smiling faces. Whole issues are resampled within year, using a common seed and 2,000 replicates.","foundation(c)\ndisplay(pd.read_csv(c.work / 'numerical-data/ch06-historical-smiling--ads.csv'))\nc.show(1, 6)"),
        ("Gender and age","Gender differences are feminine minus masculine, reported in percentage points. Age and intensity retain their ordered recorded categories. Sparse age cells are omitted from comparative plots, with counts retained in supporting data.","gender_age(c)\nc.show(7, 12)"),
        ("Industry and depiction context","The six industry facets are selected by advertisement count before smile rates are compared. Smile-rate points with fewer than 30 assessable faces are suppressed while their exact counts remain in the numerical companion.","industry_context(c)\nc.show(13, 14)"),
        ("Comparisons with existing research","Jofre and Cole's published ratio and around-1970 observation pool news and ads; our comparison uses advertising only. The financial-ad study cited as unda2024GenderStereotypes overlaps the underlying Economist archive; the unique largest face is only a central-person proxy. Broad/laughter-like intensity is compared both among all assessed faces and conditional on smiling. Every interval uses the common issue/year draws; published benchmarks are not treated as independent observations.","literature_comparisons(c)\nc.show(28, 30)"),
        ("Focused follow-up: the cosmetics exception","The same industry contrast is recomputed separately for photographs and other depictions.","display(pd.read_csv(c.work / 'cosmetics-depiction-followup.csv'))")]),
      "chapter06_relational_analysis":notebook("Relational smiling and visual prominence",[
        ("Within-ad comparisons and visual prominence","Smile concentration is first evaluated among fully assessed advertisements with at least two faces, including a descriptive independence benchmark that retains decade-specific face smile shares and observed face counts per ad. Adult pairs then require exactly two recorded faces, one feminine and one masculine, both adults with assessable smiles. Largest-face ownership in mixed-gender advertisements is compared with each ad's feminine numerical share; exact ties split credit.","relational(c)\nc.show(16, 21)"),
        ("Bootstrap audit","The audit checks year-wise counts, explicit resampled rows and interval stability with an independent 5,000-draw run.","audit_bootstrap(c)\ndisplay(pd.read_csv(c.work / 'bootstrap-audit.csv'))"),
        ("Artifact checks","Check the 16 retained chapter outputs and complete numerical companion. Source notebooks remain output-free in Git; executed copies stay in private output storage.","finalize(c)")])}
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
