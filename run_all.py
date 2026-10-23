"""Exécuter les trois notebooks dans trois noyaux neufs, puis vérifier leurs sorties."""
from pathlib import Path
import sys, time, json, os
from jupyter_client import KernelManager
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
ROOT=Path(__file__).resolve().parents[1]
# Noyau explicite avec le Python qui exécute ce script (compatible environnement virtuel).
kernel_dir=ROOT/'.runtime_kernel'
kernel_dir.mkdir(exist_ok=True)
(kernel_dir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Projet AI4I','language':'python'}))
manager=KernelSpecManager()
manager.install_kernel_spec(str(kernel_dir),kernel_name='ai4i-project',user=True)
report=[]
for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    start=time.time()
    print('Exécution :',path.name,flush=True)
    nb=nbformat.read(path,as_version=4)
    km = KernelManager(kernel_name='ai4i-project', transport='ipc') if os.environ.get('AI4I_IPC') == '1' else None
    NotebookClient(nb,km=km,timeout=1800,kernel_name='ai4i-project',resources={'metadata':{'path':str(ROOT/'notebooks')}}).execute()
    nb.metadata.kernelspec={'name':'python3','language':'python','display_name':'Python 3'}
    nbformat.validate(nb)
    assert all(o.output_type!='error' for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]))
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
    nbformat.write(nb,path)
    report.append({'notebook':path.name,'status':'PASS','seconds':round(time.time()-start,2),'code_cells':sum(c.cell_type=='code' for c in nb.cells)})
    print(report[-1],flush=True)
(ROOT/'results/execution_report.json').write_text(json.dumps(report,indent=2))
