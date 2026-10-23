"""Exécution isolée sans sockets : une nouvelle session IPython par notebook.
Alternative à run_all.py pour les environnements interdisant les sockets Jupyter.
Capture sorties texte, HTML et figures PNG au format notebook standard.
"""
from pathlib import Path
import sys, os, time, json, subprocess
ROOT=Path(__file__).resolve().parents[1]
if len(sys.argv)==1:
    report=[]
    for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
        start=time.time()
        subprocess.run([sys.executable,__file__,str(path)],check=True)
        report.append({'notebook':path.name,'status':'PASS','seconds':round(time.time()-start,2),'method':'Fresh Python process + IPython; all code cells executed in order'})
    (ROOT/'results/execution_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
else:
    import nbformat, base64, io
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from IPython.display import display
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from joblib import parallel_backend
    shell=InteractiveShell.instance()
    shell.run_cell('from IPython.display import display')
    def show(*args,**kwargs):
        for no in plt.get_fignums():
            buf=io.BytesIO(); plt.figure(no).savefig(buf,format='png',bbox_inches='tight')
            display({'image/png':base64.b64encode(buf.getvalue()).decode()},raw=True)
        plt.close('all')
    plt.show=show
    path=Path(sys.argv[1]); nb=nbformat.read(path,as_version=4)
    os.chdir(ROOT/'notebooks')
    count=0
    for cell in nb.cells:
        if cell.cell_type!='code': continue
        count+=1
        print(path.name,'cell',count,flush=True)
        with parallel_backend('threading'), capture_output(display=True) as captured:
            result=shell.run_cell(cell.source,store_history=False)
        if result.error_before_exec or result.error_in_exec:
            print(captured.stdout,captured.stderr)
            raise RuntimeError(f'{path.name} cellule {count} : {result.error_before_exec or result.error_in_exec}')
        outputs=[]
        if captured.stdout: outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr: outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for o in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data',data=o.data,metadata=o.metadata))
        cell.outputs=outputs; cell.execution_count=count
    nbformat.validate(nb); nbformat.write(nb,path)
    print(path.name,'PASS',flush=True)
