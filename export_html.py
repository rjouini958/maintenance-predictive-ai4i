"""Exporter des copies de lecture des notebooks déjà exécutés, sans recalcul."""
from pathlib import Path
import nbformat
from nbconvert import HTMLExporter
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'lecture'; folder.mkdir(exist_ok=True)
exporter=HTMLExporter(template_name='classic')
exporter.exclude_input_prompt=True
exporter.exclude_output_prompt=True
for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    nb=nbformat.read(path,4)
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
    html,_=exporter.from_notebook_node(nb)
    (folder/(path.stem+'.html')).write_text(html,encoding='utf-8')
    print(path.stem+'.html')
