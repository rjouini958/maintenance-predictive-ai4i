"""Contrôles indépendants sur les résultats exportés, sans nouvel entraînement."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import nbformat
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from project import load_data,split_indices,FEATURES,MODES,PhysicalFeatures,metrics,rebuild_selected_pipeline,annual_scenario
r=ROOT/'results'
df=load_data(); train,test=split_indices(df)
oof=pd.read_csv(r/'oof_predictions.csv')
pred=pd.read_csv(r/'test_predictions.csv')
choice=json.loads((r/'selection.json').read_text())
score=json.loads((r/'test_metrics.json').read_text())
assert set(oof.source_index)==set(train)
assert set(pred.udi)==set(df.iloc[test].udi)
assert not set(oof.source_index)&set(test)
assert not oof.isna().any().any()
assert len(oof)==8000 and len(pred)==2000
assert np.array_equal(oof.target,df.iloc[train].machine_failure)
assert np.array_equal(pred.target,df.iloc[test].machine_failure)
recomputed=metrics(pred.target,pred.score,choice['threshold'])
for k,v in score.items():
    assert np.isclose(v,recomputed[k]),(k,v,recomputed[k])
assert np.array_equal(pred.prediction,pred.score>=choice['threshold'])
rebuilt=rebuild_selected_pipeline(choice).fit(df.iloc[train][FEATURES],df.iloc[train].machine_failure)
assert np.allclose(rebuilt.predict_proba(df.iloc[test][FEATURES])[:,1],pred.score)
thresholds=pd.read_csv(r/'threshold_search.csv')
feasible=thresholds[(thresholds.recall>=.8)&(thresholds.precision>=1/3)]
if choice['constraints_oof_met']:
    assert np.isclose(choice['oof_metrics']['cost'],feasible.cost.min())
assert not set(FEATURES)&set(MODES+['udi','product_id','machine_failure'])
# Vérifier la conversion physique avec un cas dont la puissance est connue.
x=df[FEATURES].iloc[:1].copy(); x['rot_speed']=60.; x['torque']=1.
assert np.isclose(PhysicalFeatures().transform(x).power_w.iloc[0],2*np.pi)
# Les combinaisons impossibles doivent être refusées, sans faux ROI calculé.
for bad_precision in [0, -0.1, 1.1]:
    try:
        annual_scenario('invalide',.03,.8,bad_precision,3000)
    except ValueError:
        pass
    else:
        raise AssertionError('Précision invalide acceptée')
roi=pd.read_csv(r/'roi_scenarios.csv')
assert np.allclose(roi.gain,roi.cost_without-roi.cost_with)
assert np.allclose(roi.net_gain,roi.gain-55000)
assert np.allclose(roi.roi_pct,100*roi.net_gain/55000)
assert (roi.fp>=0).all() and (roi.fn>=0).all()
for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    nb=nbformat.read(path,4); nbformat.validate(nb)
    for c in nb.cells:
        if c.cell_type=='code':
            assert c.execution_count is not None
            assert not any(o.output_type=='error' for o in c.outputs)
print('PASS : partitions, hors-pli, absence de champs de fuite, seuil, métriques, unités, ROI et notebooks exécutés.')
(r/'verification.txt').write_text('PASS — partitions, hors-pli, exclusion des fuites, choix du seuil, métriques test, conversion W, calculs ROI, validation nbformat et absence de cellules en erreur.\n',encoding='utf-8')
