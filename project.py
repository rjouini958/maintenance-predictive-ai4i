"""Fonctions partagées : données, pipeline et coûts. Aucune lecture du test à l'import."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
RENAME = {'UDI':'udi','Product ID':'product_id','Type':'type','Air temperature [K]':'air_temp','Process temperature [K]':'process_temp','Rotational speed [rpm]':'rot_speed','Torque [Nm]':'torque','Tool wear [min]':'tool_wear','Machine failure':'machine_failure'}
RAW_NUM = ['air_temp','process_temp','rot_speed','torque','tool_wear']
FEATURES = ['type'] + RAW_NUM
MODES = ['TWF','HDF','PWF','OSF','RNF']
ENGINEERED = ['temp_gap','power_w','wear_torque']

def load_data():
    path = ROOT / 'data/raw/ai4i2020.csv'
    meta = ROOT / 'data/source.json'
    if meta.exists():
        import json
        assert hashlib.sha256(path.read_bytes()).hexdigest() == json.loads(meta.read_text())['sha256'], 'Le CSV a changé : vérifier sa provenance.'
    df = pd.read_csv(path).rename(columns=RENAME)
    assert df.shape == (10000,14) and df.machine_failure.sum() == 339
    assert df.udi.is_unique and set(df.machine_failure.unique()) == {0,1}
    return df

def split_indices(df):
    return train_test_split(np.arange(len(df)), test_size=.2, stratify=df.machine_failure, random_state=SEED)

class PhysicalFeatures(TransformerMixin, BaseEstimator):
    """Relations physiques fixées avant apprentissage, sans accès à la cible."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        out = X[FEATURES].copy()
        out['temp_gap'] = out.process_temp - out.air_temp
        out['power_w'] = out.torque * out.rot_speed * (2*np.pi/60)
        out['wear_torque'] = out.tool_wear * out.torque
        return out

def make_pipeline(model):
    num = Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    cat = Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
    prep = ColumnTransformer([('numeric',num,RAW_NUM+ENGINEERED),('category',cat,['type'])])
    return Pipeline([('physics',PhysicalFeatures()),('prep',prep),('model',model)])

def metrics(y, p, threshold=.5):
    pred = np.asarray(p) >= threshold
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
        recall=float(recall_score(y,pred,zero_division=0)),precision=float(precision_score(y,pred,zero_division=0)),
        f1=float(f1_score(y,pred,zero_division=0)),ap=float(average_precision_score(y,p)),accuracy=float(accuracy_score(y,pred)),
        cost=float(fn*3000+fp*150+tp*500))

def annual_scenario(name, prevalence, recall, precision, failure_cost, cycles=5000, project_cost=55000, effectiveness=1.):
    """Les effectifs sont des espérances annuelles, pas des observations industrielles."""
    if not (0 <= prevalence <= 1 and 0 <= recall <= 1 and 0 < precision <= 1 and 0 <= effectiveness <= 1):
        raise ValueError('Prévalence/rappel/efficacité entre 0 et 1 ; précision dans ]0, 1].')
    if cycles <= 0 or project_cost <= 0 or failure_cost < 0:
        raise ValueError('Cycles et coût projet positifs ; coût de panne non négatif.')
    tp = cycles*prevalence*recall
    fn = cycles*prevalence-tp
    fp = tp*(1/precision-1) if precision else 0.
    if fp > cycles*(1-prevalence):
        raise ValueError('Scénario irréalisable : trop de faux positifs.')
    without = cycles*prevalence*failure_cost
    with_model = (fn+(1-effectiveness)*tp)*failure_cost+tp*500+fp*150
    gain = without-with_model
    return dict(scenario=name,recall=recall,precision=precision,effectiveness=effectiveness,failure_cost=failure_cost,tp=tp,fp=fp,fn=fn,cost_without=without,cost_with=with_model,gain=gain,net_gain=gain-project_cost,roi_pct=100*(gain-project_cost)/project_cost,payback_months=12*project_cost/gain if gain>0 else np.nan)


def rebuild_selected_pipeline(selection):
    """Reconstruire le choix gelé sans désérialiser un modèle binaire.

    Les hyperparamètres viennent uniquement du notebook 02 ; aucune recherche
    ne peut être déclenchée par cette fonction lors de l'évaluation du test.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    factories = {
        'Baseline': lambda: DummyClassifier(strategy='most_frequent'),
        'Logistique': lambda: LogisticRegression(max_iter=2000,class_weight='balanced',random_state=SEED),
        'Logistique réglée': lambda: LogisticRegression(max_iter=2000,class_weight='balanced',random_state=SEED),
        'Arbre': lambda: DecisionTreeClassifier(random_state=SEED),
        'Arbre réglé': lambda: DecisionTreeClassifier(class_weight='balanced',random_state=SEED),
        'Random Forest': lambda: RandomForestClassifier(n_estimators=100,random_state=SEED,n_jobs=1),
        'Random Forest réglée': lambda: RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=SEED,n_jobs=1),
    }
    if selection['model'] not in factories:
        raise ValueError('Modèle inconnu dans la sélection.')
    return make_pipeline(factories[selection['model']]()).set_params(**selection['params'])
