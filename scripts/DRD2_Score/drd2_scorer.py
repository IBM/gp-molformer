# Modified from https://github.com/wengong-jin/iclr19-graph2graph/blob/master/props/drd2_scorer.py
#!/usr/bin/env python
import numpy as np
from rdkit import rdBase
from rdkit.Chem import AllChem
from sklearn import svm
import os.path as op

rdBase.DisableLog("rdApp.error")

"""Scores based on an ECFP classifier for activity."""

clf_model = None
def load_model():
    global clf_model
    clf_model = svm.SVC()
    name = op.join(op.dirname(__file__), "clf_py27.npz")
    clf_model.__dict__.update(np.load(name, allow_pickle=True))


def get_score(mol):
    if clf_model is None:
        load_model()

    # mol = Chem.MolFromSmiles(smile)
    if mol:
        fp = fingerprints_from_mol(mol)
        score = clf_model.predict_proba(fp)[:, 1]
        return float(score)
    return 0.0


def fingerprints_from_mol(mol):
    fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    size = 2048
    nfp = np.zeros((1, size), np.int32)
    for idx, v in fp.GetNonzeroElements().items():
        nidx = idx % size
        nfp[0, nidx] += int(v)
    return nfp
