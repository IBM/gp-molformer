# requires python==2.7, rdkit==2017.09, numpy==1.13, scipy==0.19.1, scikit-learn==0.19.1
from __future__ import unicode_literals
import os.path as op
import pickle

import numpy as np


name = op.join(op.dirname(__file__), "clf_py27.pkl")
with open(name, "rb") as f:
    clf_model = pickle.load(f)
clf_model.kernel = unicode(clf_model.kernel)
clf_model._n_support = clf_model.n_support_
clf_model._probA = clf_model.probA_
clf_model._probB = clf_model.probB_
np.savez(name[:-3] + "npz", **clf_model.__dict__)
