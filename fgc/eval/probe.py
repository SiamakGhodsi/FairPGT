"""How much sensitive information do the node embeddings carry?

Logistic-regression probe predicting S from the embeddings, 5-fold stratified CV,
balanced accuracy (0.5 = chance for two balanced-weighted groups). The embeddings are
standardized; the same probe is applied to every method that exposes embeddings.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sensitive_probe(Z: np.ndarray, S: np.ndarray, seed: int = 0, max_n: int = 20000) -> float:
    rng = np.random.default_rng(seed)
    if len(S) > max_n:
        idx = rng.choice(len(S), max_n, replace=False)
        Z, S = Z[idx], S[idx]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, Z, S, cv=cv, scoring="balanced_accuracy").mean())
