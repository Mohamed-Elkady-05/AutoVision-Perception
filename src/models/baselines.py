import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class BaselineModels:
    """
    A wrapper class for baseline classifiers.
    Currently supports KNN and GNB.
    """

    def __init__(self, knn_k=5, use_scaler=True, use_pca=False, pca_components=50):
        self.knn_k = knn_k
        self.use_scaler = use_scaler
        self.use_pca = use_pca
        self.pca_components = pca_components

        self.scaler = StandardScaler() if use_scaler else None
        self.pca = PCA(n_components=pca_components) if use_pca else None

        # supported classifiers
        self.knn = KNeighborsClassifier(n_neighbors=self.knn_k)
        self.nb = GaussianNB()

        self._fitted = False

    def _prepare(self, X, fit=False):
        Xp = X
        if self.scaler is not None:
            if fit:
                Xp = self.scaler.fit_transform(Xp)
            else:
                Xp = self.scaler.transform(Xp)
        if self.pca is not None:
            if fit:
                Xp = self.pca.fit_transform(Xp)
            else:
                Xp = self.pca.transform(Xp)
        return Xp

    def fit(self, X, y):
        Xp = self._prepare(X, fit=True)
        self.knn.fit(Xp, y)
        self.nb.fit(Xp, y)
        self._fitted = True

    def predict(self, X, model='knn'):
        if not self._fitted:
            raise RuntimeError('Models not fitted. Call fit() first.')
        Xp = self._prepare(X, fit=False)
        if model == 'knn':
            return self.knn.predict(Xp)
        elif model == 'nb' or model == 'naivebayes':
            return self.nb.predict(Xp)
        else:
            raise ValueError('Unknown model: choose "knn" or "nb"')

    def evaluate(self, X, y, model='knn', average='macro'):
        y_pred = self.predict(X, model=model)
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average=average, zero_division=0),
            'recall': recall_score(y, y_pred, average=average, zero_division=0),
            'f1': f1_score(y, y_pred, average=average, zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred),
        }
        return metrics
