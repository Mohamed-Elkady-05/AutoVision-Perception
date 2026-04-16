import numpy as np
from sklearn.ensemble import RandomForestClassifier ,  GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from sklearn.preprocessing import StandardScaler

class ensemble():
  def __init__(self, n_estimators=100, use_scaler=True):
    self.n_estimators = n_estimators
    self.use_scaler = use_scaler
    self.scaler = StandardScaler() if use_scaler else None
    
    # Bagging
    self.rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    # Boosting
    self.gb = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    self.ada = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)

    self.models = [self.rf, self.gb, self.ada]
  
  def _prepare(self, X, fit=False):
        if self.scaler is not None:
            if fit:
                return self.scaler.fit_transform(X)
            return self.scaler.transform(X)
        return X

  def fit(self, X, y):
    X = self._prepare(X, fit=True)
    for model in self.models:
      model.fit(X, y)
  
  def predict(self, X , model='rf'):
    Xp = self._prepare(X,False)

    if model == 'rf':
      return self.model[0].predict(Xp)
    elif model == 'gb':
      return self.model[1].predict(Xp)
    elif model == 'ada':
      return self.model[2].predict(Xp)
    
  def evaluate(self, X, y, model='rf'):
    y_pred = self.predict(X, model)
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred)
    }
