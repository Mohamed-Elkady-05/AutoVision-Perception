import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from sklearn.preprocessing import StandardScaler

class Ensemble_models():
  def __init__(self, n_estimators=100, use_scaler=True, max_depth=5):
    self.n_estimators = n_estimators
    self.use_scaler = use_scaler
    self.scaler = StandardScaler() if use_scaler else None
    
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, max_depth=max_depth, n_jobs=-1)
    gb = HistGradientBoostingClassifier(max_iter=self.n_estimators, max_depth=max_depth, random_state=42)
    ada = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
    
    self.voting = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('ada', ada),
                ('gb', gb)
            ],
            voting='soft',
            n_jobs=-1 
        )
  
  def _prepare(self, X, fit=False):
        if self.scaler is not None:
            if fit:
                return self.scaler.fit_transform(X)
            return self.scaler.transform(X)
        return X

  def fit(self, X, y):
    X = self._prepare(X, fit=True)
    self.voting.fit(X, y)
  
  def predict(self, X, model='rf'):
    Xp = self._prepare(X, False)

    if model == 'rf':
      return self.voting.named_estimators_['rf'].predict(Xp)
    elif model == 'voting':
      return self.voting.predict(Xp)
    
  def evaluate(self, X, y, model='voting'):
    y_pred = self.predict(X, model)
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y, y_pred, average='macro', zero_division=0)
    }, y_pred
