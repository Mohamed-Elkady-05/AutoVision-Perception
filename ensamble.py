import numpy as np
from sklearn.ensemble import RandomForestClassifier ,  GradientBoostingClassifier, AdaBoostClassifier ,VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from sklearn.preprocessing import StandardScaler

class Ensemble_models():
  def __init__(self, n_estimators=100, use_scaler=True):
    self.n_estimators = n_estimators
    self.use_scaler = use_scaler
    self.scaler = StandardScaler() if use_scaler else None
    
    # Bagging
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    # Boosting
    gb = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    ada = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
    voting = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('ada', ada),
                ('gb', gb)
            ],
            voting='soft'  
        )

    self.models = [rf,voting]

    
  
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
      return self.models[0].predict(Xp)
<<<<<<< HEAD
    elif model == 'voting':
      return self.models[1].predict(Xp)
=======
    elif model == 'gb':
      return self.models[1].predict(Xp)
    elif model == 'ada':
      return self.models[2].predict(Xp)
>>>>>>> 982a40c62cf16eacca85f32256eaaf7cc4bb4d5d
    
  def evaluate(self, X, y, model='voting'):
    y_pred = self.predict(X, model)
    return {
        'accuracy': accuracy_score(y, y_pred),
<<<<<<< HEAD
        'precision': precision_score(y, y_pred, average = 'macro'),
        'recall': recall_score(y, y_pred, average = 'macro'),
        'f1': f1_score(y, y_pred, average = 'macro')
    }, y_pred
=======
        'precision': precision_score(y, y_pred, average='macro'),
        'recall': recall_score(y, y_pred, average='macro'),
        'f1': f1_score(y, y_pred, average='macro')
    }
>>>>>>> 982a40c62cf16eacca85f32256eaaf7cc4bb4d5d
