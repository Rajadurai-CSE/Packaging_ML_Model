#Custom Transformation

from sklearn.base import BaseEstimator,TransformerMixin

from math import log

class MeanImputer(BaseEstimator,TransformerMixin):
  def __init__(self,variables=None):
    self.variables = variables

  def fit(self,X,y=None):
    self.dict_ = {}
    for col in self.variables:
      self.dict_[col] = X[col].mean()
    return self
  
  def transform(self,X):
    for col in self.variables:
      X[col].fillna(self.dict_[col],inplace=True)
    return X


class ModeImputer(BaseEstimator,TransformerMixin):
  def __init__(self,variables=None):
    self.variables = variables

  def fit(self,X,y=None):
    self.dict_ = {}
    for col in self.variables:
      self.dict_[col] = X[col].mode()[0]
    return self
  
  def transform(self,X):
    for col in self.variables:
      X[col].fillna(self.dict_[col],inplace=True)
    return X
  


class DropColumn(BaseEstimator,TransformerMixin):
  def __init__(self,variables_to_drop=None):
    self.variables_to_drop = variables_to_drop

  def fit(self,X,y=None):
    return self
  
  def transform(self,X):
    X.drop([self.variables_to_drop],axis=1,inplace = True)
    return X
  



class DomainProcessing(BaseEstimator,TransformerMixin):
  def __init__(self,variable_to_modify=None,variable_to_add = None):
    self.variable_to_modify = variable_to_modify
    self.variable_to_add = variable_to_add

  def fit(self,X,y=None):
    return self
  
  def transform(self,X):
    X[self.variable_to_modify] = X[self.variable_to_add] + X[self.variable_to_modify]
    return X
  

class logprocessing(BaseEstimator,TransformerMixin):
  def __init__(self,variables = None):
    self.variables = variables

  def fit(self,X,y=None):
    return self
  
  def transform(self,X):
    for col in self.variables:
      X[col] = X[col].apply(lambda x: log(x))
    return X
  


