import pandas as pd
import numpy as np
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset,save_pipeline
import prediction_model.processing.preprocessing as pp
import  prediction_model.pipeline as pp

def training():
  df = load_dataset(config.TRAIN_FILE)
  target = df[config.TARGET_FEATURE].map({'N':0,'Y':1})
  pp.pipeline.fit(df,target)
  save_pipeline(pp.pipeline)

if __name__ == 'main':
  training()
