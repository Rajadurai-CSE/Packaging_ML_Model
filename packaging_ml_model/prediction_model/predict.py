import pandas as pd
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset,load_pipeline
import prediction_model.processing.preprocessing as pp
import prediction_model.pipeline as pp

def generate_prediction():
  df = load_dataset(config.TEST_FILE)
  model = load_pipeline()
  prediction = model.predict(df[config.FEATURES])
  funct = lambda x: 'Y' if x==1 else 'N'
  result = list(funct(prediction))
  return result

if __name__ == 'main':
  generate_prediction()
