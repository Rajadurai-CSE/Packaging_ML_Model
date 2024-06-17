import os
import pandas as pd
import joblib
from prediction_model.config import config

#Load Datasets
def load_dataset(file_name):
  filepath = os.path.join(config.DATAPATH,file_name)
  df = pd.read_csv(filepath)
  return df

#Serialization
def save_pipeline(pipeline_to_save):
  save_path = os.path.join(config.SAVED_MODEL,config.MODEL_NAME)
  joblib.dump(pipeline_to_save,save_path)
  print(f'The model has been saved under the name {config.MODEL_NAME}')


#Deserialization
def load_pipeline():
  load_path = os.path.join(config.SAVED_MODEL,config.MODEL_NAME)
  model = joblib.load(load_path)
  return model

