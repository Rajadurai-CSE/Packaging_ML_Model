from sklearn.pipeline import Pipeline
from prediction_model.config import config
import prediction_model.processing.preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from math import log

pipeline = Pipeline(
  ('Mean Imputation',pp.MeanImputer(config.NUM_FEATURES)),
  ('Mode Imputation',pp.ModeImputer(config.CAT_FEATURES)),
  ('Modify Features',pp.DomainProcessing(config.FEATURE_TO_MODIFY,config.FEATURE_TO_ADD)),
  ('Delete Features',pp.DropColumn(config.DROP_FEATURE)),
  ('Log Transformation',pp.logprocessing(config.LOG_FEATURES)),
  ('MinMaxScalar',MinMaxScaler()),
  ('Logisitic Classifier',LogisticRegression(random_state=0))
)