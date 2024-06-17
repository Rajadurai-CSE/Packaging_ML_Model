#Configuration File

import os
import pathlib
import prediction_model

ROOT_DIRECTORY = pathlib.Path(prediction_model.__file__)

DATAPATH = os.path.join(ROOT_DIRECTORY,"datasets")

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

MODEL_NAME = 'model.pkl'
SAVED_MODEL = os.path.join(ROOT_DIRECTORY,"trained_models")

TARGET_FEATURE = 'Loan_Status'

FEATURES = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area','Applicant_Income']

NUM_FEATURES = ['Applicant_Income', 'LoanAmount', 'Loan_Amount_Term']

CAT_FEATURES = ['Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Property_Area',
'Credit_History'
 ]

FEATURE_TO_MODIFY = 'ApplicantIncome'
FEATURE_TO_ADD = 'CoapplicantIncome'

DROP_FEATURE = 'CoapplicantIncome'
LOG_FEATURES = ['ApplicantIncome','LoanAmount']


