"""This module contains constants that will be used for Pipeline"""

import os
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACTS_DIR: str = os.path.join("artifacts", TIMESTAMP)

MODELS_DIRECTORY = "models"

"""
DATA INGESTION related constant
"""
DATA_INGESTION_ARTIFACTS_DIR: str = "DataIngestionArtifacts"
S3_ZIP_FILE_NAME: str = "data.zip"
CSV_DATA_FILE_NAME = "ner.csv"
DATA_BUCKET_NAME: str = "name-entity-recognition-data-28052024"


"""
DATA TRANSFORMATION related constant
"""
DATA_TRANSFORMATION_ARTIFACTS_DIR: str = "DataTransformationArtifacts"
LABELS_IDS_FILE_NAME: str = "labels_to_ids.pkl"
IDS_TO_LABELS_FILE_NAME: str = "ids_to_labels.pkl"
DF_TRAIN_FILE_NAME: str = "df_train.pkl"
DF_VAL_FILE_NAME:str = "df_val.pkl"
DF_TEST_FILE_NAME: str = "df_test.pkl"
UNIQUE_LABELS_FILE_NAME: str = "unique_labels.pkl"




"""
MODEL TRAINER related constant
"""
MODEL_TRAINING_ARTIFACTS_DIR = "ModelTrainingArtifacts"
BERT_MODEL_INSTANCE_NAME = "bert_model.pt"
TOKENIZER_FILE_NAME = "tokenizer.pkl"
MODEL_TRAINER_LEARNING_RATE = 5e-3
MODEL_TRAINER_NO_EPOCHS: int = 1
MODEL_TRAINER_BATCH_SIZE: int = 2


"""
MODEL PUSHER related constanta
"""
MODEL_EVALUATION_ARTIFACTS_DIR = "ModelEvaluationArtifacts"

"""
MODEL PUSHER related constants
"""
MODEL_BUCKET_NAME: str = "name-entity-recognition-model-30052024"


"""
APPLICATION Constants
"""
APP_HOST = "0.0.0.0"
APP_PORT = 8080