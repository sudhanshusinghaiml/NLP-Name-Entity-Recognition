"""This module contains configuration entity for all Components"""

from dataclasses import dataclass
import os
from src.name_entity_recognition.constants import *

@dataclass
class DataIngestionConfig:
    """Data Ingestion Configurations"""
    def __init__(self):
        self.data_ingestion_artifacts_dir = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)

        self.s3_data_file_path = os.path.join(self.data_ingestion_artifacts_dir, S3_ZIP_FILE_NAME)

        self.output_file_path = self.data_ingestion_artifacts_dir

        self.csv_data_file_path = os.path.join(self.data_ingestion_artifacts_dir, CSV_DATA_FILE_NAME)

        self.data_storage_bucket = DATA_BUCKET_NAME


@dataclass
class DataTransformationConfig:
    """Data Transformation Configurations"""
    def __init__(self):
        self.data_transformation_artifacts_dir = os.path.join(ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR)

        self.labels_to_ids_path = os.path.join(self.data_transformation_artifacts_dir, LABELS_TO_IDS_FILE_NAME)

        self.ids_to_labels_path = os.path.join(self.data_transformation_artifacts_dir, IDS_TO_LABELS_FILE_NAME)

        self.ids_to_labels_aws_path = os.path.join(self.data_transformation_artifacts_dir)

        self.df_train_path = os.path.join(self.data_transformation_artifacts_dir, DF_TRAIN_FILE_NAME)

        self.df_val_path = os.path.join(self.data_transformation_artifacts_dir, DF_VAL_FILE_NAME)

        self.df_test_path = os.path.join(self.data_transformation_artifacts_dir, DF_TEST_FILE_NAME)

        self.unique_labels_path = os.path.join(self.data_transformation_artifacts_dir, UNIQUE_LABELS_FILE_NAME)

        self.data_storage_bucket = DATA_BUCKET_NAME

        self.ids_to_labels_file_name = IDS_TO_LABELS_FILE_NAME

    
@dataclass
class ModelTrainingConfig:
    """Model Training Configurations"""
    def __init__(self):
        self.model_training_artifacts_dir = os.path.join(ARTIFACTS_DIR, MODEL_TRAINING_ARTIFACTS_DIR)

        self.bert_model_instance_path = os.path.join(self.model_training_artifacts_dir, AWS_MODEL_NAME)

        self.tokenizer_file_path = os.path.join(self.model_training_artifacts_dir, TOKENIZER_FILE_NAME)

        self.tokenizer_file_storage_path = os.path.join(self.model_training_artifacts_dir)

        self.model_training_learning_rate = MODEL_TRAINING_LEARNING_RATE
        self.model_training_epochs = MODEL_TRAINING_NO_EPOCHS
        self.model_training_batch_size = MODEL_TRAINING_BATCH_SIZE


@dataclass
class ModelEvaluationConfig:
    """Model Evaluation Configurations"""
    def __init__(self):
        self.model_evaluation_artifacts_dir = os.path.join(ARTIFACTS_DIR, MODEL_EVALUATION_ARTIFACTS_DIR)
        
        self.aws_model_path = os.getcwd()
        
        self.aws_local_path = AWS_MODEL_NAME


@dataclass
class ModelPusherConfig:
    """Model Pusher Configurations"""
    def __init__(self):
        self.bucket_name = MODEL_BUCKET_NAME
        self.model_name = AWS_MODEL_NAME
        self.upload_model_path = os.path.join(ARTIFACTS_DIR, MODEL_TRAINING_ARTIFACTS_DIR)


@dataclass
class ModelPredictorConfig:
    """Model Predictor Configurations"""
    def __init__(self):
        self.model_predictor_artifacts_dir = os.path.join(ARTIFACTS_DIR, MODEL_PREDICTOR_ARTIFACTS_DIR)
        self.tokenizer_local_path = os.path.join(self.model_predictor_artifacts_dir, TOKENIZER_FILE_NAME)
        self.ids_to_labels_local_path = os.path.join(self.model_predictor_artifacts_dir, IDS_TO_LABELS_FILE_NAME)
        self.best_model_dir = os.path.join(self.model_predictor_artifacts_dir, BEST_MODEL_DIRECTORY)
        self.best_model_path = os.path.join(self.best_model_dir, AWS_MODEL_NAME)